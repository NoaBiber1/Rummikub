"""Simulation layer: who plays, against whom, and what happens to the weights.

NOT AN ENTRY POINT. This module is driven entirely by a config dict handed in
from seed_sweep.py, which is the only runnable file in the project. There is
no main(), no argv parsing and no menu here.

    simulation(config)      the external API: `training_iterations` blocks of
                            (train for train_episodes_per_block, then test for
                            test_episodes_per_block)
    run_self_play_training  one training block: learner vs its ONE training
                            opponent, weights move
    run_test_simulation     one test block: learner vs ANY opponent, weights
                            frozen

Everything measured or printed lives in evaluation.py.
"""
import os
import random
import time
from collections import deque

import numpy as np
import torch

import evaluation as ev
import learning as learn
from game_env import GE
from greedy_alg import GreedySolution
from ilp_solution import validate_budget
from replay_buffer import ReplayBuffer
from q_model import MLP

online_net = None
opponent_net = None
target_net = None

# ONE solver for the whole process. __init__ enumerates 1173 melds and builds
# the LP variables (~13ms, measured); reset() is the only per-turn work, so
# rebuilding it per turn - or even per game - would be pure waste. It is
# stateful (reset() then solve()), hence single-threaded use only.
greedy_solver = GreedySolution()

# EVERY game in this project is 1-vs-1, training and testing alike.
# Training: the model and the single opponent it trains against (opponent_net,
# drawn from the snapshot pool).
# Testing: the model and ONE baseline. A test block exists to measure the main
# agent against an isolated baseline, so seating two baselines at the same
# table would have them play each other and score a game the model barely
# influenced. Opponents are benchmarked one after another, never together.
PLAYERS_PER_GAME = 2

# x = [board(53), hand(53), action(53)].
VECTOR_LEN = 53
HAND_START = VECTOR_LEN
ACTION_START = 2 * VECTOR_LEN

# [STALE-FIX] Exploration used to be split: the start (INITIAL_EPSILON=1.0)
# and floor (MIN_EPSILON=0.05) were hardcoded module constants and only the
# decay rate was a config key. All three are now config keys - DEFAULTS
# below ("epsilon", "epsilon_decay", "epsilon_min") - with these exact values
# as defaults, so an existing caller that doesn't set them trains identically
# to before. The names are kept here as a comment, not code, so nothing can
# import the old constants by accident.
#   old INITIAL_EPSILON -> DEFAULTS["epsilon"]      (1.0)
#   old MIN_EPSILON     -> DEFAULTS["epsilon_min"]  (0.05)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "online_net.pt")

# Every knob the loops read. A caller's config is merged over this, so
# seed_sweep.py only has to name what it actually varies.
DEFAULTS = dict(
    # --- simulation flow ---
    training_iterations=10,
    # 0 is legal and means "never train": the test blocks then measure the
    # network exactly as _build_nets initialised it. That is how the untrained
    # FLOOR is measured (see _cfg's flow-key validation). Anything > 0 trains
    # normally.
    train_episodes_per_block=100,
    test_episodes_per_block=20,
    # --- ILP action-set budget (game_env.GE -> ilp_solution.ILP_solutions) ---
    # [STALE-FIX] Used to be two hardcoded dicts in ilp_solution.py
    # (DEFAULT_BUDGET=24/4/4, TRAINING_BUDGET=12/2/2) with GE always
    # constructing ILP_solutions() with no argument - silently getting
    # TRAINING_BUDGET on every call this project ever actually made. Both
    # constants are gone; `budget` is a required, explicit argument the whole
    # way down (ILP_solutions.__init__ raises if it's missing). This default
    # is TRAINING_BUDGET's old values, so an existing config that doesn't set
    # "budget" behaves identically to before. See for_claude.md's budget
    # config section for the full propagation path and what max_actions/
    # alt_counts/alts_per_count each control (ilp_solution.py's
    # build_action_set). Validated in _cfg via ilp_solution.validate_budget
    # (shape + int-floor check; max_actions >= 1, alt_counts/alts_per_count
    # >= 0 - see validate_budget's docstring for why the two floors differ)
    # BEFORE any GE is built, not after.
    budget=dict(max_actions=12, alt_counts=2, alts_per_count=2),
    # --- learning (swept from seed_sweep.py) ---
    # Gradient updates per environment step (the replay ratio). tau is
    # compensated for this, so sweeping it varies the replay ratio ALONE and
    # not the target-drift-per-turn that used to ride along with it.
    # "DQN" or "DDQN" (0/1 also accepted) - learning.py owns the difference.
    learning_method="DQN",
    # 1 = standard 1-step TD. n > 1 propagates the terminal reward n turns
    # back per update instead of one - see the note in _play_episode.
    n_step=1,
    # Potential-based reward shaping (Ng et al. 1999). False reproduces an
    # unshaped run exactly. Provably policy-invariant, so this changes how
    # fast the agent learns, not what it should converge to.
    reward_shaping=True,
    gamma=0.99,
    lr=0.001,
    tau=0.005,
    updates_per_step=4,
    # Exploration schedule for the LEARNER only (select_x via main_policy in
    # run_self_play_training) - the training opponent's own rate is the
    # separate train_opponent_epsilon key below, and test blocks always play
    # greedily (epsilon fixed at 0.0, not a config key - see
    # run_test_simulation). epsilon is the starting value, epsilon_decay the
    # per-episode multiplicative decay, epsilon_min the floor:
    #   epsilon = max(epsilon_min, epsilon * epsilon_decay)   every episode
    # All three used to be a mix of a config key (epsilon_decay alone) and
    # two hardcoded module constants; promoted to config so any of the three
    # can be swept without editing this file. Validated together in _cfg
    # (0 <= epsilon_min <= epsilon <= 1, 0 < epsilon_decay <= 1).
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05,
    # --- replay ---
    # CAPACITY of the run's single replay buffer. simulation() builds one and
    # reuses it across every training block, so this is a sliding window over
    # the whole run's experience - roughly the most recent ~500 episodes at
    # ~40 turns/episode. [STALE-FIX] the buffer used to be rebuilt per block,
    # which capped occupancy at one block's worth and made every capacity above
    # that identical; see run_self_play_training.
    buffer_size=20000,
    min_buffer_size=500,
    batch_size=32,
    # --- self-play opponent pool ---
    opponent_update_every=50,
    opponent_pool_size=5,
    train_opponent_epsilon=0.0,
    # --- testing ---
    # A LIST: the model is scored against every one of these each test block,
    # sequentially and in isolation. A bare string is accepted and wrapped.
    test_opponents=("random",),
    eval_seed_base=0,
    # --- run-level ---
    seed=None,
    checkpoint_path=None,
    log_every=50,
    reward_window=200,
    quiet=True,
)


def _cfg(config):
    """Caller's config over DEFAULTS. Unknown keys raise: a typo'd knob that
    silently does nothing is the most expensive kind of experiment bug."""
    config = dict(config or {})
    unknown = set(config) - set(DEFAULTS) - {"name"}
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **config}
    # Validate the method HERE, not at the first update. Warmup means the
    # first train_step_batch is min_buffer_size episodes into the run, and a
    # sweep that dies there has already burned the cell.
    learn.resolve_learning_method(cfg["learning_method"])
    if not isinstance(cfg["n_step"], int) or isinstance(cfg["n_step"], bool) \
            or cfg["n_step"] < 1:
        raise ValueError(f"n_step must be an integer >= 1, got {cfg['n_step']!r}")
    # The three flow keys, checked here so a malformed run shape fails at
    # config time rather than as an IndexError several blocks in.
    #   training_iterations >= 1   at least one (train -> test) block, or the
    #                              run produces no measurement at all
    #   train_episodes_per_block >= 0
    #       0 IS LEGAL and is a supported configuration, not a degenerate one:
    #       it evaluates the network exactly as initialised, which is the
    #       untrained FLOOR every later "beats the floor by 2 SE" claim is
    #       measured against. simulation() carries epsilon forward with a guard
    #       for precisely this case.
    #   test_episodes_per_block >= 0
    #       0 skips measurement entirely - legal, but the run returns empty
    #       `blocks`, so anything scoring it must tolerate that.
    for key, floor in (("training_iterations", 1),
                       ("train_episodes_per_block", 0),
                       ("test_episodes_per_block", 0)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < floor:
            raise ValueError(
                f"{key} must be an integer >= {floor}, got {value!r}")
    # Validates updates_per_step (int >= 1) and tau (in [0,1]) together, since
    # the target-drift compensation is only defined for both.
    learn.effective_tau(cfg["tau"], cfg["updates_per_step"])
    # Same fail-loudly treatment for the exploration schedule: a floor above
    # the start, or a start/floor outside [0, 1], is a config that can never
    # decay the way epsilon_decay implies - catch it here, not 500+ episodes
    # into warmup.
    if not 0.0 <= cfg["epsilon_min"] <= cfg["epsilon"] <= 1.0:
        raise ValueError(
            f"epsilon config must satisfy 0 <= epsilon_min <= epsilon <= 1, "
            f"got epsilon={cfg['epsilon']!r}, epsilon_min={cfg['epsilon_min']!r}"
        )
    if not 0.0 < cfg["epsilon_decay"] <= 1.0:
        raise ValueError(
            f"epsilon_decay must lie in (0, 1], got {cfg['epsilon_decay']!r}"
        )
    # Cheap shape/floor check on the ILP action-set budget (no melds
    # generated - see validate_budget's docstring), run BEFORE any GE gets
    # built rather than letting a malformed budget surface as a KeyError deep
    # inside build_action_set on the first turn of the first episode.
    # Normalizes cfg["budget"] to exactly {max_actions, alt_counts,
    # alts_per_count} so every downstream GE(...) call gets a clean dict.
    cfg["budget"] = validate_budget(cfg["budget"])
    # buffer_size is the replay buffer's CAPACITY (deque(maxlen=buffer_size),
    # section 8) - validated here for the same reason as everything above:
    # a bad value should fail before any training time is spent, not surface
    # as a mysterious "loss is always NaN" 500 episodes into a sweep cell.
    if not isinstance(cfg["buffer_size"], int) or isinstance(cfg["buffer_size"], bool) \
            or cfg["buffer_size"] < 1:
        raise ValueError(
            f"buffer_size must be an integer >= 1, got {cfg['buffer_size']!r}")
    # buffer_size < min_buffer_size is not merely a bad tuning choice: the
    # buffer is a deque(maxlen=buffer_size), so len(replay_buffer) can never
    # exceed buffer_size. If that ceiling sits below min_buffer_size, the
    # `len(replay_buffer) < cfg["min_buffer_size"]` warmup check in
    # run_self_play_training's `update` NEVER passes - every batched update is
    # skipped for the ENTIRE run, silently: no exception, no NaN, just a
    # training loop that runs to completion having trained on nothing. Catch
    # it here instead of leaving it to be noticed as an oddly flat reward
    # curve after burning the full run.
    if cfg["buffer_size"] < cfg["min_buffer_size"]:
        raise ValueError(
            f"buffer_size ({cfg['buffer_size']!r}) must be >= min_buffer_size "
            f"({cfg['min_buffer_size']!r}) - otherwise warmup can never "
            f"complete and the run silently never trains"
        )
    return cfg


def _as_list(value):
    """One opponent or many. A bare string is a single opponent, not an
    iterable of characters."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return [value]
    return list(value)


def save_checkpoint(net, path=CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)


def load_checkpoint(net, path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        return False
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return True

# ---------------------------------------------------------------- policies
# A policy is just callable(valid_x_list) -> x. Everything anyone can play
# against reduces to one.

def select_x(valid_x_list, epsilon, net):
    """Epsilon-greedy over candidate inputs, scored by `net`. The net is an
    explicit argument, so the same function serves the learner and any
    model-backed opponent."""
    if np.random.rand() < epsilon:
        return valid_x_list[np.random.choice(len(valid_x_list))]

    with torch.no_grad():
        q_values = [
            net(x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)).item()
            for x in valid_x_list
        ]
    return valid_x_list[int(np.argmax(q_values)) if q_values else 0]


def random_opponent(valid_x_list):
    """Uniform over legal actions. The fixed, non-moving yardstick."""
    return valid_x_list[np.random.choice(len(valid_x_list))]


def greedy_opponent(valid_x_list):
    """Maximizes TILES PLACED, full stop. No Q-values, no net, no lookahead:
    this is a fixed baseline, and the moment it consulted a network it would
    stop being one.

    board/hand are identical across every candidate x, so the position is read
    off the first one, handed to the global greedy_solver via reset(), and the
    action segment of the X it returns matched back to the candidate carrying
    that action. solve() always returns a playable X - "nothing playable" is
    the all-zero draw action, not a None to branch on.

    [GOTCHA] The match can miss. GE's candidates come from ILP_solutions under
    a BUDGET (4.2 of the project notes) and break ties by point value, while
    greedy_alg maximizes tile count alone and breaks ties arbitrarily, so an
    equally-large action the solver picks may not be among the ones offered.
    The fallback is the richest action actually on the table - same tile
    count, so the greedy POLICY is unaffected; this is why it can't assert.

    [GOTCHA] .round() before slicing, exactly as GE.get_valid_x_list does
    before calling the ILP. Counts live in float tensors and
    GreedySolution.reset coerces with int(), which TRUNCATES - one accumulated
    0.9999999 would silently become a 0 tile and change the solver's answer.
    """
    position = torch.as_tensor(valid_x_list[0], dtype=torch.float32).round()
    greedy_solver.reset(hand_tails=position[HAND_START:ACTION_START],
                        board_tails=position[:HAND_START])
    # solve() returns a plain list (greedy_alg stays torch-free); the tensor
    # conversion belongs here, where torch is already a dependency.
    wanted = torch.as_tensor(greedy_solver.solve(), dtype=torch.float32)[ACTION_START:]

    for x in valid_x_list:
        if torch.equal(torch.as_tensor(x, dtype=torch.float32)[ACTION_START:], wanted):
            return x
    return max(valid_x_list, key=lambda x: float(torch.as_tensor(x)[ACTION_START:].sum()))


def model_opponent(net, epsilon=0.0):
    """Greedy (or epsilon-greedy) play through a net - a frozen checkpoint, a
    pool snapshot, or the live opponent_net."""
    net.eval()
    return lambda valid_x_list: select_x(valid_x_list, epsilon, net)


def _resolve_opponent(opponent, epsilon=0.0):
    """Accepts "random", "greedy", an nn.Module, a checkpoint path, or a
    ready-made policy callable. Returns (policy, label).

    [GOTCHA] Building an MLP here resets torch's global RNG (see the MLP note
    in the project file), so opponents are always resolved BEFORE the episode
    loop starts seeding - never inside it."""
    if isinstance(opponent, torch.nn.Module):
        return model_opponent(opponent, epsilon), "saved model"
    if isinstance(opponent, str):
        named = {"random": random_opponent, "greedy": greedy_opponent}
        if opponent.lower() in named:
            return named[opponent.lower()], opponent.lower()
        net = MLP()
        if not load_checkpoint(net, opponent):
            raise FileNotFoundError(f"no checkpoint at {opponent}")
        return model_opponent(net, epsilon), f"saved model ({opponent})"
    if callable(opponent):
        return opponent, getattr(opponent, "__name__", "custom opponent")
    raise TypeError(f"unsupported opponent: {opponent!r}")

# ------------------------------------------------------------- one episode

def _take_turn(ge, policy, valid_x_list=None):
    """One turn for whichever player is to move: get_valid_x_list -> policy ->
    play. Returns the x played, or None if there was nothing to play.

    `valid_x_list` is an OPTIONAL already-computed legal set for the state the
    game is in RIGHT NOW. None means "compute it" - the original behaviour and
    what every opponent turn still does. Passing one skips ge.get_valid_x_list(),
    which is the single most expensive call in the project (~3.45 ILP solves,
    ~66ms at budget 12/2/2), so it must be the legal set of the CURRENT mover in
    the CURRENT position, not a stale one. _play_episode is the only caller that
    passes it; see the note there for why the state cannot have moved.

    [GOTCHA] `[]` is a MEANINGFUL value here, not "no cache": it means the
    caller already established there is nothing legal. That is why the sentinel
    is None and the check below is `is None`, not falsiness - `if not
    valid_x_list: valid_x_list = ge.get_valid_x_list()` would re-run the solver
    on exactly the position where the answer is known and known to be empty.
    """
    if valid_x_list is None:
        valid_x_list = ge.get_valid_x_list()
    elif VERIFY_ACTION_CACHE:
        _assert_cache_matches(ge, valid_x_list)
    if not valid_x_list:
        return None
    chosen_x = policy(valid_x_list)
    ge.play(chosen_x)
    return chosen_x


# Off by default: the check costs a full extra get_valid_x_list (i.e. it
# restores exactly the duplicate solve the cache exists to remove), so it is a
# verification mode, not a safety net to leave on. Flip it for a short run
# after touching _play_episode's turn order or GE's state handling, confirm it
# stays silent, flip it back. Also exercised by
# test_action_cache_equivalence.py.
VERIFY_ACTION_CACHE = False


def _assert_cache_matches(ge, cached):
    """The invariant the cache rests on: the position has not moved since the
    cached list was built. Compares the [board|hand] prefix that every
    candidate x carries (ACTION_START = 2*VECTOR_LEN) against the live game."""
    fresh = ge.get_valid_x_list()
    if len(fresh) != len(cached):
        raise AssertionError(
            f"[VERIFY_ACTION_CACHE] cached legal set has {len(cached)} actions, "
            f"recomputing gives {len(fresh)} - the cached list does not belong "
            f"to this state")
    for a, b in zip(fresh, cached):
        if not torch.equal(torch.as_tensor(a)[:ACTION_START],
                           torch.as_tensor(b)[:ACTION_START]):
            raise AssertionError(
                "[VERIFY_ACTION_CACHE] cached candidates carry a different "
                "[board|hand] prefix than the live position")


def _opponent_turns(ge, policy, count):
    """`count` consecutive opponent turns, stopping early on a finished or
    stuck game."""
    for _ in range(count):
        if ge.is_Done() or _take_turn(ge, policy) is None:
            return


def _seed_all(seed):
    """All THREE RNG streams this project can consume.
    torch   GE.shuffle() - deck order, and therefore who gets which hand.
    numpy   select_x()'s epsilon-greedy draws and random_opponent.
    random  the stdlib module, which ReplayBuffer.sample() uses.
    [GOTCHA] `random` was NOT seeded until it was noticed that two runs of an
    identical config diverged from the first gradient step: batch COMPOSITION
    was coming from an unseeded stdlib RNG, so every "reproducible" training
    run was in fact reproducible only in its deck and its exploration, not in
    what it actually learned from. Seeding one or two of the three is worse
    than seeding none, because it looks controlled."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _seed_episode(seeds, itr):
    """Common random numbers: pin every RNG stream that affects this episode
    before it starts. Seeding a subset still lets the others diverge and
    silently reintroduces the variance this is meant to remove."""
    if seeds is None:
        return
    _seed_all(seeds[(itr - 1) % len(seeds)])


def _play_episode(ge, episode, main_player, main_policy, opponent_policy,
                  opponents, update, n_step=1, gamma=0.99, shaping=False):
    """One episode from the learner's seat, shared by both loops.

    `update(x, rewards, next_valid_x_list, done, episode) -> (loss, q_pred)` is
    called once per EMITTED transition and is the ONLY thing that differs
    between training and testing: it owns whatever happens to the weights, and
    its own sanity checking (training legitimately returns NaN during buffer
    warmup, which is not a fire-alarm condition).

    ONE SOLVER CALL PER STATE. The legal set built for s' after the opponents
    reply is handed straight to the next iteration's _take_turn instead of
    being rebuilt there. s' and "the state the learner acts from next" are the
    same state, so this is a pure de-duplication, not a cache with an
    invalidation policy - see the block comment at the bottom of the loop for
    the argument, and simulation.VERIFY_ACTION_CACHE for the switch that
    re-checks it at runtime.

    N-STEP AGGREGATION LIVES HERE, in `pending`. A turn's (x, reward) is held
    until n_step rewards have actually been observed after it; only then is it
    emitted, paired with the legal set at S_{t+n} - so no consumer can ever
    see a transition whose future hasn't happened yet, and the replay buffer
    physically cannot contain one. Doing it here rather than in the buffer
    means the test loop (which has no buffer) gets the same targets, and
    ReplayBuffer stays a dumb ring.

    WHY IT MATTERS IN THIS GAME: GE.get_reward returns 0 for every non-final
    turn (section 3's `not self.done` guard) - the entire signal is one number
    at the end. With n_step=1 that number moves backwards one turn per update,
    through a bootstrap chain that is mostly noise early on. n_step=n carries
    it n turns per update. This is the single biggest lever on how fast
    anything is learned here, which is why it's a config key.

    At the episode's end the remaining window is FLUSHED: each leftover
    transition is emitted with its truncated reward tail and the final
    (next_valid, done). Truncation is not a special case downstream - the
    target uses each sequence's own length as the discount exponent.

    REWARD SHAPING (`shaping`, config["reward_shaping"]). GE pays out only at
    the end, so the per-turn reward is optionally augmented with the
    potential-based term

        F(s, s') = gamma * PHI(s') - PHI(s)

    exactly as Ng et al. (1999) require, using GE.potential (which returns 0
    at terminal states - the condition the theorem needs). PHI(s) is sampled
    BEFORE the learner acts and PHI(s') after the opponents have replied, so
    s' is the next state the learner actually decides from: the same state
    whose legal set is stored in the transition.
    Three properties worth not breaking:
      - The sanity check runs on the TRUE reward, before shaping. A shaped
        reward legitimately exceeds MAX_POSSIBLE_REWARD's game-outcome bound,
        so checking the shaped value would either false-alarm or need a bound
        loose enough to stop catching anything.
      - record_episode still stores ge.get_reward(), unshaped, so
        episode_rewards and the vs-random / vs-greedy benchmarks keep
        measuring game outcomes and stay comparable across shaping settings.
      - PBRS composes with n-step for free: sum_j gamma^j F_{t+j} telescopes
        to gamma^k PHI(s_{t+k}) - PHI(s_t), so a k-step shaped return is
        itself a valid potential-based term. No special handling needed.

    Returns (losses, qs, turns). Emissions equal turns over the episode
    (everything pending is flushed), just delayed by up to n_step-1.
    """
    _opponent_turns(ge, opponent_policy, main_player)  # seats before the learner

    losses, qs, turns = [], [], 0
    pending = deque()
    last_next_valid_x_list = []
    # The learner's legal set for the state it is about to act from. None on
    # the first pass (nothing computed yet); afterwards it is the SAME list
    # object the transition was built against - see the note below.
    valid_x_list = None

    def emit(next_valid_x_list, done):
        x, _ = pending[0]
        rewards = [r for _, r in pending]
        loss, q_pred = update(x, rewards, next_valid_x_list, done, episode)
        losses.append(loss)
        qs.append(q_pred)
        pending.popleft()

    while not ge.is_Done():
        phi = ge.potential(main_player) if shaping else 0.0

        main_chosen_x = _take_turn(ge, main_policy, valid_x_list)
        if main_chosen_x is None:
            break

        _opponent_turns(ge, opponent_policy, opponents)

        next_valid_x_list = [] if ge.is_Done() else ge.get_valid_x_list()
        reward = ge.get_reward(main_player)
        ev.check_reward(reward, episode)  # the TRUE reward, before shaping
        if shaping:
            # PHI(s') is 0 at a terminal state, enforced inside GE.potential.
            reward = reward + gamma * ge.potential(main_player) - phi
        turns += 1

        pending.append((main_chosen_x, reward))
        last_next_valid_x_list = next_valid_x_list
        if len(pending) == n_step:
            emit(next_valid_x_list, ge.is_Done())

        # THE LEGAL SET COMPUTED ABOVE IS THE ONE THE NEXT TURN NEEDS. s' (the
        # state whose legal set goes into the transition) and the state the
        # learner acts from next are THE SAME STATE - the loop does nothing
        # between here and the next _take_turn that can move the game. What
        # runs in between is: ge.get_reward / ge.potential (both read-only),
        # ev.check_reward (pure), and `update`, which touches the replay buffer
        # and the nets, never the GE. So handing this list forward is the same
        # list get_valid_x_list would rebuild, not an approximation of it.
        #
        # This removes ~1 of every 3 solver calls per round: a round used to
        # cost the learner's set (in _take_turn), the opponent's set, then the
        # learner's set AGAIN (here). Now it costs two.
        #
        # [GOTCHA] Do NOT "simplify" this to `valid_x_list = next_valid_x_list
        # or None`. An empty next_valid_x_list only ever happens when the game
        # is done (GE.get_valid_x_list sets done itself when it comes back
        # empty, section 9), and `while not ge.is_Done()` then ends the loop -
        # so [] is never consumed. Mapping it to None would instead re-run the
        # solver on a finished game.
        valid_x_list = next_valid_x_list

    # GE.get_valid_x_list() sets done itself when it comes back empty, so a
    # `main_chosen_x is None` break leaves the game finished too - read done
    # now rather than trusting the snapshot from the last completed turn.
    done = ge.is_Done()
    while pending:
        emit([] if done else last_next_valid_x_list, done)

    return losses, qs, turns


def _won(ge, main_player):
    winner = ge.get_winner()
    return None if winner is None else (winner == main_player)


def _snapshot(net):
    """A pool entry must be a CLONE: state_dict() hands back references to the
    live parameter tensors, which the optimizer updates in place."""
    return {k: v.clone().detach() for k, v in net.state_dict().items()}

# ----------------------------------------------------------------- training

def run_self_play_training(config=None, epsilon=None, episodes=None,
                           replay_buffer=None):
    """ONE training block: train online_net by self-play against ONE opponent,
    opponent_net, which is periodically reloaded from a pool of past
    online_net snapshots.

    Always 2 seats (PLAYERS_PER_GAME) - a training episode is the learner and
    the thing it learns against, nothing else. Seat order alternates via
    `itr % 2` so the learner sees both first and second move. Every
    main-player transition goes to the replay buffer and, past warmup, drives
    updates_per_step batched updates.

    The DQN/DDQN choice is config["learning_method"] and the bootstrap horizon
    is config["n_step"]; both go straight through to train_step_batch - see
    learning.py for the target math and _play_episode for where the n-step
    window is accumulated.

    EPSILON. `epsilon=None` (the default) means "start of run": resolved to
    config["epsilon"] below, same as `episodes=None` resolving to
    config["train_episodes_per_block"]. Passing an explicit value is how the
    caller carries the schedule ACROSS blocks - simulation() feeds back
    history["epsilons"][-1] so exploration decays over the whole run instead
    of restarting at config["epsilon"] every block. Per-episode decay uses
    config["epsilon_decay"] and floors at config["epsilon_min"]; all three
    are validated together in _cfg.

    RETURNS `history` (a dict).
    """
    cfg = _cfg(config)
    episodes = cfg["train_episodes_per_block"] if episodes is None else episodes
    epsilon = cfg["epsilon"] if epsilon is None else epsilon
    quiet = cfg["quiet"]

    ge = GE(PLAYERS_PER_GAME, cfg["budget"])
    opponent_snapshots = [_snapshot(online_net)]
    # RUN-SCOPED, NOT BLOCK-SCOPED. simulation() builds one buffer and passes
    # the same object into every block; `None` means "standalone call", which
    # keeps this function usable on its own.
    #
    # [STALE-FIX] This used to be an unconditional
    # `ReplayBuffer(cfg["buffer_size"])` here, so the buffer was rebuilt at the
    # start of EVERY training block. Section 8 describes buffer_size as "the
    # most recent ~500 episodes: a genuine sliding window, not the whole run" -
    # which a per-block buffer cannot be. At the default 10 blocks x 100
    # episodes, a block only ever collected ~3,700 transitions, so every
    # capacity at or above that was identical and nothing was ever evicted:
    # buffer_size was not a live knob and a sweep over it compared three copies
    # of the same configuration. Rebuilding also re-paid min_buffer_size warmup
    # every block (~14 episodes of 100, ~14% of training spent not training)
    # and threw away every transition the previous block had paid an ILP solve
    # to collect.
    if replay_buffer is None:
        replay_buffer = ReplayBuffer(cfg["buffer_size"])
    history = ev.new_history()
    start_time = time.time()

    # epsilon is read at call time, not closure-creation time: it decays.
    def main_policy(valid_x_list):
        return select_x(valid_x_list, epsilon, online_net)

    def opponent_policy(valid_x_list):
        return select_x(valid_x_list, cfg["train_opponent_epsilon"], opponent_net)

    def update(x, rewards, next_valid_x_list, done, episode):
        # `rewards` is already an n-step sequence and next_valid_x_list is the
        # legal set n turns later (_play_episode), so nothing incomplete can
        # reach the buffer.
        replay_buffer.push(x, rewards, next_valid_x_list, done)
        if len(replay_buffer) < cfg["min_buffer_size"]:
            return float("nan"), float("nan")  # warmup: nothing trains yet
        # EXACTLY updates_per_step updates per environment step, each on an
        # independently sampled batch. Each one drifts the target by
        # effective_tau, so the whole step drifts it by cfg["tau"].
        step_losses, step_qs = [], []
        for _ in range(cfg["updates_per_step"]):
            batch = replay_buffer.sample(cfg["batch_size"])
            q_pred, loss = learn.train_step_batch(
                online_net, target_net, batch, cfg["gamma"], cfg["lr"], cfg["tau"],
                updates_per_step=cfg["updates_per_step"],
                n_step=cfg["n_step"], learning_method=cfg["learning_method"])
            ev.check_loss_and_q(loss, q_pred, episode)
            step_losses.append(loss)
            step_qs.append(q_pred)
        return float(np.mean(step_losses)), float(np.mean(step_qs))

    for itr in range(1, episodes + 1):
        ge.reset()
        main_player = itr % PLAYERS_PER_GAME

        losses, qs, turns = _play_episode(
            ge, itr, main_player, main_policy, opponent_policy, 1, update,
            n_step=cfg["n_step"], gamma=cfg["gamma"],
            shaping=cfg["reward_shaping"])

        epsilon = max(cfg["epsilon_min"], epsilon * cfg["epsilon_decay"])
        ev.record_episode(history, itr, float(ge.get_reward(main_player)),
                          _won(ge, main_player), losses, qs, epsilon, turns)

        if not quiet and (itr % cfg["log_every"] == 0 or itr == episodes):
            ev.print_progress(itr, episodes, history, cfg["reward_window"], cfg["log_every"])

        if itr % cfg["opponent_update_every"] == 0:
            opponent_snapshots.append(_snapshot(online_net))
            if len(opponent_snapshots) > cfg["opponent_pool_size"]:
                opponent_snapshots.pop(0)
            opponent_net.load_state_dict(
                opponent_snapshots[np.random.randint(len(opponent_snapshots))])
            opponent_net.eval()

    if not quiet:
        elapsed = time.time() - start_time
        print(f"[training block] {episodes} episodes in {elapsed:.1f}s "
              f"({episodes / max(elapsed, 1e-9):.2f} episodes/s)")

    return history

# ------------------------------------------------------------------ testing

def _test_block(net, cfg, opponent_policy, episodes, seeds):
    """ONE test block: the main agent against ONE resolved baseline, 1-vs-1
    for `episodes` games. Returns a history."""
    ge = GE(PLAYERS_PER_GAME, cfg["budget"])
    history = ev.new_history()

    def main_policy(valid_x_list):
        return select_x(valid_x_list, 0.0, net)

    def update(x, rewards, next_valid_x_list, done, episode):
        q_pred, loss = learn.train_step(
            net, target_net, x, rewards, next_valid_x_list, done,
            cfg["gamma"], cfg["lr"], cfg["tau"],
            # updates_per_step is left at 1: this path never soft-updates
            # (skeep_progress=True below), so there is no drift to compensate.
            n_step=cfg["n_step"],
            learning_method=cfg["learning_method"],
            skeep_progress=True,  # compute loss/Q, skip the update
        )
        ev.check_loss_and_q(loss, q_pred, episode)
        return loss, q_pred

    for itr in range(1, episodes + 1):
        _seed_episode(seeds, itr)
        ge.reset()
        main_player = itr % PLAYERS_PER_GAME

        losses, qs, turns = _play_episode(
            ge, itr, main_player, main_policy, opponent_policy, 1, update,
            n_step=cfg["n_step"], gamma=cfg["gamma"],
            shaping=cfg["reward_shaping"])

        ev.record_episode(history, itr, float(ge.get_reward(main_player)),
                          _won(ge, main_player), losses, qs, 0.0, turns)

    return history


def run_test_simulation(net, config=None, opponents=None, episodes=None,
                        seeds=None):
    """ONE test block PER OPPONENT: play `net` GREEDILY (epsilon fixed at 0)
    for `episodes` games against each opponent in turn. No replay buffer, no
    weight updates, no opponent pool, no epsilon.

    `opponents` defaults to config["test_opponents"] and is a LIST; a bare
    string is accepted and wrapped. Each entry is anything _resolve_opponent
    understands: "random", "greedy", an nn.Module, a checkpoint path, or a
    policy callable.

    STRICTLY SEQUENTIAL, STRICTLY 1-vs-1: main agent vs opponents[0] for
    `episodes` games, then main agent vs opponents[1], and so on. Two
    baselines are never at the same table - a game between random and greedy
    measures them against each other, not the model, and would contaminate
    the very number this phase exists to produce.

    RETURNS {label: history}, keyed by _resolve_opponent's label - one
    complete, separately-scored history per opponent, never pooled. Pooling
    them would average a baseline the agent beats with one it loses to and
    report the mean as skill.

    Loss/Q are still computed per turn (train_step with skeep_progress=True) so
    the fire-alarm checks cover testing too and the history schema matches
    training's - but nothing is written back to any net. learning_method and
    n_step are passed through as well: the reported loss should be measured
    against the same target the run is training towards, or it isn't the same
    number.

    COMMON RANDOM NUMBERS: seeds defaults to
    range(eval_seed_base, eval_seed_base + episodes) and the SAME list is
    reused for every opponent, so vs-random and vs-greedy are played on
    identical decks. Every comparison in the project is therefore paired:
    across opponents, across blocks, across training seeds, across configs.
    Pass an explicit `seeds` list if you deliberately want an unpaired sample.
    """
    global target_net

    cfg = _cfg(config)
    episodes = cfg["test_episodes_per_block"] if episodes is None else episodes
    opponents = _as_list(cfg["test_opponents"] if opponents is None else opponents)

    net.eval()
    # Resolve every opponent BEFORE the seeded loop: a checkpoint entry
    # constructs an MLP, which resets torch's global RNG and would otherwise
    # shift the decks of whichever opponent happened to come after it.
    resolved = [_resolve_opponent(o) for o in opponents]
    if target_net is None:
        target_net = MLP()
        target_net.load_state_dict(net.state_dict())
    target_net.eval()

    if seeds is None:
        seeds = list(range(cfg["eval_seed_base"], cfg["eval_seed_base"] + episodes))

    histories = {}
    for policy, label in resolved:
        histories[label] = _test_block(net, cfg, policy, episodes, seeds)
        if not cfg["quiet"]:
            ev.print_summary(histories[label], f"test vs {label} opponent")
    return histories


# --------------------------------------------------------------------- API

def _merge_history(combined, block):
    """Append a block's per-episode lists onto the run-level history.
    test_evals is skipped: that series is BLOCK-level, owned by simulation()
    below, and no single block produces it."""
    for key, values in block.items():
        if key != "test_evals":
            combined[key].extend(values)


def _build_nets(seed):
    """online/opponent/target, opponent and target synced to online's initial
    weights. Neither is ever trained directly - opponent_net only via
    load_state_dict from the snapshot pool, target_net only via
    soft_update_target inside the train steps.

    [GOTCHA] MLP's seed is passed EXPLICITLY. Its default (42) would make
    every "random" init identical across sweep seeds, quietly collapsing the
    run-to-run variance seed_sweep.py exists to measure."""
    nets = [MLP() if seed is None else MLP(seed=seed) for _ in range(3)]
    online, opponent, target = nets
    for net in (opponent, target):
        net.load_state_dict(online.state_dict())
        net.eval()
    return online, opponent, target


def simulation(config):
    """The external API. Runs config["training_iterations"] blocks of
    (train train_episodes_per_block episodes -> test test_episodes_per_block
    episodes) and returns everything a caller needs to score the run.

    Interleaving train and test in blocks IS the periodic-benchmark mechanism:
    the learner is measured against a fixed opponent on a fixed set of decks
    at regular intervals, so the test curve is comparable across blocks and
    across configs. There is deliberately no second benchmark buried inside
    the training loop.

    Owns the nets and the RNG: seeds both streams from config["seed"] once,
    then builds the three nets. Training runs free-running from there; only
    the TEST blocks reseed per episode (CRN, see run_test_simulation).

    Testing is MULTI-OPPONENT but never multi-player: every entry of
    config["test_opponents"] gets its own 1-vs-1 block of
    test_episodes_per_block games against the main agent, run one after
    another, scored separately and kept separately end to end. Beating random is table stakes; beating greedy
    is the first result that says the net learned something a one-line
    heuristic doesn't already do, and a single pooled number would hide
    exactly that distinction.

    RETURNS {"config", "train_history", "test_histories", "blocks"}, where
    `test_histories` and `blocks` are both keyed by opponent label:
    {label: [one per block]}. `blocks` entries are evaluation.summarize()
    dicts tagged with the cumulative training-episode count.
    """
    global online_net, opponent_net, target_net

    cfg = _cfg(config)
    quiet = cfg["quiet"]

    if cfg["seed"] is not None:
        _seed_all(cfg["seed"])
    online_net, opponent_net, target_net = _build_nets(cfg["seed"])

    train_history = ev.new_history()
    test_histories = {}
    blocks = {}
    epsilon = cfg["epsilon"]
    # ONE buffer for the whole run, handed to every block (section 8). Built
    # here rather than inside run_self_play_training so that buffer_size means
    # what section 8 says it means - a sliding window over recent EXPERIENCE -
    # instead of resetting every block and capping out at one block's worth.
    replay_buffer = ReplayBuffer(cfg["buffer_size"])

    for block in range(1, cfg["training_iterations"] + 1):
        _merge_history(train_history, run_self_play_training(
            config, epsilon=epsilon, replay_buffer=replay_buffer))
        # Carry exploration forward: the schedule spans the whole run, not one
        # block. record_episode stored the post-decay value every episode.
        #
        # The guard is for train_episodes_per_block=0, the "evaluate an
        # untrained net" configuration: no episode ran, so no epsilon was
        # recorded, and [-1] on the empty list used to raise IndexError before
        # the first test block ever executed.
        if train_history["epsilons"]:
            epsilon = train_history["epsilons"][-1]

        episodes_so_far = len(train_history["episode_rewards"])
        if not quiet:
            print(f"[block {block}/{cfg['training_iterations']}] epsilon={epsilon:.3f}")

        for label, history in run_test_simulation(online_net, config).items():
            test_histories.setdefault(label, []).append(history)
            stats = {"episode": episodes_so_far, "opponent": label,
                     **ev.summarize(history)}
            blocks.setdefault(label, []).append(stats)
            train_history["test_evals"].append(stats)
            if not quiet:
                ev.print_eval_checkpoint(episodes_so_far, stats, label)

        if not quiet:
            ev.plot_training_curves(
                train_history,
                title=f"{cfg.get('name', 'run')} (block {block}/{cfg['training_iterations']})",
                reward_window=cfg["reward_window"],
            )

    if cfg["checkpoint_path"]:
        save_checkpoint(online_net, cfg["checkpoint_path"])
        if not quiet:
            print(f"[checkpoint saved to {cfg['checkpoint_path']}]")

    return {
        "config": cfg,
        "train_history": train_history,
        "test_histories": test_histories,
        "blocks": blocks,
    }