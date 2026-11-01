"""Simulation layer: who plays, against whom, and what happens to the
weights.

NOT AN ENTRY POINT - driven entirely by a config dict from seed_sweep.py.
No main(), no argv, no menu.

    simulation(config)      training_iterations blocks of (train, test)
    run_self_play_training  one training block; weights move
    run_test_simulation     one test block per opponent; weights frozen

Nothing is reported while a block runs: each loop feeds an evaluation.py
accumulator and returns ONE record when the block ends.
"""
import os
import random
from collections import deque

import numpy as np
import torch

import evaluation as ev
import learning as learn
from game_env import GE
from greedy_alg import GreedySolution
from ilp_solution import validate_budget
from replay_buffer import ReplayBuffer
from q_model import MLP, Q_INIT_MODES as MLP_INIT_MODES

online_net = None
opponent_net = None
target_net = None

greedy_solver = GreedySolution()

PLAYERS_PER_GAME = 2

VECTOR_LEN = 53
HAND_START = VECTOR_LEN
ACTION_START = 2 * VECTOR_LEN


CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "online_net.pt")

_REQUIRED = object()

AUTO = "auto"

TRAIN_OPPONENT_KINDS = ("self", "greedy", "random")

DEFAULTS = dict(
    training_iterations=_REQUIRED,
    train_episodes_per_block=_REQUIRED,
    test_episodes_per_block=_REQUIRED,
    budget=_REQUIRED,
    learning_method=_REQUIRED,
    n_step=_REQUIRED,
    reward_shaping=_REQUIRED,
    gamma=_REQUIRED,
    lr=_REQUIRED,
    tau=_REQUIRED,
    updates_per_step=_REQUIRED,
    epsilon=_REQUIRED,
    epsilon_decay=_REQUIRED,
    epsilon_min=_REQUIRED,
    buffer_size=_REQUIRED,
    min_buffer_size=500,
    batch_size=128,
    huber_delta=None,
    grad_clip=None,
    target_clip=None,
    q_hidden_dim=256,
    q_features=True,
    q_layer_norm=True,
    q_init="kaiming",
    opponent_update_every=50,
    opponent_pool_size=5,
    train_opponent_epsilon=0.0,
    train_opponent_mix=None,
    test_opponents=_REQUIRED,
    eval_seed_base=_REQUIRED,
    seed=None,
    checkpoint_path=None,
)


def _cfg(config):
    """Caller's config merged over DEFAULTS, validated, and returned.

    Unknown keys raise ('name' tolerated). Values are checked at CONFIG time
    rather than at first use, because warmup puts the first update
    min_buffer_size episodes into a cell that has already burned its time.
    """
    config = dict(config or {})
    unknown = set(config) - set(DEFAULTS) - {"name"}
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **config}
    learn.resolve_learning_method(cfg["learning_method"])
    if not isinstance(cfg["n_step"], int) or isinstance(cfg["n_step"], bool) \
            or cfg["n_step"] < 1:
        raise ValueError(f"n_step must be an integer >= 1, got {cfg['n_step']!r}")
    for key, floor in (("training_iterations", 1),
                       ("train_episodes_per_block", 0),
                       ("test_episodes_per_block", 0)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < floor:
            raise ValueError(
                f"{key} must be an integer >= {floor}, got {value!r}")
    if not (cfg["train_episodes_per_block"] or cfg["test_episodes_per_block"]):
        raise ValueError(
            "train_episodes_per_block and test_episodes_per_block are both 0 - "
            "the run would neither train nor measure anything")
    learn.effective_tau(cfg["tau"], cfg["updates_per_step"])
    if not 0.0 <= cfg["epsilon_min"] <= cfg["epsilon"] <= 1.0:
        raise ValueError(
            f"epsilon config must satisfy 0 <= epsilon_min <= epsilon <= 1, "
            f"got epsilon={cfg['epsilon']!r}, epsilon_min={cfg['epsilon_min']!r}"
        )
    if not 0.0 < cfg["epsilon_decay"] <= 1.0:
        raise ValueError(
            f"epsilon_decay must lie in (0, 1], got {cfg['epsilon_decay']!r}"
        )
    cfg["budget"] = validate_budget(cfg["budget"])
    if not isinstance(cfg["buffer_size"], int) or isinstance(cfg["buffer_size"], bool) \
            or cfg["buffer_size"] < 1:
        raise ValueError(
            f"buffer_size must be an integer >= 1, got {cfg['buffer_size']!r}")
    if cfg["buffer_size"] < cfg["min_buffer_size"]:
        raise ValueError(
            f"buffer_size ({cfg['buffer_size']!r}) must be >= min_buffer_size "
            f"({cfg['min_buffer_size']!r}) - otherwise warmup can never "
            f"complete and the run silently never trains"
        )
    cfg["target_clip"] = _resolve_target_clip(cfg)
    for key in ("huber_delta", "grad_clip", "target_clip"):
        learn._validate_positive(cfg[key], key)
    if not isinstance(cfg["q_hidden_dim"], int) \
            or isinstance(cfg["q_hidden_dim"], bool) or cfg["q_hidden_dim"] < 1:
        raise ValueError(
            f"q_hidden_dim must be an integer >= 1, got {cfg['q_hidden_dim']!r}")
    for key in ("q_features", "q_layer_norm"):
        if not isinstance(cfg[key], bool):
            raise TypeError(f"{key} must be a bool, got {cfg[key]!r}")
    if cfg["q_init"] not in MLP_INIT_MODES:
        raise ValueError(
            f"q_init must be one of {MLP_INIT_MODES}, got {cfg['q_init']!r}")
    cfg["train_opponent_mix"] = _resolve_opponent_mix(cfg["train_opponent_mix"])
    _check_seed_separation(cfg)
    return cfg


def _check_seed_separation(cfg):
    """Raise when the training seed lands inside the eval seed range.

    The two streams are separate by construction - training is seeded
    once and never again, evaluation is pinned per episode - but they are
    drawn from one integer line, and a training seed that collides with
    an eval seed puts a training deal and a test deal on the same shuffle.
    That is not a crash, it is a quiet leak of the eval set into training,
    and the config that caused it would still be reported as clean.
    """
    seed, base = cfg["seed"], cfg["eval_seed_base"]
    episodes = cfg["test_episodes_per_block"]
    if seed is None or not episodes:
        return
    if not isinstance(base, int) or isinstance(base, bool):
        raise TypeError(f"eval_seed_base must be an integer, got {base!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer or None, got {seed!r}")
    if base <= seed < base + episodes:
        raise ValueError(
            f"training seed {seed} lies inside the evaluation seed range "
            f"[{base}, {base + episodes}) - training and evaluation draw "
            f"from one integer line, so this trains on a deck the agent is "
            f"then scored on. Move `seed` or `eval_seed_base` apart")


def _resolve_target_clip(cfg):
    """`target_clip` as a number or None, with 'auto' resolved HERE.

    Resolved at config time rather than at first use, so `full_config` -
    and therefore the cache fingerprint - records the bound that was
    actually applied instead of the word that stood for it.

    'auto' is the largest value the discounted return can physically take:
    the zero-sum payoff over a deck worth MAX_DECK_VALUE, on get_reward's
    1/100 scale. With shaping on, the shaped return also carries a
    potential term bounded by the same quantity, so the budget doubles.
    """
    value = cfg["target_clip"]
    if value != AUTO:
        return value
    bound = ev.MAX_POSSIBLE_REWARD
    return 2.0 * bound if cfg["reward_shaping"] else bound


def _resolve_opponent_mix(mix):
    """Validate a training-opponent mix and return it as normalised weights.

    None means pure self-play, which is what the pipeline has always done.
    A dict maps any of TRAIN_OPPONENT_KINDS to a non-negative weight;
    'self' is the snapshot pool, the other two are the fixed baselines.
    Normalising here means the recorded config shows the sampling
    probabilities rather than whatever arbitrary scale they were written on.
    """
    if mix is None:
        return None
    if not isinstance(mix, dict):
        raise TypeError(
            f"train_opponent_mix must be None or a dict over "
            f"{TRAIN_OPPONENT_KINDS}, got {mix!r}")
    unknown = sorted(set(mix) - set(TRAIN_OPPONENT_KINDS))
    if unknown:
        raise KeyError(
            f"train_opponent_mix has unknown opponent kinds {unknown} - "
            f"expected any of {list(TRAIN_OPPONENT_KINDS)}")
    for kind, weight in mix.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                or weight < 0 or weight != weight:
            raise ValueError(
                f"train_opponent_mix[{kind!r}] must be a number >= 0, "
                f"got {weight!r}")
    total = float(sum(mix.values()))
    if total <= 0:
        raise ValueError(
            f"train_opponent_mix weights sum to {total} - at least one "
            f"opponent kind has to be reachable or no training game has an "
            f"opponent at all")
    return {kind: float(mix[kind]) / total
            for kind in TRAIN_OPPONENT_KINDS if mix.get(kind, 0) > 0}


def resolved_config(config):
    """The full config a run will use: DEFAULTS under `config`, validated.

    The same dict `simulation()` returns as result['config'], exposed so a
    caller can resolve and validate a cell WITHOUT running it - which is
    what lets seed_sweep fingerprint a cell for the cache before deciding
    whether it has to be launched at all.
    """
    return _cfg(config)


def _as_list(value):
    """One opponent or many. A bare string is a single opponent, not an
    iterable of characters.
    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return [value]
    return list(value)


def save_checkpoint(net, path=CHECKPOINT_PATH):
    """Save a net's state dict to `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)


def load_checkpoint(net, path=CHECKPOINT_PATH):
    """Load a state dict into `net`; returns False when the path is absent."""
    if not os.path.exists(path):
        return False
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return True


def select_x(valid_x_list, epsilon, net):
    """Epsilon-greedy over candidate inputs, scored by `net`. The net is an
    explicit argument, so one function serves the learner and any
    model-backed opponent.

    The candidate set is scored in ONE batched forward pass rather than one
    pass per candidate. A decision costs ~7 candidate evaluations and every
    turn of every training and test game makes one, so this is the same
    arithmetic through a shape torch is actually built for.

    > [GOTCHA] batching is not bit-identical to looping: a batched matmul
    > accumulates in a different order, so two candidates that used to tie
    > exactly can now differ in the last ULP and `argmax` can pick the other
    > one. With the network as it was initialised - Q spread 7e-5 across a
    > whole candidate set - that was a real risk. With a properly scaled
    > init it is not, but a comparison across this change is still a
    > comparison across a behaviour change.
    """
    if not valid_x_list:
        raise ValueError(
            "select_x got an empty candidate list - an empty legal set means "
            "the game is over and _take_turn is supposed to return before "
            "reaching a policy")
    if np.random.rand() < epsilon:
        return valid_x_list[np.random.choice(len(valid_x_list))]

    with torch.no_grad():
        candidates = torch.stack([
            x if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=torch.float32)
            for x in valid_x_list
        ])
        q_values = net(candidates).reshape(-1)
    return valid_x_list[int(torch.argmax(q_values))]


def random_opponent(valid_x_list):
    """Uniform over legal actions. The fixed, non-moving yardstick."""
    return valid_x_list[np.random.choice(len(valid_x_list))]


def greedy_opponent(valid_x_list):
    """Maximizes TILES PLACED, full stop - no Q-values, no net, no lookahead.

    Reads the position off the first candidate, solves it with the global
    greedy_solver and matches the returned action segment back to a
    candidate. The match can miss (GE's candidates come from a budgeted
    ILP that breaks ties by value, greedy_alg breaks them arbitrarily), so
    it falls back to the richest candidate on offer - same tile count, so
    the policy is unaffected.
    """
    position = torch.as_tensor(valid_x_list[0], dtype=torch.float32).round()
    greedy_solver.reset(hand_tails=position[HAND_START:ACTION_START],
                        board_tails=position[:HAND_START])
    wanted = torch.as_tensor(greedy_solver.solve(), dtype=torch.float32)[ACTION_START:]

    for x in valid_x_list:
        if torch.equal(torch.as_tensor(x, dtype=torch.float32)[ACTION_START:], wanted):
            return x
    return max(valid_x_list, key=lambda x: float(torch.as_tensor(x)[ACTION_START:].sum()))


def model_opponent(net, epsilon=0.0):
    """Greedy (or epsilon-greedy) play through a net: a frozen checkpoint, a
    pool snapshot, or the live opponent_net.
    """
    net.eval()
    return lambda valid_x_list: select_x(valid_x_list, epsilon, net)


def net_kwargs(cfg):
    """The architecture arguments every MLP in a run is built from.

    One place, so the online net, the target net, the self-play opponent
    and any net loaded from a checkpoint cannot end up with different
    shapes - a mismatch that surfaces as a load_state_dict error at best
    and as a silently different opponent at worst.
    """
    return dict(hidden_dim=cfg["q_hidden_dim"], features=cfg["q_features"],
                layer_norm=cfg["q_layer_norm"], init=cfg["q_init"])


def _resolve_opponent(opponent, epsilon=0.0, arch=None):
    """Resolve 'random', 'greedy', an nn.Module, a checkpoint path or a
    callable to (policy, label). Building an MLP here reseeds torch
    globally, so callers resolve BEFORE a seeded episode loop.

    `arch` is net_kwargs(cfg); a checkpoint is loaded into a net built to
    the run's own architecture, so a checkpoint saved under a different one
    fails loudly at load_state_dict instead of quietly playing as something
    else.
    """
    if isinstance(opponent, torch.nn.Module):
        return model_opponent(opponent, epsilon), "saved model"
    if isinstance(opponent, str):
        named = {"random": random_opponent, "greedy": greedy_opponent}
        if opponent.lower() in named:
            return named[opponent.lower()], opponent.lower()
        net = MLP(**(arch or {}))
        if not load_checkpoint(net, opponent):
            raise FileNotFoundError(f"no checkpoint at {opponent}")
        return model_opponent(net, epsilon), f"saved model ({opponent})"
    if callable(opponent):
        return opponent, getattr(opponent, "__name__", "custom opponent")
    raise TypeError(f"unsupported opponent: {opponent!r}")


def _take_turn(ge, policy, valid_x_list=None):
    """One turn for whichever player is to move: legal set, policy, play.
    Returns the x played, or None if there was nothing to play.

    `valid_x_list` is an optional already-computed legal set for the state
    the game is in RIGHT NOW; None means compute it. The sentinel is None
    and the test is `is None`, because [] is meaningful - the caller has
    already established that nothing is legal.
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


VERIFY_ACTION_CACHE = False


def _assert_cache_matches(ge, cached):
    """VERIFY_ACTION_CACHE check: the cached legal set still belongs to the
    live position, compared on the [board | hand] prefix every candidate
    carries.
    """
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
    stuck game.
    """
    for _ in range(count):
        if ge.is_Done() or _take_turn(ge, policy) is None:
            return


def _seed_all(seed):
    """Seed all THREE RNG streams: torch (deck order), numpy (exploration and
    random_opponent), stdlib random (replay batch composition). Seeding a
    subset is worse than seeding none, because it looks controlled.

    Two callers only: `simulation()` once at the start of a run, and
    `_seed_episode` inside a test block, whose effect never outlives the
    block. Do not add a third in the training path.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _seed_episode(seeds, itr):
    """Common random numbers: pin every stream for episode `itr` from a seed
    list, before the deal.

    Called from the TEST loop only. Training never reseeds: see
    `_rng_state` for why the two are not allowed to share a stream.
    """
    if seeds is None:
        return
    _seed_all(seeds[(itr - 1) % len(seeds)])


def _rng_state():
    """A snapshot of all THREE RNG streams, for the eval phase to restore.

    Evaluation has to pin its streams per episode (common random numbers
    are what make every comparison in this project paired), and training
    must never be reseeded at all - a training stream that restarts from
    a fixed point every time a test block fires deals the SAME hands in
    every block after the first, which is a data-diversity bug, not a
    reproducibility feature.

    Both requirements hold at once only if the eval phase puts back what
    it found. `run_test_simulation` snapshots here, seeds freely, and
    restores in a `finally`, so training sees one uninterrupted stream
    from `simulation()`'s single `_seed_all` to the end of the run and
    the deal a training episode gets no longer depends on the eval
    cadence, the opponent list, or how many decisions the last test game
    happened to make.
    """
    return (torch.get_rng_state(), np.random.get_state(), random.getstate())


def _restore_rng_state(state):
    """Put back a `_rng_state` snapshot, all three streams."""
    torch_state, numpy_state, python_state = state
    torch.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def _play_episode(ge, episode, main_player, main_policy, opponent_policy,
                  opponents, update, n_step=1, gamma=0.99, shaping=False):
    """One episode from the learner's seat, shared by both loops.
    Returns (losses, turns).

    `update(x, rewards, next_valid_x_list, done, episode) -> loss` is the
    ONLY thing that differs between training and testing; it owns the
    weights and its own sanity checking.

    Owns two things besides the turn order. N-STEP AGGREGATION: a turn is
    held in `pending` until n_step rewards have been observed after it, so
    no consumer can see a transition whose future has not happened; the
    window is flushed at the end with truncated tails. REWARD SHAPING (when
    `shaping`): the per-turn reward becomes r + gamma*PHI(s') - PHI(s), with
    PHI(s) sampled before the learner acts and PHI(s') after the opponents
    reply.
    """
    _opponent_turns(ge, opponent_policy, main_player)

    losses, turns = [], 0
    pending = deque()
    last_next_valid_x_list = []
    valid_x_list = None

    def emit(next_valid_x_list, done):
        """Emit the oldest pending transition with its reward window."""
        x, _ = pending[0]
        rewards = [r for _, r in pending]
        losses.append(update(x, rewards, next_valid_x_list, done, episode))
        pending.popleft()

    while not ge.is_Done():
        phi = ge.potential(main_player) if shaping else 0.0

        main_chosen_x = _take_turn(ge, main_policy, valid_x_list)
        if main_chosen_x is None:
            break

        _opponent_turns(ge, opponent_policy, opponents)

        next_valid_x_list = [] if ge.is_Done() else ge.get_valid_x_list()
        reward = ge.get_reward(main_player)
        ev.check_reward(reward, episode)
        if shaping:
            reward = reward + gamma * ge.potential(main_player) - phi
        turns += 1

        pending.append((main_chosen_x, reward))
        last_next_valid_x_list = next_valid_x_list
        if len(pending) == n_step:
            emit(next_valid_x_list, ge.is_Done())

        valid_x_list = next_valid_x_list

    done = ge.is_Done()
    while pending:
        emit([] if done else last_next_valid_x_list, done)

    return losses, turns


def _won(ge, main_player):
    """True / False / None: did the main player win?"""
    winner = ge.get_winner()
    return None if winner is None else (winner == main_player)


def _snapshot(net):
    """A detached CLONE of a net's state dict - state_dict() alone hands back
    references the optimizer keeps updating in place.
    """
    return {k: v.clone().detach() for k, v in net.state_dict().items()}


def run_self_play_training(config=None, epsilon=None, episodes=None,
                           replay_buffer=None, block=1, episode_offset=0):
    """ONE training block: online_net vs opponent_net, a periodically reloaded
    snapshot of itself. Always 2 seats, alternating on itr % 2.

    `epsilon=None` means start of run; passing a value is how simulation()
    carries the schedule ACROSS blocks. `replay_buffer=None` builds a local
    one, which is only for standalone calls - the run's buffer is owned by
    simulation(). `block` and `episode_offset` only label the row.

    Returns (train block record, the epsilon the schedule reached).
    """
    cfg = _cfg(config)
    episodes = cfg["train_episodes_per_block"] if episodes is None else episodes
    epsilon = cfg["epsilon"] if epsilon is None else epsilon

    ge = GE(PLAYERS_PER_GAME, cfg["budget"])
    opponent_snapshots = [_snapshot(online_net)]
    if replay_buffer is None:
        replay_buffer = ReplayBuffer(cfg["buffer_size"])
    block_log = ev.TrainBlockAccumulator()

    # The learner is the ONLY net that is ever in training mode. A test
    # block and `model_opponent` both put a net into eval, and a net left
    # there is silently a different function the moment this trunk grows a
    # dropout or batchnorm layer, so the mode is asserted here rather than
    # assumed to have survived the previous phase.
    online_net.train()
    opponent_net.eval()
    if target_net is not None:
        target_net.eval()

    def main_policy(valid_x_list):
        """The learner's move: epsilon-greedy through online_net.

        epsilon is read at CALL time, so the policy tracks the decay.
        """
        return select_x(valid_x_list, epsilon, online_net)

    def self_policy(valid_x_list):
        """The training opponent's move, through opponent_net."""
        return select_x(valid_x_list, cfg["train_opponent_epsilon"], opponent_net)

    mix = cfg["train_opponent_mix"]
    policies_by_kind = {"self": self_policy, "greedy": greedy_opponent,
                        "random": random_opponent}
    mix_kinds = list(mix) if mix else []
    mix_weights = [mix[k] for k in mix_kinds] if mix else []

    def pick_opponent_policy():
        """This episode's training opponent.

        Pure self-play when `train_opponent_mix` is None, which is what the
        pipeline has always done. With a mix, the kind is drawn PER EPISODE
        and not per turn, so a game is played against one opponent from
        start to finish and the transitions in it describe a coherent
        adversary rather than a chimera that changes identity mid-game.
        """
        if not mix_kinds:
            return self_policy
        return policies_by_kind[
            mix_kinds[int(np.random.choice(len(mix_kinds), p=mix_weights))]]

    def update(x, rewards, next_valid_x_list, done, episode):
        """Store the transition and, past warmup, run updates_per_step batched
        updates on independently sampled batches. Returns the step's mean loss,
        or NaN during warmup - no update happened, so there is no loss.
        """
        replay_buffer.push(x, rewards, next_valid_x_list, done)
        if len(replay_buffer) < cfg["min_buffer_size"]:
            return float("nan")
        step_losses = []
        for _ in range(cfg["updates_per_step"]):
            batch = replay_buffer.sample(cfg["batch_size"])
            q_pred, loss = learn.train_step_batch(
                online_net, target_net, batch, cfg["gamma"], cfg["lr"], cfg["tau"],
                updates_per_step=cfg["updates_per_step"],
                n_step=cfg["n_step"], learning_method=cfg["learning_method"],
                huber_delta=cfg["huber_delta"], grad_clip=cfg["grad_clip"],
                target_clip=cfg["target_clip"])
            ev.check_loss_and_q(loss, q_pred, episode)
            step_losses.append(loss)
        return float(np.mean(step_losses))

    for itr in range(1, episodes + 1):
        ge.reset()
        main_player = itr % PLAYERS_PER_GAME

        losses, _turns = _play_episode(
            ge, itr, main_player, main_policy, pick_opponent_policy(), 1, update,
            n_step=cfg["n_step"], gamma=cfg["gamma"],
            shaping=cfg["reward_shaping"])

        epsilon = max(cfg["epsilon_min"], epsilon * cfg["epsilon_decay"])
        block_log.add_episode(losses)

        if itr % cfg["opponent_update_every"] == 0:
            opponent_snapshots.append(_snapshot(online_net))
            if len(opponent_snapshots) > cfg["opponent_pool_size"]:
                opponent_snapshots.pop(0)
            opponent_net.load_state_dict(
                opponent_snapshots[np.random.randint(len(opponent_snapshots))])
            opponent_net.eval()

    return block_log.record(block=block,
                            episode=episode_offset + episodes), epsilon


def _test_block(net, cfg, opponent_policy, label, episodes, seeds,
                block=1, episode=0):
    """ONE test block: the main agent vs ONE resolved baseline, 1-vs-1 for
    `episodes` games. Returns one evaluation.py test-block record.

    avg_reward is the mean TRUE terminal payoff over the games that actually
    finished, always unshaped; win_rate is over decided games. Loss and Q are
    still computed and checked so the fire alarm covers testing, but nothing
    is written back to any net and neither number is reported.
    """
    ge = GE(PLAYERS_PER_GAME, cfg["budget"])
    block_log = ev.TestBlockAccumulator(label)

    def main_policy(valid_x_list):
        """The agent's move: greedy through `net`."""
        return select_x(valid_x_list, 0.0, net)

    def update(x, rewards, next_valid_x_list, done, episode_index):
        """Compute loss/Q for the fire alarm without touching any weights."""
        q_pred, loss = learn.train_step(
            net, target_net, x, rewards, next_valid_x_list, done,
            cfg["gamma"], cfg["lr"], cfg["tau"],
            n_step=cfg["n_step"],
            learning_method=cfg["learning_method"],
            skeep_progress=True,
            huber_delta=cfg["huber_delta"], grad_clip=cfg["grad_clip"],
            target_clip=cfg["target_clip"],
        )
        ev.check_loss_and_q(loss, q_pred, episode_index)
        return loss

    for itr in range(1, episodes + 1):
        _seed_episode(seeds, itr)
        ge.reset()
        main_player = itr % PLAYERS_PER_GAME

        _play_episode(ge, itr, main_player, main_policy, opponent_policy, 1,
                      update, n_step=cfg["n_step"], gamma=cfg["gamma"],
                      shaping=cfg["reward_shaping"])

        block_log.add_game(float(ge.get_reward(main_player)),
                           _won(ge, main_player), ge.is_Done(), episode=itr)

    return block_log.record(block=block, episode=episode)


def run_test_simulation(net, config=None, opponents=None, episodes=None,
                        seeds=None, block=1, episode=0):
    """ONE test block PER OPPONENT, each `episodes` games, played greedily.
    No buffer, no updates, no pool, no epsilon.

    `opponents` defaults to config['test_opponents'] and is a list; each
    entry is anything _resolve_opponent understands. Strictly sequential and
    strictly 1-vs-1: two baselines are never at the same table. Returns
    {label: record}, never pooled.

    COMMON RANDOM NUMBERS: seeds default to
    range(eval_seed_base, eval_seed_base + episodes) and the SAME list is
    reused for every opponent, so every comparison in the project is paired.
    Every episode pins all three streams before its deal, so the first pin
    lands immediately before the first test game of the phase.

    RNG ISOLATION: that pinning, and any net this function builds, are
    undone on the way out - the whole phase runs inside a snapshot taken
    at entry (see `_rng_state`). Evaluation is reproducible AND training
    is never reseeded, which used to be mutually exclusive here.
    """
    global target_net

    cfg = _cfg(config)
    episodes = cfg["test_episodes_per_block"] if episodes is None else episodes
    opponents = _as_list(cfg["test_opponents"] if opponents is None else opponents)

    # Everything from here to the `finally` may seed a stream or flip a
    # net's mode: `_resolve_opponent` builds an MLP for a checkpoint path
    # (and MLP.__init__ seeds torch GLOBALLY), `_test_block` pins all three
    # streams before every game, and scoring needs `net` in eval. None of
    # that is allowed to escape into the training phase, so the state is
    # snapshotted first and put back unconditionally.
    entry_rng = _rng_state()
    was_training = net.training
    try:
        net.eval()
        arch = net_kwargs(cfg)
        resolved = [_resolve_opponent(o, arch=arch) for o in opponents]
        if target_net is None:
            target_net = MLP(seed=cfg["seed"] if cfg["seed"] is not None else 42,
                             **arch)
            target_net.load_state_dict(net.state_dict())
        target_net.eval()

        if seeds is None:
            seeds = list(
                range(cfg["eval_seed_base"], cfg["eval_seed_base"] + episodes))

        records = {}
        for policy, label in resolved:
            if label in records:
                raise ValueError(
                    f"two test opponents resolved to the same label {label!r} - "
                    f"the second block would overwrite the first, and the config "
                    f"would silently be measured against one fewer baseline")
            records[label] = _test_block(net, cfg, policy, label, episodes, seeds,
                                         block=block, episode=episode)
    finally:
        net.train(was_training)
        _restore_rng_state(entry_rng)
    return records


def _build_nets(cfg):
    """online/opponent/target nets, the latter two synced to online's initial
    weights. MLP's seed is passed EXPLICITLY: its default would make every
    init in a sweep identical and collapse the variance being measured.

    All three are built from net_kwargs(cfg), so the architecture a sweep
    varies reaches every net a run holds.
    """
    seed = cfg["seed"]
    arch = net_kwargs(cfg)
    nets = [MLP(**arch) if seed is None else MLP(seed=seed, **arch)
            for _ in range(3)]
    online, opponent, target = nets
    for net in (opponent, target):
        net.load_state_dict(online.state_dict())
        net.eval()
    return online, opponent, target


def simulation(config):
    """The external API: training_iterations blocks of (train, then test).

    Owns the nets, the RNG seeding, the run's single replay buffer and the
    epsilon schedule across blocks. Returns {'config', 'schema', 'train',
    'test'} - the merged config actually used plus evaluation.RunLog's dict,
    validated before it is returned. Either phase is None when it never ran.
    Prints nothing.
    """
    global online_net, opponent_net, target_net

    cfg = _cfg(config)

    # Nets FIRST, then the one and only seeding of the run. MLP.__init__
    # calls torch.manual_seed itself, so building three nets resets the
    # torch stream three times; seeding after them means the training
    # stream starts at exactly `seed` instead of at whatever the last
    # weight draw left behind. The weights are unaffected - each MLP seeds
    # itself - and from this line to the end of the run NOTHING reseeds
    # training. Test blocks pin and restore their own streams.
    online_net, opponent_net, target_net = _build_nets(cfg)
    if cfg["seed"] is not None:
        _seed_all(cfg["seed"])

    run_log = ev.RunLog()
    epsilon = cfg["epsilon"]
    episodes_so_far = 0
    replay_buffer = ReplayBuffer(cfg["buffer_size"])

    for block in range(1, cfg["training_iterations"] + 1):
        if cfg["train_episodes_per_block"]:
            record, epsilon = run_self_play_training(
                config, epsilon=epsilon, replay_buffer=replay_buffer,
                block=block, episode_offset=episodes_so_far)
            episodes_so_far = record["episode"]
            run_log.add_train_block(record)

        if cfg["test_episodes_per_block"]:
            for record in run_test_simulation(
                    online_net, config, block=block,
                    episode=episodes_so_far).values():
                run_log.add_test_block(record)

    if cfg["checkpoint_path"]:
        save_checkpoint(online_net, cfg["checkpoint_path"])

    return {"config": cfg, **run_log.to_dict()}