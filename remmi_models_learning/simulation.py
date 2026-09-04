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
from q_model import MLP

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

DEFAULTS = dict(
    training_iterations=10,
    train_episodes_per_block=100,
    test_episodes_per_block=20,
    budget=dict(max_actions=12, alt_counts=2, alts_per_count=2),
    learning_method="DQN",
    n_step=1,
    reward_shaping=True,
    gamma=0.99,
    lr=0.001,
    tau=0.005,
    updates_per_step=4,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05,
    buffer_size=20000,
    min_buffer_size=500,
    batch_size=32,
    opponent_update_every=50,
    opponent_pool_size=5,
    train_opponent_epsilon=0.0,
    test_opponents=("random",),
    eval_seed_base=0,
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
    return cfg


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
    """
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


def _resolve_opponent(opponent, epsilon=0.0):
    """Resolve 'random', 'greedy', an nn.Module, a checkpoint path or a
    callable to (policy, label). Building an MLP here reseeds torch
    globally, so callers resolve BEFORE a seeded episode loop.
    """
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
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _seed_episode(seeds, itr):
    """Common random numbers: pin every stream for episode `itr` from a seed
    list, before the deal.
    """
    if seeds is None:
        return
    _seed_all(seeds[(itr - 1) % len(seeds)])


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

    def main_policy(valid_x_list):
        """The learner's move: epsilon-greedy through online_net.

        epsilon is read at CALL time, so the policy tracks the decay.
        """
        return select_x(valid_x_list, epsilon, online_net)

    def opponent_policy(valid_x_list):
        """The training opponent's move, through opponent_net."""
        return select_x(valid_x_list, cfg["train_opponent_epsilon"], opponent_net)

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
                n_step=cfg["n_step"], learning_method=cfg["learning_method"])
            ev.check_loss_and_q(loss, q_pred, episode)
            step_losses.append(loss)
        return float(np.mean(step_losses))

    for itr in range(1, episodes + 1):
        ge.reset()
        main_player = itr % PLAYERS_PER_GAME

        losses, _turns = _play_episode(
            ge, itr, main_player, main_policy, opponent_policy, 1, update,
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
    """
    global target_net

    cfg = _cfg(config)
    episodes = cfg["test_episodes_per_block"] if episodes is None else episodes
    opponents = _as_list(cfg["test_opponents"] if opponents is None else opponents)

    net.eval()
    resolved = [_resolve_opponent(o) for o in opponents]
    if target_net is None:
        target_net = MLP()
        target_net.load_state_dict(net.state_dict())
    target_net.eval()

    if seeds is None:
        seeds = list(range(cfg["eval_seed_base"], cfg["eval_seed_base"] + episodes))

    records = {}
    for policy, label in resolved:
        if label in records:
            raise ValueError(
                f"two test opponents resolved to the same label {label!r} - "
                f"the second block would overwrite the first, and the config "
                f"would silently be measured against one fewer baseline")
        records[label] = _test_block(net, cfg, policy, label, episodes, seeds,
                                     block=block, episode=episode)
    return records


def _build_nets(seed):
    """online/opponent/target nets, the latter two synced to online's initial
    weights. MLP's seed is passed EXPLICITLY: its default would make every
    init in a sweep identical and collapse the variance being measured.
    """
    nets = [MLP() if seed is None else MLP(seed=seed) for _ in range(3)]
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

    if cfg["seed"] is not None:
        _seed_all(cfg["seed"])
    online_net, opponent_net, target_net = _build_nets(cfg["seed"])

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