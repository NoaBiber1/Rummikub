"""STEP 0 - the baseline every later performance claim is measured against.

Runs one fixed, small, fully-seeded config end to end through
simulation() and records what it cost: wall time, main-player turns, and
the two solver counters (GE.get_valid_x_list and ILP_solutions._solve).

    python bench_step0.py --label before      # unmodified tree
    python bench_step0.py --label after       # after a change
    python bench_step0.py --compare before after

COMPARE THE RATES (ms/turn, solves/turn), NOT wall_seconds: episodes end
when a rack empties, so anything that changes the policy changes the
turn count and a shorter game would read as a speedup. Results go to
checkpoints/bench/<label>.json. The config is FIXED - editing it
invalidates every saved baseline. This is not a benchmark of training
quality and must never be read as one.
"""
import argparse
import json
import os
import time

STEP0_CONFIG = dict(
    name="step0",
    training_iterations=2,
    train_episodes_per_block=15,
    test_episodes_per_block=5,
    test_opponents=["random", "greedy"],
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
    seed=0,
    eval_seed_base=10_000,
)


class Counters:
    """Work counters filled by the instrumented call sites."""

    def __init__(self):
        """Start every counter at zero."""
        self.valid_x_calls = 0
        self.ilp_solves = 0
        self.greedy_solves = 0
        self.ilp_seconds = 0.0
        self.turns = 0
        self.episodes = 0


def instrument(counters):
    """Wrap the four call sites that do or count solver work. Returns undo().

    Turns and episodes are counted here, off _play_episode's own return
    value, since a run now reports one row per block and no turn count.
    """
    import game_env
    import greedy_alg
    import ilp_solution
    import simulation

    ge_orig = game_env.GE.get_valid_x_list
    ilp_orig = ilp_solution.ILP_solutions._solve
    greedy_orig = greedy_alg.GreedySolution.solve
    episode_orig = simulation._play_episode

    def ge_wrapped(self):
        """Count a legal-set build and time it."""
        counters.valid_x_calls += 1
        t0 = time.perf_counter()
        try:
            return ge_orig(self)
        finally:
            counters.ilp_seconds += time.perf_counter() - t0

    def ilp_wrapped(self, *a, **k):
        """Count one ILP solve."""
        counters.ilp_solves += 1
        return ilp_orig(self, *a, **k)

    def greedy_wrapped(self, *a, **k):
        """Count one greedy solve."""
        counters.greedy_solves += 1
        return greedy_orig(self, *a, **k)

    def episode_wrapped(*a, **k):
        """Count one episode and its turns."""
        losses, turns = episode_orig(*a, **k)
        counters.episodes += 1
        counters.turns += turns
        return losses, turns

    game_env.GE.get_valid_x_list = ge_wrapped
    ilp_solution.ILP_solutions._solve = ilp_wrapped
    greedy_alg.GreedySolution.solve = greedy_wrapped
    simulation._play_episode = episode_wrapped

    def undo():
        """Restore every wrapped call site."""
        game_env.GE.get_valid_x_list = ge_orig
        ilp_solution.ILP_solutions._solve = ilp_orig
        greedy_alg.GreedySolution.solve = greedy_orig
        simulation._play_episode = episode_orig

    return undo


def measure(config=None):
    """Run the step-0 config under instrumentation; returns the stats dict."""
    import simulation as sim

    cfg = dict(STEP0_CONFIG if config is None else config)
    counters = Counters()
    undo = instrument(counters)
    try:
        t0 = time.perf_counter()
        result = sim.simulation(cfg)
        wall = time.perf_counter() - t0
    finally:
        undo()

    turns, episodes = counters.turns, counters.episodes
    solves = max(counters.ilp_solves, 1)
    blocks = len(result["train"]["blocks"]) if result["train"] else 0
    return {
        "wall_seconds": round(wall, 3),
        "ilp_seconds": round(counters.ilp_seconds, 3),
        "ilp_share_pct": round(100.0 * counters.ilp_seconds / wall, 1),
        "turns": turns,
        "episodes": episodes,
        "blocks": blocks,
        "valid_x_calls": counters.valid_x_calls,
        "ilp_solves": counters.ilp_solves,
        "greedy_solves": counters.greedy_solves,
        "ms_per_turn": round(1000.0 * wall / max(turns, 1), 2),
        "ms_per_solve": round(1000.0 * counters.ilp_seconds / solves, 2),
        "valid_x_calls_per_turn": round(counters.valid_x_calls / max(turns, 1), 3),
        "solves_per_valid_x_call": round(solves / max(counters.valid_x_calls, 1), 3),
        "solves_per_turn": round(solves / max(turns, 1), 3),
        "config": cfg,
    }


def _out_path(label):
    """Path of a saved measurement label, created if needed."""
    import simulation as sim
    d = os.path.join(sim.CHECKPOINT_DIR, "bench")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{label}.json")


KEYS = ("wall_seconds", "ilp_seconds", "ilp_share_pct", "turns", "episodes",
        "valid_x_calls", "ilp_solves", "ms_per_turn", "ms_per_solve",
        "valid_x_calls_per_turn", "solves_per_turn")


def report(stats, label):
    """Print one measurement."""
    print(f"\n=== step-0 baseline [{label}] ===")
    for k in KEYS:
        print(f"  {k:<24} {stats[k]}")
    print()


def compare(a_label, b_label):
    """Print a saved A-vs-B table, flagging a differing turn count."""
    a = json.load(open(_out_path(a_label)))
    b = json.load(open(_out_path(b_label)))
    if a["config"] != b["config"]:
        print("!! configs differ - these two runs are NOT comparable\n")
    print(f"\n=== {a_label} -> {b_label} ===")
    print(f"  {'metric':<24} {a_label:>12} {b_label:>12} {'change':>12}")
    for k in KEYS:
        x, y = a[k], b[k]
        change = f"{(y - x) / x * 100:+.1f}%" if x else "n/a"
        print(f"  {k:<24} {x:>12} {y:>12} {change:>12}")
    if a["turns"] != b["turns"]:
        print("\n  NOTE: turn counts differ, so the two runs did not play the "
              "same games.\n  Compare ms_per_turn and solves_per_turn, not "
              "wall_seconds - and if the\n  change was meant to be "
              "behaviour-preserving, this is a red flag.")
    print()


def main():
    """CLI: measure and save a label, or compare two saved ones."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="before", help="name this measurement")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="print a saved A-vs-B table and exit")
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    stats = measure()
    path = _out_path(args.label)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    report(stats, args.label)
    print(f"[saved {path}]")


if __name__ == "__main__":
    main()