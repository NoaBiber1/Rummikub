"""STEP 0 - the baseline every later performance claim is measured against.

Runs one fixed, small, fully-seeded config end to end through simulation() and
records what it cost: wall time split by phase, main-player turns, and the two
counters that matter for solver work - GE.get_valid_x_list() calls and
ILP_solutions._solve() calls.

WHY A SEPARATE HARNESS AND NOT "time python seed_sweep.py". Wall time alone
cannot tell a real speedup from a shorter game. Episodes here end when a rack
empties, so anything that changes the policy changes the number of turns, and
turns are what the solver is paid per. Reporting ms/TURN and solves/TURN
alongside the total makes a regression that merely shortened the games visible
instead of looking like a win. The solve counter also pins down WHERE a change
landed: fix 1 should move calls/turn and leave ms/solve alone; an in-process
solver would do the opposite.

USAGE
    python bench_step0.py --label before          # on the unmodified tree
    python bench_step0.py --label after           # after a change
    python bench_step0.py --compare before after

Results go to checkpoints/bench/<label>.json. The config below is FIXED
deliberately - editing it invalidates every previously saved baseline, so if
you must, use a new label namespace rather than overwriting.

[GOTCHA] This is not a benchmark of training quality and must never be read as
one. 30 training episodes is far too few to learn anything; the run exists to
be cheap, deterministic and representative of the per-turn work mix (both the
warmup and post-warmup replay paths, and both a train and a test block).
"""
import argparse
import json
import os
import time

# The baseline config. Small, seeded, and crosses the min_buffer_size warmup
# threshold part-way through so both replay paths are exercised.
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
    quiet=True,
)


class Counters:
    def __init__(self):
        self.valid_x_calls = 0
        self.ilp_solves = 0
        self.greedy_solves = 0
        self.ilp_seconds = 0.0


def instrument(counters):
    """Wrap the three call sites that do solver work. Returns an undo()."""
    import game_env
    import greedy_alg
    import ilp_solution

    ge_orig = game_env.GE.get_valid_x_list
    ilp_orig = ilp_solution.ILP_solutions._solve
    greedy_orig = greedy_alg.GreedySolution.solve

    def ge_wrapped(self):
        counters.valid_x_calls += 1
        t0 = time.perf_counter()
        try:
            return ge_orig(self)
        finally:
            counters.ilp_seconds += time.perf_counter() - t0

    def ilp_wrapped(self, *a, **k):
        counters.ilp_solves += 1
        return ilp_orig(self, *a, **k)

    def greedy_wrapped(self, *a, **k):
        counters.greedy_solves += 1
        return greedy_orig(self, *a, **k)

    game_env.GE.get_valid_x_list = ge_wrapped
    ilp_solution.ILP_solutions._solve = ilp_wrapped
    greedy_alg.GreedySolution.solve = greedy_wrapped

    def undo():
        game_env.GE.get_valid_x_list = ge_orig
        ilp_solution.ILP_solutions._solve = ilp_orig
        greedy_alg.GreedySolution.solve = greedy_orig

    return undo


def _turns(result):
    """Main-player turns over the whole run: every training episode plus every
    test episode of every opponent. history["turns"] is one entry per episode
    (evaluation.record_episode)."""
    total = sum(result["train_history"]["turns"])
    for histories in result["test_histories"].values():
        for h in histories:
            total += sum(h["turns"])
    return total


def measure(config=None):
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

    turns = _turns(result)
    # [GOTCHA] len(history) is the number of KEYS in the history dict, not the
    # number of episodes. Every per-episode list works, but "turns" is the one
    # _turns() already reads, so use it here too and keep the two counts
    # sourced from the same place.
    episodes = (len(result["train_history"]["turns"])
                + sum(len(h["turns"])
                      for hs in result["test_histories"].values() for h in hs))
    solves = max(counters.ilp_solves, 1)
    return {
        "wall_seconds": round(wall, 3),
        "ilp_seconds": round(counters.ilp_seconds, 3),
        "ilp_share_pct": round(100.0 * counters.ilp_seconds / wall, 1),
        "turns": turns,
        "episodes": episodes,
        "valid_x_calls": counters.valid_x_calls,
        "ilp_solves": counters.ilp_solves,
        "greedy_solves": counters.greedy_solves,
        # the rates - these are the numbers to compare, not wall_seconds
        "ms_per_turn": round(1000.0 * wall / max(turns, 1), 2),
        "ms_per_solve": round(1000.0 * counters.ilp_seconds / solves, 2),
        "valid_x_calls_per_turn": round(counters.valid_x_calls / max(turns, 1), 3),
        "solves_per_valid_x_call": round(solves / max(counters.valid_x_calls, 1), 3),
        # the headline for fix 1
        "solves_per_turn": round(solves / max(turns, 1), 3),
        "config": cfg,
    }


def _out_path(label):
    import simulation as sim
    d = os.path.join(sim.CHECKPOINT_DIR, "bench")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{label}.json")


KEYS = ("wall_seconds", "ilp_seconds", "ilp_share_pct", "turns", "episodes",
        "valid_x_calls", "ilp_solves", "ms_per_turn", "ms_per_solve",
        "valid_x_calls_per_turn", "solves_per_turn")


def report(stats, label):
    print(f"\n=== step-0 baseline [{label}] ===")
    for k in KEYS:
        print(f"  {k:<24} {stats[k]}")
    print()


def compare(a_label, b_label):
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
