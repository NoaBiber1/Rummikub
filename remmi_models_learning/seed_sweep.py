"""
THE ENTRY POINT. Everything in this project is run from here: simulation.py
is a pure module with no main(), and every knob it reads comes from the config
dicts below.

Compare multiple training configs (DQN vs DDQN, n_step, different
gamma/lr/tau/updates_per_step/epsilon/epsilon_decay/epsilon_min/budget/
buffer_size, or in future different
architectures / replay strategies) with enough seeds and enough statistical care to trust the
result.

Every cell is scored against EVERY opponent in test_opponents (random,
greedy, ...) on the same decks, and the results are kept separate: beating
random is table stakes, beating greedy is the first real result, and a pooled
number would hide which of the two just changed.

Two variance-reduction pieces work together here:
  - Each config is trained with >=3 different TRAINING seeds (run-to-run DQN
    variance is large; one seed per config is how people convince themselves
    of effects that aren't there).
  - Every test block uses the SAME fixed eval seed list across every config
    and seed (common random numbers - see simulation.run_test_simulation), so
    deck luck is paired out of the config-vs-config comparison rather than
    averaged over.

This intentionally does NOT run a full factorial grid. Screen candidates with
a reduced episode budget first; promote only finalists to a full-length run -
see section 2 / section 10.3 of the project notes for why.

Usage (from the project root, with both the project root and agent/ on
PYTHONPATH - the same layout simulation.py itself requires):

    python seed_sweep.py

Edit CONFIGS below to add/remove configs and BASE_CONFIG for anything shared.
Results are written as JSON to checkpoints/sweep/<config_name>/seed<seed>.json
and a summary table is printed (and returned) at the end.
"""
import json
import os

import numpy as np

import parallel_sweep as ps
import simulation as sim
from simulation import simulation

# --- shared by every config; override any key per-config below -------------
# The three flow keys define the shape of a run: training_iterations blocks of
# (train_episodes_per_block training games -> test_episodes_per_block test
# games). Total training games per cell = iterations * train_episodes_per_block.
BASE_CONFIG = dict(
    training_iterations=10,
    train_episodes_per_block=100,
    test_episodes_per_block=20,
    # ILP action-set budget for game_env.GE's solver (ilp_solution.py).
    # max_actions: hard cap on the returned action list. alt_counts: how many
    # top commitment-ladder rungs get tier-2 alternatives. alts_per_count:
    # alternatives per rung. [STALE-FIX] Used to be two hardcoded dicts in
    # ilp_solution.py (DEFAULT_BUDGET=24/4/4, TRAINING_BUDGET=12/2/2) with
    # GE always silently getting TRAINING_BUDGET; both are gone, budget is
    # now required all the way down to ILP_solutions.__init__. These are
    # TRAINING_BUDGET's old values, so leaving this out of a per-config dict
    # reproduces the old always-used behavior exactly. See
    # simulation.DEFAULTS["budget"] and for_claude.md's budget config
    # section for the full propagation path and the actions/ms trade-off
    # table (richer budget = more actions per turn, more ILP solve time).
    budget=dict(max_actions=12, alt_counts=2, alts_per_count=2),
    # "DQN" (target net both picks and prices the next action) or "DDQN"
    # (online net picks, target net prices - less overestimation bias). 0/1
    # also accepted. See learning.py.
    learning_method="DQN",
    # Bootstrap horizon. 1 = standard 1-step TD. Rewards here are 0 until the
    # game ends, so n_step controls how many turns the one real number travels
    # back per update - a strong candidate axis to sweep.
    n_step=1,
    # Potential-based reward shaping (Ng et al. 1999) with PHI = score
    # differential. Policy-invariant, so this is a LEARNING-SPEED axis, not a
    # different objective: sweep it, don't assume it.
    reward_shaping=True,
    gamma=0.99,
    lr=0.001,
    tau=0.005,          # target drift per ENVIRONMENT STEP, not per update
    # Gradient updates per environment step (the replay ratio). tau below is
    # the drift budget PER TURN and is compensated for this value, so changing
    # it varies the replay ratio alone - see simulation/learning notes.
    updates_per_step=4,
    # Exploration schedule for the learner: start value, per-episode decay,
    # and floor. All three are explicit config keys (simulation.DEFAULTS
    # carries the same values as its own defaults, so leaving any of them out
    # of a per-config dict below still reproduces the old hardcoded
    # INITIAL_EPSILON/MIN_EPSILON behavior exactly) - see the note in
    # simulation.py next to DEFAULTS and for_claude.md section 5.1.
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05,
    # Replay buffer CAPACITY (a fresh-per-training-block deque, section 8).
    # [STALE-FIX] Was already a simulation.DEFAULTS key with no hardcoded
    # fallback in the buffer's own construction - `ReplayBuffer(cfg[
    # "buffer_size"])` was already dynamic - but it wasn't explicitly named
    # here, so a sweep could vary every other knob without ever touching the
    # one that sets how much experience the agent replays from. 20000 is
    # simulation.DEFAULTS' value (same default either way), roughly the most
    # recent ~500 episodes at ~35-45 turns/episode - a sliding window, not the
    # whole run: early experience from a much weaker policy should age out.
    # MUST stay >= min_buffer_size (simulation.DEFAULTS, 500) - simulation._cfg
    # now enforces this (13), since a buffer_size below the warmup threshold
    # means len(buffer) can never reach it and the run silently never trains.
    buffer_size=20000,
    # Each is benchmarked against the main agent in its own 1-vs-1 block,
    # sequentially. Baselines never meet each other. Cost is linear in the
    # length of this list.
    test_opponents=["random", "greedy"],
    quiet=True,
)

# Which opponent's avg reward ranks the configs. The others are still measured
# and printed - this only decides the sort order, and picking one explicitly
# beats letting list order decide it silently.
SCORE_OPPONENT = "random"

# --- edit this to add/remove configs ---------------------------------------

# One axis at a time (see the docstring), so swap the list rather than crossing
# it with the one above:
#   CONFIGS = [dict(name="dqn", learning_method="DQN"),
#              dict(name="ddqn", learning_method="DDQN")]
#   CONFIGS = [dict(name=f"n{n}", n_step=n) for n in (1, 3, 5)]
#   CONFIGS = [dict(name="shaped", reward_shaping=True),
#              dict(name="unshaped", reward_shaping=False)]
#   CONFIGS = [dict(name=f"u{u}", updates_per_step=u) for u in (1, 2, 4, 8)]
#   CONFIGS = [dict(name=f"budget_{b['max_actions']}", budget=b) for b in
#              (dict(max_actions=6, alt_counts=1, alts_per_count=1),
#               dict(max_actions=12, alt_counts=2, alts_per_count=2),
#               dict(max_actions=24, alt_counts=4, alts_per_count=4))]
#   CONFIGS = [dict(name=f"buf{b}", buffer_size=b) for b in (5000, 20000, 50000)]
#              # keep min_buffer_size (simulation.DEFAULTS, 500) below the
#              # smallest buffer_size tried, or override it alongside
# The sweep run by a bare `python seed_sweep.py`. MUST exist even if it only
# holds one all-defaults cell: run_sweep's signature below is
# `def run_sweep(configs=CONFIGS, ...)`, and Python evaluates default arguments
# at DEFINITION time, so an undefined name here is an ImportError-class
# failure - `NameError: name 'CONFIGS' is not defined` raised while the module
# is still being read, before any of it runs. Nothing in the project imports
# without it.
CONFIGS = [dict(name="baseline")]

TRAINING_SEEDS = [0, 1, 2]          # >=3, per-config training seeds
# Default first eval seed. Every config and every training seed reuses the same
# eval seed list, which is what makes the comparison paired (CRN, section 12.1),
# so this is deliberately a single module-level value rather than something a
# per-config dict sets casually. A config CAN now override it (build_config
# below) - that exists for one purpose, a HELD-OUT eval set that was never used
# for selection, and using it for anything else silently unpairs the sweep.
EVAL_SEED_BASE = 10_000


def build_config(config, seed, overrides=None):
    """One (config, seed) cell's full config dict.

    PRECEDENCE, lowest to highest:
        BASE_CONFIG  <  the config's own keys  <  overrides
    then two sweep-level pins are applied on top. simulation._cfg fills in the
    rest and raises on an unknown key, so a typo here fails loudly instead of
    silently doing nothing.

    THE TWO PINS, and why they are not ordinary keys:

    `seed` is owned by run_sweep's `seeds=` argument, not by a config. The
    entire point of the sweep is to run EVERY config across the SAME set of
    training seeds so the across-seed spread (summarize() below) measures
    run-to-run training variance rather than a difference in which seeds each
    config happened to get. A config that pinned its own seed would collapse
    that comparison while still printing an SE, so it is rejected rather than
    silently ignored.

    `eval_seed_base` defaults to the module-level EVAL_SEED_BASE for the same
    reason - every cell must be scored on the same decks or the comparison
    stops being paired. It IS overridable, because one legitimate case needs
    it: a FINAL HELD-OUT evaluation on seeds that were never used for
    selection. Selecting and reporting on the same eval seeds biases the
    reported number upward, and there is no other way to get a clean holdout.
    Set it deliberately, per config, and never mid-sweep.

    [GOTCHA] `name` is not a config key at all - run_sweep reads it off the
    per-config dict to pick the output directory and the result label, and
    simulation._cfg merely tolerates it. In `overrides` it would rename every
    cell's plot title while leaving the directories untouched, so it is
    rejected too.
    """
    overrides = dict(overrides or {})
    for source, label in ((config, "config"), (overrides, "overrides")):
        if "seed" in source:
            raise KeyError(
                f"{label} sets 'seed' - training seeds come from run_sweep's "
                f"seeds= argument, so a per-config seed would silently break "
                f"the across-seed variance estimate this sweep exists to make")
    if "name" in overrides:
        raise KeyError(
            "overrides sets 'name' - the config name selects the output "
            "directory and is read off the per-config dict, so setting it here "
            "would relabel results without moving them")

    merged = {**BASE_CONFIG, **config, **overrides}
    return {
        **merged,
        "seed": seed,
        "eval_seed_base": merged.get("eval_seed_base", EVAL_SEED_BASE),
    }


def _run_one(config, seed, overrides=None):
    """Run one (config, seed) cell end to end and return a dict ready to
    json.dump. simulation() owns the nets, the RNG seeding and the
    train/test block loop; this only scores what comes back."""
    full_config = build_config(config, seed, overrides)
    result = simulation(full_config)

    # {opponent_label: [one summarize() dict per block]}
    blocks = result["blocks"]
    rewards = result["train_history"]["episode_rewards"]
    window = result["config"]["reward_window"]

    return {
        "config": config["name"],
        "seed": seed,
        # Recorded because it is now overridable (build_config): two results
        # scored on different eval seeds are not paired, and without this in
        # the JSON there is no way to tell afterwards which is which.
        "eval_seed_base": full_config["eval_seed_base"],
        "training_iterations": full_config["training_iterations"],
        "train_episodes_per_block": full_config["train_episodes_per_block"],
        "test_episodes_per_block": full_config["test_episodes_per_block"],
        "hyperparams": {k: full_config[k] for k in
                        ("learning_method", "n_step", "reward_shaping",
                         "gamma", "lr", "tau", "updates_per_step",
                         "epsilon", "epsilon_decay", "epsilon_min",
                         "budget", "buffer_size")},
        # Final block, per opponent. score_of() below reads this.
        "eval": {label: {"avg_reward": series[-1]["avg_reward"],
                         "avg_reward_se": series[-1]["avg_reward_se"],
                         "win_rate": series[-1]["win_rate"],
                         "n": series[-1]["n"]}
                 for label, series in blocks.items()},
        # The whole learning curve, not just its endpoint: a config that peaks
        # early and decays looks identical to a steady one at the last block.
        "test_curves": {label: [{"episode": b["episode"],
                                 "avg_reward": b["avg_reward"],
                                 "win_rate": b["win_rate"]} for b in series]
                        for label, series in blocks.items()},
        "train_final_reward_median": float(np.median(rewards[-window:]))
        if rewards else float("nan"),
    }


def score_of(result, opponent=None):
    """A cell's headline number: final-block avg reward vs `opponent`.
    Falls back to the only opponent present when SCORE_OPPONENT wasn't in
    test_opponents, so changing the list can't silently produce a KeyError
    mid-sweep."""
    evals = result["eval"]
    if opponent is None:
        opponent = SCORE_OPPONENT
    if opponent not in evals:
        opponent = next(iter(evals))
    return evals[opponent]["avg_reward"]


def run_sweep(configs=CONFIGS, seeds=TRAINING_SEEDS, overrides=None,
              out_dir=None, quiet=True, workers=1, resume=True):
    """Run every (config, seed) cell and summarise.

    `workers` is how many cells run AT ONCE, each in its own subprocess (see
    parallel_sweep for why subprocesses and not multiprocessing). 1 is the
    default and reproduces the original serial behaviour; 'auto' uses one per
    core. Cells are independent - separate processes, separate seeds, separate
    output files, and nothing shared but the ILP's temp directory, which is
    uuid-prefixed - so a parallel sweep returns the same results as a serial
    one. Only the wall-clock and the interleaving of the progress lines change.

    `resume` skips cells whose result JSON already exists. A campaign of this
    length gets interrupted, and re-running a nine-hour block to recover the
    two cells that had not finished is a mistake you only need to make once.
    Delete a config's directory to force it to re-run.

    A cell that FAILS is reported and skipped, not fatal. One diverging lr is
    an expected outcome of a sweep - the test plan's own rule is "discard that
    cell" - and it should not cost the other 26. Failures are listed again
    before the summary, with their tracebacks in seed<N>.error.json, and are
    absent from the table rather than being scored as zeros.
    """
    if out_dir is None:
        out_dir = os.path.join(sim.CHECKPOINT_DIR, "sweep")
    overrides = {**(overrides or {}), "quiet": quiet}
    workers = ps.resolve_workers(workers)

    cells = [(config, seed) for config in configs for seed in seeds]
    for config in configs:
        print(f"[sweep] config={config['name']} "
              f"({ {k: v for k, v in config.items() if k != 'name'} })")

    all_results, failures = ps.run_cells(cells, out_dir, overrides, workers,
                                         resume=resume)

    if failures:
        print("\n=== FAILED CELLS (absent from the summary below) ===")
        for (name, seed), info in sorted(failures.items()):
            print(f"  {name} seed={seed}  rc={info['returncode']}  "
                  f"log={info['log']}")
        print("  These are MISSING DATA, not zeros. A config whose cells "
              "failed has a smaller\n  n_seeds in the table below; check that "
              "before reading its SE.\n")

    summary = summarize(all_results)
    print_summary(summary)
    return all_results, summary


def _mean_se(values):
    n = len(values)
    return dict(
        n_seeds=n,
        mean=float(np.mean(values)),
        se=float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        per_seed=values,
    )


def summarize(all_results):
    """Mean/SE of final avg reward ACROSS TRAINING SEEDS, per config AND per
    opponent. This is a second, outer layer of variance on top of the per-seed
    eval SE: run-to-run DQN training variance, not sampling noise within one
    eval. Opponents stay separate here too - a config can improve against
    random while going nowhere against greedy, and that IS the finding.
    """
    by_config = {}
    for r in all_results:
        for label, e in r["eval"].items():
            by_config.setdefault(r["config"], {}).setdefault(label, []).append(
                e["avg_reward"])

    return {name: {label: _mean_se(values) for label, values in by_opponent.items()}
            for name, by_opponent in by_config.items()}


def _ranking_opponent(summary):
    """SCORE_OPPONENT if it was actually tested, else the first label seen."""
    labels = next(iter(summary.values())).keys() if summary else []
    return SCORE_OPPONENT if SCORE_OPPONENT in labels else next(iter(labels), None)


def print_summary(summary):
    if not summary:
        return
    ranked_by = _ranking_opponent(summary)
    print(f"\n=== config comparison (mean final avg-reward across training seeds, "
          f"ranked vs {ranked_by}) ===")
    rows = sorted(summary.items(), key=lambda kv: -kv[1][ranked_by]["mean"])
    for name, by_opponent in rows:
        print(f"  {name}")
        for label, s in by_opponent.items():
            marker = "*" if label == ranked_by else " "
            print(f"   {marker} vs {label:<10} {s['mean']:>8.2f}  "
                  f"(SE +/-{s['se']:.2f}, n={s['n_seeds']} seeds)  "
                  f"per-seed={['%.2f' % v for v in s['per_seed']]}")
    if len(rows) >= 2:
        top, second = rows[0][1][ranked_by], rows[1][1][ranked_by]
        gap = top["mean"] - second["mean"]
        combined_se = (top["se"] ** 2 + second["se"] ** 2) ** 0.5
        note = "likely real" if combined_se == combined_se and gap > 2 * combined_se else "not distinguishable from noise at ~2 SE"
        print(f"\n  top vs runner-up gap (vs {ranked_by}): {gap:.2f} "
              f"(combined SE {combined_se:.2f}) -> {note}")
    print()


if __name__ == "__main__":
    run_sweep()