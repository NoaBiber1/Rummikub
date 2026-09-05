"""THE ENTRY POINT. Everything in this project is run from here.

Compares training configs (DQN vs DDQN, n_step, gamma/lr/tau/
updates_per_step/epsilon/budget/buffer_size, ...) across >= 3 training
seeds each, with every test block played on the same fixed eval decks
(common random numbers) so deck luck is paired out of the comparison.
Every cell is scored against every opponent separately - never pooled.

    python seed_sweep.py            # runs CONFIGS x TRAINING_SEEDS
    python aggregate.py <dir> --plots   # re-analyse without retraining

Per sweep, in checkpoints/sweep/<config>/: seed<N>.json (that cell's
block log), aggregate.json (mean + variance across seeds per block) and
plots/*.png. Nothing is printed or plotted while a cell runs.
"""
import math
import os

import aggregate as agg
import evaluation as ev
import parallel_sweep as ps
import plots
import simulation as sim
from simulation import simulation

BASE_CONFIG = dict(
    training_iterations=10,
    train_episodes_per_block=100,
    test_episodes_per_block=30,
    budget=dict(max_actions=12, alt_counts=2, alts_per_count=2),
    learning_method="DQN",
    n_step=1,
    reward_shaping=False,
    gamma=0.99,
    lr=0.001,
    tau=0.005,
    updates_per_step=4,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05,
    buffer_size=20000,
    test_opponents=["random", "greedy"],
)

SCORE_OPPONENT = "random"


CONFIGS = [dict(name="baseline")]

TRAINING_SEEDS = [0, 1, 2]
EVAL_SEED_BASE = 10_000


def build_config(config, seed, overrides=None):
    """One (config, seed) cell's full config dict.

    Precedence: BASE_CONFIG < the config's own keys < overrides, then the
    two sweep-level pins. 'seed' in a config or in overrides is rejected
    (training seeds belong to run_sweep), and so is 'name' in overrides
    (it selects the output directory). eval_seed_base defaults to
    EVAL_SEED_BASE and is overridable only for a held-out eval set.
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
    """Run one (config, seed) cell and return the dict for its per-seed JSON:
    the run log, plus metadata, hyperparams and the derived final-block
    'eval'. Block series are copied through unchanged and unsummarised.
    """
    full_config = build_config(config, seed, overrides)
    result = simulation(full_config)
    run_log = {key: result[key] for key in ("schema", "train", "test")}

    return {
        **run_log,
        "config": config["name"],
        "seed": seed,
        "eval_seed_base": full_config["eval_seed_base"],
        "training_iterations": full_config["training_iterations"],
        "train_episodes_per_block": full_config["train_episodes_per_block"],
        "test_episodes_per_block": full_config["test_episodes_per_block"],
        "hyperparams": {k: full_config[k] for k in
                        ("learning_method", "n_step", "reward_shaping",
                         "gamma", "lr", "tau", "updates_per_step",
                         "epsilon", "epsilon_decay", "epsilon_min",
                         "budget", "buffer_size")},
        "eval": ev.final_test_metrics(run_log),
    }


def score_of(result, opponent=None):
    """A cell's headline number: final-block avg reward vs `opponent`.

    Falls back to whichever opponent is present, so editing test_opponents
    cannot raise mid-sweep. None when nothing was measured - which is not a
    score of 0.
    """
    evals = result.get("eval") or {}
    if not evals:
        return None
    if opponent is None:
        opponent = SCORE_OPPONENT
    if opponent not in evals:
        opponent = next(iter(evals))
    return evals[opponent]["avg_reward"]


def run_sweep(configs=CONFIGS, seeds=TRAINING_SEEDS, overrides=None,
              out_dir=None, workers=1, resume=True, make_plots=True):
    """Run every (config, seed) cell, aggregate, plot and summarise.

    `workers` cells run at once, each in its own subprocess. `resume` skips
    cells whose result JSON already exists. A failed cell is reported and
    skipped, never scored as a zero. Aggregation and figures run at the end,
    over the finished block logs. Returns (results, summary, aggregates).
    """
    if out_dir is None:
        out_dir = os.path.join(sim.CHECKPOINT_DIR, "sweep")
    overrides = dict(overrides or {})
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

    aggregates = aggregate_results(all_results, out_dir, make_plots=make_plots)
    summary = summarize(all_results)
    print_summary(summary)
    return all_results, summary, aggregates


def aggregate_results(all_results, out_dir, make_plots=True):
    """Per config: write aggregate.json and draw the figures; returns
    {config: aggregate}. A config that cannot be aggregated is reported and
    skipped rather than costing the whole summary table.
    """
    by_config = {}
    for r in all_results:
        by_config.setdefault(r["config"], []).append(r)

    aggregates = {}
    for name, results in sorted(by_config.items()):
        config_dir = os.path.join(out_dir, name)
        try:
            aggregates[name] = agg.aggregate_seeds(results, config_name=name)
            path = agg.write_aggregate(aggregates[name], config_dir)
            print(f"[aggregate] {name}: {aggregates[name]['n_seeds']} seeds "
                  f"-> {path}")
        except (ValueError, RuntimeError) as exc:
            print(f"[aggregate] SKIP {name}: {exc}")
            continue
        if make_plots:
            plots.plot_config(results, os.path.join(config_dir, "plots"), name,
                              aggregate=aggregates[name])
    return aggregates


def _mean_se(values):
    """Mean, SE and per-seed values across seeds, from aggregate.mean_variance
    so the table and aggregate.json cannot disagree. SE is None for one
    seed, where the spread is undefined.
    """
    stats = agg.mean_variance(values, what="final avg_reward")
    n = stats["n_seeds"]
    return dict(
        n_seeds=n,
        mean=stats["mean"],
        se=(stats["std"] / math.sqrt(n)) if stats["std"] is not None else None,
        per_seed=list(values),
    )


def summarize(all_results):
    """Mean/SE of final avg reward ACROSS TRAINING SEEDS, per config and per
    opponent - the outer layer of variance, on top of within-block sampling
    noise. Cells whose final block measured nothing are dropped, not zeroed.
    """
    by_config = {}
    for r in all_results:
        for label, e in r["eval"].items():
            if e["avg_reward"] is None:
                continue
            by_config.setdefault(r["config"], {}).setdefault(label, []).append(
                e["avg_reward"])

    return {name: {label: _mean_se(values) for label, values in by_opponent.items()}
            for name, by_opponent in by_config.items()}


def _ranking_opponent(summary):
    """SCORE_OPPONENT if it was actually tested, else the first label seen."""
    labels = next(iter(summary.values())).keys() if summary else []
    return SCORE_OPPONENT if SCORE_OPPONENT in labels else next(iter(labels), None)


def print_summary(summary):
    """The config comparison table, plus the top-two gap against its combined
    SE. A single-seed SE prints as n/a rather than as a false precision.
    """
    if not summary:
        return
    ranked_by = _ranking_opponent(summary)
    def rank_key(item):
        """Sort key: best mean first, missing-opponent configs last."""
        stats = item[1].get(ranked_by)
        return -stats["mean"] if stats else float("inf")

    print(f"\n=== config comparison (mean final avg-reward across training seeds, "
          f"ranked vs {ranked_by}) ===")
    rows = sorted(summary.items(), key=rank_key)
    for name, by_opponent in rows:
        print(f"  {name}")
        for label, s in by_opponent.items():
            marker = "*" if label == ranked_by else " "
            se = f"{s['se']:.2f}" if s["se"] is not None else "n/a"
            print(f"   {marker} vs {label:<10} {s['mean']:>8.2f}  "
                  f"(SE +/-{se}, n={s['n_seeds']} seeds)  "
                  f"per-seed={['%.2f' % v for v in s['per_seed']]}")
    ranked = [r for r in rows if ranked_by in r[1]]
    if len(ranked) >= 2:
        top, second = ranked[0][1][ranked_by], ranked[1][1][ranked_by]
        gap = top["mean"] - second["mean"]
        if top["se"] is None or second["se"] is None:
            print(f"\n  top vs runner-up gap (vs {ranked_by}): {gap:.2f} "
                  f"(no combined SE - at least one config has a single seed, "
                  f"so the gap cannot be compared with anything)")
        else:
            combined_se = (top["se"] ** 2 + second["se"] ** 2) ** 0.5
            note = ("likely real" if gap > 2 * combined_se
                    else "not distinguishable from noise at ~2 SE")
            print(f"\n  top vs runner-up gap (vs {ranked_by}): {gap:.2f} "
                  f"(combined SE {combined_se:.2f}) -> {note}")
    print()


if __name__ == "__main__":
    run_sweep()