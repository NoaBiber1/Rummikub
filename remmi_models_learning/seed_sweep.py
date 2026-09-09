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

Every finished cell is also published to checkpoints/cells/, keyed by a
fingerprint of the config it ran under, its seed and the code that ran
it (see cell_cache), so two cells that are the same work under two names
are computed once and a stale result is never mistaken for a current one.
"""
import math
import os

import aggregate as agg
import cell_cache as cc
import evaluation as ev
import parallel_sweep as ps
import plots
import simulation as sim
from simulation import simulation

# The eight-seed run that diverged used learning_method="DQN", n_step=1,
# reward_shaping=False, gamma=0.99 and no brakes at all. Every line below
# that carries a "was" is a diagnosed cause, not a tuning preference; the
# reasoning is in DIVERGENCE.md and each one is a live config key, so an
# ablation is a one-line edit rather than a fork.
BASE_CONFIG = dict(
    training_iterations=10,
    train_episodes_per_block=200,
    test_episodes_per_block=100,
    budget=dict(max_actions=12, alt_counts=2, alts_per_count=2),

    # was "DQN". A smaller lever than it first looked: measured on 1,022 real
    # legal sets with a properly lagged target, DDQN removes 11% of the
    # maximiser's premium. The premium DOES grow with |A| (0.00126 at |A|=2
    # to 0.00434 at |A|=7) but it ANTI-correlates with hand size (-0.361), so
    # the "max-bias drives hoarding" story is refuted - see DIVERGENCE.md §4.
    # Kept because it costs one forward pass and 11% is not nothing, NOT
    # because it is expected to fix the collapse.
    learning_method="DDQN",

    # was 1. get_reward is 0 on every turn but the last of a ~45-turn game,
    # so at n_step=1 almost every target is bootstrap-on-bootstrap.
    n_step=3,

    # was False. PHI = (opponent hand score - own hand score)/100 is already
    # implemented, already verified to telescope, and is 0 at terminals, so
    # it is policy-invariant (Ng et al. 1999) - it cannot change which
    # policy is optimal, only how fast the approximator finds it. It also
    # prices a draw honestly turn by turn: taking a tile raises your own
    # hand score, so PHI falls.
    reward_shaping=True,

    # was 0.99. UNVALIDATED - a hypothesis, not a measurement. 1/(1-gamma)
    # = 100 against a ~45-turn episode, so any per-backup bias is amplified
    # 100x before the horizon is even reached; 0.97 still covers the episode
    # (horizon ~33) at a third the amplification and discounts a distant
    # win, which is the right preference in a game you want to end quickly.
    # Sweep it last, after the changes that ARE measured.
    gamma=0.97,
    lr=3e-4,
    tau=0.001,
    updates_per_step=4,
    epsilon=1.0,
    epsilon_decay=0.99701,
    epsilon_min=0.05,
    buffer_size=10000,

    # The three brakes, all previously absent. A SAFETY NET, not a cure:
    # measured, they are INERT in the normal regime (TD errors ~1e-5, so
    # Huber == MSE, no gradient reaches 10, no target approaches the bound)
    # and an offline arm carrying them came out bit-identical to legacy.
    # They engage once divergence is already large, which turns a run that
    # would silently burn sixty hours into one that stays bounded.
    # target_clip="auto" resolves at config time to the largest return the
    # game can physically produce (MAX_POSSIBLE_REWARD, doubled when shaping
    # adds a potential term), so full_config records the number applied.
    huber_delta=1.0,
    grad_clip=10.0,
    target_clip="auto",

    # Architecture. THE ONE CHANGE WITH A DIRECTLY MEASURED BEFORE/AFTER:
    # the old N(0, 0.01) draw made the net a constant function (output std
    # 8.5e-5, Q spread 7.3e-5 across twelve candidates in one state, 52% of
    # fc1 dead), so the greedy policy ranked actions by numerical noise.
    # Each is separately ablatable: q_init="legacy" restores the old draw,
    # q_features=False the raw [board|hand|action] input, q_layer_norm=False
    # the unnormalised trunk. Ablate q_init FIRST - see DIVERGENCE.md §10.
    q_features=True,
    q_layer_norm=True,
    q_init="kaiming",

    test_opponents=["random", "greedy"],
)

SCORE_OPPONENT = "random"
CONFIGS = [dict(name="baseline")]
TRAINING_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
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
    if "checkpoint_path" in overrides:
        raise KeyError(
            "overrides sets 'checkpoint_path' - overrides apply to EVERY cell, "
            "so every one of them would save its weights to the same file "
            "while the others were doing the same, and what survived would "
            "belong to no cell in particular. Put it on a single config that "
            "runs one seed when a run really has to leave a checkpoint")

    merged = {**BASE_CONFIG, **config, **overrides}
    return {
        **merged,
        "seed": seed,
        "eval_seed_base": merged.get("eval_seed_base", EVAL_SEED_BASE),
    }


def _plan_cells(configs, seeds, overrides):
    """Every (config, seed) cell of a sweep, resolved and fingerprinted.

    Resolving in the parent rather than in the worker does two things. It
    validates every config through `_cfg` before a single subprocess has
    been paid for, so a typo fails in the first second of a sweep instead
    of the first second of a cell. And it produces the fingerprint the
    cache is keyed on, which has to be known BEFORE a cell is launched
    for the cache to save anything at all.
    """
    cells = []
    for config in configs:
        for seed in seeds:
            full_config = sim.resolved_config(
                build_config(config, seed, overrides))
            cells.append(cc.Cell(config=config, seed=seed,
                                 full_config=full_config,
                                 fingerprint=cc.fingerprint(full_config, seed)))
    return cells


def _run_one(config, seed, overrides=None):
    """Run one (config, seed) cell and return the dict for its per-seed JSON.

    The run log, the config the run ACTUALLY used, that config's cache
    fingerprint and code version, and the derived final-block 'eval'.
    Block series are copied through unchanged and unsummarised.

    'full_config' is simulation's own resolved config, so every key that
    shapes a run reaches the disk - including the ones no sweep varies
    (`min_buffer_size`, `batch_size`, the opponent pool), which used to be
    absent and left a saved cell impossible to tell apart from one run
    under different values. It is also what the fingerprint is taken
    over, which is what makes the cache safe to trust.
    """
    result = simulation(build_config(config, seed, overrides))
    full_config = result["config"]
    run_log = {key: result[key] for key in ("schema", "train", "test")}

    return {
        **run_log,
        "config": config["name"],
        "seed": seed,
        cc.FULL_CONFIG_KEY: full_config,
        cc.FINGERPRINT_KEY: cc.fingerprint(full_config, seed),
        cc.CODE_VERSION_KEY: cc.CODE_VERSION,
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
    cells whose result JSON already exists AND still matches the fingerprint
    of the config and code in front of it; a result left by an older tree is
    re-run rather than served. A cell that another sweep already ran under a
    different name is copied from the cache instead of being computed again.
    A failed cell is reported and skipped, never scored as a zero.
    Aggregation and figures run at the end, over the finished block logs.
    Returns (results, summary, aggregates).
    """
    if out_dir is None:
        out_dir = os.path.join(sim.CHECKPOINT_DIR, "sweep")
    overrides = dict(overrides or {})
    workers = ps.resolve_workers(workers)

    for config in configs:
        print(f"[sweep] config={config['name']} "
              f"({ {k: v for k, v in config.items() if k != 'name'} })")
    cells = _plan_cells(configs, seeds, overrides)
    print(f"[sweep] code version {cc.CODE_VERSION}, cache {cc.cache_dir()}")

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