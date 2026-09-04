"""Across-SEED aggregation: many per-seed block logs -> one file per config.

For each block and metric, the MEAN across seeds and the SPREAD across
seeds (sample variance, ddof=1) - run-to-run training variance, not one
block's sampling noise. Every input is re-validated on the way in and
every statistic is computed twice by independent routes.

    per-seed    checkpoints/sweep/<config>/seed<N>.json
    aggregate   checkpoints/sweep/<config>/aggregate.json
"""
import json
import math
import os

import numpy as np

import evaluation as ev

CHECK_RTOL = 1e-9
CHECK_ATOL = 1e-12

VARIANCE_DDOF = 1


def _close(a, b):
    """True if two floats agree within CHECK_RTOL / CHECK_ATOL."""
    return abs(a - b) <= CHECK_ATOL + CHECK_RTOL * max(abs(a), abs(b))


def mean_variance(values, seeds=None, what="value"):
    """Mean, variance and std of one metric at one block, across seeds.

    Returns {mean, variance, std, n_seeds, per_seed}, always with the same
    keys. n=0 gives all-null; n=1 gives a mean with a null variance, since
    one observation says nothing about spread.
    """
    values = list(values)
    seeds = list(seeds) if seeds is not None else list(range(len(values)))
    if len(seeds) != len(values):
        raise ValueError(
            f"mean_variance({what}): {len(values)} values but {len(seeds)} "
            f"seed labels - the two lists are built together and must stay "
            f"aligned or per_seed attributes numbers to the wrong run")
    per_seed = {str(s): v for s, v in zip(seeds, values)}
    n = len(values)
    if n == 0:
        return {"mean": None, "variance": None, "std": None,
                "n_seeds": 0, "per_seed": {}}

    mean = ev.strict_mean(values, what)
    np_mean = float(np.mean(values))
    if not _close(mean, np_mean):
        raise RuntimeError(
            f"[math check] mean of {what} disagrees between fsum ({mean}) and "
            f"numpy ({np_mean}) over n={n} values")

    if n == 1:
        return {"mean": mean, "variance": None, "std": None,
                "n_seeds": 1, "per_seed": per_seed}

    variance = math.fsum((v - mean) ** 2 for v in values) / (n - VARIANCE_DDOF)
    np_variance = float(np.var(values, ddof=VARIANCE_DDOF))
    if not _close(variance, np_variance):
        raise RuntimeError(
            f"[math check] variance of {what} disagrees between the two-pass "
            f"formula ({variance}) and numpy ({np_variance}) over n={n}")
    if variance < 0.0:
        raise RuntimeError(
            f"[math check] variance of {what} is negative ({variance})")
    if max(values) - min(values) == 0.0 and variance != 0.0:
        raise RuntimeError(
            f"[math check] every seed reported the same {what} "
            f"({values[0]}) but the variance came out {variance}, not 0")

    std = math.sqrt(variance)
    if not _close(std * std, variance):
        raise RuntimeError(
            f"[math check] std of {what} ({std}) does not square back to its "
            f"variance ({variance})")
    return {"mean": mean, "variance": variance, "std": std,
            "n_seeds": n, "per_seed": per_seed}


def _totals(count_dicts, seeds):
    """Counts summed, not averaged -> {key: {total, per_seed}}."""
    keys = set(count_dicts[0])
    for d in count_dicts[1:]:
        if set(d) != keys:
            raise ValueError(f"count keys differ across seeds: {sorted(keys)} "
                             f"vs {sorted(set(d))}")
    out = {}
    for key in sorted(keys):
        values = [d[key] for d in count_dicts]
        total = sum(values)
        if total != int(np.sum(values)):
            raise RuntimeError(f"[math check] total of {key} disagrees with numpy")
        out[key] = {"total": total,
                    "per_seed": {str(s): v for s, v in zip(seeds, values)}}
    return out


def _aggregate_series(per_seed_series, seeds, kind, label=None):
    """One metric series (train, or test vs one opponent) across seeds.

    per_seed_series is [(seed, [record, ...]), ...]. Block grids must match
    exactly: two seeds whose block 4 sits at different episode counts are
    not measuring the same point on the curve.
    """
    what = f"{kind}{'' if label is None else ' vs ' + label}"
    grids = [[(r["block"], r["episode"]) for r in series]
             for _, series in per_seed_series]
    reference = grids[0]
    for (seed, _), grid in zip(per_seed_series, grids):
        if grid != reference:
            raise ValueError(
                f"{what}: seed {seed} has block grid {grid}, seed "
                f"{per_seed_series[0][0]} has {reference} - these cells were "
                f"not run with the same flow config, so they cannot be "
                f"averaged block by block (delete the config's directory and "
                f"re-run it, or aggregate the two sets separately)")

    metric_names = (ev.TRAIN_METRICS if kind == "train" else ev.TEST_METRICS)
    blocks = []
    for i, (block, episode) in enumerate(reference):
        rows = [series[i] for _, series in per_seed_series]
        metrics = {}
        for name in metric_names:
            present = [(seed, row["metrics"][name])
                       for (seed, _), row in zip(per_seed_series, rows)
                       if row["metrics"][name] is not None]
            metrics[name] = mean_variance([v for _, v in present],
                                          [s for s, _ in present],
                                          what=f"{name} @ {what} block {block}")
        blocks.append({
            "block": block,
            "episode": episode,
            **({"opponent": label} if label is not None else {}),
            "metrics": metrics,
            "counts": _totals([row["counts"] for row in rows],
                              [seed for seed, _ in per_seed_series]),
        })
    return blocks


def aggregate_seeds(results, config_name=None):
    """Aggregate per-seed cell results into ONE validated config-level dict."""
    if not results:
        raise ValueError("aggregate_seeds got no results - a config with zero "
                         "completed cells has nothing to aggregate")
    seeds = []
    for r in results:
        ev.validate_run_log(r)
        seed = r.get("seed")
        if seed is None:
            raise ValueError("every per-seed result must carry its 'seed'")
        if seed in seeds:
            raise ValueError(
                f"seed {seed} appears twice - the same cell would be counted "
                f"twice in every mean and would shrink every variance")
        seeds.append(seed)
    order = sorted(range(len(results)), key=lambda i: seeds[i])
    results = [results[i] for i in order]
    seeds = [seeds[i] for i in order]

    names = {r.get("config") for r in results if r.get("config") is not None}
    if len(names) > 1:
        raise ValueError(f"results come from more than one config: {sorted(names)}")
    if config_name is None:
        config_name = next(iter(names), None)

    for phase in ("train", "test"):
        present = [r[phase] is not None for r in results]
        if any(present) and not all(present):
            missing = [s for s, p in zip(seeds, present) if not p]
            raise ValueError(
                f"the {phase!r} phase is missing from seeds {missing} but "
                f"present in others - these cells were run with different "
                f"flow configs and cannot be aggregated together")

    out = {
        "schema": ev.SCHEMA,
        "config": config_name,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "aggregation": {
            "variance_ddof": VARIANCE_DDOF,
            "definition": "mean and sample variance ACROSS SEEDS at each "
                          "block; seeds reporting no data at a block are "
                          "dropped and counted in n_seeds",
        },
        "train": None,
        "test": None,
    }

    if results[0]["train"] is not None:
        out["train"] = {"blocks": _aggregate_series(
            [(s, r["train"]["blocks"]) for s, r in zip(seeds, results)],
            seeds, "train")}

    if results[0]["test"] is not None:
        opponent_sets = [set(r["test"]["opponents"]) for r in results]
        if any(s != opponent_sets[0] for s in opponent_sets):
            raise ValueError(
                f"seeds were tested against different opponents "
                f"({[sorted(s) for s in opponent_sets]}) - opponents are never "
                f"pooled, so there is no meaningful way to aggregate across "
                f"cells that measured different ones")
        out["test"] = {"opponents": {}}
        for label in sorted(opponent_sets[0]):
            out["test"]["opponents"][label] = _aggregate_series(
                [(s, r["test"]["opponents"][label]) for s, r in zip(seeds, results)],
                seeds, "test", label)

    return validate_aggregate(out)


def validate_aggregate(agg):
    """Re-check a finished aggregate - shapes, key sets, and the relationships
    between mean, variance, std and per_seed - and return it.
    """
    if agg.get("schema") != ev.SCHEMA:
        raise ValueError(f"aggregate schema is {agg.get('schema')!r}")
    n_seeds = agg["n_seeds"]
    if n_seeds != len(agg["seeds"]) or n_seeds < 1:
        raise ValueError(f"n_seeds={n_seeds} does not match seeds={agg['seeds']}")

    def check_block(block, metric_names, where):
        """Check one aggregated block's metric stats and count totals."""
        if set(block["metrics"]) != set(metric_names):
            raise ValueError(f"{where}: metrics must be exactly "
                             f"{sorted(metric_names)}")
        for name, stat in block["metrics"].items():
            if set(stat) != {"mean", "variance", "std", "n_seeds", "per_seed"}:
                raise ValueError(f"{where}/{name}: unexpected stat keys "
                                 f"{sorted(stat)}")
            k = stat["n_seeds"]
            if not isinstance(k, int) or k < 0 or k > n_seeds:
                raise ValueError(
                    f"{where}/{name}: n_seeds={k} outside 0..{n_seeds} - a "
                    f"metric cannot have contributions from more seeds than "
                    f"the config has")
            if len(stat["per_seed"]) != k:
                raise ValueError(
                    f"{where}/{name}: n_seeds={k} but per_seed holds "
                    f"{len(stat['per_seed'])} entries")
            if k == 0:
                if (stat["mean"], stat["variance"], stat["std"]) != (None, None, None):
                    raise ValueError(
                        f"{where}/{name}: no seed reported this metric, so "
                        f"mean/variance/std must all be null")
                continue
            values = list(stat["per_seed"].values())
            if not _close(stat["mean"], float(np.mean(values))):
                raise ValueError(
                    f"{where}/{name}: stored mean {stat['mean']} does not match "
                    f"the mean of the stored per-seed values")
            if k == 1:
                if stat["variance"] is not None or stat["std"] is not None:
                    raise ValueError(
                        f"{where}/{name}: variance is undefined for a single "
                        f"seed and must be null, not {stat['variance']!r}")
                continue
            if not _close(stat["variance"], float(np.var(values, ddof=VARIANCE_DDOF))):
                raise ValueError(
                    f"{where}/{name}: stored variance {stat['variance']} does "
                    f"not match the variance of the stored per-seed values")
            if stat["variance"] < 0 or not _close(stat["std"] ** 2, stat["variance"]):
                raise ValueError(f"{where}/{name}: std/variance inconsistent")
        for name, count in block["counts"].items():
            total = sum(count["per_seed"].values())
            if count["total"] != total:
                raise ValueError(
                    f"{where}/counts/{name}: total {count['total']} != sum of "
                    f"per-seed counts {total}")

    if agg["train"] is not None:
        for block in agg["train"]["blocks"]:
            check_block(block, ev.TRAIN_METRICS, f"train block {block['block']}")
    if agg["test"] is not None:
        for label, blocks in agg["test"]["opponents"].items():
            for block in blocks:
                check_block(block, ev.TEST_METRICS,
                            f"test block {block['block']} vs {label}")
    return agg


AGGREGATE_NAME = "aggregate.json"


def write_aggregate(agg, config_dir):
    """Write aggregate.json beside the per-seed files and return the path."""
    validate_aggregate(agg)
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, AGGREGATE_NAME)
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)
    return path


def load_seed_results(config_dir):
    """Every seed<N>.json in a config directory, sorted by seed. Failed cells
    (seed<N>.error.json) are not results and are skipped.
    """
    results = []
    for name in sorted(os.listdir(config_dir)):
        if not (name.startswith("seed") and name.endswith(".json")):
            continue
        if name.endswith(".error.json") or name == AGGREGATE_NAME:
            continue
        with open(os.path.join(config_dir, name)) as f:
            results.append(json.load(f))
    return sorted(results, key=lambda r: r.get("seed", 0))


def aggregate_config_dir(config_dir, write=True):
    """Aggregate one config directory, by default writing aggregate.json."""
    results = load_seed_results(config_dir)
    if not results:
        raise FileNotFoundError(f"no seed result files in {config_dir}")
    agg = aggregate_seeds(results, config_name=os.path.basename(config_dir))
    if write:
        path = write_aggregate(agg, config_dir)
        print(f"[aggregate] {agg['config']}: {agg['n_seeds']} seeds -> {path}")
    return agg


def aggregate_sweep_dir(sweep_dir, write=True):
    """Every config directory under a sweep root -> {config: aggregate}. A
    config that cannot be aggregated is reported and skipped.
    """
    out = {}
    for name in sorted(os.listdir(sweep_dir)):
        config_dir = os.path.join(sweep_dir, name)
        if not os.path.isdir(config_dir):
            continue
        try:
            out[name] = aggregate_config_dir(config_dir, write=write)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[aggregate] SKIP {name}: {exc}")
    return out


def main():
    """CLI: re-aggregate a sweep directory, optionally redrawing plots."""
    import argparse
    p = argparse.ArgumentParser(
        description="Re-aggregate an existing sweep directory (mean + variance "
                    "across seeds, per block). Runs nothing and trains nothing.")
    p.add_argument("sweep_dir", help="e.g. checkpoints/sweep or "
                                     "checkpoints/sweep/<config>")
    p.add_argument("--plots", action="store_true",
                   help="also (re)draw the per-config plots")
    args = p.parse_args()

    if any(n.startswith("seed") and n.endswith(".json")
           for n in os.listdir(args.sweep_dir)):
        configs = {os.path.basename(os.path.abspath(args.sweep_dir)):
                   aggregate_config_dir(args.sweep_dir)}
        dirs = {list(configs)[0]: args.sweep_dir}
    else:
        configs = aggregate_sweep_dir(args.sweep_dir)
        dirs = {name: os.path.join(args.sweep_dir, name) for name in configs}

    if args.plots:
        import plots
        for name, config_dir in dirs.items():
            plots.plot_config(load_seed_results(config_dir),
                              os.path.join(config_dir, "plots"), name,
                              aggregate=configs[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())