"""Visualisation: ONE FIGURE PER METRIC, ONE LINE PER SEED.

Drawn once at the end of a sweep from the block logs on disk, never
during training.

    train_avg_loss.png                 is the optimisation healthy?
    test_avg_reward_vs_<opponent>.png  is the agent getting better?
    test_win_rate_vs_<opponent>.png    ... and does that show as wins?

One file per opponent, never pooled. Every seed gets its own coloured
line, with the across-seed mean and a +/-1 SD band on top. A block with
no data is a NaN gap, not a zero. matplotlib is optional.
"""
import os
import re

import numpy as np

METRIC_STYLE = {
    "avg_loss": dict(title="training loss", ylabel="avg loss per block",
                     reference=None, ylim=None, logy=True),
    "avg_reward": dict(title="test avg terminal reward",
                       ylabel="avg reward (terminal games)",
                       reference=0.0, ylim=None, logy=False),
    "win_rate": dict(title="test win rate", ylabel="win rate (%)",
                     reference=50.0, ylim=(-5, 105), logy=False),
}

_matplotlib_warned = False


def _pyplot():
    """matplotlib.pyplot, or None, warned about exactly once per process."""
    global _matplotlib_warned
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        if not _matplotlib_warned:
            print("[plots] matplotlib not installed, skipping figures "
                  "(pip install matplotlib to enable them). The block-log "
                  "JSON is written either way.")
            _matplotlib_warned = True
        return None


def _safe(name):
    """An opponent label made safe to use as a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "unnamed"


def _seed_colors(plt, seeds):
    """A distinct, stable colour per seed, so one figure's legend can be read
    against another's.
    """
    n = len(seeds)
    cmap = plt.get_cmap("tab10" if n <= 10 else "turbo")
    if n <= 10:
        return {s: cmap(i % 10) for i, s in enumerate(seeds)}
    return {s: cmap(i / max(n - 1, 1)) for i, s in enumerate(seeds)}


def _series(records, metric):
    """(blocks, values) for one seed's series, None mapped to NaN so the line
    breaks at a block with no data instead of touching the axis.
    """
    blocks = [r["block"] for r in records]
    values = [np.nan if r["metrics"][metric] is None else float(r["metrics"][metric])
              for r in records]
    return blocks, values


def _draw(plt, path, seed_series, metric, title, aggregate_blocks=None):
    """One figure: a line per seed plus the across-seed mean and SD band.
    Returns the path written, or None if there was nothing to draw.
    """
    from matplotlib.ticker import MaxNLocator
    style = METRIC_STYLE[metric]
    seeds = [s for s, _ in seed_series]
    colors = _seed_colors(plt, seeds)

    fig, ax = plt.subplots(figsize=(9, 5))
    drew_anything = False
    for seed, records in seed_series:
        blocks, values = _series(records, metric)
        if not any(v == v for v in values):
            continue
        drew_anything = True
        ax.plot(blocks, values, marker="o", markersize=4, linewidth=1.4,
                color=colors[seed], label=f"seed {seed}")

    if aggregate_blocks:
        blocks = [b["block"] for b in aggregate_blocks]
        means = [np.nan if b["metrics"][metric]["mean"] is None
                 else b["metrics"][metric]["mean"] for b in aggregate_blocks]
        stds = [np.nan if b["metrics"][metric]["std"] is None
                else b["metrics"][metric]["std"] for b in aggregate_blocks]
        if any(m == m for m in means):
            ax.plot(blocks, means, color="black", linestyle="--", linewidth=2.0,
                    label=f"mean of {len(seeds)} seeds", zorder=5)
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            if any(v == v for v in lo):
                ax.fill_between(blocks, lo, hi, color="black", alpha=0.08,
                                linewidth=0, label="+/-1 SD across seeds")

    if not drew_anything:
        plt.close(fig)
        return None

    if style["reference"] is not None:
        ax.axhline(style["reference"], color="grey", linewidth=0.8, alpha=0.6)
    if style["ylim"]:
        ax.set_ylim(*style["ylim"])
    if style["logy"]:
        finite = [v for _, recs in seed_series for v in _series(recs, metric)[1]
                  if v == v and v > 0]
        if finite and max(finite) / min(finite) > 20:
            ax.set_yscale("log")
    grid = sorted({r["block"] for _, recs in seed_series for r in recs})
    if grid:
        ax.set_xlim(grid[0] - 0.4, grid[-1] + 0.4)
        if len(grid) <= 15:
            ax.set_xticks(grid)
    ax.set_xlabel("block")
    ax.set_ylabel(style["ylabel"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_title(title)
    ax.legend(fontsize="small", ncol=2 if len(seeds) > 5 else 1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_config(results, out_dir, config_name, aggregate=None, show_mean=True):
    """Every figure for ONE config; returns the list of files written.

    `results` is that config's per-seed cell dicts; `aggregate` is the
    matching aggregate, computed here if not supplied.
    """
    plt = _pyplot()
    if plt is None:
        return []
    if not results:
        return []

    results = sorted(results, key=lambda r: r.get("seed", 0))
    seeds = [r.get("seed", i) for i, r in enumerate(results)]
    if aggregate is None and show_mean:
        import aggregate as agg_mod
        aggregate = agg_mod.aggregate_seeds(results, config_name=config_name)

    written = []
    episodes_note = ""
    if results[0]["train"] is not None:
        per_block = [b["counts"]["episodes"] for b in results[0]["train"]["blocks"]]
        if per_block:
            episodes_note = f", {per_block[0]} train episodes/block"

    if results[0]["train"] is not None:
        seed_series = [(s, r["train"]["blocks"]) for s, r in zip(seeds, results)]
        path = _draw(
            plt, os.path.join(out_dir, "train_avg_loss.png"), seed_series,
            "avg_loss",
            f"{config_name} - {METRIC_STYLE['avg_loss']['title']}"
            f" ({len(seeds)} seeds{episodes_note})",
            aggregate_blocks=(aggregate or {}).get("train", {}).get("blocks")
            if aggregate else None)
        if path:
            written.append(path)

    if results[0]["test"] is not None:
        for label in sorted(results[0]["test"]["opponents"]):
            seed_series = [(s, r["test"]["opponents"][label])
                           for s, r in zip(seeds, results)]
            agg_blocks = None
            if aggregate and aggregate.get("test"):
                agg_blocks = aggregate["test"]["opponents"].get(label)
            for metric in ("avg_reward", "win_rate"):
                path = _draw(
                    plt,
                    os.path.join(out_dir, f"test_{metric}_vs_{_safe(label)}.png"),
                    seed_series, metric,
                    f"{config_name} - {METRIC_STYLE[metric]['title']} "
                    f"vs {label} ({len(seeds)} seeds)",
                    aggregate_blocks=agg_blocks)
                if path:
                    written.append(path)

    for path in written:
        print(f"[plots] wrote {path}")
    return written


def plot_sweep(results_by_config, out_dir_of, aggregates=None):
    """plot_config for every config in a finished sweep. `out_dir_of(name)`
    returns where that config's figures go.
    """
    written = []
    for name, results in results_by_config.items():
        written += plot_config(
            results, out_dir_of(name), name,
            aggregate=(aggregates or {}).get(name))
    return written