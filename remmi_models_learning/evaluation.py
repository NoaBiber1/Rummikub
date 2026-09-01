"""Measurement layer: what a run RECORDS and REPORTS, not what it plays.

Everything here works on plain numbers and the history dict. It imports no
torch, no game engine and nothing from simulation.py, so the dependency runs
one way (simulation -> evaluation) and these helpers can be exercised on
synthetic histories without a game.
"""
import math
import os

import numpy as np

# Where plot_training_curves writes by default: the same "checkpoints" folder
# simulation.py saves weights to. Recomputed here rather than imported, to
# keep the dependency one-way.
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

MAX_POSSIBLE_REWARD = 2 * 4 * sum(range(1, 14)) + 2 * 30


def _nanmean(values):
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float("nan")

# ------------------------------------------------------------- fire alarm

def check_reward(reward, episode):
    if abs(reward) > MAX_POSSIBLE_REWARD:
        raise RuntimeError(
            f"[sanity check] reward {reward} at episode {episode} exceeds the "
            f"physically possible bound of +/-{MAX_POSSIBLE_REWARD} - this is a "
            f"reward-computation bug, not an unusual game. Stopping immediately."
        )


def check_loss_and_q(loss, q_pred, episode):
    if math.isnan(loss) or math.isinf(loss):
        raise RuntimeError(
            f"[sanity check] non-finite loss ({loss}) at episode {episode} - "
            f"stopping immediately, continuing would keep training a broken network."
        )
    if math.isnan(q_pred) or math.isinf(q_pred):
        raise RuntimeError(
            f"[sanity check] non-finite Q-value ({q_pred}) at episode {episode} - "
            f"stopping immediately."
        )

# ---------------------------------------------------------------- history

def new_history():
    return {
        "episode_rewards": [],
        "wins": [],
        "avg_losses": [],
        "avg_qs": [],
        "epsilons": [],
        "turns": [],
        # One entry per (test block, opponent): a summarize() dict plus
        # "episode" (cumulative training episodes) and "opponent" (label).
        # Kept flat rather than nested so the plot can group it either way.
        "test_evals": [],
    }


def record_episode(history, episode, reward, won, losses, qs, epsilon, turns):
    """One finished episode. `won` is True/False/None (None = no winner).
    Takes numbers, not a GE - the caller reads the game, this records it."""
    check_reward(reward, episode)
    history["episode_rewards"].append(reward)
    history["wins"].append(won)
    history["avg_losses"].append(_nanmean(losses))
    history["avg_qs"].append(_nanmean(qs))
    history["epsilons"].append(epsilon)
    history["turns"].append(turns)

# ---------------------------------------------------------------- summary

def _win_rate_se(win_rate_pct, n):
    """Standard error of a win rate given as a percentage, n games."""
    if n == 0 or win_rate_pct != win_rate_pct:  # n==0 or NaN
        return float("nan")
    p = win_rate_pct / 100.0
    return math.sqrt(p * (1.0 - p) / n) * 100.0


def _reward_se(rewards):
    """Standard error of the mean of a reward sample."""
    n = len(rewards)
    if n <= 1:
        return float("nan")
    return float(np.std(rewards, ddof=1) / np.sqrt(n))


def summarize(history):
    """Headline numbers + SEs from a history. The single source both the
    periodic checkpoint and the final summary read, so they can't drift."""
    wins = [w for w in history["wins"] if w is not None]
    win_rate = (sum(wins) / len(wins) * 100.0) if wins else float("nan")
    rewards = history["episode_rewards"]
    return {
        "win_rate": win_rate,
        "win_rate_se": _win_rate_se(win_rate, len(wins)),
        "avg_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "avg_reward_se": _reward_se(rewards),
        "n": len(rewards),
    }

# --------------------------------------------------------------- printing
# Avg reward leads everywhere: it's the primary comparison metric (far lower
# variance per game than a binary win/loss, and win rate saturates once the
# agent reliably beats random, at which point it stops discriminating between
# configs). Win rate stays as a secondary sanity check.

def print_progress(itr, total_episodes, history, reward_window, log_every):
    rewards = history["episode_rewards"][-reward_window:]
    epsilons = history["epsilons"]
    print(
        f"[episode {itr:>6}/{total_episodes}] "
        f"{f'epsilon={epsilons[-1]:.3f} | ' if epsilons else ''}"
        f"reward_median(last {len(rewards)})={np.median(rewards):.2f} | "
        f"avg_loss={_nanmean(history['avg_losses'][-log_every:]):.4f} | "
        f"avg_Q={_nanmean(history['avg_qs'][-log_every:]):.3f}"
    )


def print_eval_checkpoint(episode, stats, opponent=None):
    """One test-block line, from a summarize() dict. `opponent` labels which
    baseline it was measured against - with several in play, an unlabelled
    number is worse than none."""
    label = opponent or stats.get("opponent", "?")
    print(
        f"[test @ episode {episode:>6} vs {label}] "
        f"avg_reward={stats['avg_reward']:.2f} (SE +/-{stats['avg_reward_se']:.2f}) | "
        f"win_rate={stats['win_rate']:.1f}% (SE +/-{stats['win_rate_se']:.1f}%) "
        f"(n={stats['n']})"
    )


def print_summary(history, title):
    stats = summarize(history)
    print(f"\n=== {title} ({stats['n']} episodes) ===")
    print(f"  avg reward     : {stats['avg_reward']:.2f} (SE +/-{stats['avg_reward_se']:.2f})")
    print(f"  win rate       : {stats['win_rate']:.1f}% (SE +/-{stats['win_rate_se']:.1f}%)")
    print(f"  avg game length: {np.mean(history['turns']):.1f} main-player turns")
    print(f"  avg loss       : {_nanmean(history['avg_losses']):.4f}")
    print(f"  avg Q value    : {_nanmean(history['avg_qs']):.3f}\n")

# --------------------------------------------------------------- plotting

def _rolling_median(values, window):
    values = np.asarray(values, dtype=float)
    window = max(1, int(window))
    if len(values) < window or window <= 1:
        return np.arange(1, len(values) + 1), values
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    return np.arange(window, len(values) + 1), np.median(windows, axis=1)


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.arange(1, len(values) + 1), values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.arange(window, len(values) + 1), smoothed


_matplotlib_warned = False


def plot_training_curves(history, save_path=None, title="training progress", reward_window=200):
    global _matplotlib_warned
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if not _matplotlib_warned:
            print("[plot_training_curves] matplotlib not installed, skipping plot "
                  "(pip install matplotlib to enable this)")
            _matplotlib_warned = True
        return None

    episodes = np.arange(1, len(history["episode_rewards"]) + 1)

    # One series per opponent, in first-seen order. Never pooled: a mean over
    # a baseline the agent beats and one it loses to describes neither.
    by_opponent = {}
    for entry in history.get("test_evals", []):
        by_opponent.setdefault(entry.get("opponent", "test"), []).append(entry)

    fig, axes = plt.subplots(4, 1, figsize=(9, 13))

    ep_med, med_r = _rolling_median(history["episode_rewards"], reward_window)
    axes[0].plot(episodes, history["episode_rewards"], alpha=0.15, color="tab:blue")
    axes[0].plot(ep_med, med_r, color="tab:blue")
    axes[0].set_ylabel(f"self-play reward\n(median, w={reward_window})")

    # Win rate is the secondary/sanity-check panel; avg reward is primary.
    for label, entries in by_opponent.items():
        ep = [e["episode"] for e in entries]
        axes[1].errorbar(ep, [e["win_rate"] for e in entries],
                         yerr=[e["win_rate_se"] for e in entries],
                         marker="o", capsize=3, label=label)
        axes[2].errorbar(ep, [e["avg_reward"] for e in entries],
                         yerr=[e["avg_reward_se"] for e in entries],
                         marker="o", capsize=3, label=label)
    axes[1].set_ylabel("test\nwin rate (%)")
    axes[1].set_ylim(-5, 105)
    axes[2].set_ylabel("test\navg reward (primary)")
    if len(by_opponent) > 1:
        axes[1].legend(fontsize="small")
        axes[2].legend(fontsize="small")

    ep_l, smooth_l = _rolling_mean(history["avg_losses"], max(1, len(episodes) // 50))
    axes[3].plot(ep_l, smooth_l, color="tab:red")
    axes[3].set_ylabel("loss")

    for ax in axes:
        ax.set_xlabel("episode")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is None:
        os.makedirs(PLOT_DIR, exist_ok=True)
        save_path = os.path.join(PLOT_DIR, "training_curves.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot_training_curves] saved to {save_path}")
    return save_path