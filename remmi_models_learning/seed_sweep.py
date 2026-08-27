"""
Compare multiple training configs (different gamma/lr/tau, or in future
different network architectures / replay strategies) with enough seeds and
enough statistical care to trust the result.

Two variance-reduction pieces work together here:
  - Each config is trained with >=3 different TRAINING seeds (run-to-run DQN
    variance is large; one seed per config is how people convince themselves
    of effects that aren't there).
  - Every seed's final evaluation uses the SAME fixed vs-random eval seed
    list across every config (common random numbers - see
    self_training.evaluate_model), so deck luck is paired out of the
    config-vs-config comparison rather than averaged over.

This intentionally does NOT run a full factorial grid. Screen candidates
with reduced `episodes` first; promote only finalists to a full-length run
- see section 2 / section 10.3 of the project notes for why.

Usage (from the project root, with both the project root and agent/ on
PYTHONPATH - the same layout self_training.py itself requires):

    python seed_sweep.py

Edit CONFIGS below to add/remove configs. Results are written as JSON to
checkpoints/sweep/<config_name>/seed<seed>.json and a summary table is
printed (and returned) at the end.
"""
import copy
import json
import os
import types

import numpy as np
import torch

import self_training as st
from q_model.v1_0 import MLP

# --- edit this to add/remove configs -----------------------------------
CONFIGS = [
    dict(name="gamma_0.95", gamma=0.95, lr=0.001, tau=0.005),
    dict(name="gamma_0.99", gamma=0.99, lr=0.001, tau=0.005),
    dict(name="gamma_0.995", gamma=0.995, lr=0.001, tau=0.005),
]
TRAINING_SEEDS = [0, 1, 2]          # >=3, per-config training seeds
EVAL_SEED_BASE = 10_000             # fixed across every config/seed -> CRN
PLAYERS = 2


def _train_one(config, seed, episodes, eval_episodes, players=PLAYERS,
                training_overrides=None, quiet=True):
    """Run one (config, seed) cell: train from scratch, then evaluate vs
    random with the shared CRN eval seeds. Returns a dict ready to json.dump.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    st.online_net = MLP(seed=seed)
    st.opponent_net = MLP(seed=seed)
    st.opponent_net.load_state_dict(st.online_net.state_dict())
    st.target_net = MLP(seed=seed)
    st.target_net.load_state_dict(st.online_net.state_dict())
    st.opponent_net.eval()
    st.target_net.eval()

    args = copy.deepcopy(st.TRAINING)
    args.episodes = episodes
    if training_overrides:
        for k, v in training_overrides.items():
            setattr(args, k, v)

    train_history = st.run_self_play_training(
        args, players=players,
        gamma=config["gamma"], lr=config["lr"], tau=config["tau"],
        quiet=quiet,
    )

    # Shared eval seeds across every config/seed: CRN at the comparison
    # level, not just within a single evaluate_model call.
    eval_history = st.evaluate_model(
        st.online_net, players=players, episodes=eval_episodes,
        opponent_epsilon=1.0, verbose=not quiet,
        eval_seed_base=EVAL_SEED_BASE,
    )

    eval_rewards = eval_history["episode_rewards"]
    eval_wins = [w for w in eval_history["wins"] if w is not None]
    result = {
        "config": config["name"],
        "seed": seed,
        "episodes": episodes,
        "eval_episodes": eval_episodes,
        "eval_avg_reward": float(np.mean(eval_rewards)) if eval_rewards else float("nan"),
        "eval_win_rate": (sum(eval_wins) / len(eval_wins) * 100.0) if eval_wins else float("nan"),
        "train_final_reward_median": float(np.median(
            train_history["episode_rewards"][-args.reward_window:]
        )) if train_history["episode_rewards"] else float("nan"),
    }
    return result


def run_sweep(configs=CONFIGS, seeds=TRAINING_SEEDS, episodes=1000,
              eval_episodes=100, players=PLAYERS, training_overrides=None,
              out_dir=None, quiet=True):
    if out_dir is None:
        out_dir = os.path.join(st.CHECKPOINT_DIR, "sweep")

    all_results = []
    for config in configs:
        cfg_dir = os.path.join(out_dir, config["name"])
        os.makedirs(cfg_dir, exist_ok=True)
        for seed in seeds:
            print(f"[sweep] config={config['name']} seed={seed} "
                  f"(gamma={config['gamma']} lr={config['lr']} tau={config['tau']})")
            result = _train_one(
                config, seed, episodes, eval_episodes, players=players,
                training_overrides=training_overrides, quiet=quiet,
            )
            out_path = os.path.join(cfg_dir, f"seed{seed}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)
            print(f"    -> eval_avg_reward={result['eval_avg_reward']:.2f} "
                  f"eval_win_rate={result['eval_win_rate']:.1f}%  (saved {out_path})")

    summary = summarize(all_results)
    print_summary(summary)
    return all_results, summary


def summarize(all_results):
    """Mean/SE of eval_avg_reward ACROSS TRAINING SEEDS, per config. This is
    a second, outer layer of variance on top of the per-seed eval SE:
    run-to-run DQN training variance, not sampling noise within one eval.
    """
    by_config = {}
    for r in all_results:
        by_config.setdefault(r["config"], []).append(r["eval_avg_reward"])

    summary = {}
    for name, rewards in by_config.items():
        n = len(rewards)
        mean = float(np.mean(rewards))
        se = float(np.std(rewards, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        summary[name] = dict(n_seeds=n, mean_eval_avg_reward=mean, se=se, per_seed=rewards)
    return summary


def print_summary(summary):
    print("\n=== config comparison (mean eval avg-reward across training seeds) ===")
    rows = sorted(summary.items(), key=lambda kv: -kv[1]["mean_eval_avg_reward"])
    for name, s in rows:
        print(f"  {name:<20} {s['mean_eval_avg_reward']:>8.2f}  (SE +/-{s['se']:.2f}, n={s['n_seeds']} seeds)  "
              f"per-seed={['%.2f' % v for v in s['per_seed']]}")
    if len(rows) >= 2:
        top, second = rows[0], rows[1]
        gap = top[1]["mean_eval_avg_reward"] - second[1]["mean_eval_avg_reward"]
        combined_se = (top[1]["se"] ** 2 + second[1]["se"] ** 2) ** 0.5
        note = "likely real" if combined_se == combined_se and gap > 2 * combined_se else "not distinguishable from noise at ~2 SE"
        print(f"\n  top vs runner-up gap: {gap:.2f} (combined SE {combined_se:.2f}) -> {note}")
    print()


if __name__ == "__main__":
    run_sweep()