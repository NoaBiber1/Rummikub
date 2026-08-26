import time
import types
import os
import math
import sys
import torch
import numpy as np
from q_model.v1_0 import MLP
from game_env import GE
import lerning.v1_0 as learn
from lerning.replay_buffer import ReplayBuffer

online_net = None
opponent_net = None
target_net = None
 
def select_x(valid_x_list, epsilon, is_main=False):
    if np.random.rand() < epsilon:
        idx = np.random.choice(len(valid_x_list))
        return valid_x_list[idx]
 
    q_values = []
    with torch.no_grad():
        for x in valid_x_list:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
 
            q_val = online_net(x) if is_main else opponent_net(x)
            q_values.append(q_val.item())
 
    idx = int(np.argmax(q_values)) if q_values else 0
    return valid_x_list[idx] 

def _nanmean(values):
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float("nan")

MAX_POSSIBLE_REWARD = 2 * 4 * sum(range(1, 14)) + 2 * 30

def _check_reward(reward, episode):
    if abs(reward) > MAX_POSSIBLE_REWARD:
        raise RuntimeError(
            f"[sanity check] reward {reward} at episode {episode} exceeds the "
            f"physically possible bound of +/-{MAX_POSSIBLE_REWARD} - this is a "
            f"reward-computation bug, not an unusual game. Stopping immediately."
        )

def _check_loss_and_q(loss, q_pred, episode):
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

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "online_net.pt")

def save_checkpoint(net, path=CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)

def load_checkpoint(net, path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        return False
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return True

TRAINING = types.SimpleNamespace(
    episodes=1000,
    opponent_update_every=50,
    opponent_pool_size=5,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    skeep_progress=False,
    log_every=50,
    opponent_epsilon=0.0,
    buffer_size=20000,
    min_buffer_size=500,
    batch_size=32,
    updates_per_step=4,
    reward_window=200,
    random_eval_every=500,
    random_eval_episodes=20,
)

def _take_turn(ge, epsilon, is_main):
    valid_x_list = ge.get_valid_x_list()
    if not valid_x_list:
        return None
    chosen_x = select_x(valid_x_list, epsilon, is_main)
    ge.play(chosen_x)
    return chosen_x

def _print_progress(itr, total_episodes, history, args):
    rewards = history["episode_rewards"][-args.reward_window:]
    losses = history["avg_losses"][-args.log_every:]
    qs = history["avg_qs"][-args.log_every:]
    print(
        f"[episode {itr:>6}/{total_episodes}] "
        f"epsilon={history['epsilons'][-1]:.3f} | "
        f"reward_median(last {len(rewards)})={np.median(rewards):.2f} | "
        f"avg_loss={_nanmean(losses):.4f} | "
        f"avg_Q={_nanmean(qs):.3f}"
    )

def _print_random_eval(episode, win_rate, avg_reward, n):
    print(
        f"[random-eval @ episode {episode:>6}] "
        f"win_rate={win_rate:.1f}% | avg_reward={avg_reward:.2f} (n={n})"
    )

def _print_summary(history, title):
    n = len(history["episode_rewards"])
    wins = [w for w in history["wins"] if w is not None]
    win_rate = (sum(wins) / len(wins) * 100.0) if wins else float("nan")
    win_rate_se = (
        math.sqrt((win_rate / 100.0) * (1.0 - win_rate / 100.0) / len(wins)) * 100.0
        if wins else float("nan")
    )
    avg_reward = float(np.mean(history["episode_rewards"])) if n else float("nan")
    reward_se = (
        float(np.std(history["episode_rewards"], ddof=1) / np.sqrt(n))
        if n > 1 else float("nan")
    )
    print(f"\n=== {title} ({n} episodes) ===")
    print(f"  win rate       : {win_rate:.1f}% (SE +/-{win_rate_se:.1f}%)")
    print(f"  avg reward     : {avg_reward:.2f} (SE +/-{reward_se:.2f})")
    print(f"  avg game length: {np.mean(history['turns']):.1f} main-player turns")
    print(f"  avg loss       : {_nanmean(history['avg_losses']):.4f}")
    print(f"  avg Q value    : {_nanmean(history['avg_qs']):.3f}\n")

def _rolling_median(values, window):
    values = np.asarray(values, dtype=float)
    window = max(1, int(window))
    if len(values) < window or window <= 1:
        return np.arange(1, len(values) + 1), values
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    medians = np.median(windows, axis=1)
    return np.arange(window, len(values) + 1), medians

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

    fig, axes = plt.subplots(4, 1, figsize=(9, 13))

    ep_med, med_r = _rolling_median(history["episode_rewards"], reward_window)
    axes[0].plot(episodes, history["episode_rewards"], alpha=0.15, color="tab:blue")
    axes[0].plot(ep_med, med_r, color="tab:blue")
    axes[0].set_ylabel(f"self-play reward\n(median, w={reward_window})")
    axes[0].set_xlabel("episode")

    random_eval = history.get("random_eval", [])
    if random_eval:
        re_episodes = [e["episode"] for e in random_eval]
        re_win = [e["win_rate"] for e in random_eval]
        re_reward = [e["avg_reward"] for e in random_eval]
    else:
        re_episodes, re_win, re_reward = [], [], []

    axes[1].plot(re_episodes, re_win, marker="o", color="tab:green")
    axes[1].set_ylabel("vs-random\nwin rate (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].set_xlabel("episode")

    axes[2].plot(re_episodes, re_reward, marker="o", color="tab:purple")
    axes[2].set_ylabel("vs-random\navg reward")
    axes[2].set_xlabel("episode")

    loss_window = max(1, len(episodes) // 50)

    def rolling_mean(values, w):
        values = np.array(values, dtype=float)
        if len(values) < w:
            return episodes, values
        kernel = np.ones(w) / w
        smoothed = np.convolve(values, kernel, mode="valid")
        return episodes[w - 1:], smoothed

    ep_l, smooth_l = rolling_mean(history["avg_losses"], loss_window)
    axes[3].plot(ep_l, smooth_l, color="tab:red")
    axes[3].set_ylabel("loss")
    axes[3].set_xlabel("episode")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is None:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        save_path = os.path.join(CHECKPOINT_DIR, "training_curves.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot_training_curves] saved to {save_path}")
    return save_path

def run_self_play_training(args, players=2, gamma=0.99, lr=0.001, tau=0.005, quiet=False):
    ge = GE(players)
 
    opponent_snapshots = [{k: v.clone().detach() for k, v in online_net.state_dict().items()}]
    replay_buffer = ReplayBuffer(args.buffer_size)

    history = {
        "episode_rewards": [],
        "wins": [],
        "avg_losses": [],
        "avg_qs": [],
        "epsilons": [],
        "turns": [],
        "random_eval": [],
    }
    start_time = time.time()

    for itr in range(1, args.episodes + 1):
        ge.reset()
        main_player = itr % players
 
        for i in range(0, main_player):
            if _take_turn(ge, args.opponent_epsilon, False) is None:
                break

        episode_losses = []
        episode_qs = []
        episode_turns = 0

        while not ge.is_Done():
            main_chosen_x = _take_turn(ge, args.epsilon, True)
            if main_chosen_x is None:
                break
 
            for i in range(1, players):
                if ge.is_Done():
                    break
                if _take_turn(ge, args.opponent_epsilon, False) is None:
                    break
 
            main_next_valid_x_list = []
            if not ge.is_Done():
                main_next_valid_x_list = ge.get_valid_x_list()
            reward = ge.get_reward(main_player)
            _check_reward(reward, itr)
            done = ge.is_Done()
            episode_turns += 1

            if args.skeep_progress:
                q_pred, loss = learn.train_step(
                    online_net,
                    target_net,
                    main_chosen_x,
                    reward,
                    main_next_valid_x_list,
                    done,
                    gamma,
                    lr,
                    tau,
                    True,
                )
                _check_loss_and_q(loss, q_pred, itr)
                episode_losses.append(loss)
                episode_qs.append(q_pred)
            else:
                replay_buffer.push(main_chosen_x, reward, main_next_valid_x_list, done)
                if len(replay_buffer) >= args.min_buffer_size:
                    step_losses = []
                    step_qs = []
                    for _ in range(args.updates_per_step):
                        batch = replay_buffer.sample(args.batch_size)
                        q_pred, loss = learn.train_step_batch(online_net, target_net, batch, gamma, lr, tau)
                        _check_loss_and_q(loss, q_pred, itr)
                        step_losses.append(loss)
                        step_qs.append(q_pred)
                    episode_losses.append(float(np.mean(step_losses)))
                    episode_qs.append(float(np.mean(step_qs)))
                else:
                    episode_losses.append(float("nan"))
                    episode_qs.append(float("nan"))
 
        args.epsilon = max(args.min_epsilon, args.epsilon * args.epsilon_decay)

        episode_reward = float(ge.get_reward(main_player))
        _check_reward(episode_reward, itr)
        history["episode_rewards"].append(episode_reward)
        winner = ge.get_winner()
        history["wins"].append(None if winner is None else (winner == main_player))
        history["avg_losses"].append(_nanmean(episode_losses))
        history["avg_qs"].append(_nanmean(episode_qs))
        history["epsilons"].append(args.epsilon)
        history["turns"].append(episode_turns)

        if not quiet and (itr % args.log_every == 0 or itr == args.episodes):
            _print_progress(itr, args.episodes, history, args)
            plot_training_curves(
                history,
                title=f"training progress (episode {itr}/{args.episodes})",
                reward_window=args.reward_window,
            )

        if not args.skeep_progress and args.random_eval_episodes > 0 and itr % args.random_eval_every == 0:
            eval_history = evaluate_model(
                online_net, players,
                episodes=args.random_eval_episodes,
                opponent_epsilon=1.0,
                verbose=False,
            )
            eval_wins = [w for w in eval_history["wins"] if w is not None]
            eval_win_rate = (sum(eval_wins) / len(eval_wins) * 100.0) if eval_wins else float("nan")
            eval_rewards = eval_history["episode_rewards"]
            eval_avg_reward = float(np.mean(eval_rewards)) if eval_rewards else float("nan")
            history["random_eval"].append({
                "episode": itr,
                "win_rate": eval_win_rate,
                "avg_reward": eval_avg_reward,
            })
            if not quiet:
                _print_random_eval(itr, eval_win_rate, eval_avg_reward, len(eval_rewards))
 
        if itr % args.opponent_update_every == 0:
            snapshot = {k: v.clone().detach() for k, v in online_net.state_dict().items()}
            opponent_snapshots.append(snapshot)
            if len(opponent_snapshots) > args.opponent_pool_size:
                opponent_snapshots.pop(0)
 
            snapshot = opponent_snapshots[np.random.randint(len(opponent_snapshots))]
            opponent_net.load_state_dict(snapshot)
            opponent_net.eval()

    elapsed = time.time() - start_time
    if not quiet:
        print(f"[run_self_play_training] finished {args.episodes} episodes in {elapsed:.1f}s "
              f"({args.episodes / max(elapsed, 1e-9):.2f} episodes/s)")

    return history

def evaluate_model(net, players=2, episodes=100, opponent_epsilon=0.0, log_every=20, verbose=True):
    global online_net, opponent_net, target_net
    online_net = net
    if opponent_net is None:
        opponent_net = MLP()
        opponent_net.load_state_dict(net.state_dict())
    if target_net is None:
        target_net = MLP()
        target_net.load_state_dict(net.state_dict())
    opponent_net.eval()
    target_net.eval()

    eval_args = types.SimpleNamespace(
        episodes=episodes,
        opponent_update_every=episodes + 1,
        opponent_pool_size=0,
        epsilon=0.0,
        epsilon_decay=0.0,
        min_epsilon=0.0,
        skeep_progress=True,
        log_every=log_every,
        opponent_epsilon=opponent_epsilon,
        buffer_size=1,
        min_buffer_size=1,
        batch_size=1,
        updates_per_step=1,
        reward_window=max(episodes, 1),
        random_eval_every=episodes + 1,
        random_eval_episodes=0,
    )
    history = run_self_play_training(eval_args, players, quiet=not verbose)
    if verbose:
        title = "Evaluation vs random opponent" if opponent_epsilon >= 1.0 else "Evaluation vs self (recent snapshot)"
        _print_summary(history, title)
    return history

def main(args): 
    global online_net, opponent_net, target_net
    online_net = MLP()
    opponent_net = MLP()
    opponent_net.load_state_dict(online_net.state_dict())
    opponent_net.eval()
    target_net = MLP()
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    print("Select option: ")
    print(" 1. train the model")
    print(" 2. test the model")
    print(" 3. print the model weights")
    print(" 4. exit")
    choise = input()

    match choise:
        case "1":
            train_history = run_self_play_training(TRAINING, args[1], args[4], args[5], args[6])
            save_checkpoint(online_net)
            print(f"[checkpoint saved to {CHECKPOINT_PATH}]")
            evaluate_model(online_net, players=args[1], opponent_epsilon=1.0)
        case "2":
            if load_checkpoint(online_net):
                opponent_net.load_state_dict(online_net.state_dict())
                target_net.load_state_dict(online_net.state_dict())
                print(f"[loaded checkpoint from {CHECKPOINT_PATH}]")
            else:
                print(f"[no checkpoint found at {CHECKPOINT_PATH} - testing a randomly-initialized model]")
            evaluate_model(online_net, players=args[1], opponent_epsilon=1.0)
        case "3":
            if load_checkpoint(online_net):
                print(f"[loaded checkpoint from {CHECKPOINT_PATH}]")
            else:
                print(f"[no checkpoint found at {CHECKPOINT_PATH} - printing a randomly-initialized model]")
            online_net.print_weights()
        case _:
            return

if __name__ == "__main__":
    if len(sys.argv) < 7:
        raise SystemExit(
            f"usage: python {sys.argv[0]} <players> <_> <_> <gamma> <lr> <tau>\n"
        )
    cli_args = [
        sys.argv[0],
        int(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        float(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
    ]
    main(cli_args)