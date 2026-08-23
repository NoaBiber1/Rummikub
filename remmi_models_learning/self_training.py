import time
import types
import os
import torch
import numpy as np
from .q_model.v1_0 import MLP
from .game_env import GE
import lerning.v1_0 as learn

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
    episodes=10000,
    opponent_update_every=50,
    opponent_pool_size=5,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    skeep_progress=False,
    log_every=50,
    opponent_epsilon=0.0,
)
TESTING = types.SimpleNamespace(
    episodes=100,
    opponent_update_every=101,
    opponent_pool_size=0,
    epsilon=0,
    epsilon_decay=0,
    min_epsilon=0,
    skeep_progress=True,
    log_every=20,
    opponent_epsilon=0.0,
)

def _take_turn(ge, epsilon, is_main):
    valid_x_list = ge.get_valid_x_list()
    if not valid_x_list:
        return None
    chosen_x = select_x(valid_x_list, epsilon, is_main)
    ge.play(chosen_x)
    return chosen_x

def _print_progress(itr, total_episodes, history, window):
    rewards = history["episode_rewards"][-window:]
    wins = [w for w in history["wins"][-window:] if w is not None]
    losses = history["avg_losses"][-window:]
    qs = history["avg_qs"][-window:]
    win_rate = (sum(wins) / len(wins) * 100.0) if wins else float("nan")
    print(
        f"[episode {itr:>6}/{total_episodes}] "
        f"epsilon={history['epsilons'][-1]:.3f} | "
        f"avg_reward(last {len(rewards)})={np.mean(rewards):.2f} | "
        f"win_rate(last {len(wins)})={win_rate:.1f}% | "
        f"avg_loss={np.nanmean(losses):.4f} | "
        f"avg_Q={np.nanmean(qs):.3f}"
    )

def _print_summary(history, title):
    n = len(history["episode_rewards"])
    wins = [w for w in history["wins"] if w is not None]
    win_rate = (sum(wins) / len(wins) * 100.0) if wins else float("nan")
    print(f"\n=== {title} ({n} episodes) ===")
    print(f"  win rate       : {win_rate:.1f}%")
    print(f"  avg reward     : {np.mean(history['episode_rewards']):.2f}")
    print(f"  avg game length: {np.mean(history['turns']):.1f} main-player turns")
    print(f"  avg loss       : {np.nanmean(history['avg_losses']):.4f}")
    print(f"  avg Q value    : {np.nanmean(history['avg_qs']):.3f}\n")

def plot_training_curves(history, save_path=None, title="training progress"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_training_curves] matplotlib not installed, skipping plot "
              "(pip install matplotlib to enable this)")
        return None

    episodes = np.arange(1, len(history["episode_rewards"]) + 1)
    window = max(1, len(episodes) // 50)

    def rolling(values, w):
        values = np.array(values, dtype=float)
        if len(values) < w:
            return episodes, values
        kernel = np.ones(w) / w
        smoothed = np.convolve(values, kernel, mode="valid")
        return episodes[w - 1:], smoothed

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ep_r, smooth_r = rolling(history["episode_rewards"], window)
    axes[0].plot(episodes, history["episode_rewards"], alpha=0.25, color="tab:blue")
    axes[0].plot(ep_r, smooth_r, color="tab:blue")
    axes[0].set_ylabel("reward")

    wins_numeric = [1.0 if w == True else (0.0 if w == False else np.nan) for w in history["wins"]]
    ep_w, smooth_w = rolling(wins_numeric, window)
    axes[1].plot(ep_w, smooth_w, color="tab:green")
    axes[1].set_ylabel("win rate")
    axes[1].set_ylim(-0.05, 1.05)

    ep_l, smooth_l = rolling(history["avg_losses"], window)
    axes[2].plot(ep_l, smooth_l, color="tab:red")
    axes[2].set_ylabel("loss")
    axes[2].set_xlabel("episode")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is None:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        save_path = os.path.join(CHECKPOINT_DIR, "training_curves.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot_training_curves] saved to {save_path}")
    return save_path

def run_self_play_training(args, players=2, gamma=0.99, lr=0.001, tau=0.005):
    ge = GE(players)
 
    opponent_snapshots = [{k: v.clone().detach() for k, v in online_net.state_dict().items()}]

    history = {
        "episode_rewards": [],
        "wins": [],
        "avg_losses": [],
        "avg_qs": [],
        "epsilons": [],
        "turns": [],
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
            q_pred, loss = learn.train_step(
                online_net,
                target_net,
                main_chosen_x,
                ge.get_reward(main_player),
                main_next_valid_x_list,
                ge.is_Done(),
                gamma,
                lr, 
                tau,
                args.skeep_progress
            )
            episode_losses.append(loss)
            episode_qs.append(q_pred)
 
        args.epsilon = max(args.min_epsilon, args.epsilon * args.epsilon_decay)

        history["episode_rewards"].append(float(ge.get_reward(main_player)))
        winner = ge.get_winner()
        history["wins"].append(None if winner is None else (winner == main_player))
        history["avg_losses"].append(float(np.mean(episode_losses)) if episode_losses else float("nan"))
        history["avg_qs"].append(float(np.mean(episode_qs)) if episode_qs else float("nan"))
        history["epsilons"].append(args.epsilon)
        history["turns"].append(len(episode_losses))

        if itr % args.log_every == 0 or itr == args.episodes:
            _print_progress(itr, args.episodes, history, window=args.log_every)
 
        if itr % args.opponent_update_every == 0:
            snapshot = {k: v.clone().detach() for k, v in online_net.state_dict().items()}
            opponent_snapshots.append(snapshot)
            if len(opponent_snapshots) > args.opponent_pool_size:
                opponent_snapshots.pop(0)
 
            snapshot = opponent_snapshots[np.random.randint(len(opponent_snapshots))]
            opponent_net.load_state_dict(snapshot)
            opponent_net.eval()

    elapsed = time.time() - start_time
    print(f"[run_self_play_training] finished {args.episodes} episodes in {elapsed:.1f}s "
          f"({args.episodes / max(elapsed, 1e-9):.2f} episodes/s)")

    return online_net, history

def evaluate_model(net, players=2, episodes=100, opponent_epsilon=0.0, log_every=20):
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
    )
    _, history = run_self_play_training(eval_args, players)
    title = "Evaluation vs random opponent" if opponent_epsilon >= 1.0 else "Evaluation vs self (recent snapshot)"
    _print_summary(history, title)
    return history

def __main__(args): 
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
            _, train_history = run_self_play_training(TRAINING, args[1], args[4], args[5], args[6])
            save_checkpoint(online_net)
            print(f"[checkpoint saved to {CHECKPOINT_PATH}]")
            plot_training_curves(train_history)
            evaluate_model(online_net, players=args[1], opponent_epsilon=0.0)
            evaluate_model(online_net, players=args[1], opponent_epsilon=1.0)
        case "2":
            if load_checkpoint(online_net):
                opponent_net.load_state_dict(online_net.state_dict())
                target_net.load_state_dict(online_net.state_dict())
                print(f"[loaded checkpoint from {CHECKPOINT_PATH}]")
            else:
                print(f"[no checkpoint found at {CHECKPOINT_PATH} - testing a randomly-initialized model]")
            evaluate_model(online_net, players=args[1], opponent_epsilon=0.0)
            evaluate_model(online_net, players=args[1], opponent_epsilon=1.0)
        case "3":
            if load_checkpoint(online_net):
                print(f"[loaded checkpoint from {CHECKPOINT_PATH}]")
            else:
                print(f"[no checkpoint found at {CHECKPOINT_PATH} - printing a randomly-initialized model]")
            online_net.print_weights()
        case _:
            return