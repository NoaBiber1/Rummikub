import torch
import numpy as np
from .q_model.v1_0 import MLP
from .game_env import GE
import lerning.v1_0 as lern
 
online_net = MLP()
opponent_net = MLP()
opponent_net.load_state_dict(online_net.state_dict())  # start in sync with the online net
opponent_net.eval()  # opponent net is never trained directly, only backprop is off
target_net = MLP()
target_net.load_state_dict(online_net.state_dict())  # start in sync with the online net
target_net.eval()  # target net is never trained directly, only backprop is off
 
 
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
 
 
def run_self_play_training(episodes=10000, players=2,
                            opponent_update_every=50, opponent_pool_size=5):
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05
    ge = GE(players)
 
    # start the pool with the initial (synced) weights
    opponent_snapshots = [online_net.state_dict()]
 
    for itr in range(1, episodes + 1):
        # reset game
        ge.reset()
        # define main player turn from all players
        main_player = itr % players
 
        # play the first moves until main player's turn
        for i in range(0, main_player):
            valid_x_list = ge.get_valid_x_list()
            chosen_x = select_x(valid_x_list, 0, False)
            ge.play(chosen_x)
 
        # play full turns and do train step for the main player
        while not ge.is_Done():
            # main_player
            valid_x_list = ge.get_valid_x_list()
            main_chosen_x = select_x(valid_x_list, epsilon, True)
            ge.play(main_chosen_x)
 
            # other players turns to complete full turn
            for i in range(1, players):
                if ge.is_Done():
                    break
                valid_x_list = ge.get_valid_x_list()
                chosen_x = select_x(valid_x_list, 0, False)
                ge.play(chosen_x)
 
            # get all next valid x of main player to do learning step
            main_next_valid_x_list = []
            if not ge.is_Done():
                main_next_valid_x_list = ge.get_valid_x_list()
 
            q_pred, loss = lern.train_step(
                online_net,
                target_net,
                main_chosen_x,
                ge.get_reward(main_player),
                main_next_valid_x_list,
                ge.is_Done()
            )
 
        # decay exploration once per episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
 
        # periodically improve the opponent via a rolling snapshot pool based on the online_net
        if itr % opponent_update_every == 0:
            snapshot = {k: v.clone().detach() for k, v in online_net.state_dict().items()}
            opponent_snapshots.append(snapshot)
            if len(opponent_snapshots) > opponent_pool_size:
                opponent_snapshots.pop(0)
 
            snapshot = opponent_snapshots[np.random.randint(len(opponent_snapshots))]
            opponent_net.load_state_dict(snapshot)
            opponent_net.eval()
 
    return online_net