import torch
from ..q_model.v1_0 import MLP

def soft_update_target(online_net, target_net, tau):
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

def train_step(online_net, target_net ,x,reward, next_pos_x=None, done=False, gamma=0.99, lr=0.001, tau=0.005):
    """
    x: current state-action input tensor (shape matches model's input_dim)
    reward: scalar reward for this transition
    next_pos_x: list of next state-action tensors to evaluate (e.g. all
                            possible actions from the next state) — used to get max Q
    done: bool, True if this is the terminal step
    gamma: discount factor
    lr: learning rate
    tau: target network soft-update rate, applied after the optimizer step
    """

    if next_pos_x is None:
        next_pos_x = []
        
    for param_group in online_net.optimizer.param_groups:
        param_group['lr'] = lr

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    if done:
        target = torch.tensor([[reward]], dtype=torch.float32)
    else:
        with torch.no_grad():
            q_next_values = []
            for s_next in next_pos_x:
                if not isinstance(s_next, torch.Tensor):
                    s_next = torch.tensor(s_next, dtype=torch.float32)

                q_val = target_net(s_next)
                q_next_values.append(q_val.item())

        max_q_next = max(q_next_values) if q_next_values else 0.0
        target_value = reward + gamma * max_q_next
        target = torch.tensor([[target_value]], dtype=torch.float32)

    online_net.optimizer.zero_grad()
    q_pred = online_net.forward(x)

    loss = (q_pred - target) ** 2
    loss.backward()
    online_net.optimizer.step()
    soft_update_target(online_net, target_net, tau)

    return q_pred.item(), loss.item()