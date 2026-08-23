import torch

def soft_update_target(online_net, target_net, tau):
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

def train_step(online_net, target_net ,x,reward, next_pos_x=None, done=False, gamma=0.99, lr=0.001, tau=0.005, skeep_progress=False):
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
    if not skeep_progress:
        loss.backward()
        online_net.optimizer.step()
        soft_update_target(online_net, target_net, tau)

    return q_pred.item(), loss.item()