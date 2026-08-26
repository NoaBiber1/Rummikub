import torch

def _as_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

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

def train_step_batch(online_net, target_net, batch, gamma=0.99, lr=0.001, tau=0.005):
    for param_group in online_net.optimizer.param_groups:
        param_group['lr'] = lr

    xs, rewards, next_pos_xs, dones = zip(*batch)
    x_batch = torch.stack([_as_tensor(x) for x in xs])

    with torch.no_grad():
        counts = [len(n) for n in next_pos_xs]
        flat_next = [_as_tensor(s) for n in next_pos_xs for s in n]
        if flat_next:
            flat_q = target_net(torch.stack(flat_next)).squeeze(-1)
            chunks = torch.split(flat_q, counts)
            max_q_next = torch.tensor(
                [c.max().item() if c.numel() > 0 else 0.0 for c in chunks],
                dtype=torch.float32,
            )
        else:
            max_q_next = torch.zeros(len(batch), dtype=torch.float32)

    reward_t = torch.tensor(rewards, dtype=torch.float32)
    done_t = torch.tensor(dones, dtype=torch.bool)
    target = torch.where(done_t, reward_t, reward_t + gamma * max_q_next).unsqueeze(1)

    online_net.optimizer.zero_grad()
    q_pred = online_net.forward(x_batch)

    loss = ((q_pred - target) ** 2).mean()
    loss.backward()
    online_net.optimizer.step()
    soft_update_target(online_net, target_net, tau)

    return q_pred.mean().item(), loss.item()