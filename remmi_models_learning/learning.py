"""TD update steps against a target net, plus the Polyak update.

    DQN   target = r + gamma * max_a Q_target(s', a)
    DDQN  target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
"""

import torch

DQN = 0
DDQN = 1

_METHOD_ALIASES = {"dqn": DQN, "ddqn": DDQN, 0: DQN, 1: DDQN}


def resolve_learning_method(method):
    """Accept 'DQN'/'DDQN', 0/1 or False/True and return DQN or DDQN.

    Raises on anything else rather than falling back to DQN: a typo that
    quietly trains the wrong algorithm is undetectable from the numbers.
    """
    key = method
    if isinstance(key, str):
        key = key.strip().lower()
    elif isinstance(key, bool):
        key = int(key)
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"unknown learning_method {method!r} - expected 'DQN'/'DDQN' or 0/1"
        )
    return _METHOD_ALIASES[key]


def _reward_sequence(reward, n_step):
    """A transition's rewards as a float list.

    Accepts a scalar, a list/tuple or a 1-D tensor. Raises on more than
    n_step rewards - the caller's window and the config disagree.
    """
    if isinstance(reward, torch.Tensor):
        seq = reward.reshape(-1).tolist()
    elif isinstance(reward, (list, tuple)):
        seq = list(reward)
    else:
        seq = [reward]
    if not seq:
        raise ValueError("empty reward sequence - a transition needs >= 1 reward")
    if len(seq) > n_step:
        raise ValueError(
            f"transition carries {len(seq)} rewards but n_step={n_step} - the "
            f"caller's n-step window and the config disagree"
        )
    return [float(r) for r in seq]


def _as_tensor(x):
    """x as a float32 tensor."""
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def target_drift(tau, updates_per_step=1):
    """Fraction of the target net replaced over `updates_per_step` updates."""
    return 1.0 - (1.0 - tau) ** updates_per_step


def effective_tau(tau, updates_per_step=1):
    """The per-UPDATE tau that produces `tau` total drift per environment step.

    tau_eff = 1 - (1 - tau) ** (1 / U), so config['tau'] means drift per
    TURN and sweeping updates_per_step does not sweep target staleness with
    it. U=1 returns tau exactly.
    """
    if isinstance(updates_per_step, bool) or not isinstance(updates_per_step, int) \
            or updates_per_step < 1:
        raise ValueError(
            f"updates_per_step must be an integer >= 1, got {updates_per_step!r}")
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must lie in [0, 1], got {tau!r}")
    if updates_per_step == 1:
        return tau
    return 1.0 - (1.0 - tau) ** (1.0 / updates_per_step)


def soft_update_target(online_net, target_net, tau, updates_per_step=1):
    """Polyak-update the target net in place.

    `tau` is the drift budget PER ENVIRONMENT STEP; the conversion to a
    per-update tau happens here, the single point where tau is consumed.
    """
    tau = effective_tau(tau, updates_per_step)
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


def _bootstrap(online_net, target_net, next_pos_x, method):
    """Value of the best next state-action under DQN or DDQN.

    next_pos_x is the whole legal set at the next state, so the argmax over
    actions is an argmax over that list. Returns 0.0 for an empty set, which
    is what makes `done` and 'no legal moves' bootstrap identically.
    """
    if not next_pos_x:
        return 0.0
    with torch.no_grad():
        candidates = torch.stack([_as_tensor(s) for s in next_pos_x])
        q_target = target_net(candidates).reshape(-1)
        if method == DDQN:
            q_online = online_net(candidates).reshape(-1)
            return q_target[int(q_online.argmax())].item()
        return q_target.max().item()


def train_step(online_net, target_net, x, reward, next_pos_x=None, done=False,
               gamma=0.99, lr=0.001, tau=0.005, updates_per_step=1, n_step=1,
               learning_method=DQN, skeep_progress=False):
    """One single-transition n-step update; returns (q_pred, loss).

    `reward` is the sequence of k <= n_step rewards observed after the
    action and next_pos_x is the legal set at S_t+k, so
    target = sum_j<k gamma**j * r_j (+ gamma**k * bootstrap, if not done).
    skeep_progress (sic) still computes q_pred/loss but skips backward,
    step and the soft update - the no-training pass the test loop uses.
    """
    method = resolve_learning_method(learning_method)

    for param_group in online_net.optimizer.param_groups:
        param_group['lr'] = lr

    x = _as_tensor(x)

    rewards = _reward_sequence(reward, n_step)
    k = len(rewards)
    discounts = gamma ** torch.arange(k, dtype=torch.float32)
    target_value = float((torch.tensor(rewards, dtype=torch.float32) * discounts).sum())
    if not done:
        target_value += gamma ** k * _bootstrap(
            online_net, target_net, next_pos_x or [], method)
    target = torch.tensor([[target_value]], dtype=torch.float32)

    if skeep_progress:
        with torch.no_grad():
            q_pred = online_net.forward(x)
            loss = (q_pred - target) ** 2
        return q_pred.item(), loss.item()

    online_net.optimizer.zero_grad()
    q_pred = online_net.forward(x)

    loss = (q_pred - target) ** 2
    loss.backward()
    online_net.optimizer.step()
    soft_update_target(online_net, target_net, tau, updates_per_step)

    return q_pred.item(), loss.item()


def train_step_batch(online_net, target_net, batch, gamma=0.99, lr=0.001,
                     tau=0.005, updates_per_step=1, n_step=1, learning_method=DQN):
    """One batched n-step update; returns (mean q_pred, loss).

    `batch` is ReplayBuffer.sample()'s output: a list of
    (x, rewards, next_pos_x at t+k, done). Samples may carry different
    numbers of rewards, so returns are computed on a zero-padded matrix and
    each sample's own k drives its bootstrap exponent.
    """
    method = resolve_learning_method(learning_method)

    for param_group in online_net.optimizer.param_groups:
        param_group['lr'] = lr

    xs, rewards, next_pos_xs, dones = zip(*batch)
    x_batch = torch.stack([_as_tensor(x) for x in xs])

    sequences = [_reward_sequence(r, n_step) for r in rewards]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.float32)
    width = max(len(seq) for seq in sequences)
    padded = torch.tensor(
        [seq + [0.0] * (width - len(seq)) for seq in sequences], dtype=torch.float32)
    discounts = gamma ** torch.arange(width, dtype=torch.float32)
    n_step_returns = (padded * discounts).sum(dim=1)
    bootstrap_discount = gamma ** lengths

    with torch.no_grad():
        counts = [len(n) for n in next_pos_xs]
        flat_next = [_as_tensor(s) for n in next_pos_xs for s in n]
        if flat_next:
            flat = torch.stack(flat_next)
            target_chunks = torch.split(target_net(flat).reshape(-1), counts)
            if method == DDQN:
                online_chunks = torch.split(online_net(flat).reshape(-1), counts)
                bootstrapped = [
                    t[int(o.argmax())].item() if o.numel() > 0 else 0.0
                    for o, t in zip(online_chunks, target_chunks)
                ]
            else:
                bootstrapped = [
                    c.max().item() if c.numel() > 0 else 0.0 for c in target_chunks
                ]
            max_q_next = torch.tensor(bootstrapped, dtype=torch.float32)
        else:
            max_q_next = torch.zeros(len(batch), dtype=torch.float32)

    done_t = torch.tensor(dones, dtype=torch.bool)
    target = torch.where(
        done_t,
        n_step_returns,
        n_step_returns + bootstrap_discount * max_q_next,
    ).unsqueeze(1)

    online_net.optimizer.zero_grad()
    q_pred = online_net.forward(x_batch)

    loss = ((q_pred - target) ** 2).mean()
    loss.backward()
    online_net.optimizer.step()
    soft_update_target(online_net, target_net, tau, updates_per_step)

    return q_pred.mean().item(), loss.item()