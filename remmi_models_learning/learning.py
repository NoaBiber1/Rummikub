import torch

# Learning method. Binary: 0 = standard DQN, 1 = Double DQN.
#   DQN  target = r + gamma * max_a Q_target(s', a)
#   DDQN target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
# The difference is WHICH net picks the next action and WHICH one prices it.
# DQN uses the target net for both, so any upward error in its estimate is
# selected for and propagates - the standard overestimation bias. DDQN splits
# the two roles: the online net picks, the target net values that pick, so a
# candidate has to look good to both nets to raise the target.
DQN = 0
DDQN = 1

_METHOD_ALIASES = {"dqn": DQN, "ddqn": DDQN, 0: DQN, 1: DDQN}


def resolve_learning_method(method):
    """Accept "DQN"/"DDQN" (any case), 0/1, or False/True; return DQN or DDQN.

    Config files read better with the names, code reads better with the flag,
    and callers shouldn't have to know which form this module wants. Raises on
    anything else rather than silently falling back to DQN - a typo'd method
    that quietly trains the wrong algorithm is a result you cannot detect
    afterwards from the numbers alone.
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
    """A transition's rewards as a plain float list.

    Accepts a scalar (1-step, the old shape), a list/tuple, or a 1-D tensor,
    so nothing that used to call these functions has to change. Raises on more
    than n_step rewards: that means the caller's aggregation window and the
    config disagree, which would silently discount the bootstrap by the wrong
    power of gamma and produce a plausible-looking wrong number.
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
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def target_drift(tau, updates_per_step=1):
    """Fraction of the target net replaced over `updates_per_step` consecutive
    soft updates. Each update leaves (1-tau) of the old target in place, so
    after U of them the residual is (1-tau)^U and the drift is its complement.
    This is what has to be held constant when sweeping updates_per_step."""
    return 1.0 - (1.0 - tau) ** updates_per_step


def effective_tau(tau, updates_per_step=1):
    """The per-UPDATE tau that produces `tau` total drift per environment step.

    Inverting target_drift:  1 - (1 - tau_eff)^U = tau  =>

        tau_eff = 1 - (1 - tau)^(1/U)

    so config["tau"] means "how far the target moves per TURN", independent of
    how many gradient updates that turn happens to be split into. Without this,
    sweeping updates_per_step silently sweeps target drift with it (U=4 gives
    ~4x the drift of U=1) and the sweep cannot attribute a result to either.

    The brief's U*tau is the first-order approximation of the same thing, and
    the two agree to ~0.2% at tau=0.005, U<=8 - but the exact form costs one
    pow() per update, so there is no reason to carry the approximation error.
    U=1 short-circuits to `tau` exactly, keeping the common case bit-identical
    rather than round-tripping through pow().
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
    """Polyak update. `tau` is the drift budget PER ENVIRONMENT STEP; when a
    step is split into `updates_per_step` updates, each one moves by
    effective_tau(tau, updates_per_step) so the total is unchanged.

    The conversion lives here, at the single point where tau is consumed, so a
    caller cannot apply it twice or forget it."""
    tau = effective_tau(tau, updates_per_step)
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


def _bootstrap(online_net, target_net, next_pos_x, method):
    """Value of the best next state-action, under DQN or DDQN.

    next_pos_x is the next turn's whole legal set - this project scores
    state-ACTION pairs, so "argmax over actions" is just an argmax over that
    list, and both methods need the same one forward pass per net.
    Returns 0.0 for an empty set (terminal or stuck), which is what makes
    `done` and "no legal moves" bootstrap identically.
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
    """One single-transition update, n-step.

    `reward` is the sequence of k <= n_step rewards observed after taking this
    action, and next_pos_x is the legal set at S_{t+k} - the state k steps
    later, NOT one step later. k < n_step only at an episode's tail, where the
    sequence was truncated; the exponent below uses the actual k, so truncated
    and full-window transitions are both correct without a special case.

        target = sum_{j<k} gamma^j * r_j   (+ gamma^k * bootstrap, if not done)

    skeep_progress (sic): if True, still COMPUTES q_pred/loss so the caller can
    log them, but skips backward()/step()/soft_update - which is what makes
    evaluation a true no-training pass that still reports loss/Q numbers.

    gamma/lr/tau/updates_per_step/n_step/learning_method all have defaults
    here for standalone/direct use, but every caller in this pipeline
    (simulation.py's _test_block and run_self_play_training) passes all six
    explicitly, sourced from config via simulation._cfg - these defaults are
    never actually reached by a seed_sweep.py run. A config value always
    wins; nothing here silently substitutes its own default for one the
    caller supplied.
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
    """The batched n-step update. `batch` is a list of
    (x, rewards, next_pos_x_at_t+k, done) - ReplayBuffer.sample()'s output.

    `updates_per_step` is how many times the CALLER invokes this per
    environment step. It does not change what one call does apart from the
    target update, which is scaled so that U calls drift the target by `tau`
    in total rather than by U*tau - see effective_tau.

    Samples may carry DIFFERENT numbers of rewards (tail transitions are
    truncated), so the returns are computed on a zero-padded (B, width) matrix:
    padding contributes exactly 0 to a discounted sum, and each sample's own k
    drives its bootstrap exponent. No Python loop over the batch in the math -
    only the one list comprehension that builds the padded tensor.

    gamma/lr/tau/updates_per_step/n_step/learning_method all have defaults
    here for standalone/direct use, but run_self_play_training's `update`
    (simulation.py) always passes all six explicitly from config via
    simulation._cfg - this default-carrying signature is never actually
    exercised by a seed_sweep.py run. A config value always wins over the
    default written here.
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
            # Every sample's candidates in ONE pass per net, then split back
            # per sample. DDQN costs a second pass (the online net picking),
            # not a Python loop.
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