"""The Q-network: one scalar Q for one (state, candidate-action) pair.

The raw input is still `x = [board | hand | action]`, 159 long, exactly as
`GE.get_valid_x_list` builds it and exactly as the replay buffer stores it.
Nothing outside this module ever sees a different vector. What changed is
what the network DOES with it before the first Linear layer, and how the
weights start.

TWO FIXES LIVE HERE, both diagnosed from a diverging 8-seed run:

1. `_init_weights` used to draw EVERY weight and bias from N(0, 0.01),
   ignoring fan-in. Measured on this net at 159-256-256-256-1: activation
   std collapsed 8.6e-2 -> 1.3e-2 -> 9.3e-3 and the output arrived with
   std 8.5e-5 around a constant -7.2e-3. The spread of Q across twelve
   candidate actions in one state was 7.3e-5 - the greedy policy was
   reading a constant function plus numerical noise, so early action
   selection was arbitrary. The gradient-to-weight ratio was 265x larger
   at fc4 than at fc1, so Adam scaled up the readout while the features
   under it stayed frozen random projections. He init fixes the forward
   scale; the small final layer keeps Q ~ 0 at init, which is the correct
   prior for a game whose reward is 0 until it ends.

2. `expand_features` makes the AFTERSTATE explicit. Q(s,a) here is really
   a function of where the move leaves the position - board + action and
   hand - action - and the old input made the network discover that
   subtraction across a 106-dim gap on its own. The expansion is
   invertible (board = next_board - action), so nothing is added to or
   removed from what the network can represent; the same function class
   is simply reachable in far fewer steps.

`layer_norm` is the third: normalising each hidden layer bounds how fast
the represented function can grow, which is the standard structural brake
on TD divergence. All three are constructor flags so each can be ablated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VECTOR_LEN = 53
HAND_START = VECTOR_LEN
ACTION_START = 2 * VECTOR_LEN
RAW_INPUT_DIM = 3 * VECTOR_LEN

JOKER_INDEX = 52
JOKER_VALUE = 30.0

HAND_TILES_DEALT = 14.0
DECK_TILES = 106.0
REWARD_SCALE = 100.0

N_SCALAR_FEATURES = 8
EXPANDED_INPUT_DIM = 3 * VECTOR_LEN + N_SCALAR_FEATURES

Q_INIT_MODES = ("kaiming", "legacy")

LEGACY_INIT_STD = 0.01
FINAL_LAYER_SCALE = 1e-3


def _tile_values():
    """Point value of each of the 53 tile slots: 1-13 by position, joker 30.

    The same weights `GE.hand_score` uses, so a hand's value computed here
    and a hand's value computed by the engine are the same number.
    """
    positions = torch.arange(1, VECTOR_LEN + 1, dtype=torch.float32)
    return torch.where(positions == VECTOR_LEN,
                       torch.tensor(JOKER_VALUE),
                       ((positions - 1) % 13) + 1)


TILE_VALUES = _tile_values()


def expand_features(x):
    """`[board | hand | action]` -> `[next_board | next_hand | action | scalars]`.

    Works on a single (159,) vector and on a (N, 159) batch alike.

    The three blocks are a LINEAR, INVERTIBLE re-encoding of the original
    three: `next_board = board + action` and `next_hand = hand - action`,
    so `board` and `hand` are both recoverable and no information is lost.
    What it buys is that the quantity the Q-value actually depends on -
    where the move leaves the position - is present directly instead of
    having to be synthesised from two blocks 53 apart.

    The eight scalars are likewise derived from x alone. They carry no
    privileged information (a player can count their own rack), and they
    are DESCRIPTIONS, not preferences: the network is told how many tiles
    a move places and what the rack is worth afterwards, and still has to
    learn for itself whether that is good. Putting a preference here -
    "more tiles placed is better" - would be smuggling in `greedy_alg`'s
    objective, which is the baseline this agent is measured against.
    """
    board = x[..., :HAND_START]
    hand = x[..., HAND_START:ACTION_START]
    action = x[..., ACTION_START:]

    next_board = board + action
    next_hand = hand - action

    values = TILE_VALUES.to(dtype=x.dtype, device=x.device)
    placed_tiles = action.sum(-1, keepdim=True)
    placed_value = (action * values).sum(-1, keepdim=True)
    next_hand_tiles = next_hand.sum(-1, keepdim=True)
    next_hand_value = (next_hand * values).sum(-1, keepdim=True)
    next_board_tiles = next_board.sum(-1, keepdim=True)
    jokers_placed = action[..., JOKER_INDEX:JOKER_INDEX + 1]
    is_draw = (placed_tiles == 0).to(dtype=x.dtype)
    known_fraction = (board.sum(-1, keepdim=True)
                      + hand.sum(-1, keepdim=True)) / DECK_TILES

    scalars = torch.cat([
        placed_tiles / HAND_TILES_DEALT,
        placed_value / REWARD_SCALE,
        jokers_placed,
        is_draw,
        next_hand_tiles / HAND_TILES_DEALT,
        next_hand_value / REWARD_SCALE,
        next_board_tiles / DECK_TILES,
        known_fraction,
    ], dim=-1)

    return torch.cat([next_board, next_hand, action, scalars], dim=-1)


class MLP(nn.Module):
    """Q(s,a) over the 159-dim [board | hand | action] input.

    Scores state-ACTION pairs rather than emitting one Q per action: the
    legal set changes every turn, so there is no fixed action space. Owns
    its own Adam optimizer, which train_step drives.
    """

    def __init__(self, input_dim=RAW_INPUT_DIM, hidden_dim=256, seed=42,
                 lr=0.001, features=True, layer_norm=True, init="kaiming"):
        """Build the trunk, seed torch, init the weights and add Adam.

        `seed` seeds torch GLOBALLY, so pass it explicitly: the default
        would make every net in a sweep identical and reset the RNG
        mid-run.

        `features`, `layer_norm` and `init` are the three fixes, each
        separately switchable so a sweep can attribute the result to one
        of them. `init="legacy"` restores the N(0, 0.01) draw the diverging
        run used and exists only to reproduce it.
        """
        super().__init__()
        if init not in Q_INIT_MODES:
            raise ValueError(
                f"q_init must be one of {Q_INIT_MODES}, got {init!r}")
        if input_dim != RAW_INPUT_DIM:
            raise ValueError(
                f"input_dim is the RAW x width and is fixed by the tile "
                f"encoding at {RAW_INPUT_DIM} = 3 * {VECTOR_LEN}, got "
                f"{input_dim!r} - change the encoding in game_env if this "
                f"ever needs to move, so the engine and the net cannot "
                f"disagree about what an x is")
        torch.manual_seed(seed)

        self.features = bool(features)
        self.layer_norm = bool(layer_norm)
        self.init_mode = init

        trunk_in = EXPANDED_INPUT_DIM if self.features else RAW_INPUT_DIM
        self.fc1 = nn.Linear(trunk_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.ln3 = nn.LayerNorm(hidden_dim)

        self._init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _init_weights(self):
        """He-normal hidden layers, zero biases, a deliberately small head.

        The hidden layers feed ReLUs, so the fan-in-scaled He draw is what
        keeps activation variance from collapsing layer over layer. Biases
        start at zero rather than at noise, so no unit is pushed dead
        before it has seen data. The head starts near zero so Q(s,a) ~ 0
        everywhere at step one - the correct prior when every reward is 0
        until the game ends - without that being paid for by crushing the
        features underneath it.
        """
        hidden = (self.fc1, self.fc2, self.fc3)
        if self.init_mode == "legacy":
            for layer in hidden + (self.fc4,):
                nn.init.normal_(layer.weight, mean=0.0, std=LEGACY_INIT_STD)
                nn.init.normal_(layer.bias, mean=0.0, std=LEGACY_INIT_STD)
            return
        for layer in hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc4.weight, -FINAL_LAYER_SCALE, FINAL_LAYER_SCALE)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        """Scalar Q-value for x. Accepts a single (159,) vector or an (N, 159)
        batch and returns the matching (1,) or (N, 1).
        """
        h = expand_features(x) if self.features else x
        h = F.relu(self.ln1(self.fc1(h)) if self.layer_norm else self.fc1(h))
        h = F.relu(self.ln2(self.fc2(h)) if self.layer_norm else self.fc2(h))
        h = F.relu(self.ln3(self.fc3(h)) if self.layer_norm else self.fc3(h))
        return self.fc4(h)

    def print_weights(self):
        """Print per-layer shape/mean/std/min/max and raw tensors."""
        for name, layer in [
            ("fc1", self.fc1),
            ("fc2", self.fc2),
            ("fc3", self.fc3),
            ("fc4", self.fc4),
        ]:
            w = layer.weight.data
            b = layer.bias.data
            print(f"--- {name} ---")
            print(f"  weight shape: {tuple(w.shape)}, "
                  f"mean={w.mean().item():.6f}, std={w.std().item():.6f}, "
                  f"min={w.min().item():.6f}, max={w.max().item():.6f}")
            print(f"  bias shape:   {tuple(b.shape)}, "
                  f"mean={b.mean().item():.6f}, std={b.std().item():.6f}, "
                  f"min={b.min().item():.6f}, max={b.max().item():.6f}")
            print(f"  weight values:\n{w}")
            print(f"  bias values:\n{b}")
            print()