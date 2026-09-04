"""Greedy baseline solver: the single play that places the most tiles."""

import pulp
from itertools import combinations
from collections import namedtuple

NUM_COLORS = 4
NUM_VALUES = 13
NUM_REAL_TILES = NUM_COLORS * NUM_VALUES
JOKER_INDEX = NUM_REAL_TILES
VECTOR_LEN = NUM_REAL_TILES + 1
X_LEN = 3 * VECTOR_LEN

MAX_COPIES = 2

MIN_RUN = 3
MAX_RUN = 5
MAX_JOKERS = 2

BLOCK_STARTS = tuple(range(0, NUM_REAL_TILES, NUM_VALUES))

Meld = namedtuple("Meld", ["real_indices", "num_jokers"])


def _mask(indices):
    """Bitmask of a sequence of tile indices."""
    m = 0
    for i in indices:
        m |= 1 << i
    return m


class GreedySolution:
    """Single-action max-tiles ILP.

    Melds and LP variables are built once in __init__ and reused; reset()
    only re-derives the per-turn position.
    """

    def __init__(self):
        """Enumerate melds and build the static LP variables (~13ms, once)."""
        self.melds = self._generate_melds()
        self.meld_masks = [_mask(m.real_indices) for m in self.melds]
        self.meld_jokers = [m.num_jokers for m in self.melds]
        self._build_static_vars()
        self.reset(hand_tails=None, board_tails=None)


    def _generate_melds(self):
        """Every distinct legal meld (groups plus runs), deduped."""
        seen = set()
        melds = []
        for real, nj in self._generate_groups() + self._generate_runs():
            if (real, nj) in seen:
                continue
            seen.add((real, nj))
            melds.append(Meld(real_indices=real, num_jokers=nj))
        return melds

    def _generate_groups(self):
        """Groups at value-offset v, using slots v, v+13, v+26, v+39."""
        out = []
        for v in range(NUM_VALUES):
            slots = [v + b for b in BLOCK_STARTS]
            for size in (3, 4):
                for r in range(max(1, size - MAX_JOKERS), min(size, NUM_COLORS) + 1):
                    for subset in combinations(slots, r):
                        out.append((tuple(sorted(subset)), size - r))
        return out

    def _generate_runs(self):
        """Runs of length 3-5 that cannot spill into the next colour block."""
        out = []
        for i in range(NUM_REAL_TILES):
            offset = i % NUM_VALUES
            for length in range(MIN_RUN, MAX_RUN + 1):
                if offset + length > NUM_VALUES:
                    break
                slots = tuple(range(i, i + length))
                for j in range(0, MAX_JOKERS + 1):
                    for jp in combinations(range(length), j):
                        out.append(
                            (tuple(s for p, s in enumerate(slots) if p not in jp), j)
                        )
        return out


    def _build_static_vars(self):
        """Create the x/y LP variables reused for the instance's whole life."""
        self._x_vars = [
            pulp.LpVariable(f"x_{i}", lowBound=0, upBound=MAX_COPIES, cat="Integer")
            for i in range(len(self.melds))
        ]
        self._y_vars = [
            pulp.LpVariable(f"y_{t}", lowBound=0, upBound=MAX_COPIES, cat="Integer")
            for t in range(NUM_REAL_TILES)
        ]
        self._y_joker_var = pulp.LpVariable(
            "y_joker", lowBound=0, upBound=MAX_COPIES, cat="Integer"
        )

    def reset(self, hand_tails, board_tails):
        """Set the current position from length-53 hand/board count vectors.

        Any sequence works; entries are coerced to int so a tensor element never
        leaks into the LP file handed to CBC.
        """
        if hand_tails is None or board_tails is None:
            self.hand_tails = self.board_tails = None
        else:
            self.hand_tails = [int(v) for v in hand_tails]
            self.board_tails = [int(v) for v in board_tails]
        self._base_built = False

    def _ensure_base_built(self):
        """Hand bounds plus the availability presolve, once per reset()."""
        if self._base_built:
            return
        if self.hand_tails is None or self.board_tails is None:
            raise ValueError(
                "Call reset(hand_tails, board_tails) with real data before solving."
            )
        hand, board = self.hand_tails, self.board_tails
        for t in range(NUM_REAL_TILES):
            self._y_vars[t].upBound = hand[t]
        self._y_joker_var.upBound = hand[JOKER_INDEX]

        missing = 0
        for t in range(NUM_REAL_TILES):
            if not (board[t] + hand[t]):
                missing |= 1 << t
        avail_j = board[JOKER_INDEX] + hand[JOKER_INDEX]
        masks, jokers = self.meld_masks, self.meld_jokers
        active = [
            i
            for i in range(len(masks))
            if not (masks[i] & missing) and jokers[i] <= avail_j
        ]

        terms = [[] for _ in range(NUM_REAL_TILES)]
        joker_terms = []
        for i in active:
            x = self._x_vars[i]
            for t in self.melds[i].real_indices:
                terms[t].append(x)
            if jokers[i]:
                joker_terms.append(jokers[i] * x)
        self._tile_lhs = [pulp.lpSum(v) for v in terms]
        self._joker_lhs = pulp.lpSum(joker_terms)
        self._base_built = True


    def _x(self, action):
        """One flat [board | hand | action] vector - valid_x_list's layout."""
        return self.board_tails + self.hand_tails + action

    def solve(self):
        """Return one flat X = [board(53) | hand(53) | action(53)], ALWAYS.

        Never None, never a dict. The action segment holds the tiles chosen for
        the board, or is all zeros when nothing is playable - which is this
        environment's draw action, so no caller needs a None branch. Board and
        hand come back unchanged from the last reset().
        """
        self._ensure_base_built()
        draw = [0] * VECTOR_LEN

        prob = pulp.LpProblem("rummikub_greedy_turn", pulp.LpMaximize)
        for t in range(NUM_REAL_TILES):
            prob += (
                self._tile_lhs[t] == self.board_tails[t] + self._y_vars[t],
                f"tile_{t}_balance",
            )
        prob += (
            self._joker_lhs == self.board_tails[JOKER_INDEX] + self._y_joker_var,
            "joker_balance",
        )
        prob += pulp.lpSum(self._y_vars) + self._y_joker_var

        if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))] != "Optimal":
            return self._x(draw)

        action = [int(round(v.value())) for v in self._y_vars]
        action.append(int(round(self._y_joker_var.value())))
        return self._x(action if sum(action) else draw)