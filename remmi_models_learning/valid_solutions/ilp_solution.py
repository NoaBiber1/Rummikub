import pulp
from itertools import combinations
from collections import namedtuple

NUM_COLORS = 4
NUM_VALUES = 13
NUM_REAL_TILES = NUM_COLORS * NUM_VALUES
JOKER_INDEX = NUM_REAL_TILES
VECTOR_LEN = NUM_REAL_TILES + 1

JOKER_VALUE = 30
MAX_COPIES = 2

MIN_RUN = 3
MAX_RUN = 5
MAX_JOKERS = 2

BLOCK_STARTS = tuple(range(0, NUM_REAL_TILES, NUM_VALUES))

# Big enough that one extra tile always outranks any value gain: the whole
# deck is worth 4*2*(1+..+13) + 2*30 = 788 points.
LEX_WEIGHT = 10000

Meld = namedtuple("Meld", ["real_indices", "num_jokers"])

# How many actions to build and how the budget splits. Returned array size is
# bounded by min(max_actions, N + alt_counts*alts_per_count + 2) where N is the
# most tiles playable this turn.
DEFAULT_BUDGET = dict(max_actions=24, alt_counts=4, alts_per_count=4)
TRAINING_BUDGET = dict(max_actions=12, alt_counts=2, alts_per_count=2)


def tile_point_value(index):
    return index % NUM_VALUES + 1


def _mask(indices):
    m = 0
    for i in indices:
        m |= 1 << i
    return m


class ILP_solutions:
    def __init__(self, budget=None):
        self.budget = dict(budget or TRAINING_BUDGET)
        self.melds = self._generate_melds()
        self.meld_masks = [_mask(m.real_indices) for m in self.melds]
        self.meld_jokers = [m.num_jokers for m in self.melds]
        self._build_static_vars()
        self._solve_id = 0
        self.reset(hand_tails=None, board_tails=None)

    # ----------------------------------------------------------- meld list

    def _generate_melds(self):
        seen = set()
        melds = []
        for real, nj in self._generate_groups() + self._generate_runs():
            if (real, nj) in seen:
                continue
            seen.add((real, nj))
            melds.append(Meld(real_indices=real, num_jokers=nj))
        return melds

    def _generate_groups(self):
        """A group at value-offset v uses slots v, v+13, v+26, v+39."""
        out = []
        for v in range(NUM_VALUES):
            slots = [v + b for b in BLOCK_STARTS]
            for size in (3, 4):
                for r in range(max(1, size - MAX_JOKERS), min(size, NUM_COLORS) + 1):
                    for subset in combinations(slots, r):
                        out.append((tuple(sorted(subset)), size - r))
        return out

    def _generate_runs(self):
        """A run of length L starts at slot i and covers i..i+L-1, legal only
        while i % 13 + L <= 13 so it cannot spill into the next colour block."""
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

    # ------------------------------------------------------- static LP vars

    def _build_static_vars(self):
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
        """hand_tails / board_tails are length-53 count vectors. Any sequence
        works (list, numpy array, torch tensor); entries are coerced to int so
        a tensor element never leaks into the LP file handed to CBC."""
        if hand_tails is None or board_tails is None:
            self.hand_tails = self.board_tails = None
        else:
            self.hand_tails = [int(v) for v in hand_tails]
            self.board_tails = [int(v) for v in board_tails]
        self._base_built = False

    def _ensure_base_built(self):
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

        # Availability presolve: a meld needs exactly one copy of each of its
        # real slots, so it is playable only if every one of them exists in
        # board+hand. One bitmask AND per meld. Typically leaves ~29 of 1173.
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

    def _build_base_problem(self):
        self._ensure_base_built()
        prob = pulp.LpProblem("rummikub_turn", pulp.LpMaximize)
        for t in range(NUM_REAL_TILES):
            prob += (
                self._tile_lhs[t] == self.board_tails[t] + self._y_vars[t],
                f"tile_{t}_balance",
            )
        prob += (
            self._joker_lhs == self.board_tails[JOKER_INDEX] + self._y_joker_var,
            "joker_balance",
        )
        return prob

    # ------------------------------------------------------------ objective

    def _tiles_expr(self):
        return pulp.lpSum(self._y_vars) + self._y_joker_var

    def _value_expr(self):
        return pulp.lpSum(
            self._y_vars[t] * tile_point_value(t) for t in range(NUM_REAL_TILES)
        ) + self._y_joker_var * JOKER_VALUE

    def _tiles_then_value(self):
        return LEX_WEIGHT * self._tiles_expr() + self._value_expr()

    def _extract(self, sources):
        y = [int(round(v.value())) for v in self._y_vars]
        yj = int(round(self._y_joker_var.value()))
        value = sum(y[t] * tile_point_value(t) for t in range(NUM_REAL_TILES))
        return {
            "dropped_vector": y + [yj],
            "tiles_placed": sum(y) + yj,
            "value_placed": value + yj * JOKER_VALUE,
            "jokers_placed": yj,
            "sources": list(sources),
        }

    # -------------------------------------------------------- no-good cuts

    def _add_nogoods(self, prob, exclude, tag):
        """Order encoding (y_t = sum of binaries, b_0 >= b_1) so each y-vector
        has one binary pattern and a single row excludes exactly one point.
        Slots holding a single copy are already binary and are reused as-is,
        so a typical hand adds only one or two genuinely new binaries."""
        if not exclude:
            return
        slots = {}
        for t in range(VECTOR_LEN):
            h = self.hand_tails[t]
            if h == 0:
                continue
            var = self._y_joker_var if t == JOKER_INDEX else self._y_vars[t]
            if h == 1:
                slots[t] = [var]
                continue
            bs = [pulp.LpVariable(f"b{tag}_{t}_{q}", cat="Binary") for q in range(h)]
            prob += var == pulp.lpSum(bs)
            for q in range(h - 1):
                prob += bs[q] >= bs[q + 1]
            slots[t] = bs
        for si, dv in enumerate(exclude):
            on, off = [], []
            for t, bs in slots.items():
                for q, b in enumerate(bs):
                    (on if q < dv[t] else off).append(b)
            prob += (
                pulp.lpSum([1 - b for b in on]) + pulp.lpSum(off) >= 1,
                f"nogood_{tag}_{si}",
            )

    # -------------------------------------------------------------- solving

    def _solve(self, objective, sources, tiles_eq=None, jokers_max=None,
               jokers_min=None, exclude=()):
        """One ILP solve under optional side constraints. Returns an action
        dict, or None if infeasible or if the only answer is the passive
        place-nothing solution."""
        self._solve_id += 1
        tag = str(self._solve_id)
        prob = self._build_base_problem()

        if tiles_eq is not None:
            prob += (self._tiles_expr() == tiles_eq, f"tiles_eq_{tag}")
        if jokers_max is not None:
            prob += (self._y_joker_var <= jokers_max, f"jmax_{tag}")
        if jokers_min is not None:
            prob += (self._y_joker_var >= jokers_min, f"jmin_{tag}")
        self._add_nogoods(prob, exclude, tag)

        prob += objective
        if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))] != "Optimal":
            return None
        sol = self._extract(sources)
        return sol if sol["tiles_placed"] > 0 else None

    # ------------------------------------------------------- the action set

    def build_action_set(self):
        """Return a flat list of distinct legal actions, richest first.

        Three tiers:
          1. COMMITMENT LADDER — the best play at each achievable tile count,
             from "empty the rack" down to "place the bare minimum". Tile count
             is the axis that matters: how much of the rack you commit now
             versus what you keep for future melds.
          2. COMPOSITION ALTERNATIVES — at the top rungs, further plays using
             the SAME number of tiles from different slots. Same commitment,
             different hand kept back.
          3. JOKER ANCHORS — the best play that keeps every joker and the best
             that spends one, whenever the array is otherwise one-sided.

        Returns [] when no legal move exists; the caller offers a draw instead.
        """
        if self.hand_tails is None or self.board_tails is None:
            raise ValueError(
                "Call reset(hand_tails, board_tails) with real data before solving."
            )

        cap = self.budget["max_actions"]
        actions, seen = [], {}

        def add(sol):
            if sol is None:
                return
            key = tuple(sol["dropped_vector"])
            if key in seen:
                for s in sol["sources"]:
                    if s not in seen[key]["sources"]:
                        seen[key]["sources"].append(s)
                return
            seen[key] = sol
            actions.append(sol)

        # --- tier 1: the ladder ---------------------------------------------
        top = self._solve(self._tiles_then_value(), ["max_tiles"])
        if top is None:
            return []
        add(top)

        counts = [top["tiles_placed"]]
        for k in range(top["tiles_placed"] - 1, 0, -1):
            if len(actions) >= cap:
                break
            sol = self._solve(self._value_expr(), [f"ladder_k{k}"], tiles_eq=k)
            if sol is not None:
                add(sol)
                counts.append(k)

        # --- tier 2: alternatives at the richest rungs ----------------------
        for k in counts[: self.budget["alt_counts"]]:
            excl = [a["dropped_vector"] for a in actions if a["tiles_placed"] == k]
            for _ in range(self.budget["alts_per_count"]):
                if len(actions) >= cap:
                    break
                sol = self._solve(
                    self._value_expr(), [f"alt_k{k}"], tiles_eq=k, exclude=excl
                )
                if sol is None:
                    break
                add(sol)
                excl.append(sol["dropped_vector"])

        # --- tier 3: joker anchors ------------------------------------------
        if self.hand_tails[JOKER_INDEX] > 0 and len(actions) < cap:
            spent = {a["jokers_placed"] for a in actions}
            if 0 not in spent:
                add(self._solve(
                    self._tiles_then_value(), ["keep_jokers"], jokers_max=0))
            if not any(j > 0 for j in spent) and len(actions) < cap:
                add(self._solve(
                    self._tiles_then_value(), ["spend_joker"], jokers_min=1))

        return actions[:cap]