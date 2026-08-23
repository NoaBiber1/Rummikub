import pulp
from itertools import combinations
from collections import namedtuple
from .ilp_tree import Tree

NUM_COLORS = 4
NUM_VALUES = 13
NUM_REAL_TILES = NUM_COLORS * NUM_VALUES
JOKER_INDEX = NUM_REAL_TILES
VECTOR_LEN = NUM_REAL_TILES + 1

JOKER_VALUE = 30

MAX_TREE_DEPTH = 3

def tile_index(color, value):
    return (color - 1) * NUM_VALUES + (value - 1)

def tile_point_value(index):
    return index % NUM_VALUES + 1

Meld = namedtuple("Meld", ["real_indices", "num_jokers"])

DEFAULT_CATEGORY_QUOTAS = [
    ("tiles", [8, 3]),
    ("value", [8, 3]),
    ("tiles_then_value", [6, 2]),
    ("value_then_tiles", [6, 2]),
]

TRAINING_CATEGORY_QUOTAS = [
    ("tiles", [3, 2]),
    ("value", [3, 2]),
    ("tiles_then_value", [2, 1]),
    ("value_then_tiles", [2, 1]),
]

class ILP_solutions:
    M = 10000

    def __init__(self, category_quotas=None):
        self.category_quotas = category_quotas or TRAINING_CATEGORY_QUOTAS
        self.valid_sets = self.generate_valid_sets()
        self.reset(hand_tails=None, board_tails=None)

    def reset(self, hand_tails, board_tails):
        self.hand_tails = hand_tails
        self.board_tails = board_tails
        self._base_built = False

    def generate_valid_sets(self):
        melds = self._generate_groups() + self._generate_runs()
        self.melds = melds

        self.tile_to_melds = {t: [] for t in range(NUM_REAL_TILES)}
        for i, m in enumerate(melds):
            for idx in m.real_indices:
                self.tile_to_melds[idx].append(i)

        return melds

    def _generate_groups(self):
        melds = []
        colors = list(range(1, NUM_COLORS + 1))
        for value in range(1, NUM_VALUES + 1):
            for size in (3, 4):
                for r in range(max(1, size - 2), min(size, NUM_COLORS) + 1):
                    j = size - r
                    for color_subset in combinations(colors, r):
                        real_indices = tuple(
                            sorted(tile_index(c, value) for c in color_subset)
                        )
                        melds.append(Meld(real_indices=real_indices, num_jokers=j))
        return melds

    def _generate_runs(self):
        melds = []
        for color in range(1, NUM_COLORS + 1):
            for length in range(3, 6):
                for start in range(1, NUM_VALUES - length + 2):
                    values = list(range(start, start + length))
                    indices = [tile_index(color, v) for v in values]
                    for j in range(0, min(2, length) + 1):
                        for joker_positions in combinations(range(length), j):
                            real_positions = [
                                p for p in range(length) if p not in joker_positions
                            ]
                            real_indices = tuple(indices[p] for p in real_positions)
                            melds.append(Meld(real_indices=real_indices, num_jokers=j))
        return melds

    def _ensure_base_built(self):
        if self._base_built:
            return
        if self.hand_tails is None or self.board_tails is None:
            raise ValueError(
                "Call reset(hand_tails, board_tails) with real data before solving."
            )

        self._x_vars = [
            pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer")
            for i in range(len(self.melds))
        ]
        self._y_vars = [
            pulp.LpVariable(f"y_{t}", lowBound=0, upBound=self.hand_tails[t],
                             cat="Integer")
            for t in range(NUM_REAL_TILES)
        ]
        self._y_joker_var = pulp.LpVariable(
            "y_joker", lowBound=0, upBound=self.hand_tails[JOKER_INDEX], cat="Integer"
        )

        self._tile_lhs = [
            pulp.lpSum(self._x_vars[i] for i in self.tile_to_melds.get(t, []))
            for t in range(NUM_REAL_TILES)
        ]
        self._joker_lhs = pulp.lpSum(
            self._x_vars[i] * self.melds[i].num_jokers for i in range(len(self.melds))
        )

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

        return prob, self._x_vars, self._y_vars, self._y_joker_var

    def _objective_expr(self, kind, y_vars, y_joker_var):
        tiles_expr = pulp.lpSum(y_vars[t] for t in range(NUM_REAL_TILES)) + y_joker_var
        value_expr = pulp.lpSum(
            y_vars[t] * tile_point_value(t) for t in range(NUM_REAL_TILES)
        ) + y_joker_var * JOKER_VALUE
        if kind == "tiles":
            return tiles_expr
        if kind == "value":
            return value_expr
        if kind == "tiles_then_value":
            return self.M * tiles_expr + value_expr
        if kind == "value_then_tiles":
            return self.M * value_expr + tiles_expr
        raise ValueError(f"unknown objective kind: {kind}")

    def _scalar_objective_value(self, kind, tiles_val, value_val):
        if kind == "tiles":
            return tiles_val
        if kind == "value":
            return value_val
        if kind == "tiles_then_value":
            return self.M * tiles_val + value_val
        if kind == "value_then_tiles":
            return self.M * value_val + tiles_val
        raise ValueError(f"unknown objective kind: {kind}")

    def _extract_solution(self, kind, y_vars, y_joker_var):
        y = [int(round(v.value())) for v in y_vars]
        y_joker = int(round(y_joker_var.value()))
        dropped_vector = y + [y_joker]

        tiles_val = sum(y) + y_joker
        value_val = sum(y[t] * tile_point_value(t) for t in range(NUM_REAL_TILES))
        value_val += y_joker * JOKER_VALUE

        return {
            "tiles_placed": tiles_val,
            "value_placed": value_val,
            "dropped_vector": dropped_vector,
            "objective": self._scalar_objective_value(kind, tiles_val, value_val),
        }

    def _find_next_solution(self, node, kind, siblings):
        prob, x_vars, y_vars, y_joker_var = self._build_base_problem()

        if not node.is_root():
            parent_sol = node.solution
            for t in range(NUM_REAL_TILES):
                prob += y_vars[t] <= parent_sol["dropped_vector"][t]
            prob += y_joker_var <= parent_sol["dropped_vector"][JOKER_INDEX]
            prob += (
                self._objective_expr(kind, y_vars, y_joker_var)
                <= parent_sol["objective"] - 1
            )

        for sib in siblings:
            active = [t for t in range(NUM_REAL_TILES) if sib.solution["dropped_vector"][t] > 0]
            terms = [y_vars[t] for t in active]
            rhs = sum(sib.solution["dropped_vector"][t] for t in active)
            if sib.solution["dropped_vector"][JOKER_INDEX] > 0:
                terms.append(y_joker_var)
                rhs += sib.solution["dropped_vector"][JOKER_INDEX]
            if terms:
                prob += pulp.lpSum(terms) <= rhs - 1

        prob += self._objective_expr(kind, y_vars, y_joker_var)

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            return None

        sol = self._extract_solution(kind, y_vars, y_joker_var)
        if sol["tiles_placed"] == 0:
            return None
        return sol

    def _build_tree(self, kind, branching):
        root = Tree.crate()
        self._expand_node(root, kind, branching, level=0)
        return root

    def _expand_node(self, node, kind, branching, level):
        if level >= len(branching) or level >= MAX_TREE_DEPTH:
            return
        for _ in range(branching[level]):
            sol = self._find_next_solution(node, kind, node.children)
            if sol is None:
                break
            child = node.add_child(sol)
            self._expand_node(child, kind, branching, level + 1)

    def get_action_set(self):
        trees = {}
        flat = []
        for kind, branching in self.category_quotas:
            tree = self._build_tree(kind, branching)
            trees[kind] = tree
            offset = len(flat)
            entries = tree.to_flat_list(category=kind)
            for e in entries:
                if e["parent_index"] is not None:
                    e["parent_index"] += offset
            flat.extend(entries)
        return trees, flat