"""
Rummikub turn solver: finds up to 100 candidate ACTIONS (each a length-53
"tiles dropped from hand" vector, matching the same encoding used for hand,
board, and state elsewhere in the model) spanning 4 objective categories,
each returned as a tree of related actions.

----------------------------------------------------------------------------
MODEL (board-inclusive ILP, "Part II" of the earlier formulation)
----------------------------------------------------------------------------
Data:
    T           = 52 real tile types (color 1-4 x value 1-13)
    h_t, h_J    = copies of real tile t / jokers currently in HAND
    b_t, b_J    = copies of real tile t / jokers currently on the BOARD
    S           = every combinatorially valid meld (group or run), see
                  generate_valid_sets()

Variables:
    x_s in Z>=0             how many times meld s is played
    y_t in [0, h_t]          how many copies of hand tile t get placed
    y_J in [0, h_J]          how many hand jokers get placed

Constraints (per real tile type t, plus one for jokers):
    sum_s a_{t,s} x_s == b_t + y_t      (board tiles are MANDATORY: every
                                          board tile must reappear in some
                                          meld; hand tiles are OPTIONAL,
                                          bounded by what's in hand)
    sum_s j_s x_s     == b_J + y_J

Objective (4 variants, selectable):
    "tiles"             : maximize sum_t y_t + y_J
    "value"             : maximize sum_t y_t * point_value(t) + y_J * JOKER_VALUE
    "tiles_then_value"  : lexicographic, tiles primary / value tie-break
    "value_then_tiles"  : lexicographic, value primary / tiles tie-break
    Lexicographic variants are implemented by scalarizing (M*primary +
    secondary, M=10000) rather than two-phase solving, so every category
    shares the same solve/search code.

    The value objective charges each hand tile its own face value, and each
    hand joker a fixed JOKER_VALUE (30 -- the standard Rummikub rulebook
    value for a joker held at game end). This directly measures "value of
    what came from hand" with no ambiguity, sidestepping the joker
    fungibility problem entirely (a joker's value here never depends on
    which meld it ends up in) rather than approximating it via total board
    value.

----------------------------------------------------------------------------
CANDIDATE SET (S) AND PRUNING
----------------------------------------------------------------------------
S = all groups (size 3-4, one value, distinct colors, 0-2 jokers) + all runs
of ATOMIC length 3-5 (one color, consecutive values, 0-2 jokers).

Runs of length >= 6 are deliberately excluded. Cutting any run at position 3
produces a length-3 run and a length-(L-3) run covering the exact same
slots -- same real tiles, same joker count, nothing shared or dropped.
Repeating this cut lands on pieces of length 3, 4, or 5. So any solution
using a long run has an exactly equivalent solution (identical tiles used,
identical value, identical joker usage) built from atomic runs alone --
meaning the ILP's optimum is unaffected by dropping them, while |S| shrinks
from ~7,700 to ~1,650 melds.

----------------------------------------------------------------------------
DIVERSE + SUB-ACTION TREE SEARCH (max depth 3 below the root)
----------------------------------------------------------------------------
For each of the 4 categories, a Tree (see ilp_tree.py) is grown breadth-first,
capped at MAX_TREE_DEPTH (default 3) levels below the root:
    depth 0: root (no action)
    depth 1: top-level actions -- mutually distinct solutions for that
             category's objective (found by re-solving with a "diversity"
             constraint against every previously-found sibling: the total
             usage of whichever hand tiles that sibling used must strictly
             decrease)
    depth 2: sub-actions of a depth-1 action -- hand usage elementwise <=
             the parent's, scalar objective strictly lower
    depth 3: sub-actions of a depth-2 sub-action (same rule, one level
             deeper) -- this is the deepest level searched; depth-3 nodes
             are not expanded further

A work queue holds nodes that might still have more children; popping a node
and finding its next child re-queues both the node (it may have further
siblings) and the new child (if below MAX_TREE_DEPTH, it may have its own
sub-actions), until the category's quota is hit or every node in the queue
is exhausted (infeasible on its next-child query).

----------------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------------
get_action_set() returns (trees, flat):
  trees: {category_name: Tree}, structure preserved
  flat:  <=100 dicts, each with:
    - dropped_vector : length-53 list, same convention as hand/board vectors
                        (indices 0-51 real tiles, 52 = joker) -- how many
                        copies of each tile type this action takes from hand.
                        This is the action vector.
    - tiles_placed, value_placed : the two raw objective quantities
    - objective      : the scalar this action was compared against for
                        strict-decrease / diversity constraints
    - category, depth, parent_index : where this action sits in its tree

----------------------------------------------------------------------------
PERFORMANCE
----------------------------------------------------------------------------
Building ~100 fresh LP models from scratch (recreating ~1,700 LpVariable
objects each time) dominated runtime in earlier versions. Since the variable
set and their bounds are identical across every solve within one reset()
call (only the objective and a handful of extra constraints change per
solve), variables and the 53 base-constraint LHS expressions are now built
ONCE per reset() and reused across all ~100 solves; each solve only builds a
fresh LpProblem plus that solve's small number of extra constraints. This
roughly halves per-solve overhead on top of the S-pruning above.
"""
import pulp
from itertools import combinations
from collections import namedtuple
from .ilp_tree import Tree

# ---------------------------------------------------------------------------
# Tile indexing convention (matches the comment on the hand/board vectors):
#   index 0..51  -> real tiles, index = (color-1)*13 + (value-1)
#   index 52     -> joker
# color in 1..4, value in 1..13. If your actual encoding differs (e.g.
# value-major instead of color-major), only tile_index below needs to change
# -- everything else is written in terms of it.
# ---------------------------------------------------------------------------
NUM_COLORS = 4
NUM_VALUES = 13
NUM_REAL_TILES = NUM_COLORS * NUM_VALUES  # 52
JOKER_INDEX = NUM_REAL_TILES              # 52 (the 53rd slot, 0-indexed)
VECTOR_LEN = NUM_REAL_TILES + 1           # 53

# Standard Rummikub rulebook value for a joker (used here to price a hand
# joker in the "value" objective -- see module docstring).
JOKER_VALUE = 30

# How many solution-layers deep each tree may grow below the root
# (root -> depth1 action -> depth2 sub-action -> depth3 sub-sub-action).
MAX_TREE_DEPTH = 3


def tile_index(color, value):
    """color in 1..4, value in 1..13 -> index in 0..51"""
    return (color - 1) * NUM_VALUES + (value - 1)


def tile_point_value(index):
    """index in 0..51 -> face value 1..13 (used for the 'value' objective)."""
    return index % NUM_VALUES + 1


# A candidate meld (group or run), independent of any hand/board state.
#   real_indices: tuple of tile indices (0..51) used by real (non-joker) tiles
#   num_jokers:   how many jokers this meld uses (0, 1, or 2)
# (No size/value fields: size is implicit in len(real_indices)+num_jokers and
# isn't needed anywhere; the value objective is computed directly from hand
# usage, not from melds -- see module docstring.)
Meld = namedtuple("Meld", ["real_indices", "num_jokers"])

# Objective categories and how many actions to collect for each, by default.
# Rationale for the 30/30/20/20 split: "tiles" and "value" are the two pure,
# independent objectives, and are weighted equally and heaviest since they
# represent the two ends of the tradeoff space a player actually cares about.
# The two lexicographic categories are refinements of those same two
# objectives (same primary optimum, just a different tie-break), so they
# have real but smaller marginal value on top of the pure categories -- hence
# fewer of each. This is a default, not a fixed rule: pass a different
# `category_quotas` list to ILP_solutions() to change the split (e.g. equal
# 25/25/25/25, or drop a category to 0 to skip it).
DEFAULT_CATEGORY_QUOTAS = [
    ("tiles", [6, 2, 1]),              # 6 + 6*2 + 6*2*1 = 30 nodes
    ("value", [6, 2, 1]),              # 30 nodes
    ("tiles_then_value", [4, 2, 1]),   # 4 + 4*2 + 4*2*1 = 20 nodes
    ("value_then_tiles", [4, 2, 1]),   # 20 nodes
]                                      # 100 nodes total, matching the
                                       # original 30/30/20/20 design intent
                                       # (see DEFAULT_CATEGORY_QUOTAS
                                       # rationale above), now with
                                       # guaranteed sub-action coverage at
                                       # every depth instead of leaving it to
                                       # search-order chance.

# A much smaller preset for use inside an RL training loop, where
# get_action_set() runs every turn and the full default (~130 total nodes,
# several seconds) is far too slow. Each solve costs ~40-48ms regardless of
# category (dominated by PULP_CBC_CMD spawning CBC as a subprocess per call,
# not by problem size) -- so total node count is the direct lever on
# wall-clock time.
#
# Each entry is (kind, branching), where branching = [n1, n2, n3] is an
# EXPLICIT, guaranteed node count per depth level below the root -- not a
# single quota left to emergent search-order effects. n1 = number of
# distinct top-level actions; n2 = number of sub-actions found for EACH of
# those; n3 = number of sub-sub-actions found for EACH of those. This
# matters: a single flat quota (e.g. "8 solutions total") with a
# breadth-first search essentially never reaches sub-actions in practice,
# since the root always gets re-queued for another top-level sibling before
# any child gets a turn to search its own sub-actions -- so a flat quota of
# 2 per category silently returns ZERO sub-actions, even though "the best
# move is sometimes a sub-action" is exactly why they're worth collecting.
# Explicit branching guarantees representation at every depth instead.
#
# branching=[2,1,1] -> 2 + 2*1 + 2*1*1 = 6 nodes/category (2 top-level, 2
# depth-2 subs, 2 depth-3 sub-subs), 24 total across 4 categories,
# ~1.1s/turn measured steady-state.
TRAINING_CATEGORY_QUOTAS = [
    ("tiles", [2, 1, 1]),
    ("value", [2, 1, 1]),
    ("tiles_then_value", [2, 1, 1]),
    ("value_then_tiles", [2, 1, 1]),
]


class ILP_solutions:
    # Large weight used to scalarize the two lexicographic objectives into a
    # single linear objective (M * primary + secondary). Must dominate the
    # full possible range of the secondary term -- 10000 is comfortably above
    # both the max possible tile count (106) and max possible value sum, so
    # improving the primary objective by even 1 always outranks any change
    # in the secondary term.
    M = 10000

    def __init__(self, category_quotas=None):
        """
        category_quotas: optional list of (kind, quota) pairs overriding
            DEFAULT_CATEGORY_QUOTAS (see its docstring for the reasoning
            behind the default 30/30/20/20 split, and how to change it).
            kind must be one of "tiles", "value", "tiles_then_value",
            "value_then_tiles".
        """
        self.category_quotas = category_quotas or DEFAULT_CATEGORY_QUOTAS
        self.valid_sets = self.generate_valid_sets()  # all valid melds in rummikub
        self.reset(hand_tails=None, board_tails=None)

    def reset(self, hand_tails, board_tails):
        """
        hand_tails / board_tails: length-53 vectors (indices as described above),
        each entry in {0, 1, 2} = number of copies of that tile type currently
        held / on the board.
        """
        self.hand_tails = hand_tails
        self.board_tails = board_tails
        self._base_built = False  # invalidates the cached vars/constraints below

    # -----------------------------------------------------------------
    # Candidate meld (set) generation -- independent of hand/board state
    # -----------------------------------------------------------------
    def generate_valid_sets(self):
        melds = self._generate_groups() + self._generate_runs()
        self.melds = melds

        # Precompute, for each real tile index, which melds use it -- needed
        # to build the tile-availability constraints efficiently.
        self.tile_to_melds = {t: [] for t in range(NUM_REAL_TILES)}
        for i, m in enumerate(melds):
            for idx in m.real_indices:
                self.tile_to_melds[idx].append(i)

        return melds

    def _generate_groups(self):
        """All valid groups: size 3 or 4, one value, distinct colors, 0-2 jokers."""
        melds = []
        colors = list(range(1, NUM_COLORS + 1))
        for value in range(1, NUM_VALUES + 1):
            for size in (3, 4):
                # r = number of real tiles, j = number of jokers, r + j = size
                for r in range(max(1, size - 2), min(size, NUM_COLORS) + 1):
                    j = size - r
                    for color_subset in combinations(colors, r):
                        real_indices = tuple(
                            sorted(tile_index(c, value) for c in color_subset)
                        )
                        melds.append(Meld(real_indices=real_indices, num_jokers=j))
        return melds

    def _generate_runs(self):
        """
        All valid runs of ATOMIC length (3, 4, or 5), one color, consecutive
        values, 0-2 jokers. See module docstring for why longer runs are
        safely excluded (they're always exactly replaceable by combinations
        of these atomic pieces, with zero change in tiles, value, or jokers).
        """
        melds = []
        for color in range(1, NUM_COLORS + 1):
            for length in range(3, 6):  # atomic lengths only: 3, 4, 5
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

    # -----------------------------------------------------------------
    # ILP construction
    # -----------------------------------------------------------------
    def _ensure_base_built(self):
        """Build variables and the 53 base-constraint LHS expressions once per
        reset() call, and cache them -- these are identical across every
        solve for this hand/board, so rebuilding ~1,700 LpVariable objects
        on every one of the ~100 solves would be pure waste."""
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
        """Fresh LP with the board-inclusive balance constraints (Part II model:
        board tiles are mandatory (equality), hand tiles are optional & bounded),
        reusing cached variables/expressions from _ensure_base_built()."""
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
        dropped_vector = y + [y_joker]  # length-53 action vector

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
        """Solve for the next child of `node`: if node is root, the next
        distinct top-level action; otherwise the next distinct proper
        sub-action of node.solution. Returns a solution dict, or None if
        no further distinct action exists."""
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

        # diversity: differ from every sibling already found at this level
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
            return None  # dropping nothing is never a real action -- treat
                          # it the same as "no further distinct solution"

        return sol

    def _build_tree(self, kind, branching):
        """
        branching = [n1, n2, ...] (up to MAX_TREE_DEPTH entries): n1 distinct
        top-level actions are found; for EACH of those, n2 sub-actions are
        found; for each of those, n3 sub-sub-actions, etc. This guarantees
        exact representation at every depth (up to infeasibility cutting a
        branch short), rather than leaving the top-level/sub-action split to
        emergent search-queue ordering.
        """
        root = Tree.crate()
        self._expand_node(root, kind, branching, level=0)
        return root

    def _expand_node(self, node, kind, branching, level):
        if level >= len(branching) or level >= MAX_TREE_DEPTH:
            return
        for _ in range(branching[level]):
            sol = self._find_next_solution(node, kind, node.children)
            if sol is None:
                break  # this node is exhausted -- no more distinct children
            child = node.add_child(sol)
            self._expand_node(child, kind, branching, level + 1)

    def get_action_set(self):
        """
        Builds candidate actions across 4 categories, using this instance's
        category_quotas (see DEFAULT_CATEGORY_QUOTAS / TRAINING_CATEGORY_QUOTAS
        for the format: each entry is (kind, branching), where branching
        explicitly guarantees a node count at each depth -- e.g. [10,3,2]
        means 10 distinct top-level actions, 3 sub-actions per top-level
        action, 2 sub-sub-actions per sub-action).

        Each category is grown as a tree: children of the root are mutually
        distinct actions for that category's objective; children of any
        other node are strict sub-actions of it (dropped_vector elementwise
        <=, and objective strictly lower).

        Returns:
            trees: dict {category_name: Tree}
            flat:  list of action dicts, each tagged with 'category',
                   'depth', and 'parent_index' (index into `flat` of its
                   parent, or None for a top-level action), so the tree
                   structure can be reconstructed without holding onto the
                   Tree objects.
        """
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