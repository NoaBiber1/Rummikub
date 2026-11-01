"""Greedy Rummikub move finder: place as many hand tiles as possible (PuLP 4.2 + HiGHS).

Usage::

    from rummikub_greedy import greedy_move, DRAW

    result = greedy_move(hand, board)
    if result == DRAW:
        ...
    else:
        hand, action, sBoard, numOfTails = result

Input contract: ``hand`` and ``board`` are lists of tiles, each ``"J"`` or a
tuple ``(number, color)`` (see ``rummikub_tiles``). ``board`` is flat, the
tiles on the table without any set structure. Before anything else the board
is checked to split completely and disjointly into legal sets, every tile in
exactly one set; if it does not, ``ValueError`` is raised whatever the hand.

Model, after den Hertog and Hulshof, "Solving Rummikub problems by integer
linear programming", The Computer Journal 49(6), 2006, with jokers decoupled
from the sets. Each catalogue template ``s`` (a joker-free group or run, see
``rummikub_melds``) gets an integer ``x_s`` in ``0..2`` counting its copies on
the table after the move, and each real tile type ``t`` gets an integer
``v_t`` counting the jokers that stand in for a ``t`` somewhere on the table.
With ``b_t`` and ``h_t`` the board and hand counts of ``t`` and ``a_ts`` its
multiplicity in ``s``::

    maximise    sum_s w_s x_s  +  sum_t (30 - value_t) v_t
    subject to  b_t <= sum_s a_ts x_s - v_t <= b_t + h_t      for every real type t
                b_J <= sum_t v_t <= b_J + h_J                  for the jokers

The middle expression is the number of real ``t`` tiles on the table. Its
lower bound keeps every board tile there (the board may be rearranged freely
but never taken from); its upper bound places each hand tile at most once.
The formulation is exact: every set has at least three tiles and the game has
only two jokers, so a template copy keeps a real tile however the jokers fall,
and ``v_t`` jokers can always be dealt into ``v_t`` distinct copies holding a
``t``. It needs one variable per usable template and per tile type instead of
one per joker pattern (185 templates in place of 1,173 joker-aware melds).

``w_s = LEX_WEIGHT * size_s + points_s``, so up to a constant the objective is
``LEX_WEIGHT`` per tile placed plus the penalty points placed (face value,
joker 30). ``LEX_WEIGHT`` exceeds the 788 points of a whole tile set, so one
extra tile always outweighs any gain in points: the scalarisation is exactly
lexicographic, maximising tiles first and points shed among equal-tile plays.

Board validation solves the same constraints over the board alone with
``b_t <= ... <= b_t`` (an exact partition) and no objective. A board tile that
fits no template the board can fill is rejected at once, before any solver
call. Validations are memoised, and every board this solver itself produces is
recorded as valid, since its partition was just built; in a game where those
boards come back to the solver, the check then costs nothing. Whole positions
are not memoised: an exact repeat of hand, board and meld state is vanishingly
rare in real play, so such a cache would only cost memory.

Solver: HiGHS, in process, through PuLP 4.2's API, with a zero MIP gap (a
solver that stops "within tolerance" may return a non-optimal play) and
presolve off (faster and with a shorter tail on these small models; HiGHS's
presolve also returned infeasible points on an earlier form of this model).

Initial meld (``first_meld=True``): the official rules require a player's
first play to consist of new sets built only from their own rack and worth at
least ``INITIAL_MELD_POINTS``, jokers counted at the value they represent, with
the board left untouched. That model covers the hand alone and adds a binary
``p`` with ``sum points_s x_s >= 30 p`` and ``sum size_s x_s <= |hand| p``, so
playing nothing stays feasible.
"""

import re
import threading
from functools import lru_cache
from typing import NamedTuple

import pulp

from rummikub_melds import assemble, catalogue
from rummikub_tiles import (
    JOKER,
    JOKER_POINTS,
    MAX_COPIES,
    NUM_REAL,
    NUM_TYPES,
    NUM_VALUES,
    check_copy_limits,
    count_types,
    parse_tile,
    tile_value,
)

REQUIRED_PULP = (4, 2)
DRAW = "Draw a tile"
INITIAL_MELD_POINTS = 30
LEX_WEIGHT = 1000
DEFAULT_CACHE_SIZE = 1 << 16
_ALL_REAL = (1 << NUM_REAL) - 1
_ZEROS = (0,) * NUM_TYPES


def _pulp_version(text):
    """Return the leading ``(major, minor)`` of a version string, or ``(0, 0)``."""
    match = re.match(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


if _pulp_version(pulp.__version__) < REQUIRED_PULP:
    raise ImportError(f"rummikub_greedy needs PuLP 4.2 or newer; found {pulp.__version__}")


class MoveResult(NamedTuple):
    """The four-part answer for a turn on which tiles can be played.

    Attributes:
        hand: Tiles left in the hand, in their original order.
        action: Tiles moved from the hand to the table, in hand order.
        sBoard: Every set on the table after the move, each a list of tiles in
            display order (a joker sits in the position whose value it takes);
            together they hold each board tile and each played tile exactly once.
        numOfTails: Number of tiles played, equal to ``len(action)``.
    """

    hand: list
    action: list
    sBoard: list
    numOfTails: int


def highs_solver():
    """Return the HiGHS configuration every model is solved with.

    Quiet, zero absolute and relative MIP gap, presolve off.

    Raises:
        RuntimeError: If the ``highspy`` package is not installed.
    """
    solver = pulp.HiGHS(msg=False, gapRel=0, gapAbs=0, presolve="off")
    if not solver.available():
        raise RuntimeError("the HiGHS solver is required: pip install highspy")
    return solver


def _parse_list(tiles, name):
    """Return the type indices of a list of tiles.

    Raises:
        TypeError: If ``tiles`` is not a list.
        ValueError: If any element is not a tile in the strict format.
    """
    if not isinstance(tiles, list):
        raise TypeError(f"{name} must be a list of tiles, got {type(tiles).__name__}")
    try:
        return [parse_tile(tile) for tile in tiles]
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None


def _mask_of(counts):
    """Return the bitmask of real tile types whose count is positive."""
    mask = 0
    for t in range(NUM_REAL):
        if counts[t]:
            mask |= 1 << t
    return mask


def _bits(mask):
    """Return the number of set bits in a non-negative integer."""
    return bin(mask).count("1")


class _Model:
    """Incremental builder for one PuLP problem over catalogue templates and joker slots."""

    def __init__(self, melds):
        """Start an empty maximisation problem over the given catalogue."""
        self.melds = melds
        self.problem = pulp.LpProblem("rummikub_greedy", pulp.LpMaximize)
        self.has_rows = False

    def variable(self, name, upper, category):
        """Create a variable bounded by ``0..upper`` in this problem."""
        return self.problem.add_variable(name, 0, upper, category)

    def _row(self, terms, floor, ceiling, cap, floor_binds):
        """Emit the rows ``floor <= sum(terms) <= ceiling`` that can bind.

        Args:
            terms: ``(var, coef)`` pairs.
            floor: Lower limit.
            ceiling: Upper limit.
            cap: Largest value the sum can reach within variable bounds.
            floor_binds: Whether the sum can fall below ``floor`` within bounds.
        """
        problem = self.problem
        if floor == ceiling:
            problem.addConstraint(pulp.lpSum_vars_coefs(terms) == floor)
            self.has_rows = True
            return
        upper_binds = ceiling < cap
        if not (floor_binds or upper_binds):
            return
        expression = pulp.lpSum_vars_coefs(terms)
        if floor_binds:
            problem.addConstraint(expression >= floor)
        if upper_binds:
            problem.addConstraint(expression <= ceiling)
        self.has_rows = True

    def add_block(self, prefix, active, avail, low, high):
        """Add template and joker variables for one pool of tiles, with their rows.

        Args:
            prefix: Variable-name prefix.
            active: Catalogue indices of the templates this block may use.
            avail: Copies of each type the pool holds, bounding the variables.
            low: Per-type minimum number of tiles on the table.
            high: Per-type maximum number of tiles on the table.

        Returns:
            ``(templates, jokers)``: ``(catalogue_index, variable, bound)``
            triples and ``(tile_type, variable)`` pairs.
        """
        melds = self.melds
        jokers = avail[JOKER]
        rows = [[] for _ in range(NUM_REAL)]
        cap = [0] * NUM_REAL
        templates = []
        for i in active:
            layout = melds[i].layout
            bound = MAX_COPIES
            missing = 0
            for t in layout:
                have = avail[t]
                if not have:
                    missing += 1
                if have + jokers < bound:
                    bound = have + jokers
            if missing:
                bound = min(bound, jokers // missing)
            var = self.variable(f"{prefix}{i}", bound, pulp.LpInteger)
            templates.append((i, var, bound))
            for t in layout:
                rows[t].append((var, 1))
                cap[t] += bound
        stand_ins = []
        if jokers:
            for t in range(NUM_REAL):
                if rows[t]:
                    var = self.variable(f"{prefix}j{t}", jokers, pulp.LpInteger)
                    stand_ins.append((t, var))
                    rows[t].append((var, -1))
        for t in range(NUM_REAL):
            if rows[t]:
                self._row(rows[t], low[t], high[t], cap[t], low[t] > 0 or jokers > 0)
        if stand_ins:
            terms = [(var, 1) for _, var in stand_ins]
            self._row(terms, low[JOKER], high[JOKER], jokers * len(stand_ins), low[JOKER] > 0)
        return templates, stand_ins

    def add_row(self, terms, sense, rhs):
        """Add a single linear row ``sum(coef * var) <sense> rhs``."""
        expression = pulp.lpSum_vars_coefs(terms)
        if sense == ">=":
            self.problem.addConstraint(expression >= rhs)
        else:
            self.problem.addConstraint(expression <= rhs)
        self.has_rows = True

    def solve(self, solver, objective):
        """Optimise the problem with HiGHS.

        Args:
            solver: The configured HiGHS instance.
            objective: ``(var, coef)`` pairs; empty for a pure feasibility check.

        Returns:
            ``True`` for an optimal solution, ``False`` for a proven infeasible model.

        Raises:
            RuntimeError: For any other solver outcome.
        """
        self.problem.setObjective(pulp.lpSum_vars_coefs(objective))
        status = self.problem.solve(solver).status
        if status == pulp.LpSolveStatus.Optimal:
            return True
        if status == pulp.LpSolveStatus.Infeasible:
            return False
        raise RuntimeError(f"HiGHS ended with status {getattr(status, 'name', status)}")


def _read_templates(templates):
    """Return ``(catalogue_index, copies)`` for every template variable set above zero."""
    chosen = []
    for i, var, _ in templates:
        copies = int(round(var.value() or 0))
        if copies:
            chosen.append((i, copies))
    return chosen


def _read_jokers(stand_ins):
    """Return ``(tile_type, jokers)`` for every joker variable set above zero."""
    used = []
    for t, var in stand_ins:
        count = int(round(var.value() or 0))
        if count:
            used.append((t, count))
    return used


class GreedySolver:
    """Max-tiles Rummikub move finder backed by a PuLP 4.2 integer program solved by HiGHS.

    Construction is cheap; every call to ``solve`` validates the board and
    builds a small model for the position at hand. Instances are reusable and
    may be shared between threads.
    """

    def __init__(self, cache_size=DEFAULT_CACHE_SIZE):
        """Prepare the solver configuration and the static template data.

        Args:
            cache_size: Number of board partitions and produced boards to
                memoise; ``0`` disables all memoisation.

        Raises:
            RuntimeError: If HiGHS is not installed.
        """
        self._solver = highs_solver()
        melds = catalogue()
        self._melds = melds
        self._masks = tuple(m.mask for m in melds)
        self._weights = tuple(LEX_WEIGHT * m.size + m.points for m in melds)
        self._joker_weights = tuple(JOKER_POINTS - tile_value(t) for t in range(NUM_REAL))
        self._cache_size = cache_size
        self._trusted = {}
        self._trusted_lock = threading.Lock()
        if cache_size:
            self._partition = lru_cache(maxsize=cache_size)(self._partition_board)
        else:
            self._partition = self._partition_board

    def cache_clear(self):
        """Forget every memoised board partition and produced board."""
        if hasattr(self._partition, "cache_clear"):
            self._partition.cache_clear()
        with self._trusted_lock:
            self._trusted.clear()

    def solve(self, hand, board, first_meld=False):
        """Find the play that puts the most hand tiles on the table.

        Args:
            hand: List of the player's tiles.
            board: Flat list of the tiles on the table.
            first_meld: ``True`` while the player has not yet made the initial
                meld; the play must then be new sets from the hand alone worth
                at least ``INITIAL_MELD_POINTS``, with the board untouched.

        Returns:
            A ``MoveResult`` ``(hand, action, sBoard, numOfTails)`` when at least
            one tile can be played, otherwise the string ``DRAW``.

        Raises:
            TypeError: If ``hand`` or ``board`` is not a list.
            ValueError: On a tile outside the strict format, more than two
                copies of a tile, or a board that cannot be split completely
                and disjointly into legal sets.
            RuntimeError: If HiGHS fails, or returns a solution that breaks the
                position's constraints.
        """
        hand_types = _parse_list(hand, "hand")
        board_types = _parse_list(board, "board")
        hand_counts = count_types(hand_types)
        board_counts = count_types(board_types)
        check_copy_limits(hand_counts, board_counts)
        answer = self._compute(tuple(board_counts), tuple(hand_counts), bool(first_meld))
        if answer is None:
            return DRAW
        played, layouts = answer
        remaining = list(played)
        action, action_types, new_hand = [], [], []
        for tile, t in zip(hand, hand_types):
            if remaining[t]:
                remaining[t] -= 1
                action.append(tile)
                action_types.append(t)
            else:
                new_hand.append(tile)
        pools = [[] for _ in range(NUM_TYPES)]
        for tile, t in zip(board, board_types):
            pools[t].append(tile)
        for tile, t in zip(action, action_types):
            pools[t].append(tile)
        for pool in pools:
            pool.reverse()
        s_board = [[pools[t].pop() for t in layout] for layout in layouts]
        return MoveResult(new_hand, action, s_board, len(action))

    def _usable(self, avail):
        """Return the templates the available tiles and jokers can fill, plus their coverage.

        A template is usable when the jokers in ``avail`` can stand in for
        every one of its types that ``avail`` lacks.

        Returns:
            ``(active, covered)``: catalogue indices, and the bitmask of real
            types appearing in at least one of those templates.
        """
        absent = _ALL_REAL ^ _mask_of(avail)
        jokers = avail[JOKER]
        if jokers:
            active = [i for i, mask in enumerate(self._masks) if _bits(mask & absent) <= jokers]
        else:
            active = [i for i, mask in enumerate(self._masks) if not mask & absent]
        covered = 0
        for i in active:
            covered |= self._masks[i]
        return active, covered

    def _partition_board(self, board):
        """Split the board into legal sets, or prove that it cannot be done.

        Returns:
            The display sets of one complete, disjoint partition.

        Raises:
            ValueError: If the board tiles cannot form legal sets.
        """
        active, covered = self._usable(board)
        if _mask_of(board) & ~covered or (board[JOKER] and not active):
            raise ValueError("board tiles cannot be partitioned into valid sets")
        model = _Model(self._melds)
        templates, stand_ins = model.add_block("b", active, board, board, board)
        if not model.solve(self._solver, []):
            raise ValueError("board tiles cannot be partitioned into valid sets")
        chosen, jokers = _read_templates(templates), _read_jokers(stand_ins)
        if self._placed(chosen, jokers) != list(board):
            raise RuntimeError("HiGHS returned a board partition inconsistent with the board")
        return tuple(assemble(self._deal(chosen, jokers)))

    def _validate(self, board):
        """Raise ``ValueError`` unless the board splits into legal sets; free for known boards."""
        if any(board) and board not in self._trusted:
            self._partition(board)

    def _trust(self, board, played):
        """Record the board left by a move as valid, its partition having just been built."""
        if not self._cache_size:
            return
        result = tuple(b + p for b, p in zip(board, played))
        with self._trusted_lock:
            self._trusted[result] = True
            if len(self._trusted) > self._cache_size:
                del self._trusted[next(iter(self._trusted))]

    def _compute(self, board, hand, first_meld):
        """Solve one position given as count tuples.

        Returns:
            ``None`` when nothing can be played, otherwise ``(played, layouts)``
            with ``played`` the per-type hand usage and ``layouts`` every set
            on the table after the move.
        """
        if first_meld:
            return self._compute_initial(board, hand)
        return self._compute_turn(board, hand)

    def _objective(self, templates, stand_ins):
        """Return the objective terms for one block of variables."""
        weights, joker_weights = self._weights, self._joker_weights
        terms = [(var, weights[i]) for i, var, _ in templates]
        terms.extend((var, joker_weights[t]) for t, var in stand_ins)
        return terms

    def _compute_turn(self, board, hand):
        """Solve an ordinary turn, in which the whole board may be rearranged."""
        self._validate(board)
        if not any(hand):
            return None
        avail = [b + h for b, h in zip(board, hand)]
        active, covered = self._usable(avail)
        if not (_mask_of(hand) & covered or (hand[JOKER] and active)):
            return None
        model = _Model(self._melds)
        templates, stand_ins = model.add_block("x", active, avail, board, avail)
        if model.has_rows:
            if not model.solve(self._solver, self._objective(templates, stand_ins)):
                raise RuntimeError("HiGHS declared a validated position infeasible")
            chosen, jokers = _read_templates(templates), _read_jokers(stand_ins)
        else:
            chosen, jokers = [(i, bound) for i, _, bound in templates], []
        played = tuple(p - b for p, b in zip(self._placed(chosen, jokers), board))
        if any(u < 0 or u > h for u, h in zip(played, hand)):
            raise RuntimeError("HiGHS returned a play inconsistent with the position")
        if not any(played):
            return None
        self._trust(board, played)
        return played, tuple(assemble(self._deal(chosen, jokers)))

    def _compute_initial(self, board, hand):
        """Solve an initial-meld turn: new hand-only sets worth at least 30 points."""
        board_sets = self._partition(board) if any(board) else ()
        if not any(hand):
            return None
        active, covered = self._usable(hand)
        ceiling = sum(tile_value(t) * hand[t] for t in range(NUM_REAL) if covered >> t & 1)
        ceiling += NUM_VALUES * hand[JOKER]
        if not active or ceiling < INITIAL_MELD_POINTS:
            return None
        melds = self._melds
        model = _Model(melds)
        templates, stand_ins = model.add_block("z", active, hand, _ZEROS, hand)
        play = model.variable("p", 1, pulp.LpBinary)
        model.add_row(
            [(var, melds[i].points) for i, var, _ in templates] + [(play, -INITIAL_MELD_POINTS)], ">=", 0
        )
        model.add_row([(var, melds[i].size) for i, var, _ in templates] + [(play, -sum(hand))], "<=", 0)
        if not model.solve(self._solver, self._objective(templates, stand_ins)):
            raise RuntimeError("HiGHS declared an always-feasible initial-meld model infeasible")
        chosen, jokers = _read_templates(templates), _read_jokers(stand_ins)
        played = tuple(self._placed(chosen, jokers))
        if any(u < 0 or u > h for u, h in zip(played, hand)):
            raise RuntimeError("HiGHS returned a play inconsistent with the position")
        if not any(played):
            return None
        if sum(melds[i].points * k for i, k in chosen) < INITIAL_MELD_POINTS:
            raise RuntimeError("HiGHS returned an initial meld below the points threshold")
        self._trust(board, played)
        return played, board_sets + tuple(assemble(self._deal(chosen, jokers)))

    def _placed(self, chosen, jokers):
        """Return per-type counts of the tiles a solution puts on the table, jokers included."""
        placed = [0] * NUM_TYPES
        melds = self._melds
        for i, copies in chosen:
            for t in melds[i].layout:
                placed[t] += copies
        for t, count in jokers:
            placed[t] -= count
            placed[JOKER] += count
        return placed

    def _deal(self, chosen, jokers):
        """Expand chosen templates into copies and seat each joker in a copy holding its type.

        Returns:
            A list of ``(meld, layout)`` pairs, one per copy, with ``JOKER``
            in the positions jokers take.

        Raises:
            RuntimeError: If a joker finds no free position of its type.
        """
        melds = self._melds
        copies = [(melds[i], list(melds[i].layout)) for i, k in chosen for _ in range(k)]
        for t, count in jokers:
            for _, layout in copies:
                if not count:
                    break
                if t in layout:
                    layout[layout.index(t)] = JOKER
                    count -= 1
            if count:
                raise RuntimeError("HiGHS placed a joker with no position to fill")
        return [(meld, tuple(layout)) for meld, layout in copies]


_shared = None
_shared_lock = threading.Lock()


def _shared_solver():
    """Return the lazily created process-wide ``GreedySolver``."""
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = GreedySolver()
    return _shared


def greedy_move(hand, board, first_meld=False):
    """Return the play that puts the most hand tiles on the table, or ``DRAW``.

    Convenience wrapper around a shared ``GreedySolver``; see
    ``GreedySolver.solve`` for arguments, return value and errors.
    """
    return _shared_solver().solve(hand, board, first_meld)