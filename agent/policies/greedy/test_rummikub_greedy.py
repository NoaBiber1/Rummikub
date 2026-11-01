"""Tests for the PuLP 4.2 + HiGHS greedy Rummikub solver.

Reference answers come from an exhaustive search written here from first
principles. It shares no code with the solver: it uses runs of any length
rather than the solver's atomic 3-5 catalogue, reads jokers at explicit
positions, and decides by recursion over the tile multiset instead of integer
programming. The same search decides whether a flat board can be split into
legal sets, which checks the solver's board validation. Every answer the
solver returns is also checked directly against the rules: each reported set
is legal as laid out, tiles are conserved, the hand shrinks by exactly the
tiles played, and an initial meld reaches 30.

Run with ``python -m unittest test_rummikub_greedy -v``.
"""

import random
import unittest
from collections import Counter
from functools import lru_cache
from itertools import combinations

import pulp

from rummikub_greedy import DRAW, GreedySolver, MoveResult, greedy_move

COLORS = ("red", "yellow", "blue", "black")
JOKER = "J"
FIRST_MELD_POINTS = 30
NEG = float("-inf")
R, Y, B, K = COLORS


def key(tile):
    """Return the oracle's type index for a test tile: ``(value, colour)`` or ``"J"``."""
    if tile == JOKER:
        return 52
    value, color = tile
    return COLORS.index(color) * 13 + value - 1


def counts(tiles):
    """Return the 53-slot count tuple of a list of test tiles."""
    out = [0] * 53
    for tile in tiles:
        out[key(tile)] += 1
    return tuple(out)


def sets_through(first, avail):
    """Return every legal set that contains real type ``first`` and fits in ``avail``.

    ``first`` must be the lowest-indexed real type still available, so no
    real tile of its colour lies below it and every run through it can have
    only jokers beneath it.

    Returns:
        A list of ``(types, points)`` pairs, jokers counted at the value they hold.
    """
    value, color = first % 13 + 1, first // 13
    jokers = avail[52]
    found = []
    others = [c for c in range(4) if c != color and avail[c * 13 + value - 1]]
    for r in range(len(others) + 1):
        for chosen in combinations(others, r):
            for extra in range(jokers + 1):
                size = 1 + r + extra
                if 3 <= size <= 4:
                    tiles = [first] + [c * 13 + value - 1 for c in chosen] + [52] * extra
                    found.append((tiles, value * size))
    for lead in range(jokers + 1):
        low = value - lead
        if low < 1:
            break
        stack = [(value + 1, [52] * lead + [first], lead)]
        while stack:
            nxt, tiles, used = stack.pop()
            if len(tiles) >= 3:
                found.append((tiles, sum(range(low, low + len(tiles)))))
            if nxt > 13:
                continue
            real = color * 13 + nxt - 1
            if avail[real]:
                stack.append((nxt + 1, tiles + [real], used))
            if used < jokers:
                stack.append((nxt + 1, tiles + [52], used + 1))
    return found


def take(avail, must, tiles):
    """Remove ``tiles`` from ``avail``, crediting each against the compulsory counts ``must``."""
    a, m = list(avail), list(must)
    for t in tiles:
        a[t] -= 1
        m[t] = max(m[t] - 1, 0)
    return tuple(a), tuple(m)


class OracleGaveUp(Exception):
    """Raised when the exhaustive search exceeds its state budget on a position."""


def oracle_turn(board_tiles, hand_tiles, budget=None):
    """Return the most hand tiles an ordinary turn can place, or ``None`` if the board is illegal.

    Args:
        board_tiles: Every tile on the table, as a flat list.
        hand_tiles: The player's tiles.
        budget: Maximum number of distinct search states to evaluate, or
            ``None`` for no limit.

    Raises:
        OracleGaveUp: If the search needs more than ``budget`` states.
    """
    states = [0]

    @lru_cache(maxsize=None)
    def best(avail, must):
        """Most tiles placeable from ``avail`` while placing every copy counted in ``must``."""
        states[0] += 1
        if budget is not None and states[0] > budget:
            raise OracleGaveUp
        first = next((t for t in range(52) if avail[t]), None)
        if first is None:
            return 0 if must[52] == 0 else NEG
        score = NEG
        if avail[first] > must[first]:
            a = list(avail)
            a[first] -= 1
            score = best(tuple(a), must)
        for tiles, _ in sets_through(first, avail):
            rest = best(*take(avail, must, tiles))
            if rest != NEG:
                score = max(score, len(tiles) + rest)
        return score

    board = counts(board_tiles)
    hand = counts(hand_tiles)
    total = best(tuple(b + h for b, h in zip(board, hand)), board)
    return None if total == NEG else total - len(board_tiles)


def oracle_initial(hand_tiles):
    """Return the most tiles an initial meld of new hand-only sets worth 30+ can place, or 0."""

    @lru_cache(maxsize=None)
    def best(avail, need):
        """Most tiles placeable from ``avail`` in sets worth at least ``need`` points in total."""
        first = next((t for t in range(52) if avail[t]), None)
        if first is None:
            return 0 if need == 0 else NEG
        a = list(avail)
        a[first] -= 1
        score = best(tuple(a), need)
        for tiles, points in sets_through(first, avail):
            rest = best(take(avail, avail, tiles)[0], max(0, need - points))
            if rest != NEG:
                score = max(score, len(tiles) + rest)
        return score

    total = best(counts(hand_tiles), FIRST_MELD_POINTS)
    return 0 if total == NEG else total


def laid_out_points(tiles):
    """Return the best points of a set read in the order given, or ``None`` if it is illegal.

    A group may list its tiles in any order; a run must list consecutive values
    left to right, a joker holding the value of its position.
    """
    size = len(tiles)
    reals = [t for t in tiles if t != JOKER]
    if size < 3 or not reals:
        return None
    best = None
    colors = [c for _, c in reals]
    if size <= 4 and len({v for v, _ in reals}) == 1 and len(set(colors)) == len(colors):
        best = reals[0][0] * size
    starts = {tile[0] - pos for pos, tile in enumerate(tiles) if tile != JOKER}
    if len(set(colors)) == 1 and len(starts) == 1:
        start = starts.pop()
        if start >= 1 and start + size - 1 <= 13:
            points = sum(range(start, start + size))
            best = points if best is None else max(best, points)
    return best


def penalty_points(tiles):
    """Return the end-of-game penalty value of test tiles: face value, joker 30."""
    return sum(30 if t == JOKER else t[0] for t in tiles)


def contained(part, whole):
    """Return ``True`` if multiset ``part`` fits inside multiset ``whole``."""
    have = Counter(whole)
    return all(have[t] >= k for t, k in Counter(part).items())


def is_subsequence(part, whole):
    """Return ``True`` if ``part`` appears in ``whole`` in order, by identity."""
    it = iter(whole)
    return all(any(x is y for y in it) for x in part)


def flatten(sets):
    """Return the tiles of a list of sets as one flat list."""
    return [tile for tile_set in sets for tile in tile_set]


def check_move(case, hand, board, result, first_meld=False):
    """Assert every rule-level property of a solver answer and return the number of tiles played.

    ``board`` is the flat board the solver was given. For an initial meld the
    reported sets must start with a partition of the board, followed by new
    sets made only of the played tiles and worth at least 30 points.
    """
    if result == DRAW:
        case.assertEqual(result, "Draw a tile")
        return 0
    case.assertIsInstance(result, MoveResult)
    case.assertEqual(len(result), 4)
    new_hand, action, s_board, played = result
    case.assertEqual(played, len(action))
    case.assertGreater(played, 0)
    case.assertEqual(Counter(new_hand) + Counter(action), Counter(hand))
    case.assertTrue(is_subsequence(new_hand, hand))
    case.assertTrue(is_subsequence(action, hand))
    case.assertEqual(Counter(flatten(s_board)), Counter(board) + Counter(action))
    for tile_set in s_board:
        case.assertIsNotNone(laid_out_points(tile_set), f"illegal set {tile_set}")
    if first_meld:
        splits = [
            k
            for k in range(len(s_board) + 1)
            if Counter(flatten(s_board[:k])) == Counter(board)
            and Counter(flatten(s_board[k:])) == Counter(action)
        ]
        case.assertTrue(splits, "initial meld must leave the board sets apart from the new ones")
        new_sets = s_board[splits[0]:]
        case.assertGreaterEqual(sum(laid_out_points(s) for s in new_sets), FIRST_MELD_POINTS)
    return played


def random_set(rng, top, colors):
    """Return a random legal set laid out in order, before any jokers are substituted."""
    if rng.random() < 0.4:
        value = rng.randint(1, top)
        chosen = rng.sample(colors, rng.choice([3, 4]) if len(colors) >= 4 else 3)
        return [(value, c) for c in chosen]
    length = rng.randint(3, min(7, top))
    start = rng.randint(1, top - length + 1)
    color = rng.choice(colors)
    return [(v, color) for v in range(start, start + length)]


def random_position(rng, top, n_colors, n_sets, hand_size, joker_rate):
    """Return ``(hand, board_sets)`` drawn without replacement from a full two-copy tile set."""
    colors = list(COLORS[:n_colors])
    pool = Counter({(v, c): 2 for c in colors for v in range(1, top + 1)})
    pool[JOKER] = 2
    board = []
    for _ in range(40 * n_sets):
        if len(board) == n_sets:
            break
        candidate = random_set(rng, top, colors)
        for _ in range(2):
            if rng.random() < joker_rate:
                candidate[rng.randrange(len(candidate))] = JOKER
        need = Counter(candidate)
        if laid_out_points(candidate) is None or any(pool[t] < k for t, k in need.items()):
            continue
        pool -= need
        board.append(candidate)
    rest = list(pool.elements())
    rng.shuffle(rest)
    return rest[:hand_size], board


class StubSolver:
    """A stand-in for HiGHS that reports a fixed status without solving anything."""

    def __init__(self, status):
        """Remember the status to report."""
        self.status = status

    def solve(self, problem, **kwargs):
        """Return an object carrying the fixed status, as PuLP 4's solve does."""
        return self


class RuleScenarios(unittest.TestCase):
    """Hand-built positions with known answers, one rule at a time."""

    def play(self, hand, board, first_meld=False):
        """Solve, check the answer against the rules, and return it."""
        result = greedy_move(hand, board, first_meld)
        check_move(self, hand, board, result, first_meld)
        return result

    def test_empty_hand_draws(self):
        """With nothing in hand the answer is the draw string."""
        self.assertEqual(self.play([], [(1, R), (2, R), (3, R)]), DRAW)
        self.assertEqual(self.play([], []), DRAW)

    def test_new_run_from_hand(self):
        """A run held entirely in hand is laid down and the stray tile kept."""
        result = self.play([(5, R), (6, R), (7, R), (1, B)], [])
        self.assertEqual(result.numOfTails, 3)
        self.assertEqual(result.hand, [(1, B)])

    def test_nothing_playable_draws(self):
        """Unrelated tiles produce the draw string."""
        self.assertEqual(self.play([(1, R), (5, B), (9, K)], [(3, Y), (4, Y), (5, Y)]), DRAW)

    def test_board_tiles_cannot_return_to_hand(self):
        """Taking R2 for a group would strand R1 and R3, so nothing is playable."""
        self.assertEqual(self.play([(2, B), (2, Y)], [(1, R), (2, R), (3, R)]), DRAW)

    def test_take_from_four_group_to_extend_run(self):
        """R5 leaves a full four-group to extend a run downward with R4."""
        board = [(5, R), (5, B), (5, Y), (5, K), (6, R), (7, R), (8, R)]
        self.assertEqual(self.play([(4, R)], board).numOfTails, 1)

    def test_split_long_run_around_new_group(self):
        """R4 leaves the middle of a seven-run to join B4 and Y4."""
        board = [(v, R) for v in range(1, 8)]
        self.assertEqual(self.play([(4, B), (4, Y)], board).numOfTails, 2)

    def test_retrieve_joker_and_reuse_it(self):
        """R4 replaces the joker, which then completes a group of sevens."""
        result = self.play([(4, R), (7, B), (7, K)], [(3, R), JOKER, (5, R)])
        self.assertEqual(result.numOfTails, 3)

    def test_two_jokers_with_one_tile(self):
        """Two jokers and one real tile form a legal set."""
        self.assertEqual(self.play([JOKER, JOKER, (7, Y)], []).numOfTails, 3)

    def test_runs_do_not_wrap(self):
        """12-13-1 is not a run, but 12-13 with a joker is."""
        self.assertEqual(self.play([(12, R), (13, R), (1, R)], []), DRAW)
        self.assertEqual(self.play([(12, R), (13, R), JOKER], []).numOfTails, 3)

    def test_group_needs_distinct_colours(self):
        """Two red fives cannot share a group; a joker rescues one of them."""
        self.assertEqual(self.play([(5, R), (5, R), (5, B)], []), DRAW)
        self.assertEqual(self.play([(5, R), (5, R), (5, B), JOKER], []).numOfTails, 3)

    def test_full_thirteen_run_is_merged(self):
        """A whole colour is reported as one 13-tile run, not as atomic pieces."""
        result = self.play([(v, K) for v in range(13, 0, -1)], [])
        self.assertEqual(result.numOfTails, 13)
        self.assertEqual(result.sBoard, [[(v, K) for v in range(1, 14)]])

    def test_extension_reported_as_single_run(self):
        """Extending a seven-run yields one eight-run on the board."""
        result = self.play([(8, R)], [(v, R) for v in range(1, 8)])
        self.assertEqual(result.sBoard, [[(v, R) for v in range(1, 9)]])

    def test_tie_broken_by_points(self):
        """Of two three-tile plays sharing the joker, the higher-scoring one is chosen."""
        result = self.play([(1, R), (2, R), (12, R), (13, R), JOKER], [])
        self.assertEqual(result.numOfTails, 3)
        self.assertEqual(Counter(result.action), Counter([(12, R), (13, R), JOKER]))

    def test_initial_meld_threshold(self):
        """The initial meld needs 30 points from the hand alone; jokers score as the tile they replace."""
        self.assertEqual(self.play([(10, R), (11, R), (12, R)], [], True).numOfTails, 3)
        self.assertEqual(self.play([(9, R), JOKER, (11, R)], [], True).numOfTails, 3)
        self.assertEqual(self.play([(8, R), JOKER, (10, R)], [], True), DRAW)
        self.assertEqual(self.play([JOKER, JOKER, (13, B)], [], True).numOfTails, 3)
        self.assertEqual(self.play([JOKER, JOKER, (5, B)], [], True), DRAW)
        self.assertEqual(self.play([(1, R), (2, R), (3, R), (9, B), (9, Y), (9, K)], [], True).numOfTails, 6)

    def test_initial_meld_ignores_board(self):
        """A tile that only fits a board set cannot count toward the initial meld."""
        board = [(10, R), (11, R), (12, R)]
        self.assertEqual(self.play([(13, R)], board, True), DRAW)
        self.assertEqual(self.play([(13, R)], board).numOfTails, 1)

    def test_initial_meld_partitions_board_first(self):
        """The board comes back as legal sets of its own, followed by the new sets."""
        board = [(1, R), (2, R), (3, R), (4, B), (4, Y), (4, K), JOKER]
        result = self.play([(13, R), (13, B), (13, Y), (2, K)], board, True)
        self.assertEqual(result.numOfTails, 3)
        self.assertEqual(Counter(flatten(result.sBoard[-1:])), Counter([(13, R), (13, B), (13, Y)]))


class InputFormat(unittest.TestCase):
    """Only "J" and (number, color) tuples in flat lists are accepted."""

    def test_rejects_every_other_tile_spelling(self):
        """Reversed tuples, strings, lists, mappings, wrong case, bools, floats and out-of-range values raise ValueError."""
        rejected = [
            ("red", 5),
            "5 red",
            "red5",
            [5, "red"],
            {"value": 5, "color": "red"},
            (5, "Red"),
            (5, "green"),
            (True, "red"),
            (5.0, "red"),
            (14, "red"),
            (0, "red"),
            (5, "red", 1),
            "j",
            "joker",
            "JOKER",
            None,
            52,
        ]
        for tile in rejected:
            with self.subTest(tile=tile):
                with self.assertRaises(ValueError):
                    greedy_move([tile], [])
                with self.assertRaises(ValueError):
                    greedy_move([], [tile, (1, R), (2, R)])

    def test_accepts_the_strict_format(self):
        """Every colour, both ends of the value range and the joker parse and play."""
        hand = [(1, c) for c in COLORS[:3]] + [(13, c) for c in COLORS] + [JOKER]
        self.assertEqual(greedy_move(hand, []).numOfTails, 8)

    def test_board_sets_are_rejected(self):
        """A board given as nested sets is not a flat list of tiles."""
        with self.assertRaises(ValueError):
            greedy_move([(1, Y)], [[(5, R), (6, R), (7, R)]])

    def test_requires_lists(self):
        """Hand and board must be lists, not tuples, strings or None."""
        for hand, board in [(((5, R),), []), ([], ((5, R), (6, R), (7, R))), ("J", []), ([], None)]:
            with self.subTest(hand=hand, board=board):
                with self.assertRaises(TypeError):
                    greedy_move(hand, board)

    def test_rejects_too_many_copies(self):
        """A third copy of any tile, or a third joker, raises ValueError."""
        with self.assertRaises(ValueError):
            greedy_move([(5, R), (5, R)], [(5, R), (5, B), (5, Y)])
        with self.assertRaises(ValueError):
            greedy_move([JOKER, JOKER], [(5, R), JOKER, (7, R)])

    def test_output_reuses_caller_objects(self):
        """Every tile returned is one of the objects passed in."""
        hand = [(4, R), (9, B)]
        board = [(1, R), (2, R), (3, R)]
        result = greedy_move(hand, board)
        self.assertIs(result.action[0], hand[0])
        self.assertIs(result.hand[0], hand[1])
        given = {id(t) for t in hand + board}
        self.assertTrue(all(id(t) in given for t in flatten(result.sBoard)))


class BoardValidation(unittest.TestCase):
    """The flat board must split completely and disjointly into legal sets."""

    def test_illegal_boards_raise_whatever_the_hand(self):
        """An unsplittable board raises even when the hand is empty or could complete it."""
        illegal = [
            [(5, R), (6, R)],
            [(5, R), (6, R), (7, R), (5, R)],
            [(5, R), (5, B), (5, B)],
            [(5, R), (6, B), (7, R)],
            [(12, R), (13, R), (1, R)],
            [(1, R), (2, R), (3, R), (4, R), (6, R), (7, R)],
            [JOKER],
            [(5, R), (6, R), (7, R), (5, R), (6, R)],
        ]
        for board in illegal:
            for hand in ([], [(7, R)], [(5, B), (5, Y), (8, R)]):
                with self.subTest(board=board, hand=hand):
                    with self.assertRaises(ValueError):
                        greedy_move(hand, board)
                    with self.assertRaises(ValueError):
                        greedy_move(hand, board, True)

    def test_legal_boards_pass(self):
        """Boards that do split, including ones needing jokers or long runs, are accepted."""
        legal = [
            [(5, R), (6, R), (7, R), (5, R), (6, R), (7, R)],
            [(1, R), JOKER, (3, R), (7, B), (7, Y), JOKER],
            [(v, B) for v in range(1, 14)],
            [(4, c) for c in COLORS] + [(5, R), (6, R)],
        ]
        for board in legal:
            with self.subTest(board=board):
                self.assertEqual(greedy_move([], board), DRAW)

    def test_matches_oracle_on_perturbed_boards(self):
        """On legal boards with a tile added, removed or swapped, validation agrees with the oracle."""
        rng = random.Random(8)
        solver = GreedySolver(cache_size=0)
        decided = illegal = 0
        for _ in range(400):
            _, sets = random_position(rng, 13, 4, rng.randint(1, 6), 0, 0.2)
            board = flatten(sets)
            pool = [(v, c) for c in COLORS for v in range(1, 14)]
            move = rng.random()
            if move < 0.35 and board:
                board.pop(rng.randrange(len(board)))
            elif move < 0.7:
                board.append(rng.choice(pool))
            elif board:
                board[rng.randrange(len(board))] = rng.choice(pool)
            if any(n > 2 for n in Counter(board).values()):
                continue
            try:
                expected_legal = oracle_turn(board, [], 60000) is not None
            except OracleGaveUp:
                continue
            decided += 1
            illegal += not expected_legal
            if expected_legal:
                self.assertEqual(solver.solve([], board), DRAW, f"legal board rejected: {board}")
            else:
                with self.assertRaises(ValueError, msg=f"illegal board accepted: {board}"):
                    solver.solve([], board)
        self.assertGreater(decided, 300)
        self.assertGreater(illegal, 100)


class Memoisation(unittest.TestCase):
    """The caches must be invisible to callers."""

    def test_cached_answers_are_independent(self):
        """Mutating a returned answer never leaks into a later identical call."""
        solver = GreedySolver()
        hand = [(5, R), (6, R), (7, R)]
        first = solver.solve(hand, [])
        first.sBoard.clear()
        first.hand.append("junk")
        second = solver.solve(hand, [])
        self.assertEqual(second.sBoard, [[(5, R), (6, R), (7, R)]])
        self.assertEqual(second.hand, [])

    def test_uncached_solver_matches_cached(self):
        """Disabling the caches changes nothing about the answers."""
        rng = random.Random(7)
        cached, plain = GreedySolver(), GreedySolver(cache_size=0)
        for _ in range(60):
            hand, sets = random_position(rng, 13, 4, rng.randint(1, 5), rng.randint(1, 10), 0.2)
            first_meld = rng.random() < 0.3
            board = flatten(sets)
            self.assertEqual(cached.solve(hand, board, first_meld), plain.solve(hand, board, first_meld))

    def test_produced_board_needs_no_revalidation(self):
        """A board this solver just produced is trusted without another partition solve."""
        solver = GreedySolver()
        result = solver.solve([(4, R), (7, B), (7, K)], [(3, R), JOKER, (5, R), (7, Y), (8, Y), (9, Y)])

        def refuse(board):
            """Fail if the board is sent for validation again."""
            raise AssertionError("produced board was validated again")

        solver._partition = refuse
        self.assertEqual(solver.solve([(1, B)], flatten(result.sBoard)), DRAW)


class SolverFailure(unittest.TestCase):
    """A HiGHS outcome other than optimal or infeasible is never turned into an answer."""

    def test_undefined_status_raises(self):
        """An undefined status surfaces as RuntimeError."""
        solver = GreedySolver(cache_size=0)
        solver._solver = StubSolver(pulp.LpSolveStatus.Undefined)
        with self.assertRaises(RuntimeError):
            solver.solve([(4, R)], [(5, R), (6, R), (7, R)])


class OracleCrossCheck(unittest.TestCase):
    """Randomised positions solved by both the solver and the exhaustive oracle."""

    def compare(self, seed, cases, top, n_colors, sets_range, hand_range, joker_rate, budget=None, share=0.9):
        """Require the solver to match the oracle on ``cases`` random positions.

        Positions on which the oracle exhausts ``budget`` are skipped; at least
        a ``share`` of them must still be verified.
        """
        rng = random.Random(seed)
        solver = GreedySolver(cache_size=0)
        verified = plays = 0
        for _ in range(cases):
            hand, sets = random_position(
                rng, top, n_colors, rng.randint(*sets_range), rng.randint(*hand_range), joker_rate
            )
            board = flatten(sets)
            got = check_move(self, hand, board, solver.solve(hand, board))
            try:
                expected = oracle_turn(board, hand, budget)
            except OracleGaveUp:
                continue
            self.assertEqual(got, expected, f"hand={hand} board={board}")
            verified += 1
            plays += got > 0
        self.assertGreaterEqual(verified, int(cases * share))
        self.assertGreater(plays, verified // 4)

    def test_dense_small_range(self):
        """Few values and colours force heavy rearrangement of the board."""
        self.compare(1, 300, 6, 3, (1, 6), (1, 8), 0.25)

    def test_full_tile_set(self):
        """Positions drawn from the complete 106-tile set."""
        self.compare(2, 300, 13, 4, (1, 6), (1, 10), 0.15)

    def test_crowded_tables(self):
        """Late-game tables of 6-10 sets with jokers about; the oracle settles about half in budget."""
        self.compare(4, 150, 13, 4, (6, 10), (4, 12), 0.3, budget=60000, share=0.4)

    def test_initial_meld(self):
        """Initial melds match the oracle and keep the board tiles in sets of their own."""
        rng = random.Random(3)
        solver = GreedySolver(cache_size=0)
        melded = 0
        for _ in range(300):
            hand, sets = random_position(rng, 13, 4, rng.randint(0, 4), rng.randint(3, 14), 0.15)
            board = flatten(sets)
            got = check_move(self, hand, board, solver.solve(hand, board, True), True)
            self.assertEqual(got, oracle_initial(hand), f"hand={hand}")
            melded += got > 0
        self.assertGreater(melded, 30)


class SelfPlay(unittest.TestCase):
    """Complete two-player games between greedy players, every turn checked."""

    def test_games(self):
        """Each turn obeys the rules, and positions the oracle settles in budget match it."""
        rng = random.Random(11)
        solver = GreedySolver()
        verified = 0
        for _ in range(6):
            deck = [(v, c) for c in COLORS for v in range(1, 14)] * 2 + [JOKER, JOKER]
            rng.shuffle(deck)
            hands = [deck[:14], deck[14:28]]
            deck = deck[28:]
            board, melded, passes, turn = [], [False, False], 0, 0
            while passes < 2 and all(hands):
                me = turn % 2
                result = solver.solve(hands[me], board, not melded[me])
                check_move(self, hands[me], board, result, not melded[me])
                if melded[me]:
                    try:
                        expected = oracle_turn(board, hands[me], 60000)
                    except OracleGaveUp:
                        expected = None
                    if expected is not None:
                        self.assertEqual(0 if result == DRAW else result.numOfTails, expected)
                        verified += 1
                if result == DRAW:
                    if deck:
                        hands[me].append(deck.pop())
                        passes = 0
                    else:
                        passes += 1
                else:
                    hands[me], board = result.hand, flatten(result.sBoard)
                    melded[me], passes = True, 0
                turn += 1
        self.assertGreater(verified, 20)


if __name__ == "__main__":
    unittest.main()
