"""CP-SAT solver: drop as many rack tiles as possible onto a legal board.

The model packs legal sets over tile *types* instead of physical tiles (the
integer program of den Hertog & Hulshof, "Solving Rummikub problems by integer
linear programming", 2006):

- A tile type is a ``(color, value)`` pair, or the joker type. Physical copies
  of one type are interchangeable, so the model counts copies instead of
  choosing between them. Enumerating physical-id combinations instead explodes
  on duplicates (two copies of 13 values give 2**13 versions of one run) and
  hits the generator caps before other colors are reached.
- Candidate sets ("templates") come from ``meld_generator`` run on one
  representative tile per type plus at most two jokers. Generated runs stop at
  five tiles: any longer run splits into two legal runs of three or more, so
  nothing is lost. Original board sets are added as their own templates so the
  board can stay as it is.
- Physical tiles are assigned to the chosen templates afterwards, board tiles
  first, reusing an original set's id where that set survives.

Moves only need to be legal at the end of the turn, so board sets may be
broken up and rebuilt freely, jokers included. The initial-meld (30 points)
rule is not modelled.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from meld_generator import generate_candidates
from models import GameState, OptimalState, Set, SetType, Tile, TileColor
from validator import classify_tiles, validate_board

logger = logging.getLogger(__name__)

# Project constraint: a solve never runs longer than this, model building
# included (the CP-SAT limit is set to whatever is left of it).
TIME_LIMIT_SECONDS = 5.0
# Most of the budget the readability tie-breaker (phase 2) may use.
_TIE_BREAK_SECONDS = 1.0
# A run of 6+ tiles is always two legal runs of 3+ tiles.
_MAX_TEMPLATE_RUN_LENGTH = 5
# A real deck has two jokers; templates never need more.
_MAX_TEMPLATE_JOKERS = 2
# Templates for a full 106-tile deck stay well below this (see tests).
_MAX_TEMPLATE_CANDIDATES = 20000

TypeKey = tuple[str, int]
_JOKER_KEY: TypeKey = (TileColor.JOKER.value, 0)


@dataclass(frozen=True, slots=True)
class _Template:
    """A legal set described by tile types in board order."""

    kind: SetType
    slots: tuple[TypeKey, ...]
    usage: dict[TypeKey, int]


# Identity equality: two copies of one template are different sets.
@dataclass(eq=False, slots=True)
class _Instance:
    """One chosen copy of a template, waiting for physical tiles."""

    kind: SetType
    slots: list[TypeKey]
    tiles: list[Tile] = field(default_factory=list)


@dataclass(eq=False, slots=True)
class _Draft:
    """A set on the new board before it gets its final id."""

    tiles: list[Tile]
    set_id: str | None = None
    # The draft carries the id of an input set (kept untouched or claimed).
    from_input: bool = False


@dataclass(slots=True)
class _BoardTile:
    tile: Tile
    set_id: str


def solve_rummikub(game_state: GameState) -> OptimalState:
    """Maximize tiles dropped from the rack onto a valid Rummikub board.

    Every board tile stays on the board, every output set is a legal RUN or
    GROUP, and the whole call is capped at ``TIME_LIMIT_SECONDS``. When no
    solution is found in time (or the result fails the final self-check) the
    input is echoed with ``tiles_used_from_rack=0``.
    """
    deadline = time.perf_counter() + TIME_LIMIT_SECONDS
    if not game_state.rack:
        return _fallback(game_state)

    board_tiles = [tile for rummikub_set in game_state.board for tile in rummikub_set.tiles]
    board_count = Counter(_type_key(tile) for tile in board_tiles)
    rack_count = Counter(_type_key(tile) for tile in game_state.rack)
    total_count = board_count + rack_count

    keepable = [
        rummikub_set
        for rummikub_set in game_state.board
        if classify_tiles(rummikub_set.tiles) != SetType.INVALID
    ]

    model = cp_model.CpModel()

    # Decision: how many copies of each template go on the final board. A
    # template can be used at most as often as its scarcest tile type allows.
    copies: list[tuple[_Template, cp_model.IntVar]] = []
    for index, template in enumerate(_ALL_TEMPLATES):
        upper = min(total_count[key] // count for key, count in template.usage.items())
        if upper > 0:
            copies.append((template, model.new_int_var(0, upper, f"template_{index}")))

    # Decision: keep an original board set exactly as it is (same tiles, same
    # id). Only used as a tie-breaker so the result stays readable.
    keep = [model.new_bool_var(f"keep_{index}") for index in range(len(keepable))]

    # Decision: how many rack copies of each type are placed on the board.
    placed = {
        key: model.new_int_var(0, count, f"placed_{key[0]}_{key[1]}")
        for key, count in rack_count.items()
    }

    # Tile conservation, per type: the copies used by all chosen sets equal the
    # board copies (every board tile must stay on the board) plus the rack
    # copies placed (never more than the rack holds, enforced by the domain of
    # ``placed``). Every chosen set is legal and has 3+ tiles by construction.
    used: dict[TypeKey, list] = {key: [] for key in total_count}
    for template, var in copies:
        for key, count in template.usage.items():
            used[key].append(count * var)
    for rummikub_set, keep_var in zip(keepable, keep, strict=True):
        for key, count in _set_usage(rummikub_set).items():
            used[key].append(count * keep_var)
    for key, terms in used.items():
        model.add(sum(terms) == board_count[key] + placed.get(key, 0))

    # Phase 1 objective: the number of rack tiles placed.
    total_placed = sum(placed.values())
    model.maximize(total_placed)

    # The unchanged board is always feasible; hinting it gives the search a
    # legal answer from the start.
    for keep_var in keep:
        model.add_hint(keep_var, 1)
    for _, var in copies:
        model.add_hint(var, 0)
    for var in placed.values():
        model.add_hint(var, 0)

    solver = _new_solver(deadline - time.perf_counter())
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning("Solver found no solution (%s); echoing the input.", solver.status_name(status))
        return _fallback(game_state)

    # Phase 2, inside the same budget: with the placed count fixed at its best
    # value, keep as many original sets untouched as possible so the result
    # stays readable. Two objectives one after the other solve much faster
    # than one weighted objective (measured: worst case 3.5 s -> 1.6 s). The
    # phase 1 solution is the hint, so a phase 2 timeout loses nothing.
    best = solver
    remaining_seconds = min(_TIE_BREAK_SECONDS, deadline - time.perf_counter())
    if keep and remaining_seconds > 0:
        model.add(total_placed == round(solver.objective_value))
        model.clear_hints()
        for var in [*keep, *(var for _, var in copies), *placed.values()]:
            model.add_hint(var, solver.value(var))
        model.maximize(sum(keep))
        tie_breaker = _new_solver(remaining_seconds)
        if tie_breaker.solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            best = tie_breaker

    kept_sets = [
        rummikub_set
        for rummikub_set, keep_var in zip(keepable, keep, strict=True)
        if best.value(keep_var)
    ]
    instances = [
        _Instance(kind=template.kind, slots=list(template.slots))
        for template, var in copies
        for _ in range(best.value(var))
    ]
    result = _build_result(game_state, kept_sets, instances)

    problem = _self_check(game_state, result)
    if problem is not None:
        logger.error("Solver result failed the self-check (%s); echoing the input.", problem)
        return _fallback(game_state)

    logger.info(
        "Solved: status=%s placed=%d kept=%d/%d templates=%d wall=%.3fs",
        solver.status_name(status),
        result.tiles_used_from_rack,
        len(kept_sets),
        len(keepable),
        len(copies),
        solver.wall_time + (best.wall_time if best is not solver else 0.0),
    )
    return result


def _new_solver(max_seconds: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    # Hard cap on search time: what is left of the 5-second budget, never more.
    solver.parameters.max_time_in_seconds = max(0.0, min(TIME_LIMIT_SECONDS, max_seconds))
    # One worker with a fixed seed: the same request always gets the same
    # answer, which the benchmark and the tests rely on.
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    # Presolve costs more than it saves on these small models (measured:
    # worst case 1.5 s -> 0.36 s on 97-tile states, same optimum).
    solver.parameters.cp_model_presolve = False
    return solver


def _fallback(game_state: GameState) -> OptimalState:
    """Deterministic no-move answer: the board and rack exactly as given."""
    return OptimalState(
        board=list(game_state.board),
        tiles_used_from_rack=0,
        remaining_rack=list(game_state.rack),
    )


def _is_joker(tile: Tile) -> bool:
    return tile.color == TileColor.JOKER or tile.value == 0


def _type_key(tile: Tile) -> TypeKey:
    return _JOKER_KEY if _is_joker(tile) else (tile.color.value, tile.value)


def _set_usage(rummikub_set: Set) -> Counter[TypeKey]:
    return Counter(_type_key(tile) for tile in rummikub_set.tiles)


def _deck_templates() -> tuple[_Template, ...]:
    """Every legal set over one tile per type plus two jokers.

    The sets that can be formed from any pool are exactly the ones whose tile
    types the pool holds, so this is computed once and filtered per request
    (a template's upper bound is 0 when a type is missing).
    """
    deck = [
        Tile(id=f"{color.value}-{value}", color=color, value=value)
        for color in (TileColor.RED, TileColor.BLUE, TileColor.BLACK, TileColor.YELLOW)
        for value in range(1, 14)
    ]
    deck += [
        Tile(id=f"joker-{index}", color=TileColor.JOKER, value=0)
        for index in range(_MAX_TEMPLATE_JOKERS)
    ]
    candidates = generate_candidates(
        GameState(board=[], rack=deck),
        max_candidates=_MAX_TEMPLATE_CANDIDATES,
        max_run_length=_MAX_TEMPLATE_RUN_LENGTH,
    )

    # Two templates with the same type usage are interchangeable for the
    # model (e.g. a joker standing for 5 or for 8 next to 6-7). Keep the first,
    # which the generator emits with the fewest jokers.
    by_usage: dict[tuple[tuple[TypeKey, int], ...], _Template] = {}
    for candidate in candidates:
        slots = tuple(_type_key(tile) for tile in candidate.tiles)
        usage = Counter(slots)
        by_usage.setdefault(
            tuple(sorted(usage.items())),
            _Template(kind=candidate.kind, slots=slots, usage=dict(usage)),
        )
    return tuple(by_usage.values())


_ALL_TEMPLATES = _deck_templates()


def _build_result(
    game_state: GameState,
    kept_sets: list[Set],
    instances: list[_Instance],
) -> OptimalState:
    """Assign physical tiles to the chosen sets and lay out the new board."""
    kept_ids = {rummikub_set.id for rummikub_set in kept_sets}
    original_index = {rummikub_set.id: index for index, rummikub_set in enumerate(game_state.board)}

    # Free tiles, by type. Board tiles are always consumed before rack tiles:
    # the model guarantees each type's demand covers its board copies, so
    # taking board copies first places every board tile.
    free_board: dict[TypeKey, list[_BoardTile]] = {}
    for rummikub_set in game_state.board:
        if rummikub_set.id in kept_ids:
            continue
        for tile in rummikub_set.tiles:
            free_board.setdefault(_type_key(tile), []).append(_BoardTile(tile, rummikub_set.id))
    free_rack: dict[TypeKey, list[Tile]] = {}
    for tile in game_state.rack:
        free_rack.setdefault(_type_key(tile), []).append(tile)

    # Fill the sets that overlap an input set the most first, taking that
    # set's tiles, so a set that was only extended keeps its own tiles.
    pending = list(instances)
    while pending:
        instance, anchor = max(
            ((item, _best_anchor(item, free_board)) for item in pending),
            key=lambda entry: entry[1][1],
        )
        pending.remove(instance)
        for key in instance.slots:
            instance.tiles.append(_take_tile(key, anchor[0], free_board, free_rack))

    drafts = [
        _Draft(tiles=list(rummikub_set.tiles), set_id=rummikub_set.id, from_input=True)
        for rummikub_set in kept_sets
    ]
    drafts += [_Draft(tiles=instance.tiles) for instance in instances]
    drafts = _join_runs(drafts)
    _claim_input_ids(drafts, game_state.board)

    board = [
        Set(
            id=draft.set_id or _new_set_id(draft.tiles),
            type=classify_tiles(draft.tiles),
            tiles=draft.tiles,
        )
        for draft in drafts
    ]
    # Surviving sets keep their original position; new sets go last.
    board.sort(key=lambda rummikub_set: original_index.get(rummikub_set.id, len(game_state.board)))
    placed_ids = {tile.id for rummikub_set in board for tile in rummikub_set.tiles}
    remaining = [tile for tile in game_state.rack if tile.id not in placed_ids]
    return OptimalState(
        board=board,
        tiles_used_from_rack=len(game_state.rack) - len(remaining),
        remaining_rack=remaining,
    )


def _run_bounds(tiles: list[Tile]) -> tuple[str, int, int] | None:
    """Color, first and last value of a legal run; None for anything else."""
    if classify_tiles(tiles) != SetType.RUN:
        return None
    for index, tile in enumerate(tiles):
        if not _is_joker(tile):
            start = tile.value - index
            return tile.color.value, start, start + len(tiles) - 1
    return None


def _join_runs(drafts: list[_Draft]) -> list[_Draft]:
    """Join runs that continue each other (R1-4 + R5-9 -> R1-9).

    Generated runs stop at five tiles, so a long run comes out of the model in
    pieces, and a run kept untouched can sit right next to the tiles that
    extend it. Joining changes no tile counts and gives the board a player
    would lay down. Two sets that both carry an input id stay separate.
    """
    runs: list[tuple[tuple[str, int, int], _Draft]] = []
    others: list[_Draft] = []
    for draft in drafts:
        bounds = _run_bounds(draft.tiles)
        if bounds is None:
            others.append(draft)
        else:
            runs.append((bounds, draft))

    joined: list[_Draft] = []
    ends: list[tuple[str, int]] = []
    for (color, start, end), draft in sorted(runs, key=lambda entry: entry[0][:2]):
        for index, chain in enumerate(joined):
            if ends[index] == (color, start - 1) and not (chain.from_input and draft.from_input):
                chain.tiles.extend(draft.tiles)
                if draft.from_input:
                    chain.set_id, chain.from_input = draft.set_id, True
                ends[index] = (color, end)
                break
        else:
            joined.append(draft)
            ends.append((color, end))
    return joined + others


def _claim_input_ids(drafts: list[_Draft], board: list[Set]) -> None:
    """Give a new set the id of the input set most of whose tiles it holds."""
    origin = {tile.id: rummikub_set.id for rummikub_set in board for tile in rummikub_set.tiles}
    size = {rummikub_set.id: len(rummikub_set.tiles) for rummikub_set in board}
    taken = {draft.set_id for draft in drafts if draft.set_id}
    claims = []
    for index, draft in enumerate(drafts):
        if draft.set_id:
            continue
        overlaps = Counter(origin[tile.id] for tile in draft.tiles if tile.id in origin)
        for set_id, overlap in overlaps.items():
            if overlap * 2 > size[set_id]:
                claims.append((-overlap, index, set_id))
    for _, index, set_id in sorted(claims):
        if set_id not in taken and drafts[index].set_id is None:
            drafts[index].set_id = set_id
            drafts[index].from_input = True
            taken.add(set_id)


def _new_set_id(tiles: list[Tile]) -> str:
    """Stable id for a new set, so the same request always gets the same answer."""
    key = ",".join(sorted(tile.id for tile in tiles))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rummikub-set:{key}"))


def _best_anchor(
    instance: _Instance,
    free_board: dict[TypeKey, list[_BoardTile]],
) -> tuple[str | None, int]:
    """The original set that can supply most of this instance's tiles."""
    supply: Counter[str] = Counter()
    for key, needed in Counter(instance.slots).items():
        per_set = Counter(board_tile.set_id for board_tile in free_board.get(key, []))
        for set_id, available in per_set.items():
            supply[set_id] += min(needed, available)
    if not supply:
        return None, 0
    set_id, overlap = min(supply.items(), key=lambda item: (-item[1], item[0]))
    return set_id, overlap


def _take_tile(
    key: TypeKey,
    anchor: str | None,
    free_board: dict[TypeKey, list[_BoardTile]],
    free_rack: dict[TypeKey, list[Tile]],
) -> Tile:
    board_tiles = free_board.get(key, [])
    for index, board_tile in enumerate(board_tiles):
        if board_tile.set_id == anchor:
            return board_tiles.pop(index).tile
    if board_tiles:
        return board_tiles.pop(0).tile
    return free_rack[key].pop(0)


def _self_check(game_state: GameState, result: OptimalState) -> str | None:
    """Reject any result that is not a legal end-of-turn board."""
    is_valid, message = validate_board(result.board)
    if not is_valid:
        return message

    input_ids = Counter(
        [tile.id for rummikub_set in game_state.board for tile in rummikub_set.tiles]
        + [tile.id for tile in game_state.rack]
    )
    output_board_ids = [tile.id for rummikub_set in result.board for tile in rummikub_set.tiles]
    output_ids = Counter(output_board_ids + [tile.id for tile in result.remaining_rack])
    if output_ids != input_ids:
        return "tile ids are not conserved"

    on_board = set(output_board_ids)
    if any(tile.id not in on_board for rummikub_set in game_state.board for tile in rummikub_set.tiles):
        return "a board tile left the board"
    if result.tiles_used_from_rack != sum(tile.id in on_board for tile in game_state.rack):
        return "tiles_used_from_rack does not match the board"
    return None
