"""Generate legal RUN and GROUP candidates from the tiles that actually exist.

This is the explicit search-space cut for the solver: we never invent tiles, we
never emit illegal sets, and we drop duplicate tile-id combinations. Extra Joker
substitutions (using a Joker even when a natural tile exists) are generated so
the later packing step can leave that natural free for another set.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

from models import GameState, Set, SetType, Tile, TileColor

_MIN_SET_SIZE = 3
_MAX_GROUP_SIZE = 4
_MIN_TILE_VALUE = 1
_MAX_TILE_VALUE = 13
_NATURAL_COLORS = (TileColor.RED, TileColor.BLUE, TileColor.BLACK, TileColor.YELLOW)

# Soft cap on how many candidates the packing model should see.
DEFAULT_MAX_CANDIDATES = 4000
# Hard stop while enumerating runs, so a joker-heavy pool cannot explode.
_RUN_ENUMERATION_CAP = 20000


@dataclass(frozen=True, slots=True)
class Candidate:
    """One legal set, identified by the physical tile ids it uses."""

    kind: SetType
    tiles: tuple[Tile, ...]
    original_set_id: str | None = None

    @property
    def id_key(self) -> frozenset[str]:
        return frozenset(tile.id for tile in self.tiles)

    def uses_any(self, tile_ids: set[str]) -> bool:
        return any(tile.id in tile_ids for tile in self.tiles)

    def to_set(self) -> Set:
        return Set(
            id=self.original_set_id or str(uuid.uuid4()),
            type=self.kind,
            tiles=list(self.tiles),
        )


def generate_candidate_sets(
    game_state: GameState,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[Set]:
    """Same as ``generate_candidates``, mapped to ``Set`` models."""
    return [candidate.to_set() for candidate in generate_candidates(
        game_state,
        max_candidates=max_candidates,
    )]


def generate_candidates(
    game_state: GameState,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_run_length: int = _MAX_TILE_VALUE,
) -> list[Candidate]:
    """Enumerate legal sets that can be formed from board tiles plus rack tiles.

    Priority when ``max_candidates`` is hit (groups and original board sets are
    never dropped):

    1. Original board sets (so "do nothing" / keep a set is always available)
    2. Every legal GROUP
    3. RUNs that use at least one rack tile
    4. Remaining RUNs

    ``max_run_length`` limits generated runs (not original board sets). The
    solver passes 5: any longer run splits into two legal runs of 3+ tiles.
    """
    board_tiles = [tile for rummikub_set in game_state.board for tile in rummikub_set.tiles]
    rack_tiles = list(game_state.rack)
    pool = board_tiles + rack_tiles
    rack_ids = {tile.id for tile in rack_tiles}

    original = _candidates_from_board(game_state.board)
    groups = _generate_groups(pool)
    runs = _generate_runs(pool, max_run_length)

    rack_runs = [candidate for candidate in runs if candidate.uses_any(rack_ids)]
    other_runs = [candidate for candidate in runs if not candidate.uses_any(rack_ids)]

    # Original + groups are required for a complete packing of board tiles.
    must_keep = _unique_in_order([*original, *groups])
    extra = _unique_in_order([*rack_runs, *other_runs])
    extra = [candidate for candidate in extra if candidate.id_key not in {c.id_key for c in must_keep}]

    if len(must_keep) >= max_candidates:
        return must_keep
    return must_keep + extra[: max_candidates - len(must_keep)]


def _is_joker(tile: Tile) -> bool:
    return tile.color == TileColor.JOKER or tile.value == 0


def _candidates_from_board(board: list[Set]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for rummikub_set in board:
        kind = rummikub_set.type
        if kind == SetType.INVALID:
            continue
        if len(rummikub_set.tiles) < _MIN_SET_SIZE:
            continue
        candidates.append(
            Candidate(
                kind=kind,
                tiles=tuple(rummikub_set.tiles),
                original_set_id=rummikub_set.id,
            )
        )
    return candidates


def _partition_pool(pool: list[Tile]) -> tuple[list[Tile], dict[TileColor, dict[int, list[Tile]]]]:
    """Jokers, and natural tiles indexed by color then value."""
    jokers: list[Tile] = []
    by_color: dict[TileColor, dict[int, list[Tile]]] = {
        color: {} for color in _NATURAL_COLORS
    }
    for tile in pool:
        if _is_joker(tile):
            jokers.append(tile)
            continue
        if tile.color not in by_color:
            continue
        by_color[tile.color].setdefault(tile.value, []).append(tile)
    return jokers, by_color


def _unique_in_order(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep first occurrence of each physical tile combination.

    Prefers a candidate that still carries ``original_set_id`` when a later
    duplicate is an original board set.
    """
    first_index: dict[frozenset[str], int] = {}
    ordered: list[Candidate] = []
    for candidate in candidates:
        key = candidate.id_key
        if key not in first_index:
            first_index[key] = len(ordered)
            ordered.append(candidate)
            continue
        existing = ordered[first_index[key]]
        if candidate.original_set_id and not existing.original_set_id:
            ordered[first_index[key]] = candidate
    return ordered


def _generate_groups(pool: list[Tile]) -> list[Candidate]:
    """Same value, distinct colors, size 3 or 4. Jokers fill missing colors."""
    jokers, by_color = _partition_pool(pool)
    # value -> color -> tiles of that color+value
    by_value: dict[int, dict[TileColor, list[Tile]]] = {}
    for color, ranks in by_color.items():
        for value, tiles in ranks.items():
            by_value.setdefault(value, {}).setdefault(color, []).extend(tiles)

    candidates: list[Candidate] = []
    for value in range(_MIN_TILE_VALUE, _MAX_TILE_VALUE + 1):
        color_tiles = by_value.get(value, {})
        for size in range(_MIN_SET_SIZE, _MAX_GROUP_SIZE + 1):
            for chosen in combinations(_NATURAL_COLORS, size):
                candidates.extend(
                    _group_assignments(chosen, color_tiles, jokers)
                )
    return _unique_in_order(candidates)


def _group_assignments(
    chosen: tuple[TileColor, ...],
    color_tiles: dict[TileColor, list[Tile]],
    jokers: list[Tile],
) -> list[Candidate]:
    missing = [color for color in chosen if not color_tiles.get(color)]
    present = [color for color in chosen if color_tiles.get(color)]
    extra_jokers_available = len(jokers) - len(missing)
    if extra_jokers_available < 0:
        return []

    # Fewer Jokers first: the natural-only filling, then substitutions that
    # free a natural tile for another set.
    found: list[Candidate] = []
    max_extra = min(len(present), extra_jokers_available)
    for extra in range(0, max_extra + 1):
        for replaced in combinations(present, extra):
            replaced_set = set(replaced)
            joker_slots = [
                color for color in chosen
                if color in missing or color in replaced_set
            ]
            natural_slots = [
                color for color in chosen
                if color not in missing and color not in replaced_set
            ]
            found.extend(
                _emit_groups(chosen, natural_slots, joker_slots, color_tiles, jokers)
            )
    return found


def _emit_groups(
    chosen: tuple[TileColor, ...],
    natural_slots: list[TileColor],
    joker_slots: list[TileColor],
    color_tiles: dict[TileColor, list[Tile]],
    jokers: list[Tile],
) -> list[Candidate]:
    natural_options = [color_tiles[color] for color in natural_slots]
    found: list[Candidate] = []
    for natural_pick in product(*natural_options) if natural_options else ((),):
        if _has_duplicate_ids(natural_pick):
            continue
        for joker_pick in combinations(jokers, len(joker_slots)):
            assigned: dict[TileColor, Tile] = {}
            for color, tile in zip(natural_slots, natural_pick, strict=True):
                assigned[color] = tile
            for color, tile in zip(joker_slots, joker_pick, strict=True):
                assigned[color] = tile
            tiles = tuple(assigned[color] for color in chosen)
            found.append(Candidate(kind=SetType.GROUP, tiles=tiles))
    return found


def _generate_runs(pool: list[Tile], max_length: int = _MAX_TILE_VALUE) -> list[Candidate]:
    """Same color, consecutive values in 1–13. Jokers fill holes by position."""
    jokers, by_color = _partition_pool(pool)
    candidates: list[Candidate] = []
    for color in _NATURAL_COLORS:
        ranks = by_color[color]
        if not ranks and not jokers:
            continue
        for start in range(_MIN_TILE_VALUE, _MAX_TILE_VALUE - _MIN_SET_SIZE + 2):
            longest = min(max_length, _MAX_TILE_VALUE - start + 1)
            for length in range(_MIN_SET_SIZE, longest + 1):
                values = list(range(start, start + length))
                candidates.extend(_run_assignments(values, ranks, jokers))
                if len(candidates) >= _RUN_ENUMERATION_CAP:
                    return _unique_in_order(candidates)
    return _unique_in_order(candidates)


def _run_assignments(
    values: list[int],
    ranks: dict[int, list[Tile]],
    jokers: list[Tile],
) -> list[Candidate]:
    missing = [value for value in values if not ranks.get(value)]
    present = [value for value in values if ranks.get(value)]
    extra_jokers_available = len(jokers) - len(missing)
    if extra_jokers_available < 0:
        return []

    found: list[Candidate] = []
    max_extra = min(len(present), extra_jokers_available)
    for extra in range(0, max_extra + 1):
        for replaced in combinations(present, extra):
            replaced_set = set(replaced)
            joker_slots = [
                value for value in values
                if value in missing or value in replaced_set
            ]
            natural_slots = [
                value for value in values
                if value not in missing and value not in replaced_set
            ]
            found.extend(
                _emit_runs(values, natural_slots, joker_slots, ranks, jokers)
            )
    return found


def _emit_runs(
    values: list[int],
    natural_slots: list[int],
    joker_slots: list[int],
    ranks: dict[int, list[Tile]],
    jokers: list[Tile],
) -> list[Candidate]:
    natural_options = [ranks[value] for value in natural_slots]
    found: list[Candidate] = []
    for natural_pick in product(*natural_options) if natural_options else ((),):
        if _has_duplicate_ids(natural_pick):
            continue
        for joker_pick in combinations(jokers, len(joker_slots)):
            by_value: dict[int, Tile] = {}
            for value, tile in zip(natural_slots, natural_pick, strict=True):
                by_value[value] = tile
            for value, tile in zip(joker_slots, joker_pick, strict=True):
                by_value[value] = tile
            tiles = tuple(by_value[value] for value in values)
            found.append(Candidate(kind=SetType.RUN, tiles=tiles))
    return found


def _has_duplicate_ids(tiles: tuple[Tile, ...]) -> bool:
    ids = [tile.id for tile in tiles]
    return len(ids) != len(set(ids))
