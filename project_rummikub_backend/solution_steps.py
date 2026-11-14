"""Turn the solver's final board into an ordered list of human-readable moves.

The solver only returns the finished board. This module diffs it against the
starting board and rack (tiles keep their ids, so each tile's start and end
set are known) and emits steps that can be replayed one after another:

1. ``DISBAND_SET``: an original set that is gone from the final board. Its
   tiles are lifted off the table and wait there until a later step uses them.
2. ``PLACE_FROM_RACK`` / ``MOVE_BETWEEN_SETS``: tiles added to a set that
   survives (matched by set id). One step per source (the rack, or the
   original set the tiles came from, which may be a disbanded one).
3. ``FORM_NEW_SET``: a final set with a new id; takes its tiles from wherever
   they currently are.

A tile that leaves a surviving set needs no step of its own: it is the
source of the move that puts it somewhere else. Sets whose tile ids did not
change produce no step.
"""

from __future__ import annotations

from models import Set, SetType, SolutionStep, StepType, Tile, TileColor

_RACK = None  # origin marker for tiles that start in the rack


def compute_solution_steps(
    initial_board: list[Set], initial_rack: list[Tile], final_board: list[Set]
) -> list[SolutionStep]:
    origin: dict[str, str | None] = {tile.id: _RACK for tile in initial_rack}
    initial_by_id = {s.id: s for s in initial_board}
    for rummikub_set in initial_board:
        for tile in rummikub_set.tiles:
            origin[tile.id] = rummikub_set.id
    final_ids = {s.id for s in final_board}

    steps: list[SolutionStep] = []

    def add(
        step_type: StepType,
        source: str | None,
        target: str | None,
        tiles: list[Tile],
        description: str,
    ) -> None:
        steps.append(
            SolutionStep(
                stepIndex=len(steps),
                type=step_type,
                sourceSetId=source,
                targetSetId=target,
                tiles=tiles,
                description=description,
            )
        )

    for original in initial_board:
        if original.id not in final_ids:
            add(
                StepType.DISBAND_SET,
                original.id,
                None,
                list(original.tiles),
                f"Break up {_set_label(original)}",
            )

    for final_set in final_board:
        original = initial_by_id.get(final_set.id)
        if original is None:
            continue
        existing = {t.id for t in original.tiles}
        added = [t for t in final_set.tiles if t.id not in existing]
        for source, tiles in _group_by_origin(added, origin):
            if source is _RACK:
                add(
                    StepType.PLACE_FROM_RACK,
                    None,
                    final_set.id,
                    tiles,
                    f"Place {_tiles_label(tiles)} from your rack onto {_set_label(original)}",
                )
            else:
                add(
                    StepType.MOVE_BETWEEN_SETS,
                    source,
                    final_set.id,
                    tiles,
                    f"Move {_tiles_label(tiles)} from {_set_label(initial_by_id[source])} "
                    f"onto {_set_label(original)}",
                )

    for final_set in final_board:
        if final_set.id in initial_by_id:
            continue
        parts = [
            f"{_tiles_label(tiles)} from "
            + ("your rack" if source is _RACK else _set_label(initial_by_id[source]))
            for source, tiles in _group_by_origin(final_set.tiles, origin)
        ]
        add(
            StepType.FORM_NEW_SET,
            None,
            final_set.id,
            list(final_set.tiles),
            f"Form a new {_kind(final_set)} with " + " and ".join(parts),
        )

    return steps


def _group_by_origin(
    tiles: list[Tile], origin: dict[str, str | None]
) -> list[tuple[str | None, list[Tile]]]:
    """Group tiles by where they started, in first-seen order."""
    groups: dict[str | None, list[Tile]] = {}
    for tile in tiles:
        groups.setdefault(origin[tile.id], []).append(tile)
    return list(groups.items())


def _tile_label(tile: Tile) -> str:
    if tile.color == TileColor.JOKER:
        return "Joker"
    return f"{tile.color.value.capitalize()} {tile.value}"


def _tiles_label(tiles: list[Tile]) -> str:
    return ", ".join(_tile_label(t) for t in tiles)


def _kind(rummikub_set: Set) -> str:
    return {SetType.RUN: "run", SetType.GROUP: "group"}.get(rummikub_set.type, "set")


def _set_label(rummikub_set: Set) -> str:
    return f"the {_kind(rummikub_set)} [{_tiles_label(rummikub_set.tiles)}]"
