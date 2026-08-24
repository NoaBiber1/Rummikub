from models import Set, SetType, Tile, TileColor

_MIN_SET_SIZE = 3
_MAX_GROUP_SIZE = 4
_MIN_TILE_VALUE = 1
_MAX_TILE_VALUE = 13


def validate_board(board: list[Set]) -> tuple[bool, str]:
    """Validate every set currently on the board against standard Rummikub rules.

    A board is valid when every set is either a legal RUN or a legal GROUP
    (including correct Joker substitution). Returns ``(True, "Board is valid")``
    on success, or ``(False, error_message)`` describing the first failing set.
    """
    for index, rummikub_set in enumerate(board):
        is_valid, error_message = _validate_set(rummikub_set, index)
        if not is_valid:
            return False, error_message
    return True, "Board is valid"


def _is_joker(tile: Tile) -> bool:
    return tile.color == TileColor.JOKER or tile.value == 0


def _set_label(rummikub_set: Set, index: int) -> str:
    return f"Set {index + 1} (id={rummikub_set.id})"


def _validate_set(rummikub_set: Set, index: int) -> tuple[bool, str]:
    label = _set_label(rummikub_set, index)
    tiles = rummikub_set.tiles

    if len(tiles) < _MIN_SET_SIZE:
        return False, (
            f"{label} is invalid: a set must contain at least {_MIN_SET_SIZE} tiles "
            f"(found {len(tiles)})."
        )

    if rummikub_set.type == SetType.RUN:
        return _validate_run(tiles, label)
    if rummikub_set.type == SetType.GROUP:
        return _validate_group(tiles, label)

    # type == INVALID (or any unexpected value): accept the set only if the
    # tiles themselves form a legal run or a legal group.
    run_ok, run_error = _validate_run(tiles, label)
    if run_ok:
        return True, ""
    group_ok, group_error = _validate_group(tiles, label)
    if group_ok:
        return True, ""
    return False, (
        f"{label} is invalid: tiles do not form a valid RUN or GROUP. "
        f"RUN check: {run_error} GROUP check: {group_error}"
    )


def _validate_run(tiles: list[Tile], label: str) -> tuple[bool, str]:
    """Same color, strictly consecutive values in board order. Jokers fill gaps."""
    if len(tiles) > _MAX_TILE_VALUE:
        return False, (
            f"{label} is not a valid RUN: a run cannot contain more than "
            f"{_MAX_TILE_VALUE} tiles (found {len(tiles)})."
        )

    natural_indices = [i for i, tile in enumerate(tiles) if not _is_joker(tile)]
    if not natural_indices:
        return True, ""

    colors = {tiles[i].color for i in natural_indices}
    if len(colors) > 1:
        color_list = ", ".join(sorted(color.value for color in colors))
        return False, (
            f"{label} is not a valid RUN: all non-Joker tiles must be the same color "
            f"(found {color_list})."
        )

    first_index = natural_indices[0]
    start_value = tiles[first_index].value - first_index
    end_value = start_value + len(tiles) - 1

    if start_value < _MIN_TILE_VALUE or end_value > _MAX_TILE_VALUE:
        out_of_range = start_value if start_value < _MIN_TILE_VALUE else end_value
        return False, (
            f"{label} is not a valid RUN: Joker placement would require a tile with "
            f"value {out_of_range}, which is outside the valid range "
            f"{_MIN_TILE_VALUE}-{_MAX_TILE_VALUE}."
        )

    for index in natural_indices:
        expected = start_value + index
        actual = tiles[index].value
        if actual != expected:
            return False, (
                f"{label} is not a valid RUN: values must be strictly consecutive "
                f"in board order (expected {expected} at position {index + 1}, "
                f"found {actual})."
            )

    return True, ""


def _validate_group(tiles: list[Tile], label: str) -> tuple[bool, str]:
    """Same value, strictly different colors. Jokers stand in for missing colors."""
    if len(tiles) > _MAX_GROUP_SIZE:
        return False, (
            f"{label} is not a valid GROUP: a group cannot contain more than "
            f"{_MAX_GROUP_SIZE} tiles, one per color (found {len(tiles)})."
        )

    naturals = [tile for tile in tiles if not _is_joker(tile)]
    if not naturals:
        return True, ""

    values = {tile.value for tile in naturals}
    if len(values) > 1:
        value_list = ", ".join(str(value) for value in sorted(values))
        return False, (
            f"{label} is not a valid GROUP: all non-Joker tiles must have the same "
            f"value (found {value_list})."
        )

    seen_colors: set[TileColor] = set()
    for tile in naturals:
        if tile.color in seen_colors:
            return False, (
                f"{label} is not a valid GROUP: colors must be strictly unique "
                f"(duplicate {tile.color.value})."
            )
        seen_colors.add(tile.color)

    return True, ""
