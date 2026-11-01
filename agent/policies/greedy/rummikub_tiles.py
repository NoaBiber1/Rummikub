"""Tile model for the PuLP/HiGHS greedy Rummikub solver.

A tile is exactly one of:

* ``"J"``, the joker;
* a 2-tuple ``(number, color)``, in that order, where ``number`` is an
  ``int`` from 1 to 13 and ``color`` is one of ``"red"``, ``"yellow"``,
  ``"blue"`` or ``"black"``.

Nothing else is accepted: no reversed tuples, lists, namedtuples, strings such
as ``"5 red"``, mappings, capitalised colour names, booleans or floats.

Internally each tile is a *type index*: ``color * 13 + (number - 1)`` for a
real tile, with colours in the order above, and ``52`` for the joker.
"""

COLORS = ("red", "yellow", "blue", "black")
NUM_COLORS = len(COLORS)
NUM_VALUES = 13
NUM_REAL = NUM_COLORS * NUM_VALUES
JOKER = NUM_REAL
NUM_TYPES = NUM_REAL + 1
JOKER_TILE = "J"
MAX_COPIES = 2
JOKER_POINTS = 30

_COLOR_INDEX = {name: index for index, name in enumerate(COLORS)}


def tile_index(number, color):
    """Return the type index of a real tile.

    Args:
        number: Face value, ``1..13``.
        color: Colour index, ``0..3`` in ``COLORS`` order.
    """
    return color * NUM_VALUES + number - 1


def tile_value(index):
    """Return the face value ``1..13`` of a real tile type index."""
    return index % NUM_VALUES + 1


def tile_color(index):
    """Return the colour index ``0..3`` of a real tile type index."""
    return index // NUM_VALUES


def tile_points(index):
    """Return a tile's penalty value: its face value, or ``JOKER_POINTS`` for a joker."""
    return JOKER_POINTS if index == JOKER else tile_value(index)


def canonical_tile(index):
    """Return a type index in caller format: ``(number, color)``, or ``"J"`` for the joker."""
    if index == JOKER:
        return JOKER_TILE
    return (tile_value(index), COLORS[tile_color(index)])


def parse_tile(tile):
    """Return the type index of a tile in the strict format.

    Raises:
        ValueError: If ``tile`` is anything other than ``"J"`` or a tuple
            ``(number, color)`` with an ``int`` number in ``1..13`` and one of
            the four lower-case colour names.
    """
    if type(tile) is str and tile == JOKER_TILE:
        return JOKER
    if type(tile) is tuple and len(tile) == 2:
        number, color = tile
        if type(number) is int and type(color) is str and 1 <= number <= NUM_VALUES:
            color_index = _COLOR_INDEX.get(color)
            if color_index is not None:
                return tile_index(number, color_index)
    raise ValueError(
        f"invalid tile {tile!r}: expected 'J' or (number, color) with an int number "
        f"1-{NUM_VALUES} and color one of {COLORS}"
    )


def count_types(indices):
    """Return a length-``NUM_TYPES`` list counting each type index in ``indices``."""
    counts = [0] * NUM_TYPES
    for index in indices:
        counts[index] += 1
    return counts


def check_copy_limits(hand_counts, board_counts):
    """Enforce that no tile type, joker included, exceeds ``MAX_COPIES`` in play.

    Raises:
        ValueError: Naming the first tile type that appears too often.
    """
    for index in range(NUM_TYPES):
        total = hand_counts[index] + board_counts[index]
        if total > MAX_COPIES:
            raise ValueError(
                f"{canonical_tile(index)!r} appears {total} times across hand and "
                f"board; a set has only {MAX_COPIES}"
            )
