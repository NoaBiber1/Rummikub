"""Meld templates and board assembly.

A *meld* is either a group (three or four tiles of one value in distinct
colours) or a run (three or more consecutive values of one colour, 1 always
low, never wrapping past 13). A joker may stand in for any tile.

The solver chooses from a fixed catalogue of *templates*: joker-free melds
given as the tile types filling their positions. It holds

* every group of size 3 or 4;
* every run of length 3, 4 or 5;

185 templates in all. Jokers are not part of the catalogue: the solver lets
a joker fill any position of any template (see ``rummikub_greedy``), which
replaces the hundreds of joker patterns a joker-aware catalogue would need.

Runs of length six or more are omitted on purpose. Any run of length
``L >= 6`` splits into consecutive pieces of lengths 3-5, and since a set has
only two jokers every piece of length three or more keeps a real tile, so
each piece is a legal run on its own. Dropping long runs therefore changes
nothing about which tile multisets can be laid out. ``assemble`` re-joins
consecutive pieces so the reported board shows the long runs a player would
actually lay down.
"""

from functools import lru_cache
from itertools import combinations
from typing import NamedTuple

from rummikub_tiles import JOKER, NUM_COLORS, NUM_VALUES, tile_index, tile_value

MIN_MELD = 3
MAX_GROUP = 4
MAX_ATOMIC_RUN = 5
GROUP = "group"
RUN = "run"


class Meld(NamedTuple):
    """One joker-free template.

    Attributes:
        layout: Real tile type indices in display order.
        size: Number of tiles.
        points: Sum of the face values of the positions.
        kind: ``GROUP`` or ``RUN``.
        color: Colour index of a run; ``-1`` for a group.
        start: First value of a run, or the shared value of a group.
        mask: Bitmask with bit ``t`` set for every type ``t`` in ``layout``.
    """

    layout: tuple
    size: int
    points: int
    kind: str
    color: int
    start: int
    mask: int


def _make(layout, kind, color, start):
    """Build a ``Meld`` from its display layout and descriptive fields."""
    mask = 0
    for t in layout:
        mask |= 1 << t
    points = sum(tile_value(t) for t in layout)
    return Meld(layout, len(layout), points, kind, color, start, mask)


@lru_cache(maxsize=None)
def catalogue():
    """Return every template as a tuple of ``Meld``: groups by value, then runs by colour and start."""
    melds = []
    for value in range(1, NUM_VALUES + 1):
        for size in (MIN_MELD, MAX_GROUP):
            for colors in combinations(range(NUM_COLORS), size):
                layout = tuple(tile_index(value, c) for c in colors)
                melds.append(_make(layout, GROUP, -1, value))
    for color in range(NUM_COLORS):
        for start in range(1, NUM_VALUES + 1):
            for length in range(MIN_MELD, MAX_ATOMIC_RUN + 1):
                if start + length - 1 > NUM_VALUES:
                    break
                layout = tuple(tile_index(v, color) for v in range(start, start + length))
                melds.append(_make(layout, RUN, color, start))
    return tuple(melds)


def _placement_order(pair):
    """Sort key for ``(meld, layout)`` pairs: start value, then size, then layout."""
    meld, layout = pair
    return meld.start, meld.size, layout


def _jokers_last(layout):
    """Return a group's layout with its real tiles first and its jokers after them."""
    return tuple(t for t in layout if t != JOKER) + tuple(t for t in layout if t == JOKER)


def _chain_runs(runs):
    """Join runs of one colour end-to-start into the longest chains available.

    A run starting at value ``v`` is appended to any chain that currently ends
    at ``v - 1``; concatenating consecutive runs always yields a legal run.

    Args:
        runs: List of ``(meld, layout)`` pairs of one colour.

    Returns:
        A list of layouts ordered by start value.
    """
    open_by_end = {}
    chains = []
    for meld, layout in sorted(runs, key=_placement_order):
        waiting = open_by_end.get(meld.start - 1)
        if waiting:
            chain = waiting.pop()
        else:
            chain = []
            chains.append(chain)
        chain.extend(layout)
        open_by_end.setdefault(meld.start + meld.size - 1, []).append(chain)
    return [tuple(chain) for chain in chains]


def assemble(placed):
    """Turn placed template copies into display sets.

    Groups come first, ordered by value, each listing its jokers last; runs
    follow, ordered by colour then start, with consecutive runs of one colour
    merged into single long runs and each joker in the position it fills.

    Args:
        placed: Iterable of ``(meld, layout)`` pairs, one per copy on the
            table, where ``layout`` is the template's layout with ``JOKER``
            in any position a joker fills.

    Returns:
        A list of layouts, each a tuple of type indices in display order.
    """
    groups = []
    runs = [[] for _ in range(NUM_COLORS)]
    for meld, layout in placed:
        if meld.kind == GROUP:
            groups.append((meld, layout))
        else:
            runs[meld.color].append((meld, layout))
    sets = [_jokers_last(layout) for _, layout in sorted(groups, key=_placement_order)]
    for color_runs in runs:
        sets.extend(_chain_runs(color_runs))
    return sets
