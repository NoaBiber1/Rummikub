"""Latency benchmark for the PuLP 4.2 + HiGHS greedy Rummikub solver.

Positions are harvested from complete greedy-versus-greedy games, so they
cover the whole game from opening melds to crowded late-game tables. The
script then times three situations:

* cold: every cache off, so each call also solves the board-validation model;
* in game: positions replayed in order with caches on, so a board this solver
  produced on the previous turn is trusted without re-validation;
* repeat: every position asked a second time, served from the memo.

Run with ``python bench_rummikub_greedy.py [games] [seed]``.
"""

import random
import statistics
import sys
import time

import pulp

from rummikub_greedy import DRAW, GreedySolver
from rummikub_tiles import COLORS


def harvest(games, seed):
    """Play ``games`` greedy self-play games and return every ``(hand, board, first_meld)`` faced."""
    rng = random.Random(seed)
    player = GreedySolver()
    positions = []
    for _ in range(games):
        deck = [(v, c) for c in COLORS for v in range(1, 14)] * 2 + ["J", "J"]
        rng.shuffle(deck)
        hands = [deck[:14], deck[14:28]]
        deck = deck[28:]
        board, melded, passes, turn = [], [False, False], 0, 0
        while passes < 2 and all(hands):
            me = turn % 2
            positions.append((list(hands[me]), list(board), not melded[me]))
            result = player.solve(hands[me], board, not melded[me])
            if result == DRAW:
                if deck:
                    hands[me].append(deck.pop())
                    passes = 0
                else:
                    passes += 1
            else:
                hands[me] = result.hand
                board = [tile for tile_set in result.sBoard for tile in tile_set]
                melded[me], passes = True, 0
            turn += 1
    return positions


def percentile(sorted_values, fraction):
    """Return the nearest-rank percentile of an ascending list."""
    index = min(len(sorted_values) - 1, max(0, round(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def timings(engine, positions):
    """Return per-position wall times in milliseconds for one pass over ``positions``."""
    times = []
    for hand, board, first_meld in positions:
        start = time.perf_counter()
        engine.solve(hand, board, first_meld)
        times.append(1000 * (time.perf_counter() - start))
    return times


def report(label, times):
    """Print one row of mean, median, tail percentiles and worst case."""
    times = sorted(times)
    row = [
        statistics.fmean(times),
        statistics.median(times),
        percentile(times, 0.95),
        percentile(times, 0.99),
        times[-1],
    ]
    print(f"{label:10s}" + "".join(f"{v:8.2f}" for v in row))


def main():
    """Harvest positions, time the three situations, and print a summary table."""
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    positions = harvest(games, seed)
    crowded = sum(1 for _, board, _ in positions if len(board) >= 40)
    print(
        f"PuLP {pulp.__version__} + HiGHS: {len(positions)} positions from {games} games, "
        f"{crowded} with 40+ board tiles"
    )
    print(f"{'':10s}{'mean':>8s}{'median':>8s}{'p95':>8s}{'p99':>8s}{'max':>8s}   ms/position")
    report("cold", timings(GreedySolver(cache_size=0), positions))
    engine = GreedySolver()
    report("in game", timings(engine, positions))
    report("repeat", timings(engine, positions))


if __name__ == "__main__":
    main()
