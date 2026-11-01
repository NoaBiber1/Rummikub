"""Baseline-vs-baseline matches. NO NETWORK IS INVOLVED, EVER.

A MEASUREMENT SCRIPT, not a second entry point: it trains nothing,
builds no MLP and writes no weights. It fills the one hole the training
pipeline cannot - greedy vs random - because simulation is
agent-vs-ONE-baseline by construction.

    python baseline_match.py                     # greedy vs random, 200
    python baseline_match.py --games 500
    python baseline_match.py --main random --opponent random   # ~0

--eval-seed-base defaults to seed_sweep.EVAL_SEED_BASE, so these games
are played on THE SAME DECKS as every test block in a sweep and the
greedy bar is paired with the agent's vs-greedy score. Reporting goes
through evaluation.TestBlockAccumulator, so the numbers share their
definitions and denominators with the sweep's.
"""
import argparse
import json
import os

import evaluation as ev
import simulation as sim

POLICIES = {
    "greedy": sim.greedy_opponent,
    "random": sim.random_opponent,
}

DEFAULT_BUDGET = dict(max_actions=12, alt_counts=2, alts_per_count=2)


def _no_update(x, rewards, next_valid_x_list, done, episode):
    """_play_episode's `update` hook, neutered. Returns NaN, the honest 'no
    update happened' sentinel, rather than a 0.0 that would average in as a
    real loss.
    """
    return float("nan")


def play_match(main="greedy", opponent="random", games=200, eval_seed_base=None,
               budget=None):
    """`games` 1-vs-1 games of `main` against `opponent`.

    Returns (block record, accumulator, total turns). Scores are from MAIN'S
    SEAT; the payoff is zero-sum, so the opponent's number is its negation.
    Seats alternate on itr % 2, as both pipeline loops do.
    """
    from game_env import GE

    if main not in POLICIES or opponent not in POLICIES:
        raise ValueError(
            f"policies must be one of {sorted(POLICIES)} - this script is "
            f"net-free by design, so model-backed opponents are not accepted; "
            f"score those through seed_sweep.py instead")

    budget = DEFAULT_BUDGET if budget is None else budget
    if eval_seed_base is None:
        import seed_sweep
        eval_seed_base = seed_sweep.EVAL_SEED_BASE
    seeds = list(range(eval_seed_base, eval_seed_base + games))

    main_policy, opponent_policy = POLICIES[main], POLICIES[opponent]
    ge = GE(sim.PLAYERS_PER_GAME, budget)
    block_log = ev.TestBlockAccumulator(f"{main} vs {opponent}")
    total_turns = 0

    for itr in range(1, games + 1):
        sim._seed_episode(seeds, itr)
        ge.reset()
        main_player = itr % sim.PLAYERS_PER_GAME

        _losses, turns = sim._play_episode(
            ge, itr, main_player, main_policy, opponent_policy,
            opponents=1, update=_no_update,
            n_step=1, gamma=0.99, shaping=False)
        total_turns += turns

        block_log.add_game(float(ge.get_reward(main_player)),
                           sim._won(ge, main_player), ge.is_Done(), episode=itr)

    return block_log.record(block=1, episode=0), block_log, total_turns


def summarize_match(record, block_log, main, opponent, eval_seed_base, games,
                    total_turns):
    """The saved JSON: the block record's two metrics, the two SEs computed
    from the accumulator's own sample, the counts and the average length.
    """
    counts = record["counts"]
    return {
        "main": main,
        "opponent": opponent,
        "games": games,
        "eval_seed_base": eval_seed_base,
        "avg_reward": record["metrics"]["avg_reward"],
        "avg_reward_se": block_log.reward_se(),
        "win_rate": record["metrics"]["win_rate"],
        "win_rate_se": block_log.win_rate_se(),
        "terminal_games": counts["terminal_games"],
        "decided_games": counts["decided_games"],
        "avg_turns": total_turns / max(games, 1),
    }


def print_match(result):
    """Print one match result."""
    def num(value, fmt):
        """Format a value, or 'n/a' when it is None."""
        return "n/a" if value is None else format(value, fmt)

    print(f"\n=== {result['main']} (main) vs {result['opponent']} "
          f"({result['games']} games) ===")
    print(f"  avg reward     : {num(result['avg_reward'], '.2f')} "
          f"(SE +/-{num(result['avg_reward_se'], '.2f')}) over "
          f"{result['terminal_games']} finished games")
    print(f"  win rate       : {num(result['win_rate'], '.1f')}% "
          f"(SE +/-{num(result['win_rate_se'], '.1f')}%) over "
          f"{result['decided_games']} decided games")
    print(f"  avg game length: {result['avg_turns']:.1f} main-player turns\n")


def _out_path(main, opponent):
    """Path of the JSON file for a match, created if needed."""
    d = os.path.join(sim.CHECKPOINT_DIR, "baselines")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{main}_vs_{opponent}.json")


def main():
    """CLI: play one match, save it, and apply the step-1 gate (greedy must
    clear random by more than 2 SE, or the bar the campaign is measured
    against is not a bar).
    """
    p = argparse.ArgumentParser(
        description="Baseline-vs-baseline match. Trains nothing.")
    p.add_argument("--main", default="greedy", choices=sorted(POLICIES))
    p.add_argument("--opponent", default="random", choices=sorted(POLICIES))
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--eval-seed-base", type=int, default=None,
                   help="default: seed_sweep.EVAL_SEED_BASE (paired with the "
                        "sweep's test blocks)")
    p.add_argument("--max-actions", type=int, default=12)
    p.add_argument("--alt-counts", type=int, default=2)
    p.add_argument("--alts-per-count", type=int, default=2)
    args = p.parse_args()

    budget = dict(max_actions=args.max_actions, alt_counts=args.alt_counts,
                  alts_per_count=args.alts_per_count)
    if args.eval_seed_base is None:
        import seed_sweep
        args.eval_seed_base = seed_sweep.EVAL_SEED_BASE

    print(f"[baseline match] {args.main} vs {args.opponent}, {args.games} games, "
          f"eval_seed_base={args.eval_seed_base}, budget={budget}")

    record, block_log, total_turns = play_match(
        args.main, args.opponent, args.games, args.eval_seed_base, budget)
    result = summarize_match(record, block_log, args.main, args.opponent,
                             args.eval_seed_base, args.games, total_turns)
    print_match(result)

    path = _out_path(args.main, args.opponent)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved {path}]")

    if {args.main, args.opponent} == {"greedy", "random"}:
        margin, se = result["avg_reward"], result["avg_reward_se"]
        if margin is None or se is None:
            print("\n[step 1 gate] NOT MEASURED: fewer than two games produced "
                  "a terminal reward, so there is no margin and no SE. Check "
                  "the budget and the engine before reading anything into this.")
            return 1
        margin *= (1 if args.main == "greedy" else -1)
        verdict = "PASS" if margin > 2 * se else "FAIL"
        print(f"\n[step 1 gate] greedy over random: {margin:+.3f} "
              f"(2 SE = {2 * se:.3f}) -> {verdict}")
        if verdict == "FAIL":
            print("  Greedy does not clearly beat random. The greedy baseline "
                  "or the eval harness is wrong - not the agent. Stop here.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())