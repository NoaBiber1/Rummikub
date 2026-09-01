"""Baseline-vs-baseline matches. NO NETWORK IS INVOLVED, EVER.

Fills the one hole in Step 1 of the test plan that the training pipeline
cannot: `greedy vs random`. simulation.run_test_simulation is main-agent-vs-ONE-
baseline by construction - PLAYERS_PER_GAME is 2, and section 5.1 records that
the multi-seat path was removed on purpose, because seating two baselines at
the same table scores a game the model barely influenced. So there is no config
that makes two baselines play each other, and there should not be: that is a
different measurement, and it belongs in a different file.

This is a MEASUREMENT SCRIPT, not a second entry point. It trains nothing,
constructs no MLP, loads no checkpoint and writes no weights. seed_sweep.py
remains the only way to run training.

WHAT IT IS FOR. Step 1 needs a reference floor (an untrained net) and a
reference BAR (greedy). The bar is only meaningful if greedy actually beats
random by a clear margin - if it doesn't, the greedy baseline or the eval
harness is broken, and every later "the agent beat greedy" claim is measuring
something else. That check cannot be made from inside the sweep.

    python baseline_match.py                       # greedy vs random, 200 games
    python baseline_match.py --games 500
    python baseline_match.py --main random --opponent random   # control: ~0

COMMON RANDOM NUMBERS. --eval-seed-base defaults to seed_sweep.EVAL_SEED_BASE,
so these games are played on THE SAME DECKS as every test block in the sweep.
That is the point: the greedy bar and the agent's vs-greedy score are then
paired, and the difference between them is not deck luck.
"""
import argparse
import json
import os

import evaluation as ev
import simulation as sim

# The policies this script will seat. Deliberately only the two net-free ones:
# anything model-backed would need an MLP, which is exactly what this script
# exists not to touch.
POLICIES = {
    "greedy": sim.greedy_opponent,
    "random": sim.random_opponent,
}

DEFAULT_BUDGET = dict(max_actions=12, alt_counts=2, alts_per_count=2)


def _no_update(x, rewards, next_valid_x_list, done, episode):
    """_play_episode's `update` hook, neutered.

    Returning NaN rather than 0.0 is not cosmetic: evaluation._nanmean drops
    NaNs, so avg_losses/avg_qs come out as NaN ("no data") instead of a fake
    zero that would print as a real, very low loss for a run that never
    computed one."""
    return float("nan"), float("nan")


def play_match(main="greedy", opponent="random", games=200, eval_seed_base=None,
               budget=None, quiet=False):
    """`games` 1-vs-1 games of `main` against `opponent`. Returns a history.

    Scores are reported FROM MAIN'S SEAT. The payoff is zero-sum, so the
    opponent's number is the negation of this one; there is no second row to
    measure.

    Seats alternate on itr % 2, exactly as run_self_play_training and
    _test_block do, so neither policy gets the first-move advantage for the
    whole match.
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
    history = ev.new_history()

    for itr in range(1, games + 1):
        sim._seed_episode(seeds, itr)
        ge.reset()
        main_player = itr % sim.PLAYERS_PER_GAME

        losses, qs, turns = sim._play_episode(
            ge, itr, main_player, main_policy, opponent_policy,
            opponents=1, update=_no_update,
            # n_step/gamma only shape the target `update` would have consumed,
            # and it consumes nothing. shaping stays OFF so the recorded reward
            # is the true game outcome, comparable with every sweep number.
            n_step=1, gamma=0.99, shaping=False)

        ev.record_episode(history, itr, float(ge.get_reward(main_player)),
                          sim._won(ge, main_player), losses, qs, 0.0, turns)

        if not quiet and itr % 50 == 0:
            print(f"  [{itr}/{games}]")

    return history


def summarize_match(history, main, opponent, eval_seed_base, games):
    stats = ev.summarize(history)
    return {
        "main": main,
        "opponent": opponent,
        "games": games,
        "eval_seed_base": eval_seed_base,
        "avg_reward": stats["avg_reward"],
        "avg_reward_se": stats["avg_reward_se"],
        "win_rate": stats["win_rate"],
        "win_rate_se": stats["win_rate_se"],
        "n": stats["n"],
        "avg_turns": float(sum(history["turns"]) / max(len(history["turns"]), 1)),
    }


def _out_path(main, opponent):
    d = os.path.join(sim.CHECKPOINT_DIR, "baselines")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{main}_vs_{opponent}.json")


def main():
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
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    budget = dict(max_actions=args.max_actions, alt_counts=args.alt_counts,
                  alts_per_count=args.alts_per_count)
    if args.eval_seed_base is None:
        import seed_sweep
        args.eval_seed_base = seed_sweep.EVAL_SEED_BASE

    print(f"[baseline match] {args.main} vs {args.opponent}, {args.games} games, "
          f"eval_seed_base={args.eval_seed_base}, budget={budget}")

    history = play_match(args.main, args.opponent, args.games,
                         args.eval_seed_base, budget, args.quiet)

    ev.print_summary(history, f"{args.main} (main) vs {args.opponent}")
    result = summarize_match(history, args.main, args.opponent,
                             args.eval_seed_base, args.games)

    path = _out_path(args.main, args.opponent)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved {path}]")

    # The Step 1 gate, stated where it is measured rather than left to a
    # reader: greedy must clear random by more than 2 SE, or the bar the whole
    # campaign is measured against is not a bar.
    if {args.main, args.opponent} == {"greedy", "random"}:
        margin = result["avg_reward"] * (1 if args.main == "greedy" else -1)
        se = result["avg_reward_se"]
        verdict = "PASS" if margin > 2 * se else "FAIL"
        print(f"\n[step 1 gate] greedy over random: {margin:+.3f} "
              f"(2 SE = {2 * se:.3f}) -> {verdict}")
        if verdict == "FAIL":
            print("  Greedy does not clearly beat random. The greedy baseline "
                  "or the eval harness is wrong - not the agent. Stop here.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
