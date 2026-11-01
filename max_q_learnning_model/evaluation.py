"""Measurement layer: what a run RECORDS, one row per BLOCK.

Plain numbers only - no torch, no game engine, nothing from
simulation.py - so the dependency runs one way and every check here is
testable on synthetic data.

    TRAIN block -> {'avg_loss'}                    and nothing else
    TEST  block -> {'avg_reward', 'win_rate'}      and nothing else

Counts (the denominators) sit outside 'metrics'. A metric with no data
is None - never 0.0, never NaN.
"""
import math

MAX_DECK_VALUE = 2 * 4 * sum(range(1, 14)) + 2 * 30

REWARD_SCALE = 100.0

MAX_POSSIBLE_REWARD = MAX_DECK_VALUE / REWARD_SCALE

REWARD_BOUND_TOLERANCE = 1e-6

SCHEMA = "block-log/1"

TRAIN_METRICS = ("avg_loss",)
TEST_METRICS = ("avg_reward", "win_rate")

TRAIN_COUNTS = ("episodes", "updates")
TEST_COUNTS = ("games", "terminal_games", "decided_games")

MEAN_BOUND_TOLERANCE = 1e-9


def check_reward(reward, episode):
    """Raise if `reward` is non-finite or outside the scaled zero-sum bound.

    Call with the TRUE reward, never a shaped one: a potential-based term
    legitimately pushes a per-turn reward outside this range.
    """
    if math.isnan(reward) or math.isinf(reward):
        raise RuntimeError(
            f"[sanity check] non-finite reward ({reward}) at episode {episode} "
            f"- this is a reward-computation bug. Stopping immediately."
        )
    if abs(reward) > MAX_POSSIBLE_REWARD + REWARD_BOUND_TOLERANCE:
        raise RuntimeError(
            f"[sanity check] reward {reward} at episode {episode} is outside "
            f"the physically possible range "
            f"[-{MAX_POSSIBLE_REWARD}, +{MAX_POSSIBLE_REWARD}] - this is a "
            f"reward-computation bug, not an unusual game. Stopping immediately."
        )


def check_loss_and_q(loss, q_pred, episode):
    """Raise on a non-finite loss or Q-value."""
    if math.isnan(loss) or math.isinf(loss):
        raise RuntimeError(
            f"[sanity check] non-finite loss ({loss}) at episode {episode} - "
            f"stopping immediately, continuing would keep training a broken network."
        )
    if math.isnan(q_pred) or math.isinf(q_pred):
        raise RuntimeError(
            f"[sanity check] non-finite Q-value ({q_pred}) at episode {episode} - "
            f"stopping immediately."
        )


def is_finite_number(value):
    """True for a finite int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def strict_mean(values, what="value"):
    """Mean of a NON-EMPTY sample of finite numbers, checked three ways.

    Non-empty, all entries finite, and the result inside [min, max] of the
    sample - the check that catches a wrong denominator. math.fsum, so the
    result does not depend on collection order.
    """
    values = list(values)
    n = len(values)
    if n == 0:
        raise ValueError(
            f"strict_mean({what}) called with an empty sample - the caller "
            f"must report None for 'no data', not divide by zero")
    for v in values:
        if not is_finite_number(v):
            raise ValueError(
                f"strict_mean({what}) got a non-finite entry {v!r} - "
                f"non-finite values must be filtered (warmup) or raised on "
                f"(fire alarm) before they reach an average")
    mean = math.fsum(values) / n
    lo, hi = min(values), max(values)
    if not (lo - MEAN_BOUND_TOLERANCE <= mean <= hi + MEAN_BOUND_TOLERANCE):
        raise RuntimeError(
            f"[math check] mean of {what} is {mean}, outside the sample range "
            f"[{lo}, {hi}] over n={n} - the sum and the denominator disagree")
    return mean


def mean_or_none(values, what="value"):
    """strict_mean, with an empty sample mapped to None."""
    values = list(values)
    return strict_mean(values, what) if values else None


def sample_se(values, what="value"):
    """Standard error of the mean, ddof=1. None for n < 2, where the sample
    variance is undefined.
    """
    values = list(values)
    n = len(values)
    if n < 2:
        return None
    mean = strict_mean(values, what)
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    if variance < 0:
        raise RuntimeError(
            f"[math check] variance of {what} came out negative ({variance})")
    return math.sqrt(variance / n)


def proportion_se(rate_pct, n):
    """SE of a percentage from n Bernoulli trials, in percentage points.
    None for n <= 0; 0.0 at a 0% or 100% rate is correct, not missing.
    """
    if n <= 0:
        return None
    if not 0.0 <= rate_pct <= 100.0:
        raise ValueError(f"rate must be a percentage in [0, 100], got {rate_pct!r}")
    p = rate_pct / 100.0
    return math.sqrt(p * (1.0 - p) / n) * 100.0


class TrainBlockAccumulator:
    """Collects the one thing a training block reports: its average loss.

    Averaged over environment STEPS, so a long episode contributes in
    proportion to the training it drove. Warmup NaNs are counted as
    `skipped` and dropped; any other non-finite loss raises.
    """

    def __init__(self):
        """Start an empty training block."""
        self.losses = []
        self.episodes = 0
        self.skipped = 0

    def add_episode(self, losses):
        """Add one episode's per-step losses."""
        self.episodes += 1
        for loss in losses:
            if not isinstance(loss, (int, float)) or isinstance(loss, bool):
                raise TypeError(
                    f"a training block takes numeric losses, got {loss!r} "
                    f"({type(loss).__name__})")
            if math.isnan(loss):
                self.skipped += 1
                continue
            if not math.isfinite(loss):
                raise ValueError(
                    f"training block got a non-finite loss {loss!r} that is "
                    f"not a warmup NaN - check_loss_and_q should have fired "
                    f"on it at the source")
            self.losses.append(float(loss))

    def record(self, block, episode):
        """The block's validated train row.

        `episode` is the CUMULATIVE training-episode count at the end of the
        block - the x axis every curve is drawn against.
        """
        rec = {
            "block": block,
            "episode": episode,
            "metrics": {"avg_loss": mean_or_none(self.losses, "avg_loss")},
            "counts": {"episodes": self.episodes, "updates": len(self.losses)},
        }
        return validate_block_record(rec, "train")


class TestBlockAccumulator:
    """Collects a test block: average terminal reward and win rate.

    The reward is averaged over TERMINAL games only (an unfinished game is a
    non-measurement, not a draw) and the win rate over DECIDED ones, each
    with its own denominator. The reward stored is the true, unshaped
    outcome.
    """

    def __init__(self, opponent):
        """Start an empty test block against one opponent label."""
        if not isinstance(opponent, str) or not opponent:
            raise ValueError(f"opponent label must be a non-empty string, "
                             f"got {opponent!r}")
        self.opponent = opponent
        self.rewards = []
        self.wins = []
        self.games = 0

    def add_game(self, reward, won, terminal, episode=None):
        """Record one test game.

        reward   GE.get_reward(main_player), the true scaled payoff
        won      True / False / None (None = the game was not decided)
        terminal whether the game actually ended (GE.is_Done())
        """
        self.games += 1
        check_reward(reward, episode if episode is not None else self.games)
        if terminal:
            self.rewards.append(float(reward))
        elif reward != 0.0:
            raise RuntimeError(
                f"[sanity check] non-terminal game reported reward {reward} "
                f"(expected 0.0) vs {self.opponent} - get_reward and is_Done "
                f"disagree about whether this game finished")
        if won is not None:
            self.wins.append(bool(won))

    def record(self, block, episode):
        """The block's validated test row."""
        wins = self.wins
        win_rate = (100.0 * sum(wins) / len(wins)) if wins else None
        rec = {
            "block": block,
            "episode": episode,
            "opponent": self.opponent,
            "metrics": {
                "avg_reward": mean_or_none(self.rewards, "avg_reward"),
                "win_rate": win_rate,
            },
            "counts": {
                "games": self.games,
                "terminal_games": len(self.rewards),
                "decided_games": len(wins),
            },
        }
        return validate_block_record(rec, "test")

    def reward_se(self):
        """Within-block SE of the terminal rewards, or None below n=2."""
        return sample_se(self.rewards, "avg_reward")

    def win_rate_se(self):
        """Within-block SE of the win rate, in percentage points.
        None when no game was decided.
        """
        wins = self.wins
        if not wins:
            return None
        return proportion_se(100.0 * sum(wins) / len(wins), len(wins))


class RunLog:
    """The block log of ONE cell: one config, one training seed.

        {'schema', 'train': {'blocks': [...]} | None,
                   'test':  {'opponents': {label: [...]}} | None}

    A phase is null when it never ran, so 'no training phase' cannot be
    confused with 'trained and produced no rows'.
    """

    def __init__(self):
        """Start an empty run log."""
        self.train_blocks = []
        self.test_blocks = {}

    def add_train_block(self, record):
        """Append a validated train block row."""
        self.train_blocks.append(validate_block_record(record, "train"))

    def add_test_block(self, record):
        """Append a validated test block row under its opponent label."""
        record = validate_block_record(record, "test")
        self.test_blocks.setdefault(record["opponent"], []).append(record)

    def to_dict(self):
        """The validated run-log dict."""
        d = {
            "schema": SCHEMA,
            "train": {"blocks": list(self.train_blocks)} if self.train_blocks else None,
            "test": ({"opponents": {k: list(v) for k, v in self.test_blocks.items()}}
                     if self.test_blocks else None),
        }
        return validate_run_log(d)

    def final_test_metrics(self):
        """Final-block metrics per opponent - what a sweep ranks a cell on.

        Derived rather than stored, so it cannot drift from the series it
        summarises.
        """
        return final_test_metrics(self.to_dict())


def final_test_metrics(run_log):
    """{opponent: {avg_reward, win_rate, games}} from a run log DICT's last
    block, so a result read back from JSON is scored by the same code.
    {} when the run had no test phase.
    """
    if not run_log.get("test"):
        return {}
    out = {}
    for label, series in run_log["test"]["opponents"].items():
        last = series[-1]
        out[label] = {**last["metrics"], "games": last["counts"]["games"]}
    return out


def _check_count(value, name, where):
    """Raise unless `value` is an int >= 0."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: count {name} must be an int >= 0, got {value!r}")


def _check_metric(value, name, where, lo=None, hi=None):
    """Raise unless `value` is None or a finite number within [lo, hi]."""
    if value is None:
        return
    if not is_finite_number(value):
        raise ValueError(
            f"{where}: metric {name} must be a finite number or None "
            f"(None = no data for this block), got {value!r}")
    if lo is not None and value < lo - MEAN_BOUND_TOLERANCE:
        raise ValueError(f"{where}: metric {name}={value} is below its floor {lo}")
    if hi is not None and value > hi + MEAN_BOUND_TOLERANCE:
        raise ValueError(f"{where}: metric {name}={value} is above its ceiling {hi}")


def validate_block_record(record, kind):
    """Check one block row and return it; raises on anything malformed.

    The metric key set is checked EXACTLY, and every metric must be None if
    and only if its denominator is 0.
    """
    if kind not in ("train", "test"):
        raise ValueError(f"unknown block kind {kind!r}")
    where = f"{kind} block {record.get('block', '?')}"

    for key in ("block", "episode", "metrics", "counts"):
        if key not in record:
            raise ValueError(f"{where}: missing required key {key!r}")
    _check_count(record["block"], "block", where)
    if record["block"] < 1:
        raise ValueError(f"{where}: block index is 1-based, got {record['block']!r}")
    _check_count(record["episode"], "episode", where)

    expected_metrics = set(TRAIN_METRICS if kind == "train" else TEST_METRICS)
    got_metrics = set(record["metrics"])
    if got_metrics != expected_metrics:
        raise ValueError(
            f"{where}: metrics must be exactly {sorted(expected_metrics)}, got "
            f"{sorted(got_metrics)} - the block log's columns are a contract, "
            f"and every aggregate and plot downstream is built from them")

    expected_counts = set(TRAIN_COUNTS if kind == "train" else TEST_COUNTS)
    if set(record["counts"]) != expected_counts:
        raise ValueError(
            f"{where}: counts must be exactly {sorted(expected_counts)}, got "
            f"{sorted(record['counts'])}")
    for name, value in record["counts"].items():
        _check_count(value, name, where)

    if kind == "train":
        _check_metric(record["metrics"]["avg_loss"], "avg_loss", where, lo=0.0)
        if (record["metrics"]["avg_loss"] is None) != (record["counts"]["updates"] == 0):
            raise ValueError(
                f"{where}: avg_loss is "
                f"{'None' if record['metrics']['avg_loss'] is None else 'set'} but "
                f"updates={record['counts']['updates']} - a block reports a loss "
                f"if and only if at least one gradient update produced one")
    else:
        if not isinstance(record.get("opponent"), str) or not record["opponent"]:
            raise ValueError(f"{where}: test rows need a non-empty opponent label")
        _check_metric(record["metrics"]["avg_reward"], "avg_reward", where,
                      lo=-MAX_POSSIBLE_REWARD - REWARD_BOUND_TOLERANCE,
                      hi=MAX_POSSIBLE_REWARD + REWARD_BOUND_TOLERANCE)
        _check_metric(record["metrics"]["win_rate"], "win_rate", where,
                      lo=0.0, hi=100.0)
        counts = record["counts"]
        if counts["terminal_games"] > counts["games"] or \
                counts["decided_games"] > counts["games"]:
            raise ValueError(
                f"{where}: terminal_games={counts['terminal_games']} / "
                f"decided_games={counts['decided_games']} exceed "
                f"games={counts['games']}")
        if counts["decided_games"] > counts["terminal_games"]:
            raise ValueError(
                f"{where}: decided_games={counts['decided_games']} exceeds "
                f"terminal_games={counts['terminal_games']} - a game cannot "
                f"have a winner without having finished")
        if (record["metrics"]["avg_reward"] is None) != (counts["terminal_games"] == 0):
            raise ValueError(
                f"{where}: avg_reward and terminal_games disagree - a block "
                f"reports a reward if and only if at least one game finished")
        if (record["metrics"]["win_rate"] is None) != (counts["decided_games"] == 0):
            raise ValueError(
                f"{where}: win_rate and decided_games disagree - a block "
                f"reports a win rate if and only if at least one game was decided")
    return record


def _validate_series(series, kind, label=""):
    """Check one phase's series: valid rows, block indices 1..N with no gaps,
    non-decreasing cumulative episode counts.
    """
    where = f"{kind} series{' vs ' + label if label else ''}"
    if not series:
        raise ValueError(f"{where}: empty - a phase that ran produces rows, "
                         f"a phase that did not run must be null")
    for i, record in enumerate(series, start=1):
        validate_block_record(record, kind)
        if record["block"] != i:
            raise ValueError(
                f"{where}: block indices must be 1..N with no gaps, found "
                f"{record['block']} at position {i}")
    episodes = [r["episode"] for r in series]
    if any(b < a for a, b in zip(episodes, episodes[1:])):
        raise ValueError(
            f"{where}: cumulative episode counts must be non-decreasing, got "
            f"{episodes} (they can repeat only when a block trains 0 episodes)")


def validate_run_log(d):
    """Whole-cell check, run before the log is written and again when it is
    read back for aggregation. Returns the dict.
    """
    if d.get("schema") != SCHEMA:
        raise ValueError(
            f"run log schema is {d.get('schema')!r}, expected {SCHEMA!r} - "
            f"refusing to mix block logs written by different trees "
            f"(delete the stale result files and re-run those cells)")
    if d["train"] is None and d["test"] is None:
        raise ValueError(
            "run log has neither a train nor a test phase - nothing was "
            "measured, which is a config bug (both episodes-per-block are 0)")

    if d["train"] is not None:
        _validate_series(d["train"]["blocks"], "train")
    if d["test"] is not None:
        by_opponent = d["test"]["opponents"]
        if not by_opponent:
            raise ValueError("test phase present but no opponents recorded")
        grids = {}
        for label, series in by_opponent.items():
            _validate_series(series, "test", label)
            grids[label] = [(r["block"], r["episode"]) for r in series]
        reference_label, reference = next(iter(grids.items()))
        for label, grid in grids.items():
            if grid != reference:
                raise ValueError(
                    f"opponent {label!r} was measured on a different block grid "
                    f"than {reference_label!r} ({grid} vs {reference}) - "
                    f"per-block comparisons across opponents assume one grid")
    if d["train"] is not None and d["test"] is not None:
        n_train = len(d["train"]["blocks"])
        n_test = len(next(iter(d["test"]["opponents"].values())))
        if n_train != n_test:
            raise ValueError(
                f"{n_train} train blocks but {n_test} test blocks - the loop "
                f"runs them in pairs, so a mismatch means one phase failed "
                f"part-way through")
    return d