"""Content-addressed cache for sweep cells.

A cell's numbers are a pure function of three things: the config it
actually ran under (`simulation.resolved_config`'s output, so every key
DEFAULTS supplies is included, not only the ones a sweep varies), the
training seed, and the version of the code that produced them.
`fingerprint` hashes exactly those three and nothing else - in
particular NOT the config's `name`, so two cells that differ only in
what they are called share one fingerprint and are computed once.

A finished cell is published to `cache_dir()/<fingerprint>.json` and
copied back into a sweep's own directory on a hit, so `aggregate.py` and
`plots.py` keep reading the per-config layout they have always read.

CODE_VERSION digests RESULT_SOURCES, the modules that can change a
cell's numbers - not the whole tree, because editing `plots.py` or
`aggregate.py` must not invalidate sixty hours of training. Set
SWEEP_CODE_VERSION to pin it by hand when a change to one of those
modules is known to be behaviour-preserving; that is a promise the cache
cannot check on your behalf.
"""
import hashlib
import json
import os
import tempfile
from collections import namedtuple

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_SOURCES = (
    "evaluation.py",
    "game_env.py",
    "greedy_alg.py",
    "ilp_solution.py",
    "learning.py",
    "q_model.py",
    "replay_buffer.py",
    "seed_sweep.py",
    "simulation.py",
)

VERSIONED_PACKAGES = ("numpy", "pulp", "torch")

CODE_VERSION_ENV = "SWEEP_CODE_VERSION"
CACHE_DIR_ENV = "SWEEP_CACHE_DIR"
DEFAULT_CACHE_DIR = os.path.join(PROJECT_DIR, "checkpoints", "cells")

FINGERPRINT_KEY = "fingerprint"
CODE_VERSION_KEY = "code_version"
FULL_CONFIG_KEY = "full_config"
FINGERPRINT_LENGTH = 16


def _package_version(name):
    """Installed version of `name`, or a marker when it is absent.

    Read from the installation metadata rather than by importing, so
    resolving a fingerprint never pays for a torch import.
    """
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


def _source_digest():
    """SHA-256 over RESULT_SOURCES, newline-normalised.

    Raises when one of them is missing rather than digesting a shorter
    list: a silently narrower digest would keep serving cached results
    after the module that went missing had changed. Line endings are
    normalised so a checkout that rewrites them is not mistaken for a
    behavioural change; nothing else about the bytes is forgiven.
    """
    digest = hashlib.sha256()
    for name in RESULT_SOURCES:
        path = os.path.join(PROJECT_DIR, name)
        try:
            with open(path, "rb") as handle:
                source = handle.read().replace(b"\r\n", b"\n")
        except OSError as exc:
            raise RuntimeError(
                f"cannot digest {name} for the cache's code version ({exc}) - "
                f"RESULT_SOURCES lists the modules whose behaviour can change "
                f"a cell's numbers, so a missing one means the fingerprint "
                f"would stop noticing edits to it and stale results would be "
                f"served as if they were current") from exc
        digest.update(f"{name}:{len(source)}\0".encode())
        digest.update(source)
    return digest.hexdigest()


def _compute_code_version():
    """The identity of the code that produces a cell's numbers.

    Source digest plus the versions of the three packages that decide
    what that source computes. A pulp bump is a solver bump, and the
    optimum an ILP returns is rarely unique, so a cell solved by a
    different CBC is not the same cell.
    """
    pinned = os.environ.get(CODE_VERSION_ENV)
    if pinned:
        return pinned
    packages = "".join(
        f"{name}={_package_version(name)};" for name in VERSIONED_PACKAGES)
    payload = f"{_source_digest()}|{packages}"
    return hashlib.sha256(payload.encode()).hexdigest()[:FINGERPRINT_LENGTH]


CODE_VERSION = _compute_code_version()


def canonical_config(full_config):
    """`full_config` without `name`, as one canonical JSON string.

    Keys are sorted at every level, and JSON's own type normalisation
    means a tuple that became a list on its way through a worker's stdin
    hashes identically to the tuple it started as.
    """
    payload = {key: value for key, value in full_config.items() if key != "name"}
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise TypeError(
            f"config is not JSON-serialisable, so it can be neither "
            f"fingerprinted nor sent to a worker ({exc}). Every cell runs in "
            f"its own interpreter, so a model-backed opponent has to be named "
            f"by checkpoint path rather than passed as a live object") from exc


def fingerprint(full_config, seed):
    """The cache key of one cell: its full config, its seed, CODE_VERSION.

    The seed is already inside `full_config`; it is hashed again because
    it is the axis a sweep varies deliberately and a cache key that
    depended on it only indirectly would be one refactor away from
    silently pooling seeds.
    """
    payload = "|".join((canonical_config(full_config), str(seed), CODE_VERSION))
    return hashlib.sha256(payload.encode()).hexdigest()[:FINGERPRINT_LENGTH]


class Cell(namedtuple("Cell", "config seed full_config fingerprint")):
    """One (config, seed) unit of work, resolved and fingerprinted.

    `config` is the sweep-level dict as written, kept because that is
    what a worker is sent; `full_config` is what the run will actually
    use, and is what the fingerprint is taken over.
    """

    __slots__ = ()

    @property
    def name(self):
        """The config name, which selects the output directory."""
        return self.config["name"]

    @property
    def key(self):
        """(name, seed): unique within a sweep, and the result-dict key."""
        return self.name, self.seed


def cache_dir():
    """Where finished cells are published: SWEEP_CACHE_DIR, or the default.

    One directory for the whole project, not one per sweep - the point is
    that a cell run for step 3a is found again when step 3c asks for the
    same work under a different name.
    """
    return os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR


def cache_path(cell_fingerprint):
    """Path of the cache entry for a fingerprint."""
    return os.path.join(cache_dir(), f"{cell_fingerprint}.json")


def write_json(path, payload):
    """Write `payload` to `path` atomically, creating the directory.

    Through a temporary file in the same directory and `os.replace`, so a
    reader never sees half a result and a cell killed mid-write leaves
    the previous file intact instead of a truncated one.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def read_verified(path, cell_fingerprint):
    """The result at `path` if it carries `cell_fingerprint`, else None.

    Missing, unreadable, truncated and STALE files all come back as None,
    which every caller treats as 'not done'. The fingerprint check is
    what makes resuming safe: a result written before a config key or a
    source file changed no longer matches, so the cell is re-run instead
    of being handed back as though nothing had happened.
    """
    try:
        with open(path) as stream:
            result = json.load(stream)
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    if result.get(FINGERPRINT_KEY) != cell_fingerprint:
        return None
    return result


def publish(result):
    """Copy a finished result into the cache under its own fingerprint.

    Failures are reported and swallowed. The cache is an optimisation,
    and losing an entry must never fail a cell that has already written
    the result file the sweep actually reads.
    """
    try:
        write_json(cache_path(result[FINGERPRINT_KEY]), result)
    except OSError as exc:
        print(f"[cache] could not publish {result[FINGERPRINT_KEY]}: {exc}")


def relabel(result, cell):
    """`result` as it should appear in `cell`'s own sweep directory.

    A fingerprint deliberately ignores `name`, so a reused result can have
    been produced under a different one. `aggregate.py` groups per-seed
    files by their `config` field, so the copy written into a sweep
    directory has to carry that sweep's name - in the record, and in the
    config the record embeds.
    """
    full_config = {**result[FULL_CONFIG_KEY], "name": cell.name}
    return {**result, "config": cell.name, FULL_CONFIG_KEY: full_config}
