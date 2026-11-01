"""Parallel cell execution for seed_sweep: one subprocess per cell.

Subprocesses, not multiprocessing: spawn and forkserver both re-import
__main__, which does not exist when the entry point is stdin; fork can
deadlock a forked OpenMP pool; and one dead cell must not take the sweep
with it. Each cell gets a clean interpreter and returns its result
through the JSON file it writes, so there is no IPC and nothing to
pickle. Cells seed themselves, so a parallel sweep is bit-identical to a
serial one.

Cells are content-addressed (see cell_cache). Nothing is launched until
every cell has been resolved, fingerprinted and checked, and a cell whose
fingerprint already has a VERIFIED result is served from it rather than
computed again - from this sweep's own directory, from the central cache
where another sweep left it under a different name, or from an identical
sibling running right now. A result that no longer matches its
fingerprint is not a hit; it is re-run.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, namedtuple

import cell_cache as cc

_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

_Running = namedtuple(
    "_Running", "cell proc log out_path err_path log_path started")


def cell_paths(out_dir, name, seed):
    """(config dir, result path, error path, log path) for one cell."""
    cfg_dir = os.path.join(out_dir, name)
    stem = os.path.join(cfg_dir, f"seed{seed}")
    return cfg_dir, f"{stem}.json", f"{stem}.error.json", f"{stem}.log"


def _cell_main():
    """Child entry point: read a cell spec on stdin, run it, write the result.
    The file IS the channel back to the parent.

    The result is checked against the fingerprint the parent planned
    before it is written anywhere. A mismatch means this worker resolved
    a different config from the one the sweep checked its cache against,
    and publishing it would poison the cache for every later sweep.
    """
    import seed_sweep

    spec = json.load(sys.stdin)
    _, out_path, err_path, _ = cell_paths(
        spec["out_dir"], spec["config"]["name"], spec["seed"])
    try:
        result = seed_sweep._run_one(spec["config"], spec["seed"],
                                     spec["overrides"])
        if result[cc.FINGERPRINT_KEY] != spec["fingerprint"]:
            raise RuntimeError(
                f"this cell resolved to fingerprint "
                f"{result[cc.FINGERPRINT_KEY]} but the parent planned "
                f"{spec['fingerprint']} - the config that ran is not the one "
                f"the sweep checked its cache against, so the result is not "
                f"safe to publish")
    except BaseException:
        cc.write_json(err_path, {"config": spec["config"]["name"],
                                 "seed": spec["seed"],
                                 "traceback": traceback.format_exc()})
        raise

    cc.write_json(out_path, result)
    cc.publish(result)
    if os.path.exists(err_path):
        os.remove(err_path)


def _launch(spec, log_path):
    """Start one cell subprocess; returns (process, open log file)."""
    env = {**os.environ, **_SINGLE_THREAD_ENV}
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-c", "import parallel_sweep; parallel_sweep._cell_main()"],
        cwd=PROJECT_DIR, env=env, stdin=subprocess.PIPE,
        stdout=log, stderr=subprocess.STDOUT, text=True)
    proc.stdin.write(json.dumps(spec))
    proc.stdin.close()
    return proc, log


def _check_serializable(config, overrides):
    """Raise a usable error if a cell spec cannot be sent as JSON.

    Every cell runs in its own interpreter whatever `workers` is, so a
    config carrying a live net or a callable cannot be run at all - it
    has to be named by checkpoint path instead.
    """
    try:
        json.dumps({"config": config, "overrides": overrides})
    except TypeError as exc:
        raise TypeError(
            f"config {config.get('name', '?')!r} is not JSON-serialisable and "
            f"so cannot be sent to a worker process ({exc}). Every cell runs "
            f"in its own interpreter, so model-backed opponents have to be "
            f"named by checkpoint path rather than passed as objects.") from exc


def _validate_cells(cells, overrides):
    """Reject a cell list that cannot be run safely, before anything starts.

    Three failures are cheap here and expensive later. A repeated
    (config name, seed) maps two subprocesses onto one result file, so
    they overwrite each other while running and the survivor is then
    counted twice in every mean and shrinks every variance. A
    `checkpoint_path` shared by more than one cell has each of them
    saving its weights over the others' with nothing raising anywhere. A
    spec that cannot be serialised cannot reach a worker at all.
    """
    repeated = sorted(key for key, count in Counter(c.key for c in cells).items()
                      if count > 1)
    if repeated:
        raise ValueError(
            f"these (config, seed) cells appear more than once: {repeated} - "
            f"a cell is identified by that pair, so the duplicates would race "
            f"for one result file and then be counted twice in every mean. "
            f"Give the configs distinct names, or the sweep distinct seeds")

    owners = {}
    for cell in cells:
        path = cell.full_config.get("checkpoint_path")
        if path is not None:
            owners.setdefault(path, []).append(cell.key)
    shared = sorted((path, keys) for path, keys in owners.items() if len(keys) > 1)
    if shared:
        detail = "; ".join(f"{path!r} <- {keys}" for path, keys in shared)
        raise ValueError(
            f"checkpoint_path is shared by more than one cell ({detail}) - the "
            f"cells run at the same time and torch.save is not atomic, so they "
            f"would interleave into one file and the weights left behind would "
            f"belong to no cell in particular. A checkpoint_path belongs to a "
            f"config that runs exactly one seed")

    for cell in cells:
        _check_serializable(cell.config, overrides)


def _reuse(cell, out_dir, resume):
    """A verified previous result for `cell`, with a note saying where from.

    Two places are searched: the sweep's own result file, then the
    central cache, where a cell another sweep already ran - possibly
    under a different config name - waits under the same fingerprint.
    Both are checked against the fingerprint, so a result left by an
    older config or an older tree is a MISS and gets re-run.

    Returns (result, note) on a hit and (None, note) on a miss, where the
    note is None unless there is something the caller should say out loud.
    """
    if not resume:
        return None, None

    _, out_path, _, _ = cell_paths(out_dir, cell.name, cell.seed)
    result = cc.read_verified(out_path, cell.fingerprint)
    if result is not None:
        cc.publish(result)
        return result, f"verified result: {out_path}"

    result = cc.read_verified(cc.cache_path(cell.fingerprint), cell.fingerprint)
    if result is not None:
        return result, (f"cached {cell.fingerprint} from "
                        f"config {result['config']!r}")

    if os.path.exists(out_path):
        return None, f"config or code changed since {out_path}"
    return None, None


def _materialize(result, cell, out_dir):
    """Write a reused result into `cell`'s own result file and return it.

    Relabelled first, because the result may have been produced under
    another config's name and `aggregate.py` groups per-seed files by
    that field.
    """
    _, out_path, _, _ = cell_paths(out_dir, cell.name, cell.seed)
    labelled = cc.relabel(result, cell)
    cc.write_json(out_path, labelled)
    return labelled


def _take_reusable(pending, completed):
    """Remove and return the pending cells already computed in this sweep.

    Two cells sharing a fingerprint are the same work under two names,
    which is what a coordinate-descent plan produces whenever one step's
    arm is another step's carried-forward winner.
    """
    reusable = [cell for cell in pending if cell.fingerprint in completed]
    if reusable:
        pending[:] = [cell for cell in pending
                      if cell.fingerprint not in completed]
    return reusable


def _take_launchable(pending, in_flight, slots):
    """Remove and return up to `slots` cells that are ready to launch.

    A cell whose fingerprint is already running is left where it is: it
    would compute exactly what that process is computing, so it waits and
    reuses the answer instead. Order is otherwise preserved.
    """
    taken, blocked, keep = [], set(in_flight), []
    for cell in pending:
        if len(taken) < slots and cell.fingerprint not in blocked:
            taken.append(cell)
            blocked.add(cell.fingerprint)
        else:
            keep.append(cell)
    pending[:] = keep
    return taken


def _failure_detail(err_path):
    """The last line of a failed cell's traceback, or an empty string."""
    if not os.path.exists(err_path):
        return ""
    try:
        with open(err_path) as stream:
            return json.load(stream)["traceback"].strip().splitlines()[-1]
    except (ValueError, OSError, KeyError, IndexError):
        return ""


def run_cells(cells, out_dir, overrides, workers, resume=True, poll=2.0):
    """Run `cells` (a list of cell_cache.Cell) at most `workers` at a time.

    Returns (results, failures) with results in the ORDER GIVEN, never in
    completion order. A failed cell is reported and skipped, not fatal.
    Every cell that does not have to run is served from a fingerprint-
    verified result; `resume=False` forces recomputation but still
    collapses cells that duplicate each other inside this sweep.
    """
    _validate_cells(cells, overrides)

    pending, running = [], []
    done, failed, completed = {}, {}, {}
    total = len(cells)
    started, reused = 0, 0
    t0 = time.time()

    for cell in cells:
        result, note = _reuse(cell, out_dir, resume)
        if result is None:
            if note:
                print(f"[sweep] rerun {cell.name} seed={cell.seed} ({note})")
            pending.append(cell)
            continue
        done[cell.key] = _materialize(result, cell, out_dir)
        completed[cell.fingerprint] = result
        reused += 1
        print(f"[sweep] skip  {cell.name} seed={cell.seed} ({note})")

    if pending:
        print(f"[sweep] {len(pending)} cells to run, {workers} at a time "
              f"({len(done)} already complete)")

    while pending or running:
        for cell in _take_reusable(pending, completed):
            source = completed[cell.fingerprint]
            done[cell.key] = _materialize(source, cell, out_dir)
            reused += 1
            print(f"[sweep] dedup {cell.name} seed={cell.seed} (identical to "
                  f"{source['config']} seed={cell.seed} in this sweep)")

        in_flight = {entry.cell.fingerprint for entry in running}
        for cell in _take_launchable(pending, in_flight, workers - len(running)):
            cfg_dir, out_path, err_path, log_path = cell_paths(
                out_dir, cell.name, cell.seed)
            os.makedirs(cfg_dir, exist_ok=True)
            spec = {"config": cell.config, "seed": cell.seed,
                    "overrides": overrides, "out_dir": out_dir,
                    "fingerprint": cell.fingerprint}
            proc, log = _launch(spec, log_path)
            started += 1
            running.append(_Running(cell, proc, log, out_path, err_path,
                                    log_path, time.time()))
            print(f"[sweep] start {cell.name} seed={cell.seed} "
                  f"({started}/{total}, pid {proc.pid})")

        if running:
            time.sleep(poll)

        for entry in list(running):
            if entry.proc.poll() is None:
                continue
            running.remove(entry)
            entry.log.close()
            mins = (time.time() - entry.started) / 60.0
            result = cc.read_verified(entry.out_path, entry.cell.fingerprint)
            if entry.proc.returncode == 0 and result is not None:
                done[entry.cell.key] = result
                completed[entry.cell.fingerprint] = result
                evals = "  ".join(
                    f"vs {label} "
                    + ("n/a" if e.get("avg_reward") is None
                       else f"{e['avg_reward']:.2f}")
                    for label, e in (result.get("eval") or {}).items())
                print(f"[sweep] done  {entry.cell.name} seed={entry.cell.seed} "
                      f"({mins:.1f} min)  {evals}")
            else:
                detail = _failure_detail(entry.err_path)
                if entry.proc.returncode == 0 and not detail:
                    detail = ("exited cleanly but wrote no result matching its "
                              "fingerprint")
                failed[entry.cell.key] = {
                    "returncode": entry.proc.returncode,
                    "log": entry.log_path,
                    "error": (entry.err_path if os.path.exists(entry.err_path)
                              else None)}
                print(f"[sweep] FAIL  {entry.cell.name} seed={entry.cell.seed} "
                      f"({mins:.1f} min, rc={entry.proc.returncode}) {detail}\n"
                      f"              log: {entry.log_path}")

    results = [done[cell.key] for cell in cells if cell.key in done]
    print(f"[sweep] {len(results)}/{total} cells complete in "
          f"{(time.time() - t0) / 60.0:.1f} min wall "
          f"({started} run, {reused} reused, {len(failed)} failed)")
    return results, failed


def resolve_workers(workers):
    """workers='auto' -> one per core; otherwise validate an integer >= 1.
    Worker count is bounded by memory as often as by cores (~0.5 GB each).
    """
    if workers == "auto":
        return max(1, os.cpu_count() or 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be an integer >= 1 or 'auto', got {workers!r}")
    return workers