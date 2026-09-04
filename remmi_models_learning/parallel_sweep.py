"""Parallel cell execution for seed_sweep: one subprocess per cell.

Subprocesses, not multiprocessing: spawn and forkserver both re-import
__main__, which does not exist when the entry point is stdin; fork can
deadlock a forked OpenMP pool; and one dead cell must not take the sweep
with it. Each cell gets a clean interpreter and returns its result
through the JSON file it writes, so there is no IPC and nothing to
pickle. Cells seed themselves, so a parallel sweep is bit-identical to a
serial one.
"""
import json
import os
import subprocess
import sys
import time
import traceback

_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def cell_paths(out_dir, name, seed):
    """(config dir, result path, error path, log path) for one cell."""
    cfg_dir = os.path.join(out_dir, name)
    stem = os.path.join(cfg_dir, f"seed{seed}")
    return cfg_dir, f"{stem}.json", f"{stem}.error.json", f"{stem}.log"


def _cell_main():
    """Child entry point: read a cell spec on stdin, run it, write the result.
    The file IS the channel back to the parent.
    """
    import seed_sweep

    spec = json.load(sys.stdin)
    _, out_path, err_path, _ = cell_paths(
        spec["out_dir"], spec["config"]["name"], spec["seed"])
    try:
        result = seed_sweep._run_one(spec["config"], spec["seed"],
                                     spec["overrides"])
    except BaseException:
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, "w") as f:
            json.dump({"config": spec["config"]["name"], "seed": spec["seed"],
                       "traceback": traceback.format_exc()}, f, indent=2)
        raise

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
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
    """Raise a usable error if a cell spec cannot be sent as JSON - a config
    carrying a live net or callable has to run with workers=1.
    """
    try:
        json.dumps({"config": config, "overrides": overrides})
    except TypeError as exc:
        raise TypeError(
            f"config {config.get('name', '?')!r} is not JSON-serialisable and "
            f"so cannot be sent to a worker process ({exc}). Model-backed "
            f"opponents and callables have to run with workers=1, or be named "
            f"by checkpoint path instead of passed as objects.") from exc


def run_cells(cells, out_dir, overrides, workers, resume=True, poll=2.0):
    """Run `cells` (a list of (config, seed)) at most `workers` at a time.

    Returns (results, failures) with results in the ORDER GIVEN, never in
    completion order. A failed cell is reported and skipped, not fatal.
    """
    pending = list(cells)
    done, failed, running = {}, {}, []
    total = len(pending)
    started = 0
    t0 = time.time()

    if resume:
        still = []
        for config, seed in pending:
            _, out_path, _, _ = cell_paths(out_dir, config["name"], seed)
            if os.path.exists(out_path):
                try:
                    done[(config["name"], seed)] = json.load(open(out_path))
                    print(f"[sweep] skip  {config['name']} seed={seed} "
                          f"(already done: {out_path})")
                    continue
                except (ValueError, OSError):
                    print(f"[sweep] rerun {config['name']} seed={seed} "
                          f"(unreadable result, treating as unfinished)")
            still.append((config, seed))
        pending = still

    if pending:
        print(f"[sweep] {len(pending)} cells to run, {workers} at a time "
              f"({len(done)} already complete)")

    while pending or running:
        while pending and len(running) < workers:
            config, seed = pending.pop(0)
            _check_serializable(config, overrides)
            cfg_dir, out_path, err_path, log_path = cell_paths(
                out_dir, config["name"], seed)
            os.makedirs(cfg_dir, exist_ok=True)
            spec = {"config": config, "seed": seed, "overrides": overrides,
                    "out_dir": out_dir}
            proc, log = _launch(spec, log_path)
            started += 1
            running.append((proc, log, config, seed, out_path, err_path,
                            log_path, time.time()))
            print(f"[sweep] start {config['name']} seed={seed} "
                  f"({started}/{total}, pid {proc.pid})")

        time.sleep(poll)

        for entry in list(running):
            proc, log, config, seed, out_path, err_path, log_path, t_start = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            log.close()
            mins = (time.time() - t_start) / 60.0
            key = (config["name"], seed)
            if proc.returncode == 0 and os.path.exists(out_path):
                result = json.load(open(out_path))
                done[key] = result
                evals = "  ".join(
                    f"vs {label} "
                    + ("n/a" if e.get("avg_reward") is None
                       else f"{e['avg_reward']:.2f}")
                    for label, e in (result.get("eval") or {}).items())
                print(f"[sweep] done  {config['name']} seed={seed} "
                      f"({mins:.1f} min)  {evals}")
            else:
                detail = ""
                if os.path.exists(err_path):
                    try:
                        detail = json.load(open(err_path))["traceback"].strip()
                        detail = detail.splitlines()[-1]
                    except (ValueError, OSError, KeyError, IndexError):
                        pass
                failed[key] = {"returncode": proc.returncode, "log": log_path,
                               "error": err_path if os.path.exists(err_path) else None}
                print(f"[sweep] FAIL  {config['name']} seed={seed} "
                      f"({mins:.1f} min, rc={proc.returncode}) {detail}\n"
                      f"              log: {log_path}")

    results = [done[(c["name"], s)] for c, s in cells if (c["name"], s) in done]
    print(f"[sweep] {len(results)}/{total} cells complete in "
          f"{(time.time() - t0) / 60.0:.1f} min wall "
          f"({len(failed)} failed)")
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