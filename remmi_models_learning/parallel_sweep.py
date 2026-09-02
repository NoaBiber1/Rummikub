"""Parallel cell execution for seed_sweep.

WHY SUBPROCESSES AND NOT multiprocessing. Three constraints rule out the
obvious answer, and they were each verified rather than assumed:

  1. The project's entry pattern is `python - <<'PY' ... PY`, i.e. __main__ is
     STDIN. multiprocessing's "spawn" and "forkserver" start methods both
     re-import __main__ in the child; with no __file__ to import, both die with
     BrokenProcessPool. Measured: spawn FAILS, forkserver FAILS, fork works.
  2. That leaves "fork", which is the one start method the PyTorch docs warn
     against: forking an interpreter that has already initialised an OpenMP
     thread pool can deadlock in the child, intermittently and unreproducibly -
     the worst possible failure mode for a job that runs for days.
  3. A cell that dies (OOM, a NaN fire alarm, a CBC crash) must not take the
     other cells with it, and a sweep killed at hour six must not lose the
     cells that had already finished.

An independent `python -c "import seed_sweep; seed_sweep._cell_main()"` per
cell satisfies all three. Each cell gets a clean interpreter, a clean torch, a
clean ILP solver and its own address space; the parent never imports anything
into a child; and the result arrives through the JSON file the cell already
wrote, so no IPC and nothing to pickle. Startup costs a few seconds of torch
import against a cell that runs for hours.

WHAT IT DOES NOT CHANGE. Every cell seeds itself from its own config
(simulation._seed_all / _seed_episode), and nothing crosses process
boundaries, so a parallel sweep produces bit-identical results to a serial one.
Cell-level parallel safety of the ILP hot path - the one shared resource, via
CBC's temp files - is verified separately in test_parallel_ilp_safety.py
(pulp prefixes with uuid4().hex; 0 mismatches serial vs 4-way parallel).
"""
import json
import os
import subprocess
import sys
import time
import traceback

# One thread per worker. Without this each of N cells asks BLAS/OpenMP for as
# many threads as there are cores, so N x C threads fight over C cores and the
# sweep gets SLOWER as workers go up. The ILP is ~95% of the runtime and is a
# single-threaded subprocess anyway, so nothing here wants more than one.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def cell_paths(out_dir, name, seed):
    """Where a cell's result, error and log land. One directory per config, so
    a config's seeds sit together and a whole config can be deleted to re-run
    it."""
    cfg_dir = os.path.join(out_dir, name)
    stem = os.path.join(cfg_dir, f"seed{seed}")
    return cfg_dir, f"{stem}.json", f"{stem}.error.json", f"{stem}.log"


# --------------------------------------------------------------- the child

def _cell_main():
    """Entry point for one cell's subprocess. Reads a JSON spec on stdin,
    runs the cell, writes the result. Never returns a value to the parent -
    the file IS the channel."""
    import seed_sweep

    spec = json.load(sys.stdin)
    _, out_path, err_path, _ = cell_paths(
        spec["out_dir"], spec["config"]["name"], spec["seed"])
    try:
        result = seed_sweep._run_one(spec["config"], spec["seed"],
                                     spec["overrides"])
    except BaseException:
        # Written as a file rather than raised into the void: the parent is
        # not attached to this stderr in any useful way, and a sweep that
        # loses the traceback of the one cell that failed is a sweep you have
        # to run twice.
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, "w") as f:
            json.dump({"config": spec["config"]["name"], "seed": spec["seed"],
                       "traceback": traceback.format_exc()}, f, indent=2)
        raise

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    # Written LAST and only on success, so a half-written result from a cell
    # killed mid-run cannot be mistaken for a finished one by --resume.
    if os.path.exists(err_path):
        os.remove(err_path)


# -------------------------------------------------------------- the parent

def _launch(spec, log_path):
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
    """A cell spec crosses a process boundary as JSON, so a config carrying a
    live object (an nn.Module test opponent, a callable) cannot be sent.
    Caught here with a usable message rather than as a TypeError from inside
    json.dumps three frames down."""
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
    completion order - a summary table whose row order depended on which cell
    happened to finish first would not be diffable between runs.
    """
    pending = list(cells)
    done, failed, running = {}, {}, []
    total = len(pending)
    started = 0
    t0 = time.time()

    # Resume first, so the "N cells" the progress line counts is the number
    # actually about to run.
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
                    f"vs {label} {e['avg_reward']:.2f}"
                    for label, e in result["eval"].items())
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
                # A failed cell does NOT stop the sweep. One diverging lr is an
                # expected outcome of a sweep, not a reason to lose the other
                # 26 cells - the test plan's own rule is "discard that cell".
                print(f"[sweep] FAIL  {config['name']} seed={seed} "
                      f"({mins:.1f} min, rc={proc.returncode}) {detail}\n"
                      f"              log: {log_path}")

    results = [done[(c["name"], s)] for c, s in cells if (c["name"], s) in done]
    print(f"[sweep] {len(results)}/{total} cells complete in "
          f"{(time.time() - t0) / 60.0:.1f} min wall "
          f"({len(failed)} failed)")
    return results, failed


def resolve_workers(workers):
    """`workers='auto'` -> one per core. Deliberately not the default.

    Each cell is a full interpreter with its own torch and its own replay
    buffer, so worker count is bounded by MEMORY as often as by cores: at
    buffer_size=20000 a cell holds roughly 100-150 MB of transitions on top of
    ~300 MB of torch. Budget ~0.5 GB per worker and check before raising this.
    """
    if workers == "auto":
        return max(1, os.cpu_count() or 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be an integer >= 1 or 'auto', got {workers!r}")
    return workers
