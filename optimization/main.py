# main.py
import argparse
import logging
import os
from pathlib import Path
from config_loader import load_config
from opt_config import OptConfig
from run_opt import run
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
log = logging.getLogger(__name__)


def save_resolved_config(cfg: OptConfig, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.json").write_text(cfg.model_dump_json(indent=2))


def parse_args():
    p = argparse.ArgumentParser(
        description="Run heteropolymer simulation with Pydantic config"
    )
    p.add_argument("--config", help="Path to YAML/JSON config (optional)")
    p.add_argument("--overrides", nargs="*", help="k=v overrides parsed as YAML")
    p.add_argument(
        "--out-subdir", default=None, help="Optional subdir name under cfg.out_dir"
    )
    return p.parse_args()


# ---------- worker & init ----------
def _init_worker(numba_threads: int | None = None):
    """
    Runs once per worker process.
    Use it to cap low-level threads so multiple workers don't oversubscribe the CPU.
    """
    # If you don't use NumPy/BLAS, you can drop these.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if numba_threads is not None:
        try:
            from numba import set_num_threads

            set_num_threads(numba_threads)
        except Exception:
            pass  # numba not installed or not needed


def _worker_run(cfg_payload: dict, base_run_dir: str, idx_replicate: int) -> str:
    """
    Executed inside a worker process. Rebuild cfg from a pickle-friendly payload,
    make the per-replicate folder, and call your run().
    Returns a small string (e.g., path) so the parent can log progress.
    """
    # Rebuild cfg (if you're using pydantic):
    #   cfg = OptConfig.model_validate(cfg_payload)
    # If cfg_payload is already a dict your run() accepts, keep as-is:
    cfg = OptConfig.model_validate(cfg_payload)

    curr_run_dir = Path(base_run_dir) / str(idx_replicate)
    curr_run_dir.mkdir(parents=True, exist_ok=True)

    # Optional: per-process/per-run logfile (keeps outputs separate)
    logfile = curr_run_dir / "worker.log"
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)
    (f"Starting run: {curr_run_dir}")

    # Do the work
    run(cfg, curr_run_dir)

    log.info("Finished.")
    return str(curr_run_dir)


def main():
    args = parse_args()

    if args.config:
        cfg = load_config(args.config, args.overrides)  # strict path
        mode = "config"
    else:
        cfg = OptConfig()  # pure defaults
        # apply optional overrides even in dev
        if args.overrides:
            import yaml

            updates = {
                k: yaml.safe_load(v)
                for k, v in (s.split("=", 1) for s in args.overrides)
            }
            cfg = cfg.model_copy(update=updates)
        mode = "defaults"

    # Optional: nudge users
    if mode == "defaults":
        print(
            "[dev] Running with in-code defaults; for reproducible runs pass --config <file>."
        )

    # Compute a run directory (timestamp/condition/seed etc. if you like)
    base = Path(cfg.out_dir)
    sub = args.out_subdir or cfg.condition_name or "default"
    run_dir = base / sub
    run_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, run_dir)

    num_replicates = cfg.num_replicates
    start_idx_replicate = cfg.start_idx_replicate
    final_idx_replicate = start_idx_replicate + num_replicates

    # Picklable payload for child processes (don’t send complex objects if they aren’t picklable)
    # If OptConfig is pydantic, .model_dump() is safe; adjust if your type differs.
    cfg_payload = cfg.model_dump()

    # Tuning knobs
    max_workers = getattr(cfg, "num_workers", None) or os.cpu_count() or 4
    # If your numba kernels are parallel, give each worker few threads to avoid oversubscription.
    # For example, share cores roughly evenly:
    numba_threads_per_proc = 1

    # Submit all replicates to the pool
    futures = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(numba_threads_per_proc,),
    ) as ex:
        for idx in range(start_idx_replicate, final_idx_replicate):
            # Create the directory here in the parent, too (nice for early visibility)
            (run_dir / str(idx)).mkdir(parents=True, exist_ok=True)
            futures.append(ex.submit(_worker_run, cfg_payload, str(run_dir), idx))

        # wait for all (unordered), logging errors as they come
        for fut in as_completed(futures):
            try:
                done_path = fut.result()
                log.info(f"[OK] Completed: {done_path}")
            except Exception as e:
                log.info(f"[ERR] A worker failed: {e!r}")

    log.info("All submitted runs are done.")


if __name__ == "__main__":
    main()
