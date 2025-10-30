# main.py
import argparse
import logging
from pathlib import Path
from config_loader import load_config
from sim_config import SimConfig
from run_sim import run

logging.basicConfig(format='%(asctime)s [%(name)s] %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
log = logging.getLogger(__name__)

def save_resolved_config(cfg: SimConfig, run_dir: Path):
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


def main():
    args = parse_args()

    if args.config:
        cfg = load_config(args.config, args.overrides)  # strict path
        mode = "config"
    else:
        cfg = SimConfig()  # pure defaults
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

    for idx_replicate in range(start_idx_replicate, final_idx_replicate):
        curr_run_dir = run_dir / str(idx_replicate)
        curr_run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Starting run: {str(curr_run_dir)}")
        run(cfg, curr_run_dir)


if __name__ == "__main__":
    main()
