# config_loader.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import yaml
from pydantic import ValidationError
from sim_config import SimConfig

# Fields a rerun may change without changing what a replicate simulates.
_BOOKKEEPING_FIELDS = {
    "num_replicates",
    "start_idx_replicate",
    "out_dir",
    "condition_name",
    "gpu_device",
    "backend",
    "slurm",
    "plot_LE",
}


def _apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Overrides are k=v strings. Values parsed with YAML, so lists/tuples/numbers work:
      --overrides density=0.3 speed_list=[200,400] condition_name='"test run"'
    """
    for kv in overrides or []:
        if "=" not in kv:
            raise ValueError(f"Invalid override (missing '='): {kv}")
        k, v = kv.split("=", 1)
        # nested keys like "scheduling.max_jobs=8"
        path = k.split(".")
        cur = data
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        cur[path[-1]] = yaml.safe_load(v)
    return data


def load_config(config_path: str, overrides: List[str] | None = None) -> SimConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    merged = _apply_overrides(raw, overrides or [])

    try:
        return SimConfig.model_validate(merged)
    except ValidationError as e:
        # pretty error; fail early with specifics
        raise SystemExit(f"\nConfig validation failed:\n{e}\n")


def check_run_dir(cfg: SimConfig, run_dir: Path) -> None:
    """Refuse to reuse a run folder whose checkpoints come from a different config.

    Replicate folders are checkpoints: an existing LEFPositions_0.h5 is reused
    instead of rerunning loop extrusion, and existing blocks_*.h5 make the MD
    resume, skipping minimization and equilibration. That is right when the same
    config is rerun after an interruption, and silently wrong after a config
    change: old and new settings would be mixed in one dataset with no error.
    A missing field counts as changed, because an older resolved config may have
    run under a different implicit value (attraction_energy did, 2026-10-08).
    """
    old_path = Path(run_dir) / "config_resolved.json"
    if not old_path.exists():
        return
    has_checkpoints = any(Path(run_dir).glob("*/LEFPositions_*.h5")) or any(
        Path(run_dir).glob("*/blocks_*.h5")
    )
    if not has_checkpoints:
        return
    old = json.loads(old_path.read_text())
    new = json.loads(cfg.model_dump_json())
    changed = sorted(
        k
        for k in set(old) | set(new)
        if k not in _BOOKKEEPING_FIELDS and old.get(k, "<missing>") != new.get(k, "<missing>")
    )
    if changed:
        raise SystemExit(
            f"\n{run_dir} already holds simulations made with a different config "
            f"(changed: {', '.join(changed)}). Its replicate folders are checkpoints: "
            "LEFPositions_0.h5 would be reused instead of rerunning loop extrusion, and "
            "blocks_*.h5 would make the MD resume without setup, mixing old and new "
            "settings. Use a new condition_name or out_dir, or move the old folder "
            "away. (Rerunning the same config to resume an interrupted run is fine.)\n"
        )
