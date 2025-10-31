# config_loader.py
from __future__ import annotations
from typing import Dict, Any, List
import yaml
from pydantic import ValidationError
from opt_config import OptConfig


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


def load_config(config_path: str, overrides: List[str] | None = None) -> OptConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    merged = _apply_overrides(raw, overrides or [])

    try:
        return OptConfig.model_validate(merged)
    except ValidationError as e:
        # pretty error; fail early with specifics
        raise SystemExit(f"\nConfig validation failed:\n{e}\n")
