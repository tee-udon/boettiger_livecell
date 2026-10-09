"""MD config mistakes must fail when the config is loaded, not mid-run.

Each case is a real way to get an MD config wrong. Before these checks, each one
either crashed late (after the loop extrusion had already run) or silently
simulated something else. Two are the 2026-10-08 failures: EKExceedsError from
the implicit 3 kT attraction_energy, and old checkpoints reused after a config
change.

Needs only pydantic and PyYAML, so it runs in both the tee and py-hoomd2 envs.
Runs under pytest, or standalone with `python tests/test_md_config_guards.py`.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from pydantic import ValidationError  # noqa: E402

from config_loader import check_run_dir, load_config  # noqa: E402
from sim_config import SimConfig  # noqa: E402

# A small sticky polymer shaped like Alistair's Levine configs.
STICKY = dict(
    num_monomers=300,
    monomer_type_list=[[0, 95], [1, 5], [0, 95], [1, 5], [0, 100]],
    attraction_coefficient_matrix=[[0, 0], [0, 1.0]],
    attraction_radius=1.5,
    attraction_energy=0,
    repulsion=4,
    density=0.1,
    PBC_box=True,
)

# Alistair's YAML as sent on 2026-10-08. It never sets attraction_energy, so it
# ran with the 3 kT default and stopped with "EKExceedsError: Ek=68.8 exceeds 20".
CRASHED_ON_2026_10_08 = r"""
out_dir: F:\Alistair\2025-12_LevinePolySims\2026-10_LevinePolySims
condition_name: WT
num_replicates: 100
num_monomers: 1500
num_cohesin: 10
cohesin_speed: 1
cohesin_stall_time: 1000
cohesin_bound_lifetime: 200
cohesin_unbound_lifetime: 1
ctcf_site_location_list: [500]
ctcf_site_direction_list: ['both']
ctcf_site_stall_probability_list: [1]
ctcf_site_stall_time_list: [100000]
num_LE_steps: 1000
monomer_type_list: [[0,490],[1,10],[0,490],[1,10],[0,500]]
attraction_coefficient_matrix: [[0,0],[0,1.0]]
repulsion: 4
equilibration_timestep: 100000
num_MD_steps_per_LE: 500
attraction_radius: 1.5
density: 0.1
initial_conformation: random_walk_z
PBC_box: true
equilibrate_without_stickiness: true
backend: local
gpu_device: "0"
"""


def _rejects(fragment, **kwargs):
    """SimConfig(**kwargs) must fail with `fragment` in the message."""
    try:
        SimConfig(**kwargs)
    except ValidationError as e:
        assert fragment in str(e), str(e)
        return
    raise AssertionError(f"accepted a config that should fail: {kwargs}")


def _expect_exit(fragments, fn, *args):
    """fn(*args) must stop with SystemExit naming every fragment."""
    try:
        fn(*args)
    except SystemExit as e:
        for fragment in fragments:
            assert fragment in str(e), str(e)
        return
    raise AssertionError(f"{fn.__name__} did not stop")


# ---- attraction_energy: the 2026-10-08 crash --------------------------------


def test_sticky_config_with_explicit_attraction_energy_loads():
    assert SimConfig(**STICKY).attraction_energy == 0


def test_attraction_energy_must_be_explicit_when_attraction_is_on():
    without = {k: v for k, v in STICKY.items() if k != "attraction_energy"}
    _rejects("set attraction_energy explicitly", **without)


def test_any_explicit_attraction_energy_is_accepted():
    for value in (0, 0.3, 3):
        assert SimConfig(**{**STICKY, "attraction_energy": value}).attraction_energy == value


def test_the_config_that_crashed_now_fails_at_load():
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "as_sent.yaml"
        path.write_text(CRASHED_ON_2026_10_08)
        _expect_exit(["attraction_energy"], load_config, str(path))
        path.write_text(CRASHED_ON_2026_10_08 + "attraction_energy: 0\n")
        assert load_config(str(path)).attraction_energy == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_attraction_needs_no_attraction_energy():
    # A phantom chain, like every archived config: nothing to collapse.
    assert SimConfig(num_monomers=300, attraction_radius=0).attraction_radius == 0


def test_resolved_config_reloads():
    """config_resolved.json lists every field, so it must load back unchanged."""
    cfg = SimConfig(**STICKY)
    again = SimConfig.model_validate(json.loads(cfg.model_dump_json()))
    assert again.attraction_energy == cfg.attraction_energy


# ---- settings that are valid alone but wrong together -----------------------


def test_sticky_regions_need_attraction_radius():
    _rejects("nothing would stick", **{**STICKY, "attraction_radius": 0})


def test_attraction_radius_between_0_and_1_is_rejected():
    for r in (0.5, 1.0):
        _rejects("must be 0", **{**STICKY, "attraction_radius": r})


def test_attraction_matrix_must_be_symmetric():
    # debug/config_debug.yaml has exactly this matrix
    _rejects("symmetric", **{**STICKY, "attraction_coefficient_matrix": [[0, 1], [0, 1]]})


def test_monomer_types_must_have_no_gaps():
    gappy = [[0, 95], [2, 5], [0, 95], [2, 5], [0, 100]]
    _rejects("without gaps", **{**STICKY, "monomer_type_list": gappy})


def test_save_every_blocks_must_divide_num_LE_steps():
    _rejects("multiple of save_every_blocks", **{**STICKY, "num_LE_steps": 1000, "save_every_blocks": 3})


def test_periodic_box_must_fit_the_attraction_range():
    tiny = dict(num_monomers=20, monomer_type_list=[[0, 5], [1, 5], [0, 5], [1, 5]], density=0.9)
    _rejects("periodic box", **{**STICKY, **tiny})


def test_example_configs_load():
    paths = sorted((ROOT / "examples" / "simulation").glob("*.yaml"))
    assert paths, "no example configs found"
    for path in paths:
        try:
            load_config(str(path))
        except SystemExit as e:
            raise AssertionError(f"{path.name} does not load: {e}")


# ---- reusing an output folder ----------------------------------------------


def _folder_with_checkpoint(tmp, cfg, checkpoint="LEFPositions_0.h5"):
    run_dir = tmp / "WT"
    (run_dir / "0").mkdir(parents=True)
    (run_dir / "config_resolved.json").write_text(cfg.model_dump_json())
    (run_dir / "0" / checkpoint).write_bytes(b"")
    return run_dir


def test_rerun_of_the_same_config_may_resume():
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir = _folder_with_checkpoint(tmp, SimConfig(**STICKY), "blocks_0-99.h5")
        # bookkeeping may change: more replicates, another GPU
        check_run_dir(SimConfig(**{**STICKY, "num_replicates": 100, "gpu_device": "1"}), run_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_changed_config_cannot_reuse_old_checkpoints():
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir = _folder_with_checkpoint(tmp, SimConfig(**{**STICKY, "attraction_energy": 3}))
        _expect_exit(["attraction_energy", "new condition_name"], check_run_dir, SimConfig(**STICKY), run_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_folder_from_older_code_counts_as_changed():
    """An old resolved config lacks newer fields. Those runs used whatever the
    code then did, which for attraction_energy was not today's default."""
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = SimConfig(**STICKY)
        run_dir = _folder_with_checkpoint(tmp, cfg)
        old = json.loads((run_dir / "config_resolved.json").read_text())
        del old["attraction_energy"]
        (run_dir / "config_resolved.json").write_text(json.dumps(old))
        _expect_exit(["attraction_energy"], check_run_dir, cfg, run_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_folder_without_checkpoints_is_reusable():
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir = tmp / "WT"
        run_dir.mkdir()
        (run_dir / "config_resolved.json").write_text(
            SimConfig(**{**STICKY, "attraction_energy": 3}).model_dump_json()
        )
        check_run_dir(SimConfig(**STICKY), run_dir)
        check_run_dir(SimConfig(**STICKY), tmp / "does_not_exist")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAIL {name}: {exc}")
    print(f"\n{'FAILED' if failed else 'ALL PASSED'}")
    sys.exit(1 if failed else 0)
