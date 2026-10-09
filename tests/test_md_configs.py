"""Short MD runs of representative configs, checked for a healthy simulation.

Each config runs the real simulate_MD, and the real loop extrusion where it has
cohesin. It must finish with kinetic energy per monomer near 3/2 kT at every
saved block, never near polychrom's 20 kT crash limit. The sticky-region switch
must fire exactly when equilibrate_without_stickiness is on, and unbound
extruders must not be reported as corrupt.

It also reproduces the 2026-10-08 crash: 3 kT of uniform attraction overheats a
200-monomer chain during equilibration on CPU and GPU alike, and the error must
now say why. And a block too fast for the Windows clock must not crash.

Needs polychrom, OpenMM and numba: the tee or py-hoomd2 env. Runs on the CPU by
default; set MD_TEST_PLATFORM=CUDA to run the same cases on the GPU (tee env,
gpu_device "0" = the V100). About a minute on the CPU.
Runs under pytest, or standalone with `python tests/test_md_configs.py`.
"""

import contextlib
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

SIM_DIR = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(SIM_DIR))

import run_sim_MD  # noqa: E402  (installs the precise clock for polychrom)
from polychrom import forcekits  # noqa: E402
from polychrom.simulation import EKExceedsError, Simulation  # noqa: E402
from polychrom.starting_conformations import create_random_walk  # noqa: E402
from run_sim import downsampling_LE  # noqa: E402
from run_sim_LE import simulate_LE  # noqa: E402
from sim_config import SimConfig  # noqa: E402

PLATFORM = os.environ.get("MD_TEST_PLATFORM", "CPU")
SEED = 0

# run_sim sets the root logger to DEBUG on import; keep the console to warnings.
for _handler in logging.getLogger().handlers:
    _handler.setLevel(logging.WARNING)
for _noisy in ("numba", "matplotlib", "h5py"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# 200 monomers shaped like Alistair's Levine runs: dilute periodic box, a spread
# out random_walk_z start, firm repulsion, excluded volume on.
BASE = dict(
    num_monomers=200,
    num_cohesin=0,
    num_LE_steps=20,
    num_MD_steps_per_LE=200,
    equilibration_timestep=5000,
    save_every_blocks=1,
    density=0.1,
    collision_rate=0.03,
    initial_conformation="random_walk_z",
    PBC_box=True,
    repulsion=4,
    attraction_radius=1.5,
    attraction_energy=0,
    gpu_device="0",
)
STICKY = dict(
    monomer_type_list=[[0, 60], [1, 5], [0, 60], [1, 5], [0, 70]],
    attraction_coefficient_matrix=[[0, 0], [0, 1.0]],
)
# Short bound lifetime, so extruders unbind within the run and the unbound
# marker (-1) reaches bondUpdater.
COHESIN = dict(
    num_cohesin=3,
    cohesin_speed=1,
    cohesin_bound_lifetime=10,
    cohesin_unbound_lifetime=1,
    cohesin_stall_time=1000,
    ctcf_site_location_list=[100],
    ctcf_site_direction_list=["both"],
    ctcf_site_stall_probability_list=[1],
    ctcf_site_stall_time_list=[100000],
)


class _Probe(Simulation):
    """Records, at every saved block, the kinetic energy per monomer (kT) and
    the sticky-region parameter."""

    def __init__(self, records, **kwargs):
        super().__init__(**kwargs)
        self._records = records

    def do_block(self, *args, **kwargs):
        result = super().do_block(*args, **kwargs)
        state = self.context.getState(getEnergy=True)
        self._records.append(
            {
                "kin": state.getKineticEnergy() / self.N / self.kT,
                "sticky": self.context.getParameter(run_sim_MD._selective_attraction_param(self)),
            }
        )
        return result


@contextlib.contextmanager
def _probed_simulation(records):
    original = run_sim_MD.Simulation

    def make(**kwargs):
        kwargs["platform"] = PLATFORM
        return _Probe(records, **kwargs)

    run_sim_MD.Simulation = make
    try:
        yield
    finally:
        run_sim_MD.Simulation = original


@contextlib.contextmanager
def _captured_logs():
    lines = []

    class _Handler(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    root = logging.getLogger()
    handler, level = _Handler(logging.INFO), root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield lines
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def _simulate(**overrides):
    """Run loop extrusion (if any cohesin) and MD; return what the test needs."""
    cfg = SimConfig(**{**BASE, **overrides})
    run_dir = Path(tempfile.mkdtemp())
    try:
        if cfg.num_cohesin > 0:
            simulate_LE(cfg, run_dir, 0)
            downsampling_LE(cfg, run_dir, 0)
        records = []
        with _captured_logs() as lines, _probed_simulation(records):
            np.random.seed(SEED)
            run_sim_MD.simulate_MD(cfg, run_dir)
        frames = np.load(run_dir / "all_conformations.npy")
        lef = None
        if (run_dir / "LEFPositions_0.h5").exists():
            with h5py.File(run_dir / "LEFPositions_0.h5", "r") as f:
                lef = f["positions"][()]
        return cfg, frames, records, lines, lef
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _assert_healthy(cfg, frames, records, what):
    expected = 1 + cfg.num_LE_steps // cfg.save_every_blocks  # equilibration + LE blocks
    assert frames.shape == (expected, cfg.num_monomers, 3), (what, frames.shape)
    assert not np.isnan(frames).any(), f"{what}: NaN positions"
    kin = np.array([r["kin"] for r in records])
    assert len(kin) == expected, (what, len(kin))
    assert kin.max() < 4.0, f"{what}: kinetic energy reached {kin.max():.2f} kT per monomer"
    assert 1.2 < np.median(kin) < 1.8, f"{what}: median kinetic energy {np.median(kin):.2f} kT"


def _assert_extruders_reported_quietly(lines, lef):
    assert not any("corrupt" in m for m in lines), [m for m in lines if "corrupt" in m]
    if lef is not None and (lef == -1).any():
        assert any("hold no bond" in m for m in lines), "unbound extruders were not reported"


# ---- the 2026-10-08 crash -----------------------------------------------------


def test_uniform_attraction_overheats_and_the_error_says_why():
    """3 kT between every pair of monomers collapses the chain. polychrom stops it
    at Ek > 20; the message must now point at attraction_energy."""
    try:
        _simulate(attraction_energy=3)
    except EKExceedsError as e:
        message = str(e)
        assert "during equilibration" in message, message
        assert "attraction_energy is 3 kT" in message, message
        return
    raise AssertionError("3 kT of uniform attraction did not overheat the chain")


def test_same_chain_without_uniform_attraction_is_healthy():
    cfg, frames, records, _, _ = _simulate(attraction_energy=0)
    _assert_healthy(cfg, frames, records, "attraction_energy 0")


# ---- Levine-like sticky runs ------------------------------------------------


def test_sticky_run_with_stickiness_off_during_setup():
    cfg, frames, records, lines, lef = _simulate(**STICKY, **COHESIN, equilibrate_without_stickiness=True)
    _assert_healthy(cfg, frames, records, "sticky, setting on")
    sticky = [r["sticky"] for r in records]
    assert sticky[0] == 0, "equilibration ran with the sticky attraction on"
    assert all(s > 0 for s in sticky[1:]), "sticky attraction not back on for loop extrusion"
    assert any("attraction OFF" in m for m in lines) and any("back ON" in m for m in lines)
    _assert_extruders_reported_quietly(lines, lef)


def test_sticky_run_with_stickiness_on_throughout():
    cfg, frames, records, lines, lef = _simulate(**STICKY, **COHESIN, equilibrate_without_stickiness=False)
    _assert_healthy(cfg, frames, records, "sticky, setting off")
    assert all(r["sticky"] > 0 for r in records), "sticky attraction was switched off"
    assert not any("attraction OFF" in m for m in lines)
    _assert_extruders_reported_quietly(lines, lef)


# ---- other common setups --------------------------------------------------------


def test_excluded_volume_crumpled_without_box():
    cfg, frames, records, _, _ = _simulate(initial_conformation="crumpled", PBC_box=False, density=0.24)
    _assert_healthy(cfg, frames, records, "excluded volume, crumpled")


def test_phantom_chain_like_the_archived_configs():
    # attraction_radius 0 switches the nonbonded force off entirely
    cfg, frames, records, lines, lef = _simulate(
        attraction_radius=0, initial_conformation="crumpled", PBC_box=False, density=0.24, **COHESIN
    )
    _assert_healthy(cfg, frames, records, "phantom chain")
    _assert_extruders_reported_quietly(lines, lef)


# ---- the Windows clock ------------------------------------------------------


def test_fast_blocks_do_not_trip_the_windows_clock():
    """polychrom divides by each block's wall time to log steps per second. With
    time.time() a one-step block reads as 0 s on Windows and raises
    ZeroDivisionError; run_sim_MD gives polychrom a precise clock."""
    sim = Simulation(
        platform=PLATFORM, integrator="variableLangevin", error_tol=0.01,
        collision_rate=0.03, N=50, reporters=[], GPU="0",
    )
    # Minimized random walk, as in simulate_MD: a force-free start (a straight
    # chain) makes the variable-step integrator take one huge step and blow up.
    sim.set_data(create_random_walk(step_size=1, N=50), center=True)
    sim.add_force(forcekits.polymer_chains(sim))
    sim.local_energy_minimization()
    for _ in range(50):
        sim.do_block(steps=1)


if __name__ == "__main__":
    print(f"platform: {PLATFORM}")
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
