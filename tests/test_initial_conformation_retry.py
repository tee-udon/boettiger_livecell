"""A failed first minimization retries from a NEW start, for every initial_conformation.

simulate_MD retries local_energy_minimization from a fresh starting chain. The
retry branch used to list only crumpled and random_walk, so a random_walk_z run
whose first minimization raised died with "initial_conformation must either be
crumpled or random_walk" instead of retrying. Here the first minimization is
forced to fail, and the run must finish from a new random_walk_z chain.

Needs polychrom and OpenMM, so run it in the py-hoomd2 env. simulate_MD
hard-codes CUDA; the test switches it to the CPU platform, so no GPU is touched.
Runs under pytest, or standalone with
`python tests/test_initial_conformation_retry.py`.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import openmm
import polychrom.simulation

SIM_DIR = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(SIM_DIR))

import run_sim_MD  # noqa: E402
from sim_config import SimConfig  # noqa: E402

N = 60


def test_every_initial_conformation_builds_a_chain():
    for name in ("crumpled", "random_walk", "random_walk_z"):
        data = run_sim_MD._initial_conformation(name, N, density=0.24)
        assert data.shape == (N, 3) and np.isfinite(data).all(), name
    try:
        run_sim_MD._initial_conformation("lattice", N, density=0.24)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown initial_conformation must raise ValueError")


def test_random_walk_z_retries_after_a_failed_minimization():
    calls = []
    orig_init = polychrom.simulation.Simulation.__init__
    orig_minimize_attr = openmm.LocalEnergyMinimizer.__dict__["minimize"]
    orig_minimize = openmm.LocalEnergyMinimizer.minimize

    def cpu_init(self, **kwargs):
        kwargs["platform"] = "CPU"  # simulate_MD hard-codes CUDA
        orig_init(self, **kwargs)

    def fail_first(context, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise openmm.OpenMMException("forced failure of the first minimization")
        return orig_minimize(context, *args, **kwargs)

    cfg = SimConfig(
        num_monomers=N,
        num_cohesin=0,
        num_LE_steps=2,
        initial_conformation="random_walk_z",
        attraction_radius=1.5,
        attraction_energy=0.0,
        repulsion=1.0,
        equilibration_timestep=100,
        num_MD_steps_per_LE=10,
        plot_LE=False,
    )
    polychrom.simulation.Simulation.__init__ = cpu_init
    openmm.LocalEnergyMinimizer.minimize = staticmethod(fail_first)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_sim_MD.simulate_MD(cfg, Path(tmp))
            start = np.load(Path(tmp) / "init_conformation.npy")
            frames = np.load(Path(tmp) / "all_conformations.npy")
    finally:
        polychrom.simulation.Simulation.__init__ = orig_init
        openmm.LocalEnergyMinimizer.minimize = orig_minimize_attr

    assert len(calls) == 2, calls  # failed once, then minimized the new start
    # The saved start is the regenerated one, and it is a random_walk_z chain:
    # create_random_walk_positivez steps z by exactly 0.0015 per monomer.
    assert np.allclose(np.diff(start[:, 2]), 0.0015), "retry did not use random_walk_z"
    assert frames.shape == (cfg.num_LE_steps + 1, N, 3), frames.shape


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
