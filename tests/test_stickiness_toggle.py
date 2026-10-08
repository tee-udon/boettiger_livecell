"""Check the sticky-region switch behind equilibrate_without_stickiness.

simulate_MD zeroes one heteropolymer_SSW context parameter to switch the
attraction_coefficient_matrix term off for minimization and equilibration, then
restores it before loop extrusion. These tests fold a tiny chain into a hairpin
whose three sticky pairs sit exactly at the bottom of the attractive well, and
check that the switch removes those three wells and nothing else.

Needs polychrom and OpenMM, so run it in the py-hoomd2 env (the tee env has no
polychrom). Uses the CPU platform, so no GPU is touched. Runs under pytest, or
standalone with `python tests/test_stickiness_toggle.py`.
"""

import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(SIM_DIR))

from polychrom import forcekits, forces  # noqa: E402
from polychrom.simulation import Simulation  # noqa: E402

from run_sim_MD import _selective_attraction_param  # noqa: E402

N = 20
# Beads i and 19 - i face each other across the hairpin at distance WELL, so
# the sticky beads 0-2 and 17-19 form exactly three sticky contacts.
STICKY = [0, 1, 2, 17, 18, 19]
# Bottom of the smooth-square-well attraction for repulsionRadius=1 and
# attractionRadius=1.5: the well depth there is exactly attractionEnergy +
# matrix entry. Diagonal neighbours across the hairpin sit at sqrt(1 + 1.25^2)
# = 1.6, outside the 1.5 cutoff, so they contribute nothing.
WELL = 1.25
STICKINESS_KT = 1.0


def _hairpin_energy_sim(interaction_matrix):
    sim = Simulation(
        platform="CPU",
        integrator="variableLangevin",
        error_tol=0.01,
        collision_rate=0.03,
        N=N,
        reporters=[],
    )
    data = np.zeros((N, 3))
    data[:10, 0] = np.arange(10)  # bottom strand, beads 0..9
    data[10:, 0] = 19 - np.arange(10, N)  # top strand runs back, beads 10..19
    data[10:, 1] = WELL
    sim.set_data(data, center=False, random_offset=0, report=False)

    monomer_types = np.zeros(N, dtype=int)
    monomer_types[STICKY] = 1
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, False)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.1},
            angle_force_func=forces.angle_force,
            angle_force_kwargs={"k": 1.5},
            nonbonded_force_func=forces.heteropolymer_SSW,
            nonbonded_force_kwargs={
                "repulsionEnergy": 1.0,
                "attractionEnergy": 0.0,
                "attractionRadius": 1.5,
                "interactionMatrix": np.asarray(interaction_matrix, dtype=float),
                "monomerTypes": monomer_types,
                "extraHardParticlesIdxs": [],
            },
        )
    )
    sim._apply_forces()
    return sim


def _energy_kT(sim):
    return sim.context.getState(getEnergy=True).getPotentialEnergy() / sim.kT


def test_finds_the_selective_attraction_parameter():
    sim = _hairpin_energy_sim([[0, 0], [0, STICKINESS_KT]])
    assert _selective_attraction_param(sim) == "heteropolymer_SSW_ATTReAdd"


def test_switch_removes_exactly_the_sticky_wells():
    sim = _hairpin_energy_sim([[0, 0], [0, STICKINESS_KT]])
    param = _selective_attraction_param(sim)
    on = _energy_kT(sim)
    value = sim.context.getParameter(param)

    sim.context.setParameter(param, 0.0)
    off = _energy_kT(sim)
    sim.context.setParameter(param, value)
    restored = _energy_kT(sim)

    assert np.isclose(on - off, -3 * STICKINESS_KT, atol=1e-3), (on, off)
    assert np.isclose(restored, on, atol=1e-6), (restored, on)


def test_switched_off_equals_a_polymer_without_sticky_regions():
    """Off must mean 'no sticky regions', not 'something else reduced'."""
    sticky = _hairpin_energy_sim([[0, 0], [0, STICKINESS_KT]])
    sticky.context.setParameter(_selective_attraction_param(sticky), 0.0)
    plain = _hairpin_energy_sim([[0, 0], [0, 0]])
    assert np.isclose(_energy_kT(sticky), _energy_kT(plain), atol=1e-6)


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
