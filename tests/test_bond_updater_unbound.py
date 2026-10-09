"""bondUpdater drops the unbound sentinel quietly and warns only on corruption.

downsampling_LE writes -1 on both feet for an extruder that holds no bond this
frame, so every run with unbinding has some. They must be dropped WITHOUT the
"corrupt LEF bond" warning, which is kept for real out-of-range indices (e.g.
INT32_MAX from the run_sim_LE loop_pos dtype bug). The old warning fired on the
sentinels and sent the advisor looking for a bug that was not there.

Needs polychrom and OpenMM (run_sim_MD imports them), so run it in the
py-hoomd2 env. No GPU. Runs under pytest, or standalone with
`python tests/test_bond_updater_unbound.py`.
"""

import logging
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(SIM_DIR))

from run_sim_MD import bondUpdater  # noqa: E402

N_BEADS = 50
INT32_MAX = int(np.iinfo(np.int32).max)

# frame 0: two bound extruders; frame 1: extruder 1 is unbound (-1, -1)
UNBOUND_ONLY = [
    [[10, 11], [30, 31]],
    [[9, 12], [-1, -1]],
]


class _FakeBondForce:
    """Stands in for OpenMM's HarmonicBondForce: records the bonds added."""

    def __init__(self):
        self.bonds = []

    def addBond(self, i, j, **params):
        self.bonds.append((i, j))
        return len(self.bonds) - 1


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _setup(positions):
    """bondUpdater.setup on a (frames, LEFs, 2) array -> (bonds added, warnings)."""
    handler = _Records()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        milker = bondUpdater(np.asarray(positions, dtype=np.int32), n_beads=N_BEADS)
        milker.setParams({"length": 0.5, "k": 1.0}, {"length": 0.5, "k": 0.0})
        force = _FakeBondForce()
        milker.setup(bondForce=force, blocks=len(positions))
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
    return force.bonds, warnings


def test_unbound_sentinel_is_dropped_without_a_warning():
    bonds, warnings = _setup(UNBOUND_ONLY)
    assert warnings == [], warnings
    assert sorted(bonds) == [(9, 12), (10, 11), (30, 31)], bonds


def test_corrupt_index_is_dropped_with_a_warning():
    bonds, warnings = _setup(UNBOUND_ONLY + [[[8, 13], [INT32_MAX, 5]]])
    assert len(warnings) == 1, warnings
    assert "dropped 1 corrupt LEF bond" in warnings[0], warnings[0]
    assert sorted(bonds) == [(8, 13), (9, 12), (10, 11), (30, 31)], bonds


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
