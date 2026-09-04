"""Guard the loop-extrusion dtypes against platform drift.

spec_smc must use FIXED-WIDTH numba types. types.int_ tracks C long, which is
8 bytes on Linux/macOS but 4 bytes on Windows/MSVC, while init_SMC_sim builds
its arrays with an explicit np.int64. If the spec ever drifts back to a
platform-dependent alias, those two disagree on Windows and jitclass
construction dies with a TypingError -- on someone else's machine, not ours.

These tests are cheap and catch it here instead.

Runs under pytest, or standalone with `python tests/test_run_sim_LE_types.py`
(pytest is not installed in the py-hoomd2 env).
"""

import ast
import sys
from pathlib import Path

import numpy as np
from numba import types, typed

SIM_DIR = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(SIM_DIR))

from run_sim_LE import SMC, spec_smc, update_SMC_sim  # noqa: E402

# Aliases whose width follows the C ABI instead of being pinned: int_ and long_
# follow C long, intp follows pointer width, intc follows C int.
PLATFORM_DEPENDENT_NAMES = {
    "int_", "uint", "intp", "uintp", "intc", "uintc", "long_", "ulong",
}


def _base(t):
    """Element type for an array type, else the type itself."""
    return getattr(t, "dtype", t)


def test_spec_uses_only_fixed_width_types():
    """Scan the SOURCE, not the resolved types.

    This has to be a source-level check. On Linux `types.int_ is types.int64`
    -- the very same object -- so at runtime the two are indistinguishable and
    a set-membership test on the resolved type can never catch the alias. The
    width only diverges on Windows, where we do not run CI. The name in the
    source is the only evidence available here.
    """
    tree = ast.parse((SIM_DIR / "run_sim_LE.py").read_text())
    offenders = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in PLATFORM_DEPENDENT_NAMES
            and isinstance(node.value, ast.Name)
            and node.value.id == "types"
        }
    )
    assert not offenders, (
        "run_sim_LE.py uses platform-dependent numba aliases "
        f"{offenders}, which change width on Windows. Use types.int64 / "
        "types.float64 so the jitclass spec matches the np.int64 arrays "
        "init_SMC_sim passes into it."
    )


def test_int_fields_are_int64():
    widths = {name: _base(t) for name, t in spec_smc}
    for name in ("N_beads", "l_pos", "r_pos", "start_pos", "smc_id",
                 "ctcf_site_location_list", "ctcf_site_direction_list"):
        assert widths[name] == types.int64, f"{name} is {widths[name]}, want int64"


def _make(n_ctcf):
    """Construct an SMC the way init_SMC_sim does."""
    n_beads = 1000
    return SMC(
        N_beads=n_beads,
        bound_lifetime=100,
        unbound_lifetime=10,
        SMC_crash_lifetime=1.0,
        SMC_crash_prob=0.0,
        extrusion_sided=2,
        extrusion_rate=0.5,
        extrusion_rate_sd=0.0,
        cohesin_loading_probability_list=np.full(
            n_beads - 3, 1.0 / (n_beads - 3), dtype=np.float64
        ),
        ctcf_site_location_list=np.arange(n_ctcf, dtype=np.int64) + 10,
        ctcf_site_direction_list=np.ones(n_ctcf, dtype=np.int64),
        ctcf_site_stall_probability_list=np.full(n_ctcf, 0.5, dtype=np.float64),
        ctcf_site_stall_time_list=np.full(n_ctcf, 10.0, dtype=np.float64),
        smc_id=0,
    )


def test_constructs_with_no_ctcf_sites():
    # np.array([]) defaults to float64, so an empty CTCF system is the case
    # most likely to trip a dtype mismatch.
    assert _make(0) is not None


def test_constructs_with_ctcf_sites():
    assert _make(3) is not None


def test_loop_pos_is_int64():
    smcs = typed.List()
    smcs.append(_make(0))
    assert update_SMC_sim(smcs).dtype == np.int64


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
