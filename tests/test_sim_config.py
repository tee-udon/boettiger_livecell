# tests/test_sim_config.py

import pytest
from pydantic import ValidationError
from simulation.sim_config import SimConfig, SlurmCfg  


# -----------------------
# Basic "happy path" test
# -----------------------


def test_default_config_is_consistent():
    cfg = SimConfig()

    # num_monomers-related lists
    assert cfg.num_monomers > 0
    assert len(cfg.monomer_type_list) == cfg.num_monomers
    assert set(cfg.monomer_type_list) == {0}

    assert len(cfg.cohesin_loading_probability_list) == cfg.num_monomers
    # probabilities sum to 1
    assert pytest.approx(sum(cfg.cohesin_loading_probability_list)) == 1.0

    # attraction matrix: one monomer type → 1x1 matrix of zeros
    mat = cfg.attraction_coefficient_matrix
    assert len(mat) == 1
    assert len(mat[0]) == 1
    assert mat[0][0] == 0

    # CTCF defaults: no CTCF sites
    assert cfg.ctcf_site_location_list == []
    assert cfg.ctcf_site_direction_list == []
    assert cfg.ctcf_site_stall_probability_list == []
    assert cfg.ctcf_site_stall_time_list == []


# -----------------------
# monomer_type_list
# -----------------------


def test_monomer_type_list_default_fills_zeros():
    cfg = SimConfig(num_monomers=10, monomer_type_list=None)
    assert len(cfg.monomer_type_list) == 10
    assert set(cfg.monomer_type_list) == {0}


def test_monomer_type_list_wrong_length_raises():
    with pytest.raises(ValidationError) as excinfo:
        SimConfig(num_monomers=10, monomer_type_list=[0, 1, 2])
    assert "Length of monomer_type_list" in str(excinfo.value)


# -----------------------
# attraction_coefficient_matrix
# -----------------------


def test_attraction_matrix_default_shape_matches_unique_types():
    # two monomer types: 0 and 1
    cfg = SimConfig(
        num_monomers=4,
        monomer_type_list=[0, 1, 0, 1],
        attraction_coefficient_matrix=None,
    )
    mat = cfg.attraction_coefficient_matrix

    # 2 unique types -> 2x2 matrix
    assert len(mat) == 2
    assert all(len(row) == 2 for row in mat)
    # default is all zeros
    assert all(all(v == 0 for v in row) for row in mat)


def test_attraction_matrix_wrong_rows_raises():
    with pytest.raises(ValidationError):
        SimConfig(
            num_monomers=4,
            monomer_type_list=[0, 1, 0, 1],
            attraction_coefficient_matrix=[[0, 0, 0]],  # 1x3, should be 2x2
        )


def test_attraction_matrix_wrong_cols_raises():
    with pytest.raises(ValidationError):
        SimConfig(
            num_monomers=4,
            monomer_type_list=[0, 1, 0, 1],
            attraction_coefficient_matrix=[[0, 0], [0, 0, 0]],  # second row too long
        )


# -----------------------
# cohesin_loading_probability_list
# -----------------------


def test_loading_prob_default_uniform():
    cfg = SimConfig(num_monomers=5, cohesin_loading_probability_list=None)
    probs = cfg.cohesin_loading_probability_list
    assert len(probs) == 5
    assert pytest.approx(sum(probs)) == 1.0
    # all entries equal (uniform)
    assert len(set(probs)) == 1


def test_loading_prob_invalid_sum_raises():
    with pytest.raises(ValidationError) as excinfo:
        SimConfig(
            num_monomers=3,
            cohesin_loading_probability_list=[0.5, 0.5, 0.5],  # sums to 1.5
        )
    assert "Probabilities must sum to 1" in str(excinfo.value)


def test_loading_prob_wrong_length_raises():
    with pytest.raises(ValidationError) as excinfo:
        SimConfig(
            num_monomers=4,
            cohesin_loading_probability_list=[0.5, 0.25, 0.25],  # length 3, sum=1
        )
    assert "Length of cohesin_loading_probability_list" in str(excinfo.value)


# -----------------------
# CTCF-related validators
# -----------------------


def test_ctcf_defaults_when_no_locations():
    cfg = SimConfig(
        num_monomers=100,
        ctcf_site_location_list=None,
        ctcf_site_direction_list=None,
        ctcf_site_stall_probability_list=None,
        ctcf_site_stall_time_list=None,
    )
    assert cfg.ctcf_site_location_list == []
    assert cfg.ctcf_site_direction_list == []
    assert cfg.ctcf_site_stall_probability_list == []
    assert cfg.ctcf_site_stall_time_list == []


def test_ctcf_defaults_when_locations_given():
    cfg = SimConfig(
        num_monomers=100,
        ctcf_site_location_list=[10, 20],
        ctcf_site_direction_list=None,
        ctcf_site_stall_probability_list=None,
        ctcf_site_stall_time_list=None,
    )
    assert cfg.ctcf_site_direction_list == ["both", "both"]
    assert cfg.ctcf_site_stall_probability_list == [1, 1]
    # Permanent stall. f0e3b14 changed this default from 1e6 to np.inf.
    assert cfg.ctcf_site_stall_time_list == [float("inf"), float("inf")]


def test_ctcf_location_out_of_range_low_raises():
    with pytest.raises(ValidationError) as excinfo:
        SimConfig(num_monomers=50, ctcf_site_location_list=[-1])
    assert "CTCF site location must be within the polymer" in str(excinfo.value)


def test_ctcf_location_out_of_range_high_raises():
    # valid indices are 0..num_monomers-1
    with pytest.raises(ValidationError):
        SimConfig(num_monomers=50, ctcf_site_location_list=[50])


def test_ctcf_direction_length_mismatch_raises():
    with pytest.raises(ValidationError) as excinfo:
        SimConfig(
            num_monomers=100,
            ctcf_site_location_list=[10, 20],
            ctcf_site_direction_list=["left"],  # length 1, need 2
        )
    assert "ctcf_site_direction_list" in str(excinfo.value)


def test_ctcf_stall_probability_bounds_and_length():
    # out-of-bounds probability
    with pytest.raises(ValidationError) as excinfo:
        SimConfig(
            num_monomers=100,
            ctcf_site_location_list=[10],
            ctcf_site_stall_probability_list=[1.5],
        )
    assert "CTCF stall probability must be between 0 and 1" in str(excinfo.value)

    # wrong length
    with pytest.raises(ValidationError):
        SimConfig(
            num_monomers=100,
            ctcf_site_location_list=[10, 20],
            ctcf_site_stall_probability_list=[0.5],
        )


def test_ctcf_stall_time_bounds_and_length():
    # stall time must be >= 1
    with pytest.raises(ValidationError) as excinfo:
        SimConfig(
            num_monomers=100,
            ctcf_site_location_list=[10],
            ctcf_site_stall_time_list=[0],
        )
    assert "CTCF expected stall time should be at least 1" in str(excinfo.value)

    # wrong length
    with pytest.raises(ValidationError):
        SimConfig(
            num_monomers=100,
            ctcf_site_location_list=[10, 20],
            ctcf_site_stall_time_list=[100, 200, 300],
        )


# -----------------------
# Simple scalar field constraints
# -----------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_replicates", 0),
        ("num_monomers", 0),
        ("cohesin_speed", -0.1),  # ge=0
        ("cohesin_speed_sd", -1),  # ge=0
        ("cohesin_stall_time", 0.5),  # ge=1
        ("cohesin_stall_probability", -0.1),  # < 0
        ("cohesin_stall_probability", 1.1),  # > 1
        ("num_cohesin", -1),  # ge=0
        ("cohesin_bound_lifetime", 0),  # ge=1
        ("cohesin_unbound_lifetime", 0),  # ge=1
        ("num_LE_steps", -1),  # ge=0
        ("num_LE_steps_init", -1),  # ge=0
        ("equilibration_timestep", 0),  # ge=1
        ("num_MD_steps_per_LE", 0),  # ge=1
        ("attraction_radius", -0.1),  # ge=0
        ("density", 0),  # gt=0
        ("density", 1.5),  # le=1
        ("collision_rate", 0),  # gt=0
        ("smc_bond_dist", 0),  # gt=0
        ("smc_bond_wiggle_dist", 0),  # gt=0
    ],
)
def test_scalar_field_constraints(field, value):
    # We pass only one overridden field; others keep defaults.
    with pytest.raises(ValidationError):
        SimConfig(**{field: value})


@pytest.mark.parametrize("value", [0.0, 0.1, 0.45, 1.0, 1.5, 2.7, 5.0])
def test_cohesin_speed_accepts_any_non_negative_rate(value):
    """cohesin_speed is a mean advance per arm per round, not a step count.

    Sub-unit values are the intended way to set processivity below one monomer
    per round, and fractional values above 1 are honoured too: each arm draws
    floor(speed) certain steps plus one Bernoulli(frac(speed)) step. This field
    was ge=1, which blocked the sub-unit mechanism outright.
    """
    assert SimConfig(cohesin_speed=value).cohesin_speed == value


def test_equilibrate_without_stickiness_is_opt_in():
    """Off by default, so older configs keep the behaviour they were run with.

    Resolved configs saved before this field existed reload with the default,
    and those runs relaxed with the stickiness on.
    """
    assert SimConfig().equilibrate_without_stickiness is False
    cfg = SimConfig(equilibrate_without_stickiness=True)
    assert cfg.equilibrate_without_stickiness is True


# -----------------------
# Round-trip serialization
# -----------------------


def test_round_trip_dict_to_config():
    cfg = SimConfig()
    data = cfg.model_dump()
    new_cfg = SimConfig(**data)

    # a few key fields should be identical
    assert new_cfg.num_monomers == cfg.num_monomers
    assert new_cfg.monomer_type_list == cfg.monomer_type_list
    assert (
        new_cfg.cohesin_loading_probability_list == cfg.cohesin_loading_probability_list
    )
    assert new_cfg.ctcf_site_location_list == cfg.ctcf_site_location_list
