import numpy as np
from typing import Literal, List, Optional, Tuple, Union
from pydantic import (
    BaseModel,
    PrivateAttr,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


class SlurmCfg(BaseModel):
    partition: str = "gpu"
    account: Optional[str] = None
    time: str = "00:30:00"  # HH:MM:SS
    gpus: str = "1"  # e.g. "1" or "a100:1"
    cpus_per_task: int = 1
    mem: Optional[str] = "64G"  # e.g. "16G"
    qos: Optional[str] = None
    nodes: int = 1
    ntasks: int = 1
    constraint: Optional[str] = None
    chdir_to_run_dir: bool = True
    extra_sbatch: List[str] = []  # raw lines like ["#SBATCH --exclusive"]
    env_setup: Optional[str] = (
        None  # e.g. "module load cuda; source ~/miniconda3/etc/profile.d/conda.sh; conda activate poly"
    )
    python: str = "python"  # interpreter to call inside the job
    wait: bool = True  # block until job finishes (poll squeue/sacct)


class SimConfig(BaseModel, extra="forbid"):
    # Make sure that it validates every time the attributes have been updated
    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    # --- number of independent replicates per run
    num_replicates: int = Field(1, ge=1)

    # --- core loop extrusion simulation knobs
    num_monomers: int = Field(
        100_000,
        gt=0,
        description="Number of monomers. Approximately 1 kb/monomer.",
    )

    save_every_blocks: int = Field(
        1,
        ge=1,
        description=(
            "Save polymer positions every Nth LE step. Bonds still advance on "
            "every step, so raising this buys physical simulation time without "
            "growing all_conformations.npy. num_LE_steps must be divisible by it. "
            "One saved frame equals save_every_blocks LE steps, so a lag axis "
            "measured in frames must be rescaled to compare across values."
        ),
    )
    cohesin_speed: float = Field(
        1.0,
        ge=0,
        description=(
            "Speed of cohesin: MEAN monomers advanced per arm per LE round. Any "
            "value >= 0 is valid, fractional at any magnitude (0.1, 0.45, 1.5, 5 "
            "are all honoured exactly). Each arm independently draws "
            "floor(speed) certain steps plus one Bernoulli(frac(speed)) step per "
            "round, so below 1 this is exactly 'a speed-0.1 arm has a 1-in-10 "
            "chance of advancing each round', and waiting times are geometric. "
            "Mean loop size is ~2 * cohesin_speed * cohesin_bound_lifetime for a "
            "two-sided extruder, at EVERY speed -- binding and unbinding tick "
            "once per round and no longer scale with speed. 0 disables extrusion."
        ),
    )
    cohesin_speed_sd: float = Field(
        0.0,
        ge=0,
        description="Standard deviation of cohesin respectively (monomers/timestep). Must be positive.",
    )
    cohesin_stall_time: float = Field(
        1.0,
        ge=1,
        description="Expected number of timestep for cohesin to pause upon colliding to other condensin. Must be at least 1.",
    )
    cohesin_stall_probability: float = Field(
        1.0,
        ge=0,
        le=1,
        description="Probability of cohesin stall upon colliding other cohesin molecules.",
    )
    num_cohesin: int = Field(1000, ge=0, description="Number of cohesin in the system.")
    cohesin_bound_lifetime: float = Field(
        100000,
        ge=1,
        description="Expected number of timesteps by which cohesin are bound. Must be at least 1.",
    )
    cohesin_unbound_lifetime: float = Field(
        1,
        ge=1,
        description="Expected number of timesteps by which cohesin are unbound. Must be at least 1.",
    )
    extrusion_side: Literal[1, 2] = Field(
        2, description="Unidirectional vs bidirectional extrusion."
    )
    num_LE_steps: int = Field(
        1800,
        ge=0,
        description="Total number of loop extrusion steps. Default = 1800 steps (30 minutes).",
    )
    num_LE_steps_init: int = Field(
        0,
        ge=0,
        description="Total number of loop extrusion steps pre-exporting. This is useful for LE steady-state study.",
    )
    cohesin_loading_probability_list: Optional[
        Union[List[float], List[Tuple[float, int]]]
    ] = Field(
        None,
        description="List of cohesin loading probability along the genome. It can either be None or a list of length num_monomers or a list of lists, where each list is a length 2 vector that contains the relative cohesin loading weight and the number of monomers in the subchain. The total probabilty must sum to 1.",
    )
    ctcf_site_location_list: Optional[List[int]] = Field(
        None,
        description="List of ctcf site locations along the genome. If it is None, then simulation assumes that there is no CTCF in the system.",
    )
    ctcf_site_direction_list: Optional[List[Literal["left", "right", "both"]]] = Field(
        None,
        description="List of ctcf site directions. If it is None, then the simulation assumes that all CTCF sites can stall cohesin from both direction.",
    )
    ctcf_site_stall_probability_list: Optional[List[float]] = Field(
        None,
        description="List of ctcf site stalling probability. If it is None, then the simulation assumes that CTCF sites always stall cohesin.",
    )
    ctcf_site_stall_time_list: Optional[List[float]] = Field(
        None,
        description="List of ctcf site expeted stall time. If it is None, then the simulation assumes that CTCF sites stall cohesin forever.",
    )

    # --- core MD simulation hyperparameter
    monomer_type_list: Optional[Union[List[int], List[Tuple[int, int]]]] = Field(
        None,
        description="Monomer types description. It can either be a list of len(num_monomers) of monomer types or a list of lists. Each nested list is a length 2 vector that contains the monomer type and the number of monomers that have that type in a row. The sum of number of monomers should be equal to num_monomers. If not provided, the simulation assumes that all monomers have similar type (homopolymer).",
    )
    attraction_coefficient_matrix: Optional[List[List[float]]] = Field(
        None,
        description="Attraction coefficient between monomer types. It can either be None or an N-by-N matrix where N is the number of monomer types. If None, the simulation assumes that monomers do not attract.",
    )
    repulsion: float = Field(
        5.0, ge=0.0, description="Repulsion constant between monomers."
    )
    angle_k: float = Field(
        1.5,
        ge=0.0,
        description="Bending stiffness (angle-force k); persistence-length knob. k=1.5 flexible, k=8 stiff.",
    )
    equilibration_timestep: int = Field(
        1500,
        ge=1,
        description="Number of MD timestep at the initialization step to relax the initial conformation.",
    )
    equilibrate_without_stickiness: Literal[True, False] = Field(
        False,
        description=(
            "Switch the sticky-region attraction (the attraction_coefficient_matrix "
            "term of heteropolymer_SSW) off for energy minimization and "
            "equilibration, and back on before the first loop-extrusion step, so "
            "the starting conformation carries no stickiness bias such as a "
            "pre-formed tether. The uniform attraction_energy and excluded volume "
            "are left on. Runs resumed after preemption skip both steps, so this "
            "does not affect them."
        ),
    )
    num_MD_steps_per_LE: int = Field(
        1000, ge=1, description="Number of MD timestep between moving cohesin bonds."
    )
    attraction_radius: float = Field(
        0,
        ge=0.0,
        description="The distance in which monomer stickiness affects the surrounding. NOTE: this also sets the nonbonded cutoff in heteropolymer_SSW, so a value of 0 switches the nonbonded force off entirely, repulsion included. For active excluded volume it must be strictly greater than the repulsion radius of 1.0; 1.5 is the polychrom default. Must be 0 or above 1.0.",
    )
    attraction_energy: float = Field(
        3.0,
        ge=0.0,
        description="Base attraction well depth (kT) between ALL monomers in heteropolymer_SSW. Must be set explicitly whenever attraction_radius > 1.0, because the 3 kT default collapses the chain and typically ends the run with EKExceedsError. Set to 0 together with attraction_radius > 1.0 to obtain pure excluded volume plus only the sticky regions.",
    )
    density: float = Field(
        0.24,
        gt=0.0,
        le=1.0,
        description="The size of the initial polymer relative to the enclosing box. 1 means the polymer occupies the whole box.",
    )
    collision_rate: float = Field(
        0.03, gt=0.0, description="The rate by which monomers collide one another. "
    )
    smc_bond_dist: float = Field(
        0.5, gt=0.0, description="Average bond length between Condensin arms."
    )
    smc_bond_wiggle_dist: float = Field(
        0.2,
        gt=0.0,
        description="Standard deviation of bond lenght between Condensin arms.",
    )
    gpu_device: str = Field("0", description="GPU device used to run MD simulation.")
    initial_conformation: Literal["random_walk", "random_walk_z", "crumpled"] = Field(
        "crumpled", description="Initial conformation for MD simulation"
    )
    confinement: Optional[Literal["spherical"]] = Field(
        None, description="Confinement. Support None and spherical."
    )
    PBC_box: Literal[True, False] = Field(
        False, description="The size of periodic boundary condition."
    )

    # --- plot setting
    plot_LE: Literal[True, False] = Field(
        True, description="Plot LE simulation result for sanity check."
    )

    # --- backend setting
    backend: Literal["local", "slurm"] = Field(
        "local",
        description="Machine to run the MD simulation. Valid options are 'local' or 'slurm'.",
    )
    slurm: Optional[SlurmCfg] = None

    # --- bookkeeping
    condition_name: str = ""
    out_dir: str = "runs"
    start_idx_replicate: int = 0

    # --- constant parameters for proper internal working of simulation
    _num_cohesin_types: int = PrivateAttr(1)

    # --- validator
    # @model_validator(mode="after")
    # def check_model(self):
    #     num_monomers = self.num_monomers
    #     monomer_type_list = self.monomer_type_list

    #     if monomer_type_list is None:
    #         self.monomer_type_list = [0 for _ in range(num_monomers)]
    #     else:
    #         if len(monomer_type_list) != num_monomers:
    #             raise ValueError(
    #                 "Length of monomer_type_list must equal to num_monomers."
    #             )
    #     return self

    @field_validator("monomer_type_list", mode="after")
    def check_monomer_type_list(cls, v, info):
        num_monomers = info.data.get("num_monomers")
        # If num_monomers failed validation, bail out; the scalar field error is enough.
        if num_monomers is None:
            return v

        if v is None:
            return [0 for _ in range(num_monomers)]
        else:
            # This case means that the user uses [(monomer_type, num_monomer_subchain)] notation
            if type(v[0]) is tuple:
                monomer_type_list = []
                for v_ in v:
                    # populated the monomer type list based on the num_monomer_subchain
                    monomer_type_list += [v_[0] for _ in range(v_[1])]
                if len(monomer_type_list) != num_monomers:
                    raise ValueError(
                        "Sum of total monomers across all subchains must equal to num_monomers"
                    )
                return monomer_type_list

            if len(v) != num_monomers:
                raise ValueError(
                    "Length of monomer_type_list must equal to num_monomers."
                )
            return v

    @field_validator("attraction_coefficient_matrix", mode="after")
    def check_attraction_coefficient_matrix(cls, v, info):
        monomer_type_list = info.data.get("monomer_type_list")
        # If monomer_type_list failed validation or isn't present yet, don't do anything.
        if monomer_type_list is None:
            return v

        num_unique_monomer_types = len(set(monomer_type_list))

        if v is None:
            return [
                [0 for _ in range(num_unique_monomer_types)]
                for _ in range(num_unique_monomer_types)
            ]
        else:
            if len(v) != num_unique_monomer_types:
                raise ValueError(
                    "Number of rows should be equal to the number of unique monomer types."
                )
            for row in v:
                if len(row) != num_unique_monomer_types:
                    raise ValueError(
                        "Number of columns should be equal to the number of unique monomer types."
                    )
            return v

    @field_validator("cohesin_loading_probability_list", mode="after")
    def check_cohesin_loading_probability_list(cls, v, info):
        num_monomers = info.data.get("num_monomers")
        # If num_monomers failed validation, bail out; the scalar field error is enough.
        if num_monomers is None:
            return v

        if v is None:
            return [1 / num_monomers for _ in range(num_monomers)]
        else:
            if type(v[0]) is tuple:
                cohesin_loading_probability_list = []
                for v_ in v:
                    cohesin_loading_probability_list += [v_[0] for _ in range(v_[1])]
                # Normalize the probability
                cohesin_loading_probability_list = np.array(
                    cohesin_loading_probability_list
                )
                cohesin_loading_probability_list /= np.sum(
                    cohesin_loading_probability_list
                )
                if len(cohesin_loading_probability_list) != num_monomers:
                    raise ValueError(
                        "Sum of total monomers across all subchains must equal to num_monomers"
                    )
                return cohesin_loading_probability_list.tolist()

            if abs(sum(v) - 1) > 1e-6:  # Use epsilon for tolerance
                raise ValueError("Probabilities must sum to 1.")
            if len(v) != num_monomers:
                raise ValueError(
                    "Length of cohesin_loading_probability_list must equal to num_monomers"
                )
            return v

    @field_validator("ctcf_site_location_list", mode="after")
    def check_ctcf_site_location_list(cls, v, info):
        num_monomers = info.data.get("num_monomers")
        # If num_monomers failed validation, bail out; the scalar field error is enough.
        if num_monomers is None:
            return v

        if v is None:
            return []
        else:
            for v_ in v:
                if not (0 <= v_ < num_monomers):
                    raise ValueError("CTCF site location must be within the polymer.")
        return v

    @field_validator("ctcf_site_direction_list", mode="after")
    def check_ctcf_direction_list(cls, v, info):
        ctcf_site_location_list = info.data.get("ctcf_site_location_list")
        if ctcf_site_location_list is None:
            # Locations missing or invalid; skip to avoid hiding that error
            return v

        num_ctcf_sites = len(ctcf_site_location_list)
        if v is None:
            return ["both" for _ in range(num_ctcf_sites)]
        else:
            if len(v) != num_ctcf_sites:
                raise ValueError(
                    "Length of ctcf_site_direction_list must equal to length of ctcf_site_location_list."
                )
            return v

    @field_validator("ctcf_site_stall_probability_list", mode="after")
    def check_ctcf_site_stall_probability_list(cls, v, info):
        ctcf_site_location_list = info.data.get("ctcf_site_location_list")
        if ctcf_site_location_list is None:
            # Locations missing or invalid; skip to avoid hiding that error
            return v

        num_ctcf_sites = len(ctcf_site_location_list)
        if v is None:
            return [1 for _ in range(num_ctcf_sites)]
        else:
            if len(v) != num_ctcf_sites:
                raise ValueError(
                    "Length of ctcf_site_stall_probability_list must equal to length of ctcf_site_location_list."
                )
            for v_ in v:
                if not (0 <= v_ <= 1):
                    raise ValueError("CTCF stall probability must be between 0 and 1")

            return v

    @field_validator("ctcf_site_stall_time_list", mode="after")
    def check_ctcf_stall_time_list(cls, v, info):
        ctcf_site_location_list = info.data.get("ctcf_site_location_list")
        if ctcf_site_location_list is None:
            # Locations missing or invalid; skip to avoid hiding that error
            return v

        num_ctcf_sites = len(ctcf_site_location_list)
        if v is None:
            return [np.inf for _ in range(num_ctcf_sites)]
        else:
            if len(v) != num_ctcf_sites:
                raise ValueError(
                    "Length of ctcf_site_stall_probability_list must equal to length of ctcf_site_location_list."
                )
            for v_ in v:
                if v_ < 1:
                    raise ValueError("CTCF expected stall time should be at least 1")
            return v

    @model_validator(mode="after")
    def check_md_settings(self):
        """Settings that are each valid alone but wrong together. Caught here, at
        load time, because otherwise they crash late (after the loop extrusion has
        run) or, worse, silently simulate something else."""
        problems = []
        r = self.attraction_radius
        matrix = np.asarray(self.attraction_coefficient_matrix, dtype=float)
        types = sorted(set(self.monomer_type_list))

        if 0 < r <= 1.0:
            problems.append(
                f"attraction_radius is {r:g}. It must be 0, which switches the nonbonded "
                "force off (no excluded volume and no stickiness), or above 1.0 (1.5 is "
                "standard). In between, the cutoff slices through the repulsive core."
            )
        if r == 0 and matrix.any():
            problems.append(
                "attraction_coefficient_matrix defines sticky interactions, but "
                "attraction_radius is 0, which switches the nonbonded force off, so "
                "nothing would stick. Set attraction_radius: 1.5."
            )
        # The 2026-10-08 EKExceedsError: older configs never set attraction_energy,
        # and the code they were written for applied no uniform attraction. The
        # 3 kT default makes every monomer pair attract and the chain overheats.
        if r > 1.0 and "attraction_energy" not in self.model_fields_set:
            problems.append(
                "attraction_radius above 1.0 switches the attraction on, so set "
                "attraction_energy explicitly. It is the attraction between EVERY pair "
                "of monomers, in kT: 0 gives excluded volume plus only the sticky "
                "regions; the default of 3 collapses the whole chain and typically "
                "stops the run with EKExceedsError."
            )
        if not np.allclose(matrix, matrix.T):
            # polychrom's heteropolymer_SSW refuses it, but only at MD time
            problems.append("attraction_coefficient_matrix must be symmetric.")
        if types != list(range(len(types))):
            # polychrom sizes the interactions by the largest type, so a gap
            # (types 0 and 2 with a 2x2 matrix) fails at MD time
            problems.append(
                f"monomer types must be numbered 0, 1, 2, ... without gaps; got {types}."
            )
        if self.num_LE_steps % self.save_every_blocks != 0:
            problems.append(
                f"num_LE_steps ({self.num_LE_steps}) must be a multiple of "
                f"save_every_blocks ({self.save_every_blocks})."
            )
        if self.PBC_box and r > 0:
            box = int((self.num_monomers / self.density) ** 0.333)  # as in run_sim_MD
            if box <= 2 * r:
                problems.append(
                    f"the periodic box is {box} bond lengths (from num_monomers and "
                    f"density); it must be more than twice attraction_radius ({r:g})."
                )
        if problems:
            raise ValueError("MD settings: " + " | ".join(problems))
        return self
