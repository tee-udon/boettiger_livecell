from typing import Literal, List, Optional, Tuple
from pydantic import (
    BaseModel,
    PrivateAttr,
    Field,
    Extra,
    field_validator,
    computed_field,
)

# TODO: Options
# 1D only
# MD as well - local, slurm

# TODO: Dynamics option


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


class OptConfig(BaseModel, extra=Extra.forbid):
    # --- number of independent replicates per run
    num_replicates: int = Field(1, ge=1)

    # --- simulation type
    simulation_type: Literal["1D", "MD"] = Field(
        "1D",
        description="Simulation type used for Bayesian optimization routine. Can either be '1D' or 'MD'",
    )

    # When to calculate objective fucntions
    objective_calculation_option: Literal["Last timepoint", "All timepoints"] = Field(
        "Last timepoint",
        description="Option of when to calculate objective function. Can either be just 'Last timepoint' or 'All timepoints'",
    )

    # Timestep to calculate objective functions
    # This option is only valid when 'objective_calculation_option' == "All timepoints"
    objective_calculation_timestep: Optional[Tuple[int]] = Field(
        None,
        description="Timestep indices to calculate objective function. Only valid when 'objective_calculation_option' == 'All timepoints'",
    )

    # --- objective functions definition for each sample
    # It makes sense for "1D" option to be loss function (minimizing distance to mean/median of the loop size distribution)
    # It should be a scalar
    # On the other hand, "MD" option score function is recommended (maximizing the correlation to P(s) curve)
    # In this case, objective should be a vector
    # TODO: Make sure that this is validated against different options. Check

    # Default values are objective function for 1D optimization problem
    # Unit in Mb
    # Objective functions for Both Condensin samples
    objective_BothCond_cond1: int | float | Tuple[float] | Tuple[Tuple[float]] = 0.8
    objective_BothCond_cond2: int | float | Tuple[float] | Tuple[Tuple[float]] = 8.69

    # Objective functions for Condensin 1 only mutants
    objective_Cond1Only_cond1: int | float | Tuple[float] | Tuple[Tuple[float]] = 0.8

    # Objective functions for Condensin 2 only mutants
    objective_Cond2Only_cond2: int | float | Tuple[float] | Tuple[Tuple[float]] = 18.35

    # --- core loop extrusion simulation hyperparameter tuning
    cond1_speed_range: Tuple[float, float]
    cond2_speed_range: Tuple[float, float]

    cond1_cond1_stall_time_range: Tuple[
        float, float
    ]  # Condensin 1 stall time upon seeing other Condensin 1
    cond1_cond2_stall_time_range: Tuple[
        float, float
    ]  # Condensin 1 stall time upon seeing other Condensin 2
    cond2_cond1_stall_time_range: Tuple[
        float, float
    ]  # Condensin 2 stall time upon seeing other Condensin 1
    cond2_cond2_stall_time_range: Tuple[
        float, float
    ]  # Condensin 2 stall time upon seeing other Condensin 2

    cond1_stall_probability_range: Tuple[float.float]
    cond2_stall_probability_range: Tuple[float, float]

    cond1_num_range = Tuple[int, int]
    cond2_num_range = Tuple[int, int]

    cond1_bound_lifetime_range = Tuple[float, float]
    cond2_bound_lifetime_range = Tuple[float, float]

    cond1_unbound_lifetime_range = Tuple[float, float]
    cond2_unbound_lifetime_range = Tuple[float, float]

    # --- core loop extrusion simulation knobs (constant)
    num_monomers: int = Field(
        100_000,
        gt=0,
        description="Number of monomers per sister. Approximately 1 kb/monomer.",
    )
    condensin_speed_sd_list: Tuple[float, float] = Field(
        (0.0, 0.0),
        description="Standard deviation of Condensin 1 and 2 respectively (monomers/timestep). Must be positive.",
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
    extrusion_side_list: Tuple[Literal[1, 2], Literal[1, 2]] = Field(
        (1, 1), description="Unidirectional vs bidirectional extrusion."
    )

    # Current setting does not allow tuning of MD hyperparameters
    # --- core MD simulation hyperparameter
    repulsion: float = Field(
        5.0, ge=0.0, description="Repulsion constant between monomers."
    )
    equilibration_timestep: int = Field(
        1500,
        ge=1,
        description="Number of MD timestep at the initialization step to relax the initial conformation.",
    )
    num_MD_steps_per_LE: int = Field(
        1000, ge=1, description="Number of MD timestep between moving condensin bonds."
    )
    attraction_coefficient: float = Field(
        0.05, ge=0.0, description="Coefficient for monomer stickiness."
    )
    attraction_radius: float = Field(
        1.5,
        ge=0.0,
        description="The distance in which monomer stickiness affects the surrounding.",
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
    initial_conformation: Literal["random_walk", "crumpled"] = Field(
        "crumpled", description="Initial conformation for MD simulation"
    )

    # --- chromosome setting
    centromere_range_list: Tuple[int, int] = Field(
        (0, 0),
        description="The monomer indices of the lower and upper bounds of centromere. (0, 0) if no centromere assigned.",
    )
    centromere_type: Literal["stall", "kick"] = Field(
        "stall",
        description="Type of condensin/centromere interactions: stall - centromeres stall Condensins indefinitely; kick: centromeres unbind condensin. ",
    )
    num_sister_chromatids: Literal[1, 2] = Field(
        1, description="Number of sister chromatid in the simulation."
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

    # --- computed hyparameters from user input for downstream simulation pipeline
    # TODO: Need to calculate these values afterwards when initialize Ax
    @computed_field(return_type=int, description="Total Condensin in the system.")
    @property
    def num_condensin_total(self) -> int:
        return self.num_condensin_1 + self.num_condensin_2

    @computed_field(
        return_type=float, description="Fraction of Condensin 1 in the system."
    )
    @property
    def ratio_condensin_1(self) -> float:
        if self.num_condensin_total > 0:
            return self.num_condensin_1 / self.num_condensin_total
        else:
            return 0

    @computed_field(
        return_type=float, description="Fraction of Condensin 2 in the system."
    )
    @property
    def ratio_condensin_2(self) -> float:
        if self.num_condensin_total > 0:
            return 1 - self.ratio_condensin_1
        else:
            return 0

    @computed_field(
        return_type=Tuple[float, float], description="Tuple of Condensin fractions."
    )
    @property
    def ratio_condensin_list(self) -> Tuple[float, float]:
        return (self.ratio_condensin_1, self.ratio_condensin_2)

    @computed_field(
        return_type=Tuple[Tuple[float, float], Tuple[float, float]],
        description="Matrix of condensin stall time.",
    )
    @property
    def condensin_stall_time_matrix(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.condensin_1_stall_time_list, self.condensin_2_stall_time_list)

    @computed_field(
        return_type=Tuple[Tuple[float]], description="Attraction coefficient matrix."
    )
    @property
    def attraction_coefficient_matrix(self) -> Tuple[Tuple[float]]:
        return ((self.attraction_coefficient,),)

    @computed_field(return_type=List[int], description="Tuple of monomer types.")
    @property
    def monomer_type_list(self) -> List[int]:
        return [0 for _ in range(self.num_monomers)]

    # --- bookkeeping
    condition_name: str = ""
    out_dir: str = "runs"
    start_idx_replicate: int = 0

    # --- constant parameters for proper internal working of simulation
    _num_condensin_types: int = PrivateAttr(2)
    _num_CTCF: int = PrivateAttr(1)
    _CTCF_sites: List[int] = PrivateAttr([1])
    _CTCF_site_probability: List[int] = PrivateAttr([1])
    _CTCF_site_direction: List[int] = PrivateAttr([1])
    _CTCF_bound_lifetime: int = PrivateAttr(0)
    _CTCF_unbound_lifetime: int = PrivateAttr(1)
    _CTCF_condensin_bound_lifetime: int = PrivateAttr(
        1
    )  # Condensin does not stall on CTCFs
    _condensin_CTCF_bound_lifetime_list: Tuple[int, int] = PrivateAttr(
        (0, 0)
    )  # Condensin does not stall on CTCFs

    # --- validator
    @field_validator("condensin_speed_list")
    def check_condensin_speed_list(cls, v):
        if len(v) != 2:
            raise ValueError("condensin_speed_list must contain exactly 2 values")
        if any(val <= 0 for val in v):
            raise ValueError("All speeds must be > 0")
        return v

    @field_validator("condensin_speed_sd_list")
    def check_condensin_speed_sd_list(cls, v):
        if len(v) != 2:
            raise ValueError("condensin_speed_sd_list must contain exactly 2 values")
        if any(val < 0 for val in v):
            raise ValueError("All speed standard deviations must be >= 0")
        return v

    @field_validator("condensin_1_stall_time_list")
    def check_condensin_1_stall_time_list(cls, v):
        if len(v) != 2:
            raise ValueError(
                "condensin_1_stall_time_list must contain exactly 2 values"
            )
        if any(val < 1 for val in v):
            raise ValueError("All stall time must be >= 1")
        if v[1] != 1:
            print("Detecting Condensin 1 stall time upon Condensin 2 greater than 1...")
            print("Fixing such stall time to be 1 (no stall)")
            v[1] = 1
        return v

    @field_validator("condensin_2_stall_time_list")
    def check_condensin_2_stall_time_list(cls, v):
        if len(v) != 2:
            raise ValueError(
                "condensin_2_stall_time_list must contain exactly 2 values"
            )
        if any(val < 1 for val in v):
            raise ValueError("All stall time must be >= 1")
        return v

    @field_validator("condensin_stall_probability_list")
    def check_condensin_stall_probability_list(cls, v):
        if len(v) != 2:
            raise ValueError(
                "condensin_stall_probability_list must contain exactly 2 values"
            )
        if any(val < 0 for val in v):
            raise ValueError("All stall time must be between 0 and 1")
        if any(val > 1 for val in v):
            raise ValueError("All stall time must be between 0 and 1")
        return v

    @field_validator("centromere_range_list", mode="after")
    def check_centromere_range_list(cls, v, info):
        num_monomers = info.data["num_monomers"]
        if len(v) != 2:
            raise ValueError("centromere_range_list must contain exactly 2 values")
        if v[0] > v[1]:
            raise ValueError("Lower bound must be <= the upper bound.")
        if any(val >= num_monomers for val in v):
            raise ValueError("Centromere bound must be within the polymer")
        return v

    @field_validator("condensin_bound_lifetime_list")
    def check_condensin_bound_lifetime_list(cls, v):
        if len(v) != 2:
            raise ValueError(
                "condensin_bound_lifetime_list must contain exactly 2 values"
            )
        if any(val < 1 for val in v):
            raise ValueError("All stall time must be >= 1")
        return v

    @field_validator("condensin_unbound_lifetime_list")
    def check_condensin_unbound_lifetime_list(cls, v):
        if len(v) != 2:
            raise ValueError(
                "condensin_unbound_lifetime_list must contain exactly 2 values"
            )
        if any(val < 1 for val in v):
            raise ValueError("All stall time must be >= 1")
        return v

    @field_validator(
        "objective_BothCond_cond1",
        "objective_BothCond_cond2",
        "objective_Cond1Only_cond",
        "objective_Cond2Only_cond2",
        mode="after",
    )
    def check_objective_inputs(cls, v, info):
        simulation_type = info.data["simulation_type"]
        objective_calculation_option = info.data["objective_calculation_option"]
        objective_calculation_timestep = info.data["objective_calculation_timestep"]
        error_msg = f"For {simulation_type} simulation with {objective_calculation_option} objective calculation, "

        if simulation_type == "1D":
            if objective_calculation_option == "Last timepoint":
                if type(v) not in [int, float]:
                    raise ValueError(
                        error_msg + "input objective should either be int or float"
                    )
            elif objective_calculation_option == "All timepoints":
                if type(v) is not tuple:
                    raise ValueError(
                        error_msg + "input objective should either be an array"
                    )
                if objective_calculation_timestep is None:
                    raise ValueError(
                        error_msg
                        + "an array of timesteps for calculating objectives is required"
                    )
                if len(v) != len(objective_calculation_timestep):
                    raise ValueError(
                        error_msg
                        + "len(objective_calculation_timestep) must equal to len(objective)"
                    )

        elif simulation_type == "3D":
            if objective_calculation_option == "Last timepoint":
                if type(v) is not tuple:
                    raise ValueError(
                        error_msg + "input objective should either be an array"
                    )
            elif objective_calculation_option == "All timepoints":
                if type(v) is not tuple:
                    raise ValueError(
                        error_msg + "input objective should either be an array"
                    )
                if objective_calculation_timestep is None:
                    raise ValueError(
                        error_msg
                        + "an array of timesteps for calculating objectives is required"
                    )
                if len(v) != len(objective_calculation_timestep):
                    raise ValueError(
                        error_msg
                        + "len(objective_calculation_timestep) must equal to len(objective)"
                    )
        return v

    @field_validator(
        "cond1_speed_range",
        "cond2_speed_range",
        "cond1_cond1_stall_time_range",
        "cond1_cond2_stall_time_range",
        "cond2_cond1_stall_time_range",
        "cond2_cond2_stall_time_range",
        "cond1_num_range",
        "cond2_num_range",
    )
    def check_range(cls, v):
        if len(v) != 2:
            raise ValueError("Range must contain exactly 2 values")
        if any(val <= 0 for val in v):
            raise ValueError("All speeds must be > 0")
        if v[0] > v[1]:
            raise ValueError("Lower bound must be less than or equal to upper bound")
        return v

    @field_validator(
        "cond1_bound_lifetime_range",
        "cond2_bound_lifetime_range",
        "cond1_unbound_lifetime_range",
        "cond2_unbound_lifetime_range",
    )
    def check_range_lifetime(cls, v):
        if len(v) != 2:
            raise ValueError("Range must contain exactly 2 values")
        if any(val < 1 for val in v):
            raise ValueError("All speeds must be >= 1")
        if v[0] > v[1]:
            raise ValueError("Lower bound must be less than or equal to upper bound")
        return v

    @field_validator("cond1_stall_probability_range", "cond2_stall_probability_range")
    def check_range_probability(cls, v):
        if len(v) != 2:
            raise ValueError("Range must contain exactly 2 values")
        if any(val < 1 for val in v):
            raise ValueError("All speeds must be >= 1")
        if v[0] > v[1]:
            raise ValueError("Lower bound must be less than or equal to upper bound")
        return v
