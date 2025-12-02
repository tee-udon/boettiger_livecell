# boettiger-livecell

Simulation engine for mitotic chromosome simulation powered by [OpenMM](https://openmm.org) and [Polychrom](https://github.com/open2c/polychrom).\
Forked from [boettiger-mitotic](https://github.com/tee-udon/boettiger-mitotic.git).

## Running Simulations

### Boettiger Server

Run Anaconda Powershell Prompt **as administrator**. (`conda` requires admin access). Then activate `tee` virtual environment using the following command:

```Powershell
conda activate tee
```

If activation is successful, your Powershell Prompt should starts with `(tee)`:

```Powershell
(tee) PS: [directory]> 
```

Because this virtual environment contains all the required dependencies, we are ready to run simulation. Running simulation is very simple: you just pass the config file (`*.yaml`) that contains hyperparameters for the simulation to the main simulation driver `boettiger-livecell/main.py`. `main.py` will validate the validity of hyperparameters, coordinate LE and MD simulations, and output the results automatically. `main.py` accepts path to config file through `--config` flag.

To simulate Rouse polymer, run the following command:

Server 2

```Powershell
python "F:\Tee\boettiger-livecell\simulation\main.py" --config "F:\Tee\boettiger-livecell\tests\simulation\config_rouse_slurm.yaml"
```

This will simulate 10 independent Rouse polymers and output it in folder indicated in `out_dir` parameter in the config file, which is `F:\Tee\LiveCellSimulation\Dataset\20251121_LiveCellSimulation` in this example.

Anecdotally, this simulation should not take more than 20 minutes if your GPU does not share its resources with other GPU-intensive jobs.

## Hyperparameters

All parameters are defined using a config `yaml` file.  

### Example for Rouse Polymer:

```yaml
# Bookkeeping
out_dir: /mnt/home/tudomlumleart/ceph/04_MitoticChromosome/dataset/simulations/20251121_LiveCellSimulation
condition_name: RouseTest
start_idx_replicate: 0

# Number of independent replicates per run
num_replicates: 1

# Core loop extruion simulation hyperparameters
num_monomers: 10000 # 10 Mb 

# Cohesin-related parameters
num_cohesin: 0 
cohesin_speed: 1
cohesin_speed_sd: 0
cohesin_stall_time: 1
cohesin_stall_probability: 1
cohesin_bound_lifetime: 1
cohesin_unbound_lifetime: 1
extrusion_side: 2
cohesin_loading_probability_list: null

# CTCF-related parameters 
ctcf_site_location_list: null
ctcf_site_direction_list: null
ctcf_site_stall_probability_list: null
ctcf_site_stall_time_list: null

# Number of LE steps
num_LE_steps: 10000
num_LE_steps_init: 0

# MD simulation parameters
monomer_type_list: null
attraction_coefficient_matrix: null
repulsion: 0
equilibration_timestep: 100000 
num_MD_steps_per_LE: 1000
attraction_radius: 0 # This has to be 0 when there is no attraction!
density: 0.24
collision_rate: 0.03
initial_conformation: random_walk

# Plot for sanity check
plot_LE: true 

# backend 
backend: local

# slurm options - in case running on high-computing cluster
slurm:
  partition: gpu
  time: "01:00:00"
  gpus: "1"
  cpus_per_task: 1
  mem: "64G"
  nodes: 1
  ntasks: 1
  env_setup: |
    module load modules/2.3-20240529 cuda/12.3.2
    conda activate py-hoomd2
  python: python
  wait: false
```

### Example for Crumpled polymer with Extrusion

```yaml
# Bookkeeping
out_dir: /mnt/home/tudomlumleart/ceph/04_MitoticChromosome/dataset/simulations/20251121_LiveCellSimulation
condition_name: CrumpledExtrusionTest 
start_idx_replicate: 0

# Number of independent replicates per run
num_replicates: 1

# Core loop extruion simulation hyperparameters
num_monomers: 10000 # 10 Mb 

# Cohesin-related parameters
num_cohesin: 100
cohesin_speed: 1
cohesin_speed_sd: 0
cohesin_stall_time: 10
cohesin_stall_probability: 1
cohesin_bound_lifetime: 1000
cohesin_unbound_lifetime: 1
extrusion_side: 2
cohesin_loading_probability_list: null

# CTCF-related parameters 
ctcf_site_location_list: [4000, 6000]
ctcf_site_direction_list: ['right', 'left']
ctcf_site_stall_probability_list: [1, 1]
ctcf_site_stall_time_list: [100000, 100000]

# Number of LE steps
num_LE_steps: 10000
num_LE_steps_init: 0

# MD simulation parameters
monomer_type_list: null
attraction_coefficient_matrix: null
repulsion: 1
equilibration_timestep: 100000 
num_MD_steps_per_LE: 100
attraction_radius: 0 # This has to be 0 when there is no attraction!
density: 0.24
collision_rate: 0.03
initial_conformation: crumpled
PBC_box: False # Adding PBC_box and spherical confinement can crash fail the energy minimization process 
confinement: spherical 

# Plot for sanity check
plot_LE: true 

# backend 
backend: local

# slurm options
slurm:
  partition: gpu
  time: "01:00:00"
  gpus: "1"
  cpus_per_task: 1
  mem: "64G"
  nodes: 1
  ntasks: 1
  env_setup: |
    module load modules/2.3-20240529 cuda/12.3.2
    conda activate py-hoomd2
  python: python
  wait: false
```

### Simulation Configuration (`SimConfig`)

### General Settings

| Name                  | Type  | Default  | Constraints | Description                                  |
| --------------------- | ----- | -------- | ----------- | -------------------------------------------- |
| `num_replicates`      | `int` | `1`      | `>= 1`      | Number of independent replicates per run.    |
| `out_dir`             | `str` | `"runs"` | –           | Output directory for all runs.               |
| `condition_name`      | `str` | `""`     | –           | Optional label for the simulation condition. |
| `start_idx_replicate` | `int` | `0`      | –           | Starting index for replicate numbering.      |

### Loop Extrusion (LE) Parameters

| Name                               | Type           | Default  | Constraints                   | Description                                                                         |
| ---------------------------------- | -------------- | -------- | ----------------------------- | ----------------------------------------------------------------------------------- |
| `num_monomers`                     | `int`          | `100000` | `> 0`                         | Number of monomers (~1 kb per monomer).                                             |
| `cohesin_speed`                    | `float`        | `1.0`    | `>= 1`                        | Cohesin speed (monomers per timestep).                                              |
| `cohesin_speed_sd`                 | `float`        | `0.0`    | `>= 0`                        | Std. deviation of cohesin speed.                                                    |
| `cohesin_stall_time`               | `float`        | `1.0`    | `>= 1`                        | Expected timesteps for cohesin stall on collision.                                  |
| `cohesin_stall_probability`        | `float`        | `1.0`    | `0 ≤ p ≤ 1`                   | Probability of stall on collision.                                                  |
| `num_cohesin`                      | `int`          | `1000`   | `>= 0`                        | Number of cohesin molecules.                                                        |
| `cohesin_bound_lifetime`           | `float`        | `100000` | `>= 1`                        | Expected lifetime of bound cohesin.                                                 |
| `cohesin_unbound_lifetime`         | `float`        | `1`      | `>= 1`                        | Expected lifetime of unbound cohesin.                                               |
| `extrusion_side`                   | `Literal[1,2]` | `2`      | –                             | `1` = unidirectional, `2` = bidirectional extrusion.                                |
| `num_LE_steps`                     | `int`          | `1800`   | `>= 0`                        | Total number of LE steps (default ≈ 1800 frames).                                        |
| `num_LE_steps_init`                | `int`          | `0`      | `>= 0`                        | LE steps run before exporting (for steady state).                                   |
| `cohesin_loading_probability_list` | `list[float]?` | `null`   | len = `num_monomers`, sum = 1 | Per-monomer cohesin loading probabilities. If `null`, uniform distribution is used. |

### CTCF Site Parameters

| Name                               | Type                           | Default | Constraints                | Description                                                                                    |
| ---------------------------------- | ------------------------------ | ------- | -------------------------- | ---------------------------------------------------------------------------------------------- |
| `ctcf_site_location_list`          | `list[int]?`                   | `null`  | `0 ≤ i < num_monomers`     | Locations of CTCF sites along the genome.                                                      |
| `ctcf_site_direction_list`         | `list["left","right","both"]?` | `null`  | len = number of CTCF sites | Direction of each CTCF site. If `null`, all are `"both"`.                                      |
| `ctcf_site_stall_probability_list` | `list[float]?`                 | `null`  | `0 ≤ p ≤ 1`                | Stall probability for each CTCF site. If `null`, all are 1.                                    |
| `ctcf_site_stall_time_list`        | `list[float]?`                 | `null`  | `>= 1`                     | Expected stall time for each CTCF site. If `null`, very large value is used (permanent stall). |

### Molecular Dynamics (MD) Parameters

| Name                            | Type                                             | Default      | Constraints          | Description                                                              |
| ------------------------------- | ------------------------------------------------ | ------------ | -------------------- | ------------------------------------------------------------------------ |
| `monomer_type_list`             | `list[int]?`                                     | `null`       | len = `num_monomers` | Monomer types. If `null`, all monomers are treated as the same type.     |
| `attraction_coefficient_matrix` | `list[list[float]]?`                             | `null`       | square matrix        | Inter-type attraction coefficients. If `null`, no attraction is assumed. |
| `repulsion`                     | `float`                                          | `5.0`        | `>= 0`               | Repulsion constant between monomers.                                     |
| `equilibration_timestep`        | `int`                                            | `1500`       | `>= 1`               | MD timesteps used to relax the initial conformation.                     |
| `num_MD_steps_per_LE`           | `int`                                            | `1000`       | `>= 1`               | MD timesteps between LE updates.                                         |
| `attraction_radius`             | `float`                                          | `0`          | `>= 0`               | Radius over which monomer attraction acts.                               |
| `density`                       | `float`                                          | `0.24`       | `0 < density ≤ 1`    | Initial polymer density relative to the enclosing box.                   |
| `collision_rate`                | `float`                                          | `0.03`       | `> 0`                | Rate of monomer–monomer collisions.                                      |
| `smc_bond_dist`                 | `float`                                          | `0.5`        | `> 0`                | Mean bond length between condensin arms.                                 |
| `smc_bond_wiggle_dist`          | `float`                                          | `0.2`        | `> 0`                | Std. deviation of condensin bond length.                                 |
| `gpu_device`                    | `str`                                            | `"0"`        | –                    | GPU device used for the MD simulation.                                   |
| `initial_conformation`          | `"random_walk" \| "random_walk_z" \| "crumpled"` | `"crumpled"` | –                    | Initial polymer conformation.                                            |
| `confinement`                   | `"spherical"?`                                   | `null`       | –                    | Confinement geometry.                                                    |
| `PBC_box`                       | `bool`                                           | `false`      | –                    | Enable periodic boundary conditions. |

### Backend

| Name      | Type                 | Default   | Description                                           |
| --------- | -------------------- | --------- | ----------------------------------------------------- |
| `backend` | `"local" \| "slurm"` | `"local"` | Execution backend.                                    |
| `slurm`   | `SlurmCfg?`          | `null`    | SLURM configuration (only used if `backend="slurm"`). |

### Slurm Configuration (`SlurmCfg`)

Used only when `backend="slurm"`.

| Name               | Type        | Default      | Description                                                       |
| ------------------ | ----------- | ------------ | ----------------------------------------------------------------- |
| `partition`        | `str`       | `"gpu"`      | SLURM partition to submit jobs to.                                |
| `account`          | `str?`      | `null`       | SLURM account name, if required.                                  |
| `time`             | `str`       | `"00:30:00"` | Walltime limit in `HH:MM:SS` format.                              |
| `gpus`             | `str`       | `"1"`        | GPU request, e.g. `"1"` or `"a100:1"`.                            |
| `cpus_per_task`    | `int`       | `1`          | Number of CPUs per task.                                          |
| `mem`              | `str?`      | `"64G"`      | Memory request, e.g. `"16G"`.                                     |
| `qos`              | `str?`      | `null`       | SLURM QoS if applicable.                                          |
| `nodes`            | `int`       | `1`          | Number of nodes.                                                  |
| `ntasks`           | `int`       | `1`          | Number of tasks.                                                  |
| `constraint`       | `str?`      | `null`       | Node constraints (e.g. `"a100"`).                                 |
| `chdir_to_run_dir` | `bool`      | `true`       | Change working directory to the run directory before execution.   |
| `extra_sbatch`     | `list[str]` | `[]`         | Extra raw `#SBATCH` lines (e.g. `["#SBATCH --exclusive"]`).       |
| `env_setup`        | `str?`      | `null`       | Shell commands for environment setup (modules, conda, etc.).      |
| `python`           | `str`       | `"python"`   | Python interpreter used inside the job.                           |
| `wait`             | `bool`      | `true`       | If `true`, block until the job finishes (polls `squeue`/`sacct`). |

### Plotting

| Name      | Type   | Default | Description                                    |
| --------- | ------ | ------- | ---------------------------------------------- |
| `plot_LE` | `bool` | `true`  | Plot LE simulation result for sanity checking. |

### Automatic Defaults & Validation

The following parameters are automatically generated if not provided:

- `monomer_type_list`: initialized to all zeros.
- `attraction_coefficient_matrix`: initialized to a zero matrix.
- `cohesin_loading_probability_list`: uniform distribution over monomers.
- `ctcf_site_direction_list`: defaults to "both" for all sites.
- `ctcf_site_stall_probability_list`: defaults to 1 for all sites.
- `ctcf_site_stall_time_list`: defaults to a very large value.

All lists are strictly validated for correct lengths and physical constraints.

## Acknowledgements

This codebase stands on the shoulders of giants, including the following:

Beckwith, K. S., Brunner, A., Morero, N. R., Jungmann, R., & Ellenberg, J. (2025). Nanoscale DNA tracing reveals the self-organization mechanism of mitotic chromosomes. Cell, 188(10). https://doi.org/10.1016/j.cell.2025.02.028  

https://github.com/open2c/polychrom 