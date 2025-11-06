# run_opt.py
from __future__ import annotations
from pathlib import Path
from opt_config import OptConfig
from typing import Literal
from scipy import stats

# TODO: change this to import from subfolder
# from simulation.md_backend_slurm import run_md_slurm
# from simulation.sim_config import SimConfig
# from simulation.run_sim_LE import simulate_LE
# from simulation.run_sim_MD import simulate_MD

# HACK: I copied these files for now
import matplotlib

matplotlib.use("Agg")

from md_backend_slurm import run_md_slurm
from sim_config import SimConfig
from run_sim_LE import simulate_LE
from run_sim_MD import simulate_MD

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(name)s] %(levelname)s:%(message)s"
)
log = logging.getLogger(__name__)


def safe_savefig(fig, path, *, dpi=300, bbox_inches="tight"):
    try:
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    except Exception as e:
        log.warning("Plot save failed for %s: %r", path, e)
    finally:
        plt.close(fig)


# Parellelize this function
def run_LE(
    sim_cfg: SimConfig,
    opt_cfg: OptConfig,
    run_dir: Path,
    run_id: int,
    condition: Literal["BothCond", "Cond1Only", "Cond2Only"],
) -> float:
    curr_run_dir = run_dir / condition
    curr_run_dir.mkdir(parents=True, exist_ok=True)

    simulate_LE(sim_cfg, curr_run_dir, run_id)

    condensin_pos_files = curr_run_dir.glob(f"SMC_pos_{run_id}.npy")
    condensin_props_files = curr_run_dir.glob(f"SMC_props_{run_id}.npy")

    condensin_pos_array = np.concatenate(
        [np.load(p, mmap_mode="r") for p in condensin_pos_files], axis=0
    )
    condensin_props_array = np.concatenate(
        [np.load(p, mmap_mode="r") for p in condensin_props_files], axis=0
    )

    num_monomers = sim_cfg.num_monomers
    num_timesteps, num_condensins, _ = condensin_pos_array.shape
    # 0-1 matrix represents if monomers have been occupied by condensin arms
    loop_occupancy_array = np.zeros((num_timesteps, num_monomers), dtype=int)

    for idx_timestep, timestep in enumerate(condensin_pos_array):
        if timestep is None:
            continue
        ks = np.concatenate(timestep)  # flatten if nested
        # extract where the condensin arms and their left and right neighbors are
        neighbors = np.unique(np.clip(np.r_[ks - 1, ks, ks + 1], 0, num_monomers - 1))
        # mark where the condensins are in each timestep
        loop_occupancy_array[idx_timestep, neighbors] = 1

    # Plot kymograph of condensin occupancy
    kymograph_fpath = curr_run_dir / f"kymograph_{run_id}.png"
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(
        loop_occupancy_array[:10000, :], aspect="auto", cmap="gnuplot2", vmin=0, vmax=1
    )
    ax.set_xlim(2000, 6000)
    safe_savefig(fig, kymograph_fpath, dpi=300, bbox_inches="tight")

    # Plot loop length distribution at last timepoint
    # TODO: log objective functions for each run

    simulation_type = opt_cfg.simulation_type
    objective_calculation_timestep = opt_cfg.objective_calculation_timestep

    if simulation_type == "1D":
        if objective_calculation_timestep is None:
            timepoint_list = [-1]
        else:
            timepoint_list = objective_calculation_timestep
    elif simulation_type == "MD":
        timepoint_list = [-1]

    loss_total = 0
    if condition == "BothCond":
        condensin_types = [0, 1]
        target_value_list = [
            opt_cfg.objective_BothCond_cond1,
            opt_cfg.objective_BothCond_cond2,
        ]
    elif condition == "Cond1Only":
        condensin_types = [0]
        target_value_list = [opt_cfg.objective_Cond1Only_cond1, None]
    elif condition == "Cond2Only":
        condensin_types = [1]
        target_value_list = [None, opt_cfg.objective_Cond2Only_cond2]

    loss_records = []
    for idx_condensin_type in condensin_types:
        for timepoint in timepoint_list:
            curr_condensin_pos = condensin_pos_array[timepoint, :]

            # condensin_props_array[..., ..., 0] records the bound status
            # can be either 0 (unbound) or 1 (bound)
            condensin_bound_bool = condensin_props_array[timepoint, :, 0] == 1
            # condensin_props_array[..., ..., -1] records the condensin type
            # can be either 0 (condensin 1) or 1 (condensin 2)
            condensin_type_bool = (
                condensin_props_array[timepoint, :, -1] == idx_condensin_type
            )

            condensin_bool = (condensin_type_bool) & (condensin_bound_bool)

            # 1D location of condensin of interest
            bound_condensin_pos = curr_condensin_pos[condensin_bool]
            # Calculate loop size
            loop_size = np.squeeze(np.diff(bound_condensin_pos), axis=1)

            # Plot loop size distribution
            loop_size_fpath = (
                curr_run_dir
                / f"loop_size_{run_id}_Step{timepoint}_Cond{idx_condensin_type + 1}.png"
            )
            fig, ax = plt.subplots()
            ax.hist(loop_size)
            safe_savefig(fig, loop_size_fpath, dpi=300, bbox_inches="tight")

            curr_target = target_value_list[idx_condensin_type]
            if idx_condensin_type == 0:
                measurement = np.mean(loop_size) / 1e3  # Change to Mb
            elif idx_condensin_type == 1:
                measurement = stats.mode(loop_size)[0] / 1e3  # Change to Mb

            loss_fn = (measurement - curr_target) ** 2
            loss_total += loss_fn
            loss_records.append(
                {
                    "trial_id": run_id,
                    "condition": condition,
                    "condensin_type": idx_condensin_type,
                    "timepoint": timepoint,
                    "measurement": float(measurement),
                    "target": float(curr_target),
                    "loss_fn": float(loss_fn),
                }
            )

    df = pd.DataFrame(loss_records)
    df["loss_total"] = df["loss_fn"].sum()
    df["loss_mean"] = df["loss_fn"].mean()

    loss_mean = df["loss_fn"].mean()

    csv_fpath = curr_run_dir / f"{condition}_Trial{run_id}_Objective1D.csv"
    df.to_csv(csv_fpath, index=False)

    log.info(f"Saved losses for run {run_id} ({condition}) to {csv_fpath}")

    # If simulation_type == '1D' then we can delete all the .npy files and .h5 because it is not needed downstream
    if simulation_type == "1D":
        npy_files = curr_run_dir.glob("*.npy")
        for file in npy_files:
            file.unlink()
            log.info(f"Deleted: {file}")

    return loss_mean


def downsampling_LE(cfg: OptConfig, run_dir: Path, run_id: int) -> None:
    condensin_pos_files = run_dir.glob(f"SMC_pos_{run_id}.npy")

    condensin_pos_array = np.concatenate(
        [np.load(p, mmap_mode="r") for p in condensin_pos_files], axis=0
    )

    # This make sure that the monomer idx of second sister is not the same as the monomer idx of the first sister
    # This will help assign the force field in the MD step
    num_monomers = cfg.num_monomers
    condensin_pos_array += int(num_monomers * run_id)

    # clip such that the index is greater than 0
    condensin_pos_array = np.clip(condensin_pos_array, 1, None)

    num_timesteps, num_condensin, num_heads = condensin_pos_array.shape
    downsampling_ratio = 6

    # Save to H5py format for polychrom compatibility
    h5py_fpath = run_dir / f"LEFPositions_{run_id}.h5"
    with h5py.File(h5py_fpath, mode="w") as f:
        dset = f.create_dataset(
            "positions",
            shape=(num_timesteps // downsampling_ratio, num_condensin, num_heads),
            dtype=np.int32,
            compression="gzip",
        )
        dset[:] = condensin_pos_array[::downsampling_ratio]
        f.attrs["N"] = num_monomers
        f.attrs["LEFNum"] = num_condensin


# TODO: make sure to delete *,npy files after calculation to prevent memory leak


def run(cfg_payload: dict, run_dir: Path) -> None:
    cfg = OptConfig.model_validate(cfg_payload)

    num_sister_chromatids = cfg.num_sister_chromatids
    # backend = cfg.backend
    gpu_device = cfg.gpu_device
    num_trials = cfg.num_trials
    simulation_type = cfg.simulation_type
    objective_calculation_option = cfg.objective_calculation_option

    opt_result_fpath = run_dir / "opt_result.json"

    log.info(f"Number of Bayesian trials is {num_trials}")
    log.info(f"Simulation type is {simulation_type}")
    log.info(f"Calculate objective functions at {objective_calculation_option}")
    log.info(f"GPU device is {gpu_device}")

    if simulation_type == "1D":
        objective = "-loss_BothCond, -loss_Cond1Only, -loss_Cond2Only"  # Trying to minimize loss
    elif simulation_type == "MD":
        # log scale correlation
        objective = "reward_BothCond, reward_Cond1Only, reward_Cond2Only"  # trying to maximize score/correlation

    # Now define a new SimConfig object for downstream run
    # Transfer hyperparameters from OptConfig
    # log.debug(SimConfig.model.fields.keys())
    shared_fields = SimConfig.model_fields.keys() & OptConfig.model_fields.keys()

    sim_cfg = SimConfig(**{f: getattr(cfg, f) for f in shared_fields})

    if opt_result_fpath.exists():
        log.info("Found previous Bayesian optimization run in this directory")
        log.info("Loading the previous result...")
        client = Client.load_from_json_file(filepath=opt_result_fpath)
        client_summary = client.summarize()
        prev_num_trials = client_summary.trial_index.iloc[0]
        log.info(f"Found {prev_num_trials} previous Bayesian attempts")
        prev_num_trials += 1  # Next possible trial
        log.info("Success")
    else:
        log.info("No previous Bayesian optimization run found in this directory")
        log.info("Create a new client...")
        client = Client()
        prev_num_trials = 0
        log.info("Success")

        parameters = [
            RangeParameterConfig(
                name="cond1_speed", parameter_type="float", bounds=cfg.cond1_speed_range
            ),
            RangeParameterConfig(
                name="cond2_speed", parameter_type="float", bounds=cfg.cond2_speed_range
            ),
            RangeParameterConfig(
                name="cond1_cond1_stall_time",
                parameter_type="float",
                bounds=cfg.cond1_cond1_stall_time_range,
            ),
            RangeParameterConfig(
                name="cond1_cond2_stall_time",
                parameter_type="float",
                bounds=cfg.cond1_cond2_stall_time_range,
            ),
            RangeParameterConfig(
                name="cond2_cond1_stall_time",
                parameter_type="float",
                bounds=cfg.cond2_cond1_stall_time_range,
            ),
            RangeParameterConfig(
                name="cond2_cond2_stall_time",
                parameter_type="float",
                bounds=cfg.cond2_cond2_stall_time_range,
            ),
            RangeParameterConfig(
                name="cond1_stall_probability",
                parameter_type="float",
                bounds=cfg.cond1_stall_probability_range,
            ),
            RangeParameterConfig(
                name="cond2_stall_probability",
                parameter_type="float",
                bounds=cfg.cond2_stall_probability_range,
            ),
            RangeParameterConfig(
                name="cond1_num", parameter_type="int", bounds=cfg.cond1_num_range
            ),
            RangeParameterConfig(
                name="cond2_num", parameter_type="int", bounds=cfg.cond2_num_range
            ),
            RangeParameterConfig(
                name="cond1_bound_lifetime",
                parameter_type="float",
                bounds=cfg.cond1_bound_lifetime_range,
            ),
            RangeParameterConfig(
                name="cond2_bound_lifetime",
                parameter_type="float",
                bounds=cfg.cond2_bound_lifetime_range,
            ),
            RangeParameterConfig(
                name="cond1_unbound_lifetime",
                parameter_type="float",
                bounds=cfg.cond1_unbound_lifetime_range,
            ),
            RangeParameterConfig(
                name="cond2_unbound_lifetime",
                parameter_type="float",
                bounds=cfg.cond2_unbound_lifetime_range,
            ),
        ]

        client.configure_experiment(parameters=parameters)
        client.configure_optimization(objective=objective)

    # TODO: incorporate Ax parameter nomination in the pipeline
    # TODO: check type of objective function
    # TODO: define json filepath
    # Idea: we need to define cfg for 1D simulation based on Ax parameter nomination

    for idx_sister in range(num_sister_chromatids):
        log.info(
            f"Simulating sister {idx_sister + 1} out of {num_sister_chromatids}..."
        )

        if sim_cfg.num_condensin_total == 0:
            log.info("Number of total condensin = 0. No 1D Loop Extrusion Simulation.")
            raise ValueError(
                "No Bayesian Optimization performed since there is no Condensin in the system"
            )

        for idx_trial in range(num_trials):
            idx_trial += prev_num_trials

            log.info(
                f"Current optimization round = {idx_trial + 1} out of {num_trials + prev_num_trials}..."
            )
            # Nominate and define parameters
            trial = client.get_next_trials(max_trials=1)

            trial_params = list(trial.values())[0]

            # Now transfer nominated parameters to SimConfig object
            sim_cfg.condensin_speed_list = (
                trial_params["cond1_speed"],
                trial_params["cond2_speed"],
            )
            sim_cfg.condensin_1_stall_time_list = (
                trial_params["cond1_cond1_stall_time"],
                trial_params["cond1_cond2_stall_time"],
            )
            sim_cfg.condensin_2_stall_time_list = (
                trial_params["cond2_cond1_stall_time"],
                trial_params["cond2_cond2_stall_time"],
            )
            sim_cfg.condensin_stall_probability_list = (
                trial_params["cond1_stall_probability"],
                trial_params["cond2_stall_probability"],
            )
            sim_cfg.num_condensin_1 = trial_params["cond1_num"]
            sim_cfg.num_condensin_2 = trial_params["cond2_num"]
            sim_cfg.condensin_bound_lifetime_list = (
                trial_params["cond1_bound_lifetime"],
                trial_params["cond2_bound_lifetime"],
            )
            sim_cfg.condensin_unbound_lifetime_list = (
                trial_params["cond1_unbound_lifetime"],
                trial_params["cond2_unbound_lifetime"],
            )
            sim_cfg_Cond1Only = sim_cfg.model_copy(deep=True)
            sim_cfg_Cond1Only.num_condensin_2 = 0
            sim_cfg_Cond2Only = sim_cfg.model_copy(deep=True)
            sim_cfg_Cond2Only.num_condensin_1 = 0
            log.info("Simulating 1D Loop Extrusion...")
            # TODO: paralellize and run_dir has to be different in different conditions
            # change cfg accordingly
            with ThreadPoolExecutor(max_workers=3) as ex:
                futures = {
                    ex.submit(
                        run_LE, sim_cfg, cfg, run_dir, idx_trial, "BothCond"
                    ): "loss_BothCond",
                    ex.submit(
                        run_LE, sim_cfg_Cond1Only, cfg, run_dir, idx_trial, "Cond1Only"
                    ): "loss_Cond1Only",
                    ex.submit(
                        run_LE, sim_cfg_Cond2Only, cfg, run_dir, idx_trial, "Cond2Only"
                    ): "loss_Cond2Only",
                }

                obj_fn_dict = {}
                for fut in as_completed(futures):
                    metric = futures[fut]
                    obj_fn_dict[metric] = float(fut.result())

            # If simulation_type == '1D', save data and continue to the next trial
            if simulation_type == "1D":
                client.complete_trial(trial_index=idx_trial, raw_data=obj_fn_dict)
                log.info(f"Completed trial {idx_trial} with raw_data={obj_fn_dict}")
                client.save_to_json_file(opt_result_fpath)
                continue

            # TODO: work MD optimization part
            # TODO: work on local
            # TODO: work on slurm - this will be harder
            else:
                raise ValueError(
                    "Current version of optimization module does not support MD fitting yet."
                )

            # TODO: make sure that `1D` does not run MD simulation and return loss function right away
            # And we have to clean up the files after the simulation run
            # Because MD depends on it

            # log.info("Saving h5 files for downstream MD simulation...")
            # downsampling_LE(cfg, run_dir, idx_trial)

            # if backend == "local":
            #     log.info("Simulating molecular dynamics...")
            #     start_MD_time = time.time()
            #     simulate_MD(cfg, run_dir)
            #     end_MD_time = time.time()
            #     log.info("Finished!")
            # elif backend == "slurm":
            #     if not cfg.slurm:
            #         raise SystemExit("backend='slurm' requires cfg.slurm to be set")
            #     log.info("Simulating molecular dynamics...")
            #     start_MD_time = time.time()
            #     run_md_slurm(cfg, run_dir)
            #     end_MD_time = time.time()
            #     log.info("Finished!")
            # else:
            #     raise SystemExit(f"Unknown backend: {cfg.backend}")

            # runtime_MD = end_MD_time - start_MD_time
            # runtime_MD_fpath = run_dir / "runtime_MD_log.json"

            # record = {
            #     "timestamp": datetime.now().isoformat(),
            #     "runtime": runtime_MD,
            #     "GPU_device": cfg.gpu_device,
            # }

            # with open(runtime_MD_fpath, "a") as f:
            #     f.write(json.dumps(record) + "\n")

            # log.info(f"MD runtime logged to {runtime_MD_fpath}")


if __name__ == "__main__":
    passs
