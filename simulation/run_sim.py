# run_sim.py
from __future__ import annotations
from pathlib import Path
from sim_config import SimConfig
from run_sim_LE import simulate_LE
from run_sim_MD import simulate_MD
from md_backend_slurm import run_md_slurm
from datetime import datetime
import json
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(name)s] %(levelname)s:%(message)s"
)
log = logging.getLogger(__name__)


def plot_LE_result(cfg: SimConfig, run_dir: Path, run_id: int = 0) -> None:
    condensin_pos_files = run_dir.glob(f"SMC_pos_{run_id}.npy")
    condensin_props_files = run_dir.glob(f"SMC_props_{run_id}.npy")

    condensin_pos_array = np.concatenate(
        [np.load(p, mmap_mode="r") for p in condensin_pos_files], axis=0
    )
    condensin_props_array = np.concatenate(
        [np.load(p, mmap_mode="r") for p in condensin_props_files], axis=0
    )

    num_monomers = cfg.num_monomers
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
    kymograph_fpath = run_dir / f"kymograph_{run_id}.png"
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(
        loop_occupancy_array[:10000, :], aspect="auto", cmap="gnuplot2", vmin=0, vmax=1
    )
    ax.set_xlim(2000, 6000)
    plt.savefig(kymograph_fpath, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot loop length distribution at last timepoint
    num_condensin_types = 2
    for idx_condensin_type in range(num_condensin_types):
        timepoint = -1
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
            run_dir / f"loop_size_{run_id}_Cond{idx_condensin_type + 1}.png"
        )
        fig, ax = plt.subplots()
        ax.hist(loop_size)
        plt.savefig(loop_size_fpath, dpi=300, bbox_inches="tight")
        plt.close()


def downsampling_LE(cfg: SimConfig, run_dir: Path, run_id: int) -> None:
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


def run(cfg: SimConfig, run_dir: Path) -> None:
    # --- call your real simulation code here ---
    num_sister_chromatids = cfg.num_sister_chromatids
    plot_LE = cfg.plot_LE
    backend = cfg.backend
    gpu_device = cfg.gpu_device 
    log.info(f'GPU device is {gpu_device}')
    for idx_sister in range(num_sister_chromatids):
        log.info(f'Simulating sister {idx_sister+1} out of {num_sister_chromatids}...')
        
        if cfg.num_condensin_total == 0:
            log.info('Number of total condensin = 0. No 1D Loop Extrusion Simulation.')
            
        else:
            log.info('Simulating 1D Loop Extrusion...')
            simulate_LE(cfg, run_dir, idx_sister)

            if plot_LE:
                log.info('Plotting results from 1D Loop Extrusion for sanity check...')
                plot_LE_result(cfg, run_dir, idx_sister)
            else:
                log.warning("Program does not plot results. Change plot_LE to true if you want otherwise.")

            log.info('Saving h5 files for downstream MD simulation...')
            downsampling_LE(cfg, run_dir, idx_sister)

        if backend == "local":
            log.info("Simulating molecular dynamics...")
            start_MD_time = time.time()
            simulate_MD(cfg, run_dir)
            end_MD_time = time.time()
            log.info("Finished!")
        elif backend == "slurm":
            if not cfg.slurm:
                raise SystemExit("backend='slurm' requires cfg.slurm to be set")
            log.info("Simulating molecular dynamics...")
            start_MD_time = time.time()
            run_md_slurm(cfg, run_dir)
            end_MD_time = time.time()
            log.info("Finished!")
        else:
            raise SystemExit(f"Unknown backend: {cfg.backend}")

        runtime_MD = end_MD_time - start_MD_time
        runtime_MD_fpath = run_dir / "runtime_MD_log.json"

        record = {
            "timestamp": datetime.now().isoformat(),
            "runtime": runtime_MD,
            "GPU_device": cfg.gpu_device,
        }

        with open(runtime_MD_fpath, "a") as f:
            f.write(json.dumps(record) + "\n")

        log.info(f"MD runtime logged to {runtime_MD_fpath}")
