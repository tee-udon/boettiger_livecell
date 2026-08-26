"""Code for running polychrom polymer simulations using loop extrusion simulation as input.
This code is lightly modified from examples found in https://github.com/open2c/polychrom, please also see polychrom installation instructions found there, which is requried to run this script.

The polychrom package and this modified script are published under an MIT license:

MIT License

Copyright (c) 2019 Massachusetts Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software."""

import logging
import time
import numpy as np
import polychrom

from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic, create_random_walk
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI
import openmm

import h5py

from sim_config import SimConfig
from pathlib import Path


def _random_points_sphere(N):
    theta = np.random.uniform(0.0, 1.0, N)
    theta = 2.0 * np.pi * theta

    u = np.random.uniform(0.0, 1.0, N)
    u = 2.0 * u - 1.0

    return np.vstack([theta, u]).T


def create_random_walk_positivez(step_size, N):
    """
    Creates a freely joined chain of length N with step step_size.
    This function monotonically increases in small increments in z, to avoid knots/catenations in the chain.
    """

    theta, u = _random_points_sphere(N).T

    dx = step_size * np.sqrt(1.0 - u * u) * np.cos(theta)
    dy = step_size * np.sqrt(1.0 - u * u) * np.sin(theta)
    dz = (
        np.ones(shape=N) * 0.0015
    )  # 0.001*np.sqrt(np.mean(np.abs(dx)))#step_size * np.random.uniform(0.0, np.sqrt(np.mean(np.abs(dx))), N) #u

    x, y, z = np.cumsum(dx), np.cumsum(dy), np.cumsum(dz)

    return np.vstack([x, y, z]).T


class bondUpdater(object):
    def __init__(self, LEFpositions, n_beads=None):
        """
        :param smcTransObject: smc translocator object to work with
        :param n_beads: chain length. If given, LEF indices outside [0, n_beads)
            are treated as corrupt and dropped in setup() instead of being
            handed to OpenMM.
        """
        self.LEFpositions = LEFpositions
        self.n_beads = n_beads
        self.curtime = 0
        self.allBonds = []

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict

    def setup(self, bondForce, blocks=100, smcStepsPerBlock=1):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """

        if len(self.allBonds) != 0:
            raise ValueError(
                "Not all bonds were used; {0} sets left".format(len(self.allBonds))
            )

        self.bondForce = bondForce

        # precalculating all bonds
        allBonds = []

        loaded_positions = np.asarray(
            self.LEFpositions[self.curtime : self.curtime + blocks]
        )

        # Drop corrupt LEF indices instead of handing them to OpenMM. The LE
        # rarely emits a bad value: loop_pos in run_sim_LE.update_SMC_sim is
        # float64, and an unbound extruder can surface as the bit pattern of
        # the double 1.0, which saturates to INT32_MAX in the int32 h5. Because
        # setup() collects unique bonds over ALL frames, a single bad entry
        # anywhere used to abort the whole run with
        #   HarmonicBondForce: Illegal particle index for a bond: 2147483647
        # and the retry in simulate_MD then masked it. A dropped bond simply
        # reads as that extruder being unbound for that frame.
        valid = np.ones(loaded_positions.shape[:2], dtype=bool)
        if self.n_beads is not None:
            in_range = (loaded_positions >= 0) & (loaded_positions < self.n_beads)
            valid = in_range.all(axis=2)
            n_bad = int((~valid).sum())
            if n_bad:
                frames = np.unique(np.nonzero(~valid)[0])
                logging.warning(
                    "bondUpdater: dropped %d corrupt LEF bond(s) across %d frame(s) "
                    "(first frame %d, valid index range [0, %d)). "
                    "This is the known run_sim_LE loop_pos dtype bug.",
                    n_bad, frames.size, int(frames[0]), self.n_beads,
                )

        allBonds = [
            [
                (int(loaded_positions[i, j, 0]), int(loaded_positions[i, j, 1]))
                for j in range(loaded_positions.shape[1])
                if valid[i, j]
            ]
            for i in range(blocks)
        ]

        self.allBonds = allBonds
        # `sum(allBonds, [])` is QUADRATIC -- it rebuilds the accumulator on
        # every one of `blocks` concatenations, costing ~n_lefs * blocks^2 / 2
        # element copies. At 100 extruders x 50000 blocks that was ~40 min of
        # pure setup before MD started, and it scales with blocks^2, so a
        # 200000-block run would take hours. This is the same result in one
        # linear pass.
        uniq = set()
        for block in allBonds:
            uniq.update(block)
        self.uniqueBonds = list(uniq)

        # adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = (
                self.activeParamDict
                if (bond in self.curBonds)
                else self.inactiveParamDict
            )
            ind = bondForce.addBond(
                bond[0], bond[1], **paramset
            )  # changed from addBond
            self.bondInds.append(ind)
        self.bondToInd = {i: j for i, j in zip(self.uniqueBonds, self.bondInds)}

        self.curtime += blocks

        return self.curBonds, []

    def step(self, context, verbose=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError(
                "No bonds left to run; you should restart simulation and run setup  again"
            )

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print(
                "{0} bonds stay, {1} new bonds, {2} bonds removed".format(
                    len(bondsStay), len(bondsAdd), len(bondsRemove)
                )
            )
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(
                ind, bond[0], bond[1], **paramset
            )  # actually updating bonds
        self.bondForce.updateParametersInContext(
            context
        )  # now run this to update things in the context
        return self.curBonds, pastBonds


def simulate_MD(cfg: SimConfig, run_dir: Path):
    base_dir = run_dir

    repulsionEnergy = cfg.repulsion
    collision_rate = cfg.collision_rate
    attraction_radius = cfg.attraction_radius
    attractionEnergy = cfg.attraction_energy
    equilibration_timestep = cfg.equilibration_timestep
    initial_conformation = cfg.initial_conformation
    gpu_device = cfg.gpu_device
    N = cfg.num_monomers
    angle_k = cfg.angle_k

    interactionMatrix = np.array(cfg.attraction_coefficient_matrix)
    monomerTypes = np.array(cfg.monomer_type_list)

    # monomerTypes have to replicated to reflect the sister chromatid possibility
    monomerTypes = monomerTypes

    steps = cfg.num_MD_steps_per_LE
    density = cfg.density

    smcBondDist = cfg.smc_bond_dist
    smcBondWiggleDist = cfg.smc_bond_wiggle_dist

    lef_position_fpath = base_dir / "LEFPositions_0.h5"
    if lef_position_fpath.exists():
        lef_positions = h5py.File(lef_position_fpath, mode="r")
        LEFpositions = lef_positions["positions"]
        Nframes = LEFpositions.shape[0]
        milker = bondUpdater(LEFpositions, n_beads=N)

    else:
        Nframes = (
            cfg.num_LE_steps // 1
        )  # TODO: change this to subsampling ratio variable

    if initial_conformation == "crumpled":
        # Create a semi-dense non-catenated chain. After relaxation this resembles interphase chromatin.
        data = grow_cubic(N, int((N / density) ** 0.333))
    elif initial_conformation == "random_walk":
        data = create_random_walk(step_size=1, N=N)
    elif initial_conformation == "random_walk_z":
        data = create_random_walk_positivez(step_size=1, N=N)
    else:
        raise ValueError(
            "initial_conformation must either be crumpled or random_walk or random_walk_z"
        )

    # Save the initial conformation
    init_conformation_fpath = base_dir / "init_conformation.npy"
    np.save(init_conformation_fpath, data)  # Save the initial conformation

    # Save all the timepoints
    # Save positions every Nth LE step. Bonds still advance every step, so this
    # buys physical time without growing the trajectory: num_LE_steps 200000
    # with save_every_blocks 4 gives 4x the MD time at the same 50001 frames.
    # NOTE 1 frame is then save_every_blocks LE steps, so any lag axis measured
    # in frames must be rescaled before comparing runs with different values.
    saveEveryBlocks = cfg.save_every_blocks
    restartSimulationEveryBlocks = Nframes  # Do not restart.

    # assertions for easy managing code below
    assert (Nframes % restartSimulationEveryBlocks) == 0
    assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

    simInitsTotal = (Nframes) // restartSimulationEveryBlocks

    # overwrite=False so a requeued job does NOT delete the blocks it already
    # wrote; check_exists=False so a non-empty folder is not an error.
    reporter = HDF5Reporter(
        folder=base_dir,
        max_data_length=100,
        overwrite=False,
        check_exists=False,
        blocks_only=False,
    )

    # ---- preemption resume -------------------------------------------------
    # The blocks_*.h5 files ARE the checkpoint: save_every_blocks writes the
    # conformation as the run proceeds, so nothing extra needs saving -- only
    # reading back. polychrom's continue_trajectory() finds the last complete
    # block, returns its conformation, fixes the reporter's counters so new
    # output continues the numbering, and removes the partial trailing file
    # after re-buffering the blocks worth keeping.
    #
    # Physics: positions carry over exactly and the LEF stream is precomputed in
    # the h5, so only the velocities are re-drawn. This is Langevin at
    # collision_rate 0.03/ps, i.e. a velocity correlation time of ~33 ps against
    # ~16 ps per block -- roughly 2 blocks. Between preemptions there are
    # thousands of blocks, so the thermostat has already randomised the
    # velocities thousands of times over; re-drawing them at the boundary is
    # indistinguishable from what it does continuously anyway. The RNG stream
    # also differs after a resume, which is a different realisation of the same
    # stochastic process, not a bias.
    # ⚠️ OFF-BY-ONE, and it is PRE-EXISTING. The equilibration
    # `a.do_block(steps=equilibration_timestep)` below REPORTS a frame, so block
    # index 0 is the equilibration state and the LE blocks occupy 1..Nframes. A
    # completed run therefore holds Nframes+1 blocks -- which is why every
    # all_conformations.npy is one frame larger than num_LE_steps (50000 steps
    # -> 6,000,120,128 bytes = 50001 frames, not 50000). Frame 0 is separated
    # from frame 1 by equilibration_timestep MD steps rather than
    # num_MD_steps_per_LE, so lag analysis should treat it as suspect.
    # continue_trajectory() returns the LAST index, so the number of LE blocks
    # already done is last_block exactly -- no +1.
    resume_block = 0
    if list(base_dir.glob("blocks_*.h5")):
        last_block, cont = reporter.continue_trajectory()
        data = cont["pos"]
        resume_block = int(last_block)
        logging.warning(
            "RESUMING from LE block %d of %d (%.1f%% already done). Skipping "
            "LE, energy minimisation and equilibration.",
            resume_block, Nframes, 100.0 * resume_block / Nframes,
        )
    blocks_this_run = Nframes - resume_block
    if blocks_this_run <= 0:
        # ⛔ Nothing left to integrate. Must skip the MD loop ENTIRELY, not just
        # warn: milker.setup(blocks=0) loads no bonds and then pops from an
        # empty list (IndexError). Zeroing simInitsTotal falls straight through
        # to the assembly step, which is all a fully-complete run still needs.
        logging.warning("All %d LE blocks already present; assembling output "
                        "only, no MD.", Nframes)
        simInitsTotal = 0

    if cfg.PBC_box:
        box_size = int((N / density) ** 0.333)
        PBC_box = [box_size, box_size, box_size]
    else:
        PBC_box = False

    for iteration in range(simInitsTotal):
        # simulation parameters are defined below
        a = Simulation(
            platform="cuda",
            integrator="variableLangevin",
            error_tol=0.01,
            GPU=gpu_device,  # change here for gpu device
            collision_rate=collision_rate,
            N=len(data),
            reporters=[reporter],
            max_Ek=20,
            PBCbox=PBC_box,  # Do not use PCB box for mitotic chromosomes
            precision="single",
        )  # timestep not necessary for variableLangevin

        ############################## New code ##############################
        # center=False on a resume: the saved conformation is already the state
        # we are continuing, and recentring would shift it relative to the
        # coordinates the earlier blocks were written in. Internal distances are
        # unaffected either way, but keeping the frame consistent avoids a
        # discontinuity in any absolute-position analysis.
        a.set_data(data, center=(resume_block == 0))

        chain_tuple = [(0, None, False)]

        a.add_force(
            forcekits.polymer_chains(
                a,
                chains=chain_tuple,
                # By default the library assumes you have one polymer chain
                # If you want to make it a ring, or more than one chain, use self.setChains
                # self.setChains([(0,50,1),(50,None,0)]) will set a 50-monomer ring and a chain from monomer 50 to the end
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    "bondLength": 1.0,
                    "bondWiggleDistance": 0.1,  # Bond distance will fluctuate +- 0.05 on average
                },
                angle_force_func=forces.angle_force,
                angle_force_kwargs={
                    "k": angle_k
                    # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                    # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
                },
                
                # This needs to be properly parameterized
                # If repulsion is too high, for example, then the polymer cannot minimize the energy
                # Then the simulation fail
                nonbonded_force_func=forces.heteropolymer_SSW,
                nonbonded_force_kwargs={
                    "repulsionEnergy": repulsionEnergy,  # base repulsion energy for all monomers (function default is 3.0)
                    "attractionEnergy": attractionEnergy,  # base attraction energy for all monomers (function default is 3.0)
                    "attractionRadius": attraction_radius,
                    "interactionMatrix": interactionMatrix,
                    "monomerTypes": monomerTypes,
                    "extraHardParticlesIdxs": [],
                },
            )
        )

        if cfg.confinement is not None:
            if cfg.confinement == "spherical":
                a.add_force(forces.spherical_confinement(a, density=density))

        # ------------ initializing milker; adding bonds ---------
        # copied from addBond
        kbond = a.kbondScalingFactor / (smcBondWiggleDist**2)
        bondDist = smcBondDist * a.length_scale

        if cfg.num_cohesin > 0:
            activeParams = {"length": bondDist, "k": kbond}
            inactiveParams = {"length": bondDist, "k": 0}
            milker.setParams(activeParams, inactiveParams)

            # this step actually puts all bonds in and sets first bonds to be what they should be
            # Seek the LEF cursor to the resume point. bondUpdater.setup reads
            # LEFpositions[curtime : curtime + blocks], so setting curtime is all
            # that is needed to line the bond stream back up with the resumed
            # conformation -- the 1D trajectory itself is fixed on disk.
            milker.curtime = resume_block
            milker.setup(
                bondForce=a.force_dict["harmonic_bonds"],
                blocks=blocks_this_run,
            )

        # Initialize starting conformation and minimize the energy. Sometimes this does not work, so retry until the energy can be minimized.
        # ⛔ Skipped on a resume: the conformation loaded from the last block is
        # already relaxed, and minimising it again would drive it off the
        # trajectory we are continuing.
        if iteration == 0 and resume_block == 0:
            for i in range(100):
                try:
                    if i > 0:
                        if initial_conformation == "crumpled":
                            # Create a semi-dense non-catenated chain. After relaxation this resembles interphase chromatin.
                            data = grow_cubic(N, int((N / density) ** 0.333))
                        elif initial_conformation == "random_walk":
                            data = create_random_walk(step_size=1, N=N)
                        else:
                            raise ValueError(
                                "initial_conformation must either be crumpled or random_walk"
                            )
                        np.save(init_conformation_fpath, data)
                        a.set_data(
                            data, center=True
                        )  # loads a polymer, puts a center of mass at zero
                    a.local_energy_minimization(maxIterations=int(1e4))
                    break
                except openmm.OpenMMException:
                    # Log it. This was silent, and the retry below then raised a
                    # misleading "System object does not own its corresponding
                    # OpenMM object" that hid the real cause on every failure.
                    logging.exception(
                        "local_energy_minimization attempt %d raised; retrying", i
                    )
                    continue
        else:
            a._apply_forces()

        # ⛔ THE ONE THING THAT WOULD CORRUPT A RESUME. equilibration_timestep is
        # 100,000 MD steps -- a thousand normal blocks -- and milker.step() is NOT
        # called during it, so the LEF bonds are FROZEN throughout. Left
        # unguarded, every preemption would inject a "freeze the loops and let
        # the chain relax" episode into the middle of the trajectory, plus a
        # stretch of physical time absent from the LE/MD frame correspondence.
        if resume_block == 0:
            a.do_block(
                steps=equilibration_timestep
            )  # Initial equilibration steps to ensure that we start from a relaxed polymer state.

        for i in range(blocks_this_run):
            if i < blocks_this_run - 1 and cfg.num_cohesin > 0:
                curBonds, pastBonds = milker.step(
                    a.context
                )  # this updates bonds. You can do something with bonds here
            if i % saveEveryBlocks == (saveEveryBlocks - 1):
                a.do_block(steps=steps)
            else:
                a.integrator.step(
                    steps
                )  # do steps without getting the positions from the GPU (faster)

        del a

        reporter.blocks_only = False  # True  # Write output hdf5-files only for blocks

        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

    reporter.dump_data()

    f = list_URIs(str(base_dir))
    a = []
    for i in range(len(f)):
        a.append(load_URI(f[i])["pos"])
    arr = np.stack(a)

    all_conformation_fpath = base_dir / "all_conformations.npy"
    np.save(
        all_conformation_fpath, arr
    )  # Output all recorded conformations into a numpy file for simple use later.


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from config_loader import load_config

    p = argparse.ArgumentParser()
    p.add_argument("--from-dir", required=True)
    args = p.parse_args()
    run_dir = Path(args.from_dir)
    config_fpath = run_dir.parent / "config_resolved.json"
    # re-load resolved config from run_dir if you save it there
    cfg = load_config(str(config_fpath))  # your helper

    simulate_MD(cfg, run_dir)
