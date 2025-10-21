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

import time
import numpy as np
import polychrom

from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
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
    def __init__(self, LEFpositions):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.LEFpositions = LEFpositions
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

        loaded_positions = self.LEFpositions[self.curtime : self.curtime + blocks]
        allBonds = [
            [
                (int(loaded_positions[i, j, 0]), int(loaded_positions[i, j, 1]))
                for j in range(loaded_positions.shape[1])
            ]
            for i in range(blocks)
        ]

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))

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

    num_sister_chromatid = cfg.num_sister_chromatids
    repulsionEnergy = cfg.repulsion
    collision_rate = cfg.collision_rate
    attraction_radius = cfg.attraction_radius
    equilibration_timestep = cfg.equilibration_timestep
    gpu_device = cfg.gpu_device
    N = cfg.num_monomers
    N_per_sister_chromatid = N // num_sister_chromatid

    interactionMatrix = np.array(cfg.attraction_coefficient_matrix)
    monomerTypes = np.array(cfg.monomer_type_list)

    # monomerTypes have to replicated to reflect the sister chromatid possibility
    monomerTypes = np.tile(monomerTypes, num_sister_chromatid)

    steps = cfg.num_MD_steps_per_LE
    density = cfg.density

    smcBondDist = cfg.smc_bond_dist
    smcBondWiggleDist = cfg.smc_bond_wiggle_dist

    if cfg.num_condensin_total > 0:
        lef_position_list = []
        for idx_sister in range(num_sister_chromatid):
            lef_position_fpath = base_dir / f"LEFPositions_{idx_sister}.h5"
            lef_positions = h5py.File(
                lef_position_fpath, mode="r"
            )  # Read previously exported loop positions from 1D loop extrusion simulation.
            lef_position_list.append(lef_positions)

        # Read parameters from the loop position file:
       
        LEFpositions = np.concatenate([x["positions"] for x in lef_position_list], axis=1)
        Nframes = LEFpositions.shape[0]

        milker = bondUpdater(LEFpositions)
        

    else:
        Nframes = cfg.num_LE_steps // 6

    # Create a semi-dense non-catenated chain. After relaxation this resembles interphase chromatin.
    data = grow_cubic(N, int((N / density) ** 0.333))

    # Save the initial conformation
    init_conformation_fpath = base_dir / "init_conformation.npy"
    np.save(init_conformation_fpath, data)  # Save the initial conformation

    saveEveryBlocks = 10  # save every 1 simulation steps. Multiply by loop position sampling (typically every 1 s) for final simulation step frequency, e.g. 1x1 = 1 s.
    restartSimulationEveryBlocks = Nframes  # Do not restart.

    # assertions for easy managing code below
    assert (Nframes % restartSimulationEveryBlocks) == 0
    assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

    simInitsTotal = (Nframes) // restartSimulationEveryBlocks

    reporter = HDF5Reporter(
            folder=base_dir, max_data_length=100, overwrite=True, blocks_only=False
        )


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
            # PBCbox=[box, box, box], #Do not use PCB box for mitotic chromosomes
            precision="single",
        )  # timestep not necessary for variableLangevin

        ############################## New code ##############################
        a.set_data(data, center=True)  # loads a polymer, puts a center of mass at zero

        chain_tuple = [
            (
                idx_sister * N_per_sister_chromatid,
                (idx_sister + 1) * N_per_sister_chromatid,
                False,
            )
            for idx_sister in range(num_sister_chromatid)
        ]
        if num_sister_chromatid == 1:
            extra_bond_list = None
        else:
            cohesive_cohesin_interval = (
                1000  # every 1000 monomers, sister chromatids are connected
            )
            num_cohesive_cohesin = N_per_sister_chromatid // cohesive_cohesin_interval
            extra_bond_list = [
                (idx_cohesive_cohesin, idx_cohesive_cohesin + N_per_sister_chromatid)
                for idx_cohesive_cohesin in range(num_cohesive_cohesin)
            ]

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
                    "k": 1.5
                    # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                    # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
                },
                nonbonded_force_func=polychrom.forces.heteropolymer_SSW,
                nonbonded_force_kwargs={
                    "repulsionEnergy": repulsionEnergy,  # base repulsion energy for all monomers (function default is 3.0)
                    "attractionEnergy": 0,  # base attraction energy for all monomers (function default is 3.0)
                    "attractionRadius": attraction_radius,
                    "interactionMatrix": interactionMatrix,
                    "monomerTypes": monomerTypes,
                    "extraHardParticlesIdxs": [],
                },
                except_bonds=True,
                extra_bonds=extra_bond_list,
            )
        )

        # ------------ initializing milker; adding bonds ---------
        # copied from addBond
        kbond = a.kbondScalingFactor / (smcBondWiggleDist**2)
        bondDist = smcBondDist * a.length_scale

        if cfg.num_condensin_total > 0:
            activeParams = {"length": bondDist, "k": kbond}
            inactiveParams = {"length": bondDist, "k": 0}
            milker.setParams(activeParams, inactiveParams)

            # this step actually puts all bonds in and sets first bonds to be what they should be
            milker.setup(
                bondForce=a.force_dict["harmonic_bonds"],
                blocks=restartSimulationEveryBlocks,
            )

        # Initialize starting conformation and minimize the energy. Sometimes this does not work, so retry until the energy can be minimized.
        if iteration == 0:
            for i in range(100):
                try:
                    if i > 0:
                        data = grow_cubic(N, int((N / (0.2 * 1.2)) ** 0.333))
                        np.save(init_conformation_fpath, data)
                        a.set_data(
                            data, center=True
                        )  # loads a polymer, puts a center of mass at zero
                    a.local_energy_minimization(maxIterations=int(1e4))
                    break
                except openmm.OpenMMException:
                    continue
        else:
            a._apply_forces()

        a.do_block(
            steps=equilibration_timestep
        )  # Initial equilibration steps to ensure that we start from a relaxed polymer state.

        for i in range(restartSimulationEveryBlocks):
            if i < restartSimulationEveryBlocks - 1 and cfg.num_condensin_total > 0:
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
    config_fpath = run_dir / "config_resolved.json"
    # re-load resolved config from run_dir if you save it there
    cfg = load_config(config_fpath)  # your helper

    simulate_MD(cfg, run_dir)
