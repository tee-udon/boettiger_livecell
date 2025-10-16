from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
from sim_config import SimConfig

import numpy as np
from numba import njit, types, typed, float64
from numba.experimental import jitclass
import tqdm


@njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


spec_ctcf = [
    ("sites", types.int32[:]),
    ("probabilities", types.float32[:]),
    ("directions", types.int32[:]),
    ("avail_sites", types.boolean[:]),
    ("bound_lifetime", types.int_),
    ("unbound_lifetime", types.int_),
    ("smc_bound_lifetime", types.int_),
    ("bound", types.boolean),
    ("i", types.int_),
    ("random_numbers", types.float32[:]),
    ("position", types.int_),
    ("direction", types.int_),
    ("lifetime", float64),
    ("ctcf_id", types.int_),
    ("site_id", types.int_),
    ("bound_smc_id", types.int_),
]


@jitclass(spec_ctcf)
class CTCF:
    """Class of chromatin binding protein that can halt an SMC when encountering.
    Will bind and unbind to random sites drawn with weighted probabilites and associated directions according to an exponential decay with the bound/unbound lifetime.
    """

    def __init__(
        self,
        sites: np.ndarray,
        probabilities: np.ndarray,
        directions: np.ndarray,
        avail_sites: np.ndarray,
        bound_lifetime: int,
        unbound_lifetime: int,
        smc_bound_lifetime: int,
        ctcf_id: int = 0,
    ):
        self.sites = sites  # Array of site coordinates CTCF can occupy
        self.probabilities = probabilities  # List of probabilities of binding each site (probabilities of all sites sum to 1)
        self.directions = directions  # List of direction associated with each site.
        self.avail_sites = avail_sites
        self.bound_lifetime = bound_lifetime  # seconds, From e.g. Hansen et al, 2017
        self.unbound_lifetime = unbound_lifetime  # seconds
        self.smc_bound_lifetime = smc_bound_lifetime
        self.ctcf_id = ctcf_id
        self.i = 0
        self.random_numbers = np.random.random(10000).astype(
            np.float32
        )  # Pregenerate list of random numbers for quick access.
        self.unbind()  # Initialize in unbound state.
        self.unlink_from_site()

    def draw_random(self):
        # Draw a random number from pregenerated list.
        self.i += 1
        if self.i == len(self.random_numbers) - 1:
            self.i = 0
        return self.random_numbers[self.i]

    def update(self):
        # Bind or unbind according to lifetime.
        if (
            self.bound_lifetime == 0
        ):  # only bind if lifetime > 0 (for CTCF-free simulation purposes)
            pass
        elif self.draw_random() < 1 / self.lifetime:  # (Un)binding event
            if self.bound:
                self.unbind()
            else:
                self.bind()

    def bind(self):
        # Keep track of available binding sites, update parameters for bound state.
        if not np.any(self.avail_sites):  # No available sites to bind to.
            pass
        else:
            self.site_id = int(
                rand_choice_nb(
                    np.arange(self.sites.shape[0])[self.avail_sites],
                    self.probabilities[self.avail_sites]
                    / np.sum(self.probabilities[self.avail_sites]),
                )
            )  # Draw random (or weighted random) site.
            # if self.avail_sites[chosen_site] == True: #Check if chosen site is available
            # = chosen_site
        self.bound = True
        # self.age = 0
        self.position = self.sites[
            self.site_id
        ]  # int(np.random.choice(self.sites, p = self.probabilities)) #Draw a site weighted by the given probabilites.
        self.direction = self.directions[self.site_id]  # Set the associated direction
        self.lifetime = self.bound_lifetime  # float(np.random.exponential(self.bound_lifetime)) #Draw a bound lifetime from the lifetime distribution.

    def unbind(self):
        # Unbind the object.
        self.bound = False
        # self.age = 0
        self.lifetime = self.unbound_lifetime  # float(np.random.exponential(self.unbound_lifetime)) #Draw an unbound lifetime from the lifetime distribution.

    def unlink_from_site(self):
        # Free up the bound site.
        self.bound_smc_id = -1
        self.site_id = -1
        self.position = -1
        self.direction = 0

    def bind_smc(self, bound_smc_id):
        # Bind to SMC which could change the bound lifetime.
        self.bound_smc_id = bound_smc_id
        # self.age = 0
        self.lifetime = self.smc_bound_lifetime  # float(np.random.exponential(self.smc_bound_lifetime)) #Draw a SMC bound lifetime from the lifetime distribution.


spec_smc = [
    ("N_beads", types.int_),
    ("SMC_type", types.int_),
    ("bound_lifetime", types.int_),
    ("unbound_lifetime", types.int_),
    ("CTCF_bound_lifetime", types.int_),
    ("SMC_crash_lifetime", types.float64[:]),  # a 1d array
    ("extrusion_sided", types.int_),
    ("extrusion_rate", types.float64),
    ("extrusion_rate_sd", types.float64),
    ("extrusion_direction", types.int_),
    ("bound", types.boolean),
    # ('age', types.int_),
    ("i", types.int_),
    ("random_numbers", types.float32[:]),
    ("start_pos", types.int_),
    ("l_pos", types.int_),
    ("r_pos", types.int_),
    ("CTCF_bound", types.boolean),
    ("CTCF_bound_l", types.boolean),
    ("CTCF_bound_r", types.boolean),
    ("SMC_crash_prob", types.float64),
    ("SMC_bound_l", types.boolean),
    ("SMC_bound_r", types.boolean),
    ("current_lifetime", types.float64),
    ("smc_id", types.int_),
    ("current_stall_id_l", types.int_),
    ("current_stall_id_r", types.int_),
    ("centromere_lower_bound", types.int_),
    ("centromere_upper_bound", types.int_),
    (
        "centromere_type",
        types.int_,
    ),  # Type 0 == centromere stalls condensin infinitely, type 1 == centromere removes condensin
]


@jitclass(spec_smc)
class SMC:
    """
    SMC class. Can bind and unbind on average according to its lifetimes. While bound, each arm can step along the chain and generate loops. Can stall at CTCFs or when encountering another SMC, depending on parameters.
    """

    def __init__(
        self,
        N_beads: int,
        SMC_type: int,
        bound_lifetime: int,
        unbound_lifetime: int,
        CTCF_bound_lifetime: int,
        SMC_crash_lifetime: np.array,
        SMC_crash_prob: float,
        extrusion_sided: int,
        extrusion_rate: float,
        centromere_lower_bound: int,
        centromere_upper_bound: int,
        centromere_type: int,
        extrusion_rate_sd: float = 0,
        smc_id: int = 0,
    ):
        self.extrusion_sided = extrusion_sided  # Set if one or two-sided extrusion
        self.set_extrusion_rate(
            extrusion_rate, extrusion_rate_sd
        )  # Rate of extrusion. If 1 or more, step every increment with the given distance (must be integers). If less than one, stochastically step only in certain increments matching average extrusion rate.
        self.N_beads = N_beads  # The length of the simulated polymer.
        self.SMC_type = SMC_type  # Class of SMC for record-keeping
        self.bound_lifetime = bound_lifetime
        self.unbound_lifetime = unbound_lifetime
        self.CTCF_bound_lifetime = CTCF_bound_lifetime
        self.SMC_crash_prob = SMC_crash_prob
        n = SMC_crash_lifetime.shape[0]
        tmp = np.empty(n + 1, dtype=np.float64)
        tmp[:n] = SMC_crash_lifetime
        tmp[n] = 100000.0
        self.SMC_crash_lifetime = (
            tmp  # The last one is for centromere stall time (if it is used)
        )
        self.smc_id = smc_id
        self.unbind()  # Initialize in unbound state.
        self.i = 0
        self.random_numbers = np.random.random(10000).astype(np.float32)
        self.current_stall_id_l = (
            10000  # This is used to identify what type of SMC does it collide with
        )
        self.current_stall_id_r = (
            10000  # This is used to identify what type of SMC does it collide with
        )
        self.centromere_lower_bound = centromere_lower_bound
        self.centromere_upper_bound = centromere_upper_bound
        self.centromere_type = centromere_type
        # print(self.current_lifetime)

    def set_extrusion_rate(self, extrusion_rate, extrusion_rate_sd):
        if extrusion_rate == 0:  # For SMC-free simulation purposes.
            self.extrusion_rate = 0
        else:
            self.extrusion_rate = np.random.normal(
                loc=extrusion_rate, scale=extrusion_rate_sd
            )  # extrusion_rate #scale used for larger spread in rates. Set to 0 if no spread.
            if self.extrusion_rate > 1:
                self.extrusion_rate = np.round(self.extrusion_rate)

    def draw_random(self):
        self.i += 1
        if self.i == len(self.random_numbers) - 1:
            self.i = 0
        return self.random_numbers[self.i]

    def update(self):
        # print(self.current_lifetime)
        # Perform stochastic binding or unbding steps, and try to walk if bound.
        if not self.bound:  # and (self.age > self.current_lifetime): #Binding event
            if (
                self.bound_lifetime == 0
            ):  # only bind if lifetime > 0 (for SMC-free simulation purposes)
                pass
            elif self.draw_random() <= 1 / self.current_lifetime:
                self.bind()
        elif self.bound:  # and (self.age > self.current_lifetime): #Unbinding event
            # if self.draw_random() <= 1/self.current_lifetime
            if (
                self.draw_random() <= 1 / self.current_lifetime
            ):  # To make sure that its only unbind when lifetime is less than simulation time
                self.unbind()
            else:
                # We can change this SMC_bound_l to be the index of SMC crash lifetime
                # instead of boolean
                # print(self.SMC_crash_lifetime)
                if self.SMC_bound_l:
                    if (
                        self.draw_random()
                        < 1 / self.SMC_crash_lifetime[self.current_stall_id_l]
                    ):
                        self.SMC_bound_l = False
                        self.current_stall_id_l = 10000
                if self.SMC_bound_r:
                    if (
                        self.draw_random()
                        < 1 / self.SMC_crash_lifetime[self.current_stall_id_r]
                    ):
                        self.SMC_bound_r = False
                        self.current_stall_id_r = 10000
                self.walk()

    def bind(self):
        # Binding, updates most parameters to bound state.
        while True:
            start_pos = int(np.random.randint(0, self.N_beads - 3))
            if not (
                self.centromere_lower_bound <= start_pos <= self.centromere_upper_bound
            ):
                self.start_pos = start_pos
                break

        # Starting position is random with both arms on polymer.
        self.l_pos = self.start_pos
        self.r_pos = self.start_pos + 2
        self.bound = True
        # self.CTCF_bound = False
        self.CTCF_bound_l = False
        self.CTCF_bound_r = False
        self.SMC_bound_l = False
        self.SMC_bound_r = False
        self.current_lifetime = (
            self.bound_lifetime
        )  # Draw a new bound lifetime upon binding.
        if self.extrusion_sided == 1:
            self.extrusion_direction = np.random.choice(
                np.array([-1, 1])
            )  # Random direction chosen in one-sided extrusion.

    def unbind(self):
        # Unbinds, updates parameters to unbound state.
        self.current_lifetime = (
            self.unbound_lifetime
        )  # Draw a new unbound lifetime upon unbinding.
        self.bound = False
        self.CTCF_bound_l = False
        self.CTCF_bound_r = False
        self.SMC_bound_l = False
        self.SMC_bound_r = False
        self.l_pos = -1
        self.r_pos = -1

    def bind_ctcf(self, side):
        # Binds barrier and updates to lifetime of bound state.
        if side == "left":
            self.CTCF_bound_l = True

        elif side == "right":
            self.CTCF_bound_r = True

        # if self.CTCF_bound_l and self.CTCF_bound_r: #Needs double CTCF binding for stabilization.
        #     self.current_lifetime = self.CTCF_bound_lifetime

    def unbind_ctcf(self, side):
        # Unbinds barrier, updates lifetime to regular bound state.
        if side == "left":
            self.CTCF_bound_l = False
        elif side == "right":
            self.CTCF_bound_r = False

        # if not (self.CTCF_bound_l and self.CTCF_bound_r): #No longer CTCF bound. Here implemented such that it needs double CTCF binding for stabilization. Change to OR for single CTCF stabilization.
        #     self.current_lifetime = self.bound_lifetime

    def walk(self):
        # Walking step if not stalled at barrier or another SMC. Each arm treated separately.
        if self.extrusion_rate > 1:
            step_size = np.round(self.extrusion_rate)
        else:
            step_size = 1

        # Now when it walks if there are multiple SMCs in a row, this is buggy
        # because it should also check if there are other SMCs after

        if self.extrusion_sided == 2:
            if not (
                self.CTCF_bound_l or self.SMC_bound_l
            ):  # Only step if not CTCF/SMC bound on this side.
                if (
                    self.draw_random() < self.extrusion_rate
                ):  # Note: extrusion rate cannot be higher than 1 position per timestep in this setup, would need to add this if needed.
                    self.l_pos -= step_size

            if not (
                self.CTCF_bound_r or self.SMC_bound_r
            ):  # Only step if not CTCF/SMC bound on this side.
                if self.draw_random() < self.extrusion_rate:
                    self.r_pos += step_size

        elif self.extrusion_sided == 1:
            if self.extrusion_direction == -1:
                if not (
                    self.CTCF_bound_l or self.SMC_bound_l
                ):  # Only step if not CTCF/SMC bound on this side.
                    if (
                        self.draw_random() < self.extrusion_rate
                    ):  # Note: extrusion rate cannot be higher than 1 position per timestep in this setup, would need to add this if needed.
                        self.l_pos -= step_size

            elif self.extrusion_direction == 1:
                if not (
                    self.CTCF_bound_r or self.SMC_bound_r
                ):  # Only step if not CTCF/SMC bound on this side.
                    if self.draw_random() < self.extrusion_rate:
                        self.r_pos += step_size

        if (self.l_pos < 0) or (
            self.r_pos >= self.N_beads
        ):  # Unbind if SMC has stepped outside of polymer.
            self.unbind()

        # Unbind if SMC has stepped into centromere regions
        if (self.centromere_type == 1) and (
            (self.centromere_lower_bound <= self.l_pos <= self.centromere_upper_bound)
            or (
                self.centromere_lower_bound <= self.r_pos <= self.centromere_upper_bound
            )
        ):
            self.unbind()

        # if (self.l_pos < 0) or (self.r_pos >= self.N_beads): # If SMC steps outside of the chain, siwtch the direction instead of unbind
        #     if self.l_pos < 0: self.l_pos = 0
        #     if self.r_pos >= self.N_beads: self.r_pos = self.N_beads-1

        #     if self.extrusion_sided == 1:
        #         self.extrusion_direction *= -1  # switch the direction of extrusion

        # This might cause the accumulation of Condensin at the end and blow up the simulation


@njit(parallel=False)  # Sometimes segfaults if run in paralell...
def update_SMC_sim(CTCFs: list, SMCs: list):
    # Runs every SMC simulation step, updates all SMCs and barriers, checks for potential interactions.

    loop_pos = -np.ones(
        (len(SMCs), 2)
    )  # Initialize empty list for gathering loop positions based on SMC arm positions.
    for i in range(len(SMCs)):
        s = SMCs[i]
        if s.extrusion_rate > 1:
            capture_dist = (
                s.extrusion_rate
            )  # Prevents unintentioanl passing when using larger step sizes
        else:
            capture_dist = 1
        s.update()  # Update all SMCs to let them bind/unbind or take a step.

        if s.bound:  # Complex has to be bound to form a loop
            loop_pos[i, 0] = s.l_pos
            loop_pos[i, 1] = (
                s.r_pos
            )  # Appends the left and right arm positions of each bound SMC as a loop.

            if (
                (s.SMC_crash_prob > 0) and not (s.SMC_bound_l and s.SMC_bound_r)
            ):  # If cannot stall, or if already stalled in both arms no further checks necessary.
                l_time_so_far = 0
                r_time_so_far = 0

                if s.centromere_type == 0:
                    if (
                        np.abs(s.l_pos - s.centromere_upper_bound) <= capture_dist
                    ) and (
                        s.centromere_upper_bound <= s.l_pos
                    ):  # Left arm steps into centromere
                        s.SMC_bound_l = True
                        s.current_stall_id_l = 2  # Holder for centromere type
                        l_time_so_far = 10000000  # Infinite

                    if (
                        np.abs(s.r_pos - s.centromere_lower_bound) <= capture_dist
                    ) and (
                        s.centromere_lower_bound >= s.r_pos
                    ):  # Right arm steps into centromere
                        s.SMC_bound_r = True
                        s.current_stall_id_r = 2
                        r_time_so_far = 10000000

                    if (
                        s.centromere_lower_bound <= s.l_pos <= s.centromere_upper_bound
                    ):  # Left arm starts in centromere
                        s.SMC_bound_l = True
                        s.current_stall_id_l = 2  # Holder for centromere type
                        l_time_so_far = 100000  # Infinite

                    if (
                        s.centromere_lower_bound <= s.r_pos <= s.centromere_upper_bound
                    ):  # Right arm start in centromere
                        s.SMC_bound_r = True
                        s.current_stall_id_r = 2
                        r_time_so_far = 1000000

                for j, other_s in enumerate(SMCs):
                    if other_s.bound:
                        if j == i:
                            continue

                        # Bug comes from when there are multiple SMCs in the capture dist
                        # The final id comes from the final other SMCs that is in the capture dist
                        # I need to make sure that I only look at the SMCs that are closest to the reference
                        # pdist
                        # Also this does not take into account the directionality of the movement
                        # first we need to check the if it wants to move in that direction
                        # And this is important for unidirectional extrusion

                        # so now I want to make sure that it is the closest SMCs that I care about
                        # this still lead to jumping over

                        # i need to look at the maximum stall_time id in the capture dist
                        # now it works!

                        crash_lifetime = s.SMC_crash_lifetime[other_s.SMC_type]

                        if (
                            not s.SMC_bound_l
                            and (s.extrusion_sided == 2 or s.extrusion_direction == -1)
                            and crash_lifetime >= l_time_so_far
                        ):
                            if s.SMC_crash_prob > np.random.random():
                                if (
                                    np.abs(s.l_pos - other_s.l_pos) <= capture_dist
                                ) and (
                                    other_s.l_pos - s.l_pos < 0
                                ):  # and other_s.CTCF_bound_l:
                                    s.SMC_bound_l = True
                                    s.current_stall_id_l = other_s.SMC_type
                                    # other_s.SMC_bound_l = True #assume both stall if one does.

                                    # other_s.current_stall_id_l = s.SMC_type
                                    l_time_so_far = crash_lifetime
                                    # s.l_pos = other_s.l_pos
                                elif (
                                    np.abs(s.l_pos - other_s.r_pos) <= capture_dist
                                ) and (
                                    other_s.r_pos - s.l_pos < 0
                                ):  # and other_s.CTCF_bound_r:
                                    s.SMC_bound_l = True
                                    # other_s.SMC_bound_r = True #assume both stall if one does.
                                    s.current_stall_id_l = other_s.SMC_type
                                    # other_s.current_stall_id_r = s.SMC_type
                                    l_time_so_far = crash_lifetime
                                    # s.l_pos = other_s.r_pos
                                else:
                                    s.SMC_bound_l = False

                        if (
                            not s.SMC_bound_r
                            and (s.extrusion_sided == 2 or s.extrusion_direction == 1)
                            and crash_lifetime >= r_time_so_far
                        ):
                            if s.SMC_crash_prob > np.random.random():
                                if (
                                    np.abs(s.r_pos - other_s.l_pos) <= capture_dist
                                ) and (
                                    other_s.l_pos - s.r_pos > 0
                                ):  # and other_s.CTCF_bound_l:
                                    s.SMC_bound_r = True
                                    # other_s.SMC_bound_l = True #assume both stall if one does.
                                    s.current_stall_id_r = other_s.SMC_type
                                    # other_s.current_stall_id_l = s.SMC_type
                                    r_time_so_far = crash_lifetime
                                    # s.r_pos = other_s.l_pos
                                elif (
                                    np.abs(s.r_pos - other_s.r_pos) <= capture_dist
                                ) and (
                                    other_s.r_pos - s.r_pos > 0
                                ):  # and other_s.CTCF_bound_r:
                                    s.SMC_bound_r = True
                                    # other_s.SMC_bound_r = True #assume both stall if one does.
                                    s.current_stall_id_r = other_s.SMC_type
                                    # other_s.current_stall_id_r = s.SMC_type
                                    r_time_so_far = crash_lifetime
                                    # s.r_pos = other_s.r_pos
                                else:
                                    s.SMC_bound_r = False

    avail_sites = CTCFs[0].avail_sites
    for i in range(len(CTCFs)):
        c = CTCFs[i]
        c.avail_sites = avail_sites  # Update available sites in each CTCF
        c.update()  # Update CTCFs to let them bind/unbind.
        if c.bound:
            avail_sites[c.site_id] = (
                False  # If CTCF was bound after update step, make this site unavailable. Comment out to let multiple CTCF bind one Chip-seq site.
            )
            # if c.bound_smc_id < 0: #If CTCF not already bound to an SMC, check for this. Comment out to let CTCF bind multiple SMCs.
            for j in range(len(SMCs)):
                s = SMCs[j]
                if s.bound:
                    if s.extrusion_rate > 1:
                        capture_dist = s.extrusion_rate + 1
                    else:
                        capture_dist = 2
                    if (
                        (np.abs(s.r_pos - c.position) < capture_dist)
                        and (c.direction == -1)
                        and not s.CTCF_bound_r
                    ):  # Check if it is bound at the SMC arm and has the correct orientation, and SMC arm not already CTCF bound. Can bind to +/- 6 position (10kb capture) to avoid passing due to simulation order.
                        c.bind_smc(bound_smc_id=s.smc_id)
                        s.bind_ctcf(side="right")
                        s.r_pos = c.position
                    elif (
                        (np.abs(s.l_pos - c.position) < capture_dist)
                        and (c.direction == 1)
                        and not s.CTCF_bound_l
                    ):  # Same as above for other SMC arm.
                        c.bind_smc(bound_smc_id=s.smc_id)
                        s.bind_ctcf(side="left")
                        s.l_pos = c.position

        elif (
            (c.site_id > -1) and not c.bound
        ):  # This means CTCF was unbound during the current update step.
            avail_sites[c.site_id] = True  # Make this site available.
            if c.bound_smc_id > -1:  # If CTCF was bound to an SMC.
                s = SMCs[c.bound_smc_id]
                if c.direction == -1:
                    s.unbind_ctcf(side="right")  # Unbind CTCF from right side
                elif c.direction == 1:
                    s.unbind_ctcf(side="left")  # Unbind CTCF from left side
                # for j in range(len(SMCs)):
                #     s = SMCs[j]
                #     if (c.bound_smc_id == s.smc_id) and (s.bound):
                #         if c.direction == -1:
                #             s.unbind_ctcf(side='right') #Unbind CTCF from right side
                #         elif c.direction == 1:
                #             s.unbind_ctcf(side='left') #Unbind CTCF from left side

            c.unlink_from_site()  # Unlink the CTCF from the site, resets all site ids.

    return loop_pos


def init_SMC_sim(cfg: SimConfig) -> Tuple[CTCF, SMC]:
    # Initialize the CTCF and SMC classes for the simulation with the chosen parameters in the notebook.
    CTCFs = typed.List()
    SMCs = typed.List()

    for i in range(cfg._num_CTCF):
        CTCFs.append(
            CTCF(
                sites=np.array(cfg._CTCF_sites).astype(np.int32),
                probabilities=np.array(cfg._CTCF_site_probability).astype(np.float32),
                directions=np.array(cfg._CTCF_site_direction).astype(np.int32),
                avail_sites=np.ones(len(cfg._CTCF_sites)).astype(bool),
                bound_lifetime=cfg._CTCF_bound_lifetime,
                unbound_lifetime=cfg._CTCF_unbound_lifetime,
                smc_bound_lifetime=cfg._CTCF_condensin_bound_lifetime,
                ctcf_id=i,
            )
        )

    # Currently this case is always true (_num_condensin_types == 2). For code replicability reason.
    for i in range(cfg.num_condensin_total):
        SMC_type = int(
            np.random.choice(cfg._num_condensin_types, p=cfg.ratio_condensin_list)
        )
        SMCs.append(
            SMC(
                N_beads=cfg.num_monomers,
                SMC_type=SMC_type,
                bound_lifetime=cfg.condensin_bound_lifetime_list[SMC_type],
                unbound_lifetime=cfg.condensin_unbound_lifetime_list[SMC_type],
                CTCF_bound_lifetime=cfg._condensin_CTCF_bound_lifetime_list[SMC_type],
                SMC_crash_lifetime=np.array(
                    cfg.condensin_stall_time_matrix[SMC_type]
                ),  # Now crash lifetime is a list of list; so this will be a list instead of a number
                extrusion_sided=cfg.extrusion_side_list[SMC_type],
                extrusion_rate=cfg.condensin_speed_list[SMC_type],
                extrusion_rate_sd=cfg.condensin_speed_sd_list[SMC_type],
                SMC_crash_prob=cfg.condensin_stall_probability_list[SMC_type],
                centromere_lower_bound=cfg.centromere_range_list[0],
                centromere_upper_bound=cfg.centromere_range_list[1],
                centromere_type=0 if cfg.centromere_type == "stall" else 1,
                smc_id=i,
            )
        )

    return CTCFs, SMCs


def update_sim_objects(
    cfg: SimConfig, CTCFs: List[CTCF], SMCs: List[SMC]
) -> Tuple[List[CTCF], List[SMC]]:
    # Update function to adjust parameters without loosing simulation state.
    for i in range(cfg._num_CTCF):
        CTCFs[i].sites = np.array(cfg._CTCF_sites).astype(np.int32)
        CTCFs[i].probabilities = np.array(cfg._CTCF_site_probability).astype(np.float32)
        CTCFs[i].directions = np.array(cfg._CTCF_site_direction).astype(np.int32)
        CTCFs[i].avail_sites = np.ones(len(cfg._CTCF_sites)).astype(bool)
        CTCFs[i].bound_lifetime = cfg._CTCF_bound_lifetime
        CTCFs[i].unbound_lifetime = cfg._CTCF_unbound_lifetime
        CTCFs[i].smc_bound_lifetime = cfg._CTCF_condensin_bound_lifetime
    for i in range(cfg.num_condensin_total):
        SMC_type = SMCs[i].SMC_type
        SMCs[i].bound_lifetime = cfg.condensin_bound_lifetime_list[SMC_type]
        SMCs[i].unbound_lifetime = cfg.condensin_unbound_lifetime_list[SMC_type]
        SMCs[i].CTCF_bound_lifetime = cfg._condensin_CTCF_bound_lifetime_list[SMC_type]
        SMCs[i].SMC_crash_lifetime = np.array(cfg.condensin_stall_time_matrix[SMC_type])
        SMCs[i].SMC_crash_prob = cfg.condensin_stall_probability_list[SMC_type]
        SMCs[i].set_extrusion_rate(
            cfg.condensin_speed_list[SMC_type], cfg.condensin_speed_sd_list[SMC_type]
        )

    return CTCFs, SMCs


def simulate_LE(
    cfg: SimConfig,
    run_dir: Path,
    run_id: int,
    continue_previous: bool = False,
    CTCFs: Optional[List] = None,
    SMCs: Optional[List] = None,
):
    """Function for running SMC simulation with boundary factors (e.g. CTCF).
    Parameters for simulation set in param dict (see notebook/docs on defining the parameters)
    Saves all SMC positions.

    Args:
        params (dict): Dict of all parameters for running simulation
        run_id (int, optional): If running function in parelell, define which run this is (e.g. on HTC cluster). Defaults to 0.
    """
    CTCF_props = []
    SMC_props = []

    if (
        not continue_previous
    ):  # Initialize new objects if previous objects not supplied as arguments.
        CTCFs, SMCs = init_SMC_sim(cfg)
    else:
        if (CTCFs is None) or (SMCs is None):
            raise ValueError(
                "Need to provide previous simulation objects if continuing previous simulation"
            )
        else:
            CTCFs, SMCs = update_sim_objects(
                cfg, CTCFs, SMCs
            )  # Update objects with the provided simulation parameters.

    for i in tqdm.tqdm(
        range(cfg.num_LE_steps_init)
    ):  # Take the initialization steps, do not record loop positions.
        update_SMC_sim(CTCFs, SMCs)

    res_loop_pos = np.zeros(
        (cfg.num_LE_steps, cfg.num_condensin_total, 2), dtype=np.int64
    )  # Bond array.
    for j in tqdm.tqdm(range(cfg.num_LE_steps)):
        loop_pos = update_SMC_sim(CTCFs, SMCs)  # Calculate loops/bonds from simulation
        res_loop_pos[j] = loop_pos.copy()  # Record bonds
        CTCF_props.append(
            [(int(c.bound), c.position) for c in CTCFs]
        )  # Record CTCF properties.
        SMC_props.append(
            [
                (int(s.bound), s.bound_lifetime, s.current_lifetime, s.SMC_type)
                for s in SMCs
            ]
        )  # Record SMC properties

    base_dir = run_dir

    # Save CTCF properties into .npy files
    ctcf_props_fpath = base_dir / f"CTCF_props_{run_id}.npy"
    np.save(ctcf_props_fpath, np.array(CTCF_props))

    # Save SMC properties into .npy files
    smc_props_fpath = base_dir / f"SMC_props_{run_id}.npy"
    np.save(smc_props_fpath, np.array(SMC_props))

    # Save SMC positions into .npy files
    smc_pos_fpath = base_dir / f"SMC_pos_{run_id}.npy"
    np.save(smc_pos_fpath, res_loop_pos)

    return CTCFs, SMCs
