from __future__ import annotations
from pathlib import Path
from typing import Optional, List
from sim_config import SimConfig

import numpy as np
from numba import njit, types, typed
from numba.experimental import jitclass
from tqdm import tqdm


@njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


# Every field below uses a FIXED-WIDTH numba type on purpose. types.int_ tracks
# C long, which is 8 bytes on Linux/macOS but 4 bytes on Windows/MSVC, so a spec
# written with types.int_ silently changes width across platforms while the
# arrays handed to __init__ below are built with an explicit np.int64. On a
# Windows box those two disagree and jitclass construction dies with a
# TypingError. See tests/test_run_sim_LE_types.py, which guards this.
spec_smc = [
    ("N_beads", types.int64),
    ("SMC_type", types.int64),
    ("bound_lifetime", types.int64),
    ("unbound_lifetime", types.int64),
    ("SMC_crash_lifetime", types.float64),  # a 1d array
    ("extrusion_sided", types.int64),
    ("extrusion_rate", types.float64),
    ("extrusion_rate_sd", types.float64),
    ("extrusion_direction", types.int64),
    ("bound", types.boolean),
    # ('age', types.int64),
    ("start_pos", types.int64),
    ("l_pos", types.int64),
    ("r_pos", types.int64),
    ("SMC_crash_prob", types.float64),
    ("SMC_bound_l", types.boolean),
    ("SMC_bound_r", types.boolean),
    ("CTCF_bound_l", types.boolean),
    ("CTCF_bound_r", types.boolean),
    (
        "cohesin_loading_probability_list",
        types.float64[:],
    ),
    ("ctcf_site_location_list", types.int64[:]),
    ("ctcf_site_direction_list", types.int64[:]),
    ("ctcf_site_stall_probability_list", types.float64[:]),
    ("ctcf_site_stall_time_list", types.float64[:]),
    ("current_lifetime", types.float64),
    ("CTCF_stall_time_l", types.float64),
    ("CTCF_stall_time_r", types.float64),
    ("smc_id", types.int64),
]


@jitclass(spec_smc)
class SMC:
    """
    SMC class. Can bind and unbind on average according to its lifetimes. While bound, each arm can step along the chain and generate loops. Can stall at CTCFs or when encountering another SMC, depending on parameters.
    """

    def __init__(
        self,
        N_beads: int,
        bound_lifetime: float,
        unbound_lifetime: float,
        SMC_crash_lifetime: float,
        SMC_crash_prob: float,
        extrusion_sided: int,
        extrusion_rate: float,
        extrusion_rate_sd: float,
        cohesin_loading_probability_list: np.ndarray,
        ctcf_site_location_list: np.ndarray,
        ctcf_site_direction_list: np.ndarray,
        ctcf_site_stall_probability_list: np.ndarray,
        ctcf_site_stall_time_list: np.ndarray,
        smc_id: int,
    ):
        self.extrusion_sided = extrusion_sided  # Set if one or two-sided extrusion
        self.set_extrusion_rate(
            extrusion_rate, extrusion_rate_sd
        )  # Rate of extrusion. If 1 or more, step every increment with the given distance (must be integers). If less than one, stochastically step only in certain increments matching average extrusion rate.
        self.N_beads = N_beads  # The length of the simulated polymer.
        self.bound_lifetime = bound_lifetime
        self.unbound_lifetime = unbound_lifetime
        self.SMC_crash_prob = SMC_crash_prob
        self.cohesin_loading_probability_list = cohesin_loading_probability_list
        self.ctcf_site_location_list = ctcf_site_location_list
        self.ctcf_site_direction_list = ctcf_site_direction_list
        self.ctcf_site_stall_probability_list = ctcf_site_stall_probability_list
        self.ctcf_site_stall_time_list = ctcf_site_stall_time_list
        self.SMC_crash_lifetime = SMC_crash_lifetime

        self.smc_id = smc_id
        self.unbind()  # Initialize in unbound state.
        # print(self.current_lifetime)

    def set_extrusion_rate(self, extrusion_rate, extrusion_rate_sd):
        """Mean monomers an arm advances per LE round. Any rate >= 0 is valid.

        No longer rounded above 1: draw_step_budget() splits the rate into
        floor(rate) guaranteed steps plus one Bernoulli(frac(rate)) step, so a
        fractional rate is honoured exactly at every magnitude, not just below 1.
        """
        if extrusion_rate == 0:  # For SMC-free simulation purposes.
            self.extrusion_rate = 0.0
        else:
            self.extrusion_rate = np.random.normal(
                loc=extrusion_rate, scale=extrusion_rate_sd
            )  # scale used for larger spread in rates. Set to 0 if no spread.
            # A wide spread around a small mean can draw a NEGATIVE rate. That
            # used to disable the extruder silently for the entire run, so
            # clamp it to "never steps" rather than something meaningless.
            if self.extrusion_rate < 0.0:
                self.extrusion_rate = 0.0

    def draw_random(self):
        # A fresh draw every call. This previously indexed a fixed pool of
        # 10000 numbers generated once per SMC and cycled with period 9999. A
        # bound, unstalled extruder consumes exactly 3 numbers per round and
        # gcd(3, 9999) = 3, so each arm reread the SAME 3333 numbers forever:
        # the step pattern was exactly periodic at lag 3333, each extruder's
        # realized rate was biased by ~6%, and 3.8% of extruders drew no number
        # below 1/lifetime and could therefore NEVER unbind.
        return np.random.random()

    def draw_step_budget(self):
        """Unit steps one arm may take this round: floor(rate) + Bernoulli(frac).

        The mean is exactly extrusion_rate at every rate. Below 1 this collapses
        to a single Bernoulli(rate) draw, so waiting times stay geometric --
        the "speed 0.1 means a 1-in-10 chance of stepping each round" semantics.
        Called once per arm so the two arms stay independent.
        """
        if not self.bound or self.extrusion_rate <= 0.0:
            return 0
        n = int(self.extrusion_rate)
        if self.draw_random() < self.extrusion_rate - n:
            n += 1
        return n

    def update_state(self):
        """Binding, unbinding and stall-timer decay. Runs ONCE per LE round.

        Walking is deliberately NOT done here (see walk()). When this logic ran
        once per unit step instead, an extruder with speed 5 also unbound and
        decayed its stalls 5x faster, so speed and lifetime cancelled exactly
        and loop size was blind to any speed above 1.
        """
        # print(self.current_lifetime)
        # Perform stochastic binding or unbding steps.
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
                    if self.draw_random() < 1 / self.SMC_crash_lifetime:
                        self.SMC_bound_l = False
                if self.SMC_bound_r:
                    if self.draw_random() < 1 / self.SMC_crash_lifetime:
                        self.SMC_bound_r = False

                # self.CTCF_stall_time_* is defined below
                if self.CTCF_bound_l:
                    if self.draw_random() < 1 / self.CTCF_stall_time_l:
                        self.CTCF_bound_l = False
                if self.CTCF_bound_r:
                    if self.draw_random() < 1 / self.CTCF_stall_time_r:
                        self.CTCF_bound_r = False

    def pick_start(self, p):
        n = self.N_beads - 3

        # create the choices
        cdf = np.cumsum(p)
        r = np.random.random()
        idx = np.searchsorted(cdf, r)

        choices = np.arange(n)

        # weighted random pick
        return choices[idx]

    def bind(self):
        self.start_pos = self.pick_start(self.cohesin_loading_probability_list)

        # Starting position is random with both arms on polymer.
        self.l_pos = self.start_pos
        self.r_pos = self.start_pos + 2
        self.bound = True

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

    def walk(self, step_l, step_r):
        """Advance each arm by AT MOST ONE monomer, if it still has budget this
        round (step_l / step_r) and is not stalled at a barrier or another SMC.

        Always one monomer per call, never `extrusion_rate` monomers at once:
        update_SMC_sim re-runs collision detection between calls with
        capture_dist = 1, so a multi-monomer jump would step straight over a
        CTCF site or another extruder. A fast extruder gets more CALLS per
        round, not a longer stride.

        The per-arm Bernoulli that used to live here now lives in
        draw_step_budget(), which generalises it to rates above 1.
        """
        step_size = 1

        if self.extrusion_sided == 2:
            if step_l and not (
                self.CTCF_bound_l or self.SMC_bound_l
            ):  # Only step if not CTCF/SMC bound on this side.
                self.l_pos -= step_size

            if step_r and not (
                self.CTCF_bound_r or self.SMC_bound_r
            ):  # Only step if not CTCF/SMC bound on this side.
                self.r_pos += step_size

        elif self.extrusion_sided == 1:
            if self.extrusion_direction == -1:
                if step_l and not (
                    self.CTCF_bound_l or self.SMC_bound_l
                ):  # Only step if not CTCF/SMC bound on this side.
                    self.l_pos -= step_size

            elif self.extrusion_direction == 1:
                if step_r and not (
                    self.CTCF_bound_r or self.SMC_bound_r
                ):  # Only step if not CTCF/SMC bound on this side.
                    self.r_pos += step_size

        if (self.l_pos < 0) or (
            self.r_pos >= self.N_beads
        ):  # Unbind if SMC has stepped outside of polymer.
            self.unbind()

        # if (self.l_pos < 0) or (self.r_pos >= self.N_beads): # If SMC steps outside of the chain, siwtch the direction instead of unbind
        #     if self.l_pos < 0: self.l_pos = 0
        #     if self.r_pos >= self.N_beads: self.r_pos = self.N_beads-1

        #     if self.extrusion_sided == 1:
        #         self.extrusion_direction *= -1  # switch the direction of extrusion

        # This might cause the accumulation of Condensin at the end and blow up the simulation


@njit(parallel=False)  # Sometimes segfaults if run in paralell...
def update_SMC_sim(SMCs: list):
    # Runs every SMC simulation step, updates all SMCs and barriers, checks for potential interactions.
    # int64, matching res_loop_pos and the int32 LEFPositions h5 downstream.
    # This was float64, which fed a float into an integer bond pipeline.
    loop_pos = -np.ones(
        (len(SMCs), 2), dtype=np.int64
    )  # Initialize empty list for gathering loop positions based on SMC arm positions.
    for i in range(len(SMCs)):
        s = SMCs[i]

        # Binding, unbinding and stall decay tick exactly ONCE per LE round, so
        # the lifetime clock is a per-ROUND clock and does not scale with speed.
        # Previously this lived inside the stepping loop, so a speed-5 extruder
        # unbound 5x faster and its loop size came out identical to speed 1.
        s.update_state()

        # Independent budget per arm: floor(rate) certain steps plus one
        # Bernoulli(frac(rate)) step. Mean advance is exactly extrusion_rate,
        # and below 1 it is the plain per-arm Bernoulli it has always been.
        n_l = s.draw_step_budget()
        n_r = s.draw_step_budget()
        n_max = n_l if n_l > n_r else n_r

        # At least one pass, so an extruder that takes no step this round is
        # still recorded and still resolves its CTCF/SMC collisions.
        for k in range(max(1, n_max)):
            capture_dist = 1
            if s.bound:
                s.walk(k < n_l, k < n_r)

            if s.bound:  # Complex has to be bound to form a loop
                loop_pos[i, 0] = s.l_pos
                loop_pos[i, 1] = s.r_pos
                # Appends the left and right arm positions of each bound SMC as a loop.

                if (
                    not (s.SMC_bound_l and s.SMC_bound_r)
                    or not (s.CTCF_bound_l and s.CTCF_bound_r)
                ):  # If cannot stall, or if already stalled in both arms no further checks necessary.
                    # Check CTCF collision outdifr og yhr oopd
                    ctcf_site_location_list = s.ctcf_site_location_list
                    ctcf_site_direction_list = s.ctcf_site_direction_list
                    ctcf_site_stall_probability_list = (
                        s.ctcf_site_stall_probability_list
                    )
                    ctcf_site_stall_time_list = s.ctcf_site_stall_time_list

                    if not s.CTCF_bound_l and (
                        s.extrusion_sided == 2 or s.extrusion_direction == -1
                    ):
                        # Now check which CTCF site is collided.
                        ctcf_collision_idx_list_l = np.flatnonzero(
                            ctcf_site_location_list == s.l_pos
                        )

                        # If there is collision, meaning the location of left arm == one of the ctcf location
                        if len(ctcf_collision_idx_list_l):
                            curr_ctcf_site_idx = ctcf_collision_idx_list_l[0]
                            curr_ctcf_direction = ctcf_site_direction_list[
                                curr_ctcf_site_idx
                            ]
                            curr_ctcf_stall_probability = (
                                ctcf_site_stall_probability_list[curr_ctcf_site_idx]
                            )
                            curr_ctcf_stall_time = ctcf_site_stall_time_list[
                                curr_ctcf_site_idx
                            ]

                            # stall probability and ctcf stall from the right or both direction
                            if (
                                curr_ctcf_stall_probability > np.random.random()
                                and curr_ctcf_direction in [0, 1]
                            ):
                                s.CTCF_bound_l = True
                                s.CTCF_stall_time_l = curr_ctcf_stall_time

                    if not s.CTCF_bound_r and (
                        s.extrusion_sided == 2 or s.extrusion_direction == 1
                    ):
                        # Now check which CTCF site is collided.
                        ctcf_collision_idx_list_r = np.flatnonzero(
                            ctcf_site_location_list == s.r_pos
                        )
                        # If there is collision, meaning the location of left arm == one of the ctcf location
                        if len(ctcf_collision_idx_list_r):
                            curr_ctcf_site_idx = ctcf_collision_idx_list_r[0]
                            curr_ctcf_direction = ctcf_site_direction_list[
                                curr_ctcf_site_idx
                            ]
                            curr_ctcf_stall_probability = (
                                ctcf_site_stall_probability_list[curr_ctcf_site_idx]
                            )
                            curr_ctcf_stall_time = ctcf_site_stall_time_list[
                                curr_ctcf_site_idx
                            ]

                            # stall probability and ctcf stall from the left or both direction
                            if (
                                curr_ctcf_stall_probability > np.random.random()
                                and curr_ctcf_direction in [0, -1]
                            ):
                                s.CTCF_bound_r = True
                                s.CTCF_stall_time_r = curr_ctcf_stall_time

                    for j, other_s in enumerate(SMCs):
                        if other_s.bound:
                            if j == i:
                                continue

                            if (
                                not s.SMC_bound_l
                                and not s.CTCF_bound_l
                                and (
                                    s.extrusion_sided == 2
                                    or s.extrusion_direction == -1
                                )
                            ):
                                if s.SMC_crash_prob > np.random.random():
                                    if (
                                        np.abs(s.l_pos - other_s.l_pos) <= capture_dist
                                    ) and (other_s.l_pos - s.l_pos <= 0):
                                        s.SMC_bound_l = True

                                    elif (
                                        np.abs(s.l_pos - other_s.r_pos) <= capture_dist
                                    ) and (other_s.r_pos - s.l_pos <= 0):
                                        s.SMC_bound_l = True

                                    else:
                                        s.SMC_bound_l = False

                            if (
                                not s.SMC_bound_r
                                and not s.CTCF_bound_r
                                and (
                                    s.extrusion_sided == 2 or s.extrusion_direction == 1
                                )
                            ):
                                if s.SMC_crash_prob > np.random.random():
                                    if (
                                        np.abs(s.r_pos - other_s.l_pos) <= capture_dist
                                    ) and (other_s.l_pos - s.r_pos >= 0):
                                        s.SMC_bound_r = True

                                    elif (
                                        np.abs(s.r_pos - other_s.r_pos) <= capture_dist
                                    ) and (other_s.r_pos - s.r_pos >= 0):
                                        s.SMC_bound_r = True

                                    else:
                                        s.SMC_bound_r = False

    return loop_pos


def init_SMC_sim(cfg: SimConfig) -> List[SMC]:
    # Initialize SMC classes for the simulation with the chosen parameters in the notebook.
    SMCs = typed.List()

    mapping_direction_int = {"left": -1, "both": 0, "right": 1}

    for i in range(cfg.num_cohesin):
        SMCs.append(
            SMC(
                N_beads=cfg.num_monomers,
                extrusion_rate=cfg.cohesin_speed,
                extrusion_rate_sd=cfg.cohesin_speed_sd,
                SMC_crash_lifetime=cfg.cohesin_stall_time,
                SMC_crash_prob=cfg.cohesin_stall_probability,
                bound_lifetime=cfg.cohesin_bound_lifetime,
                unbound_lifetime=cfg.cohesin_unbound_lifetime,
                extrusion_sided=cfg.extrusion_side,
                # Every dtype below is explicit. np.array() infers a dtype from
                # the input, and that inference is both content-dependent (an
                # empty list gives float64) and platform-dependent (the default
                # integer was int32 on Windows before NumPy 2). spec_smc pins
                # fixed widths, so anything inferred here can mismatch it.
                cohesin_loading_probability_list=np.array(
                    cfg.cohesin_loading_probability_list, dtype=np.float64
                ),
                ctcf_site_location_list=np.array(
                    cfg.ctcf_site_location_list, dtype=np.int64
                ),
                ctcf_site_direction_list=np.array(
                    [mapping_direction_int[x] for x in cfg.ctcf_site_direction_list],
                    dtype=np.int64,
                ),
                ctcf_site_stall_probability_list=np.array(
                    cfg.ctcf_site_stall_probability_list, dtype=np.float64
                ),
                ctcf_site_stall_time_list=np.array(
                    cfg.ctcf_site_stall_time_list, dtype=np.float64
                ),
                smc_id=i,
            )
        )

    return SMCs


def simulate_LE(
    cfg: SimConfig,
    run_dir: Path,
    run_id: int,
    SMCs: Optional[List] = None,
):
    """Function for running SMC simulation with boundary factors (e.g. CTCF).
    Parameters for simulation set in param dict (see notebook/docs on defining the parameters)
    Saves all SMC positions.

    Args:
        params (dict): Dict of all parameters for running simulation
        run_id (int, optional): If running function in parelell, define which run this is (e.g. on HTC cluster). Defaults to 0.
    """
    SMC_props = []
    SMCs = init_SMC_sim(cfg)

    for i in tqdm(
        range(cfg.num_LE_steps_init)
    ):  # Take the initialization steps, do not record loop positions.
        update_SMC_sim(SMCs)

    res_loop_pos = np.zeros(
        (cfg.num_LE_steps, cfg.num_cohesin, 2), dtype=np.int64
    )  # Bond array.

    for j in tqdm(range(cfg.num_LE_steps)):
        loop_pos = update_SMC_sim(SMCs)  # Calculate loops/bonds from simulation
        res_loop_pos[j] = loop_pos.copy()  # Record bonds
        SMC_props.append(
            [
                (int(s.bound), s.bound_lifetime, s.current_lifetime, s.SMC_type)
                for s in SMCs
            ]
        )  # Record SMC properties

    base_dir = run_dir

    # Save SMC properties into .npy files
    smc_props_fpath = base_dir / f"SMC_props_{run_id}.npy"
    np.save(smc_props_fpath, np.array(SMC_props))

    # Save SMC positions into .npy files
    smc_pos_fpath = base_dir / f"SMC_pos_{run_id}.npy"
    np.save(smc_pos_fpath, res_loop_pos)

    return SMCs
