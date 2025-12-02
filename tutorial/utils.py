""" 
utils.py - Collection of utility functions for the project.
"""

import os 
import glob
import numpy as np 
from tqdm import tqdm 
from typing import Dict, Union
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import squareform, pdist
from scipy.interpolate import make_interp_spline


# ============== I/O utilities ==============
# ============== I/O utilities ==============
def _loader(f):
    smc_pos_folder = []
    smc_props_folder = []
    smc_pos_files = glob.glob(os.path.join(f, "SMC_pos_*.npy"))
    smc_pos_files.sort()
    smc_props_files = glob.glob(os.path.join(f, "SMC_props_*.npy"))
    smc_props_files.sort()

    polys_file = os.path.join(f, "all_conformations.npy")

    if len(smc_pos_files):
        for f in smc_pos_files:
            smc_pos_folder.append(np.load(f))
        np_smc_pos = np.concatenate(smc_pos_folder)
    else:
        np_smc_pos = []

    if len(smc_props_files):
        for f in smc_props_files:
            smc_props_folder.append(np.load(f))
        np_smc_props = np.concatenate(smc_props_folder)
    else:
        np_smc_props = []

    if os.path.exists(polys_file):
        np_polys = np.load(polys_file)
    else:
        np_polys = []

    return np_polys, np_smc_pos, np_smc_props


def load_simulation(directory: str, num_folder: int = None) -> Dict[str, np.ndarray]:
    """
    Load in the simulation from the dataset `directory`.
    This directory contains subfolders contain independent simulation trajectories.

    Args:
        directory (str): The path to the dataset directory

    Returns:
        Dict: A dictionary of polymer, smc positions, and smc properties
    """
    # List all subfolders in the directory
    subfolders = [
        f for f in glob.glob(os.path.join(directory, "*")) if os.path.isdir(f)
    ]
    subfolders.sort()

    if num_folder is None:
        num_folder = len(subfolders)

    print(f"Loading {num_folder} replicates...")
    subfolders = subfolders[:num_folder]

    polys = []
    smc_pos = []
    smc_props = []

    print("Loading in parallel...")
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_loader, subfolders))
    print("Success...")

    polys = np.array([result[0] for result in results])
    smc_pos = np.array([result[1] for result in results])
    smc_props = np.array([result[2] for result in results])

    return {"polys": polys, "smc_pos": smc_pos, "smc_props": smc_props}


# ================ polymer utilities ============
def _generate_dmap(poly: np.ndarray) -> np.ndarray: return squareform(pdist(poly))
    
def generate_pvs(polys: np.ndarray, 
                timepoint: int = -1, 
                contact_threshold: Union[int, float] = None,
                sampling_window: int = 1000,
                sampling_end: int = None) -> np.ndarray: 
    """
    Generate the curve of contact frequency as a function of genomic distance
    (Ps vs s) of a given set of polymers. 

    Args: 
        polys (np.ndarray): The (num_replicates x timepoints x num_monomers x 3) numpy n-dimensional array.
        timepoint (int): The timepoint of interest (default is -1).
        contact_threshold (int, float): The distance threshold to call an event a contact (default is None).
        sampling_window (int): The frequency in which monomers are being sampled (default is 1000).
        sampling_end (int): The index of the last monomer to be sampled (default is None).

    Returns:
        np.ndarray: An array contains average contact frequency at different genomic distance 
    """
    # Define polymer of a particular timepoint. 
    num_monomers = polys.shape[2]
    if sampling_end is None:
        sampling_end = num_monomers
    curr_poly_set = polys[:, timepoint, :sampling_end:sampling_window]
    curr_dmaps = np.array([_generate_dmap(poly) for poly in curr_poly_set])

    # If contact_threshold is not given, assume contact_threshold 
    # from average monomer distance from the first timepoint. 
    if contact_threshold is None:
        reference_poly_set = polys[:, 0, :sampling_end:sampling_window]

        # Generate pairwise distance maps.
        reference_dmaps = [_generate_dmap(poly) for poly in reference_poly_set]

        # Find the median distance map.
        median_dmap = np.median(reference_dmaps, axis=0)

        # contact_threshold is the average distance between neighboring monomers.
        contact_threshold = np.mean(np.diag(median_dmap, k=1))

    # Generate contact frequency maps from averaging the binarized distance maps based on contact threshold.
    cmap = np.mean(curr_dmaps < contact_threshold, axis=0)

    pvs = []
    num_monomers = cmap.shape[0]
    # Iterate through all genomic distance s and calculate the average contact frequency
    for i in range(num_monomers):
        pvs.append(np.mean(np.diag(cmap, i)))
    
    return np.array(pvs)


def generate_contact_map(polys: np.ndarray, 
                timepoint: int = -1, 
                contact_threshold: Union[int, float] = None,
                sampling_window: int = 1000,
                sampling_end: int = None) -> np.ndarray: 
    """
    Generate contact frequency map of a given set of polymers. 

    Args: 
        polys (np.ndarray): The (num_replicates x timepoints x num_monomers x 3) numpy n-dimensional array.
        timepoint (int): The timepoint of interest (default is -1).
        contact_threshold (int, float): The distance threshold to call an event a contact (default is None).
        sampling_window (int): The frequency in which monomers are being sampled (default is 1000).
        sampling_end (int): The index of the last monomer to be sampled (default is None).

    Returns:
        np.ndarray: A 2d array showing contact frequency
    """
    # Define polymer of a particular timepoint. 
    num_monomers = polys.shape[2]
    if sampling_end is None:
        sampling_end = num_monomers
    curr_poly_set = polys[:, timepoint, :sampling_end:sampling_window]
    curr_dmaps = np.array([_generate_dmap(poly) for poly in curr_poly_set])

    # If contact_threshold is not given, assume contact_threshold 
    # from average monomer distance from the first timepoint. 
    if contact_threshold is None:
        reference_poly_set = polys[:, 0, :sampling_end:sampling_window]

        # Generate pairwise distance maps.
        reference_dmaps = [_generate_dmap(poly) for poly in reference_poly_set]

        # Find the median distance map.
        median_dmap = np.median(reference_dmaps, axis=0)

        # contact_threshold is the average distance between neighboring monomers.
        contact_threshold = np.mean(np.diag(median_dmap, k=1))

    # Generate contact frequency maps from averaging the binarized distance maps based on contact threshold.
    cmap = np.mean(curr_dmaps < contact_threshold, axis=0)

    return cmap


def generate_dvs(polys: np.ndarray, 
                timepoint: int = -1, 
                sampling_window: int = 1000,
                sampling_end: int = None) -> np.ndarray: 
    """
    Generate the curve of euclidean distance as a function of genomic distance
    (Ds vs s) of a given set of polymers. 

    Args: 
        polys (np.ndarray): The (num_replicates x timepoints x num_monomers x 3) numpy n-dimensional array.
        timepoint (int): The timepoint of interest (default is -1).
        sampling_window (int): The frequency in which monomers are being sampled (default is 1000).
        sampling_end (int): The index of the last monomer to be sampled (default is None).

    Returns:
        np.ndarray: A 2D array contains average euclidean distance at different genomic distance for all polymers
    """
    # Define polymer of a particular timepoint. 
    num_monomers = polys.shape[2]
    if sampling_end is None:
        sampling_end = num_monomers
    curr_poly_set = polys[:, timepoint, :sampling_end:sampling_window]

    # Generate distance maps 
    curr_dmaps = np.array([_generate_dmap(poly) for poly in curr_poly_set])
    num_monomers = curr_dmaps.shape[1]

    dvs_all = []
    for dmap in curr_dmaps:
        dvs = []
        # Iterate through all genomic distance s and calculate the average euclidean distance 
        for i in range(num_monomers):
            dvs.append(np.mean(np.diag(dmap, i)))
        dvs_all.append(dvs)
        
    return np.array(dvs_all)


def spline_interpolate(poly, sampling_rate=100, spline_degree=3):
    num_points = poly.shape[0]
    num_coords = poly.shape[1]
    spl_ordinates = []
    for i in range(num_coords):
        abscissa = np.arange(num_points)
        ordinates = poly[:, i]
        
        # Cubic spline 
        # Clamped boundary condition makes sure ends do not go all over the places
        spl = make_interp_spline(abscissa, ordinates, k=spline_degree, bc_type='clamped')
        
        # Sampling 
        new_abscissa = np.linspace(0, num_points, int(num_points*sampling_rate))
            
        new_ordinates = spl(new_abscissa)
        spl_ordinates.append(new_ordinates)
        
    return np.array(spl_ordinates).T


def calculate_segment_mean(poly, window_size):
    num_monomers = poly.shape[0]
    num_segment = num_monomers // window_size
    segment_mean_poly = []
    for i in range(num_segment):
        curr_interval_start = i*window_size
        curr_interval_end = (i+1)*window_size
        segment_poly = poly[curr_interval_start:curr_interval_end, :]
        segment_centroid = np.mean(segment_poly, axis=0)
        segment_mean_poly.append(segment_centroid)
    
    return np.stack(segment_mean_poly, axis=1).T

    
def generate_axis(curr_poly, window, sampling_rate=10):
    # Find the centroid of each segment in the window 
    curr_segment_centroid = calculate_segment_mean(curr_poly, window)
    
    # Define knots for spline by adding two end to the centroid - 
    # Use this to define a chromosomal axis 
    curr_spline_knot = np.vstack([curr_poly[0, :], curr_segment_centroid, curr_poly[-1, :]])
    
    # Define an axis using spline interpolation
    curr_poly_cubic_spl = spline_interpolate(curr_spline_knot, sampling_rate=sampling_rate)
    
    return curr_poly_cubic_spl
    