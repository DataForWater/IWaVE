import multiprocessing
import numpy as np
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from scipy import optimize
from tqdm import tqdm
from typing import Tuple, List, Union

from iwave import dispersion, LazySpectrumArray, CONCURRENCY

# Create a context with the desired start method
ctx = multiprocessing.get_context("spawn")

def cost_function_velocity_depth(
    x: Tuple[float, float, float],
    measured_spectrum: np.ndarray,
    vel_indx: float,
    window_dims: Tuple[int, int, int],
    res: float,
    fps: float,
    penalty_weight: float,
    gravity_waves_switch: bool,
    turbulence_switch: bool,
    gauss_width: float,
) -> float: 
    """
    Creates a synthetic spectrum based on guessed parameters, 
    then compares it with the measured spectrum and returns a cost function for minimisation

    Parameters
    ----------
    x :  [float, float, float]
        velocity_y, velocity_x, log-depth
        tentative surface velocity components along y and x (m/s) and log of depth (m)
    measured_spectrum : np.ndarray
        measured, averaged, and normalised 3D power spectrum calculated with spectral.py
    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)
    window_dims : [int, int, int]
        [dim_t, dim_y, dim_x] window dimensions
    res : float
        image resolution (m/pxl)
    fps : float
        image acquisition rate (fps)
    penalty_weight : float
        Because of the two branches of the surface spectrum (waves and turbulence-forced patterns), the algorithm 
        may choose the wrong solution causing a strongly overestimated velocity magnitude, especially 
        when smax > 2 * the actual velocity. The penalty_weight parameter increases the inertia of the optimiser, penalising
        solutions with a higher velocity magnitude. Setting penalty_weight > 0 will produce more stable results, but may slightly
        underestimate the velocity and overestimate the depth. Setting penalty_weight = 0 will eliminate the bias, 
        but may produce more outliers. If the velocity magnitude can be predicted reasonably, setting smax < 2 * the 
        typical velocity and setting penalty_weight = 0 will provide the most accurate results.
    gravity_waves_switch : bool
        if True, gravity waves are modelled
        if False, gravity waves are NOT modelled
    turbulence_switch : bool
        if True, turbulence-generated patterns and/or floating particles are modelled
        if False, turbulence-generated patterns and/or floating particles are NOT modelled
    gauss_width : float
        width of the synthetic spectrum smoothing kernel

    Returns
    -------
    cost_function : float
        cost function to be minimised (non-dimensional)
        the cost function is defined as the inverse of the cross-correlation between the measured spectrum and the
        synthetic spectrum calculated according to the estimated flow parameters

    """
    
    depth = np.exp(x[2])    # guessed depth
    velocity = [x[0], x[1]]    # guessed velocity components

    # calculate the synthetic spectrum based on the guess velocity
    synthetic_spectrum = dispersion.intensity(
        velocity, depth, vel_indx,
        window_dims, res, fps, gauss_width,
        gravity_waves_switch, turbulence_switch
    )
    cost_function = nsp_inv(measured_spectrum, synthetic_spectrum)
    
    # add a penalisation proportional to the non-dimensionalised velocity modulus
    cost_function = cost_function*(1 + 2 * penalty_weight * np.linalg.norm(velocity)/(res*fps))
    return cost_function


def nsp_inv(
    measured_spectrum: np.ndarray,
    synthetic_spectrum: np.ndarray
) -> float:
    """
    Combine the measured and synthetic spectra and calculate the cost function (inverse of the normalised scalar product)

    Parameters
    ----------
    measured_spectrum : np.ndarray
        measured, averaged, and normalised 3D power spectrum calculated with spectral.py
    synthetic_spectrum : np.ndarray
        synthetic 3D power spectrum

    Returns
    -------
    cost : float
        cost function to be minimised (non-dimensional)
        the cost function is defined as the inverse of the cross-correlation between the measured spectrum and the
        synthetic spectrum calculated according to the estimated flow parameters

    """
    spectra_correlation = measured_spectrum * synthetic_spectrum # calculate correlation
    with np.errstate(divide='ignore', invalid='ignore'):
        cost = np.sum(synthetic_spectrum)* np.sum(measured_spectrum)  / np.sum(spectra_correlation) # calculate cost function
    return cost


def cost_function_velocity_wrapper(
    x: Tuple[float, float, float],
    *args
) -> float:
    return cost_function_velocity_depth(x, *args)
    

def optimize_single_spectrum_velocity(
    measured_spectrum: np.ndarray,
    bnds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    penalty_weight: float,
    gravity_waves_switch: bool,
    turbulence_switch: bool,
    downsample: int,
    gauss_width: float,
    kwargs: dict
) -> Tuple[float, float, float, float, float, bool, str]:
    """
    Returns:
        v, u, d, cost, quality, status, message
    """
    if downsample > 1: # reduce dimensions of spectrum (for two-step approach)
        measured_spectrum, res, fps, window_dims = dispersion.spectrum_downsample(measured_spectrum, res, fps, window_dims, downsample)
    
    
    bnds = [bnds[0], bnds[1], (np.log(bnds[2][0]), np.log(bnds[2][1]))] # log-transform depth to homogenise convergence
    opt = optimize.differential_evolution(
        cost_function_velocity_wrapper,
        bounds=bnds,
        args=(measured_spectrum, vel_indx, window_dims, res, fps, penalty_weight, gravity_waves_switch, turbulence_switch, gauss_width),
        **kwargs
    )
    status = opt.success # Boolean flag indicating if the optimizer exited successfully returned by scipy.optimizer.differential_evolution
    message = opt.message # termination message returned by scipy.optimizer.differential_evolution
        
    # define a quality metric by comparing the measured spectrum with an ideal theoretical spectrum
    quality = quality_calc(opt.x, measured_spectrum, vel_indx, window_dims, res, fps, gauss_width, gravity_waves_switch, turbulence_switch)
    cost = np.sum(opt.fun**2)
    opt.x[2] = np.exp(opt.x[2]) # transforms back optimised depth into linear scale
    
    vy, vx, d = opt.x
    return vy, vx, d, cost, quality, status, message  
    

def optimize_single_spectrum_velocity_unpack(kwargs):
    """Wrap all arguments for optimization in a single dictionary."""
    # kwargs["measured_spectrum"] = kwargs["measured_spectra"][kwargs["idx"]]
    # kwargs["bnds"] = kwargs["bnds_list"][kwargs["idx"]]
    # del kwargs["measured_spectra"]
    # del kwargs["bnds_list"]
    # del kwargs["idx"]
    return optimize_single_spectrum_velocity(**kwargs)

def silence_output():
    """Suppress output of worker functions."""
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

def optimise_velocity(
    measured_spectra: Union[np.ndarray, LazySpectrumArray],
    bnds_list: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    penalty_weight: float=1,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True,
    downsample : int=1,
    gauss_width: float=1,
    chunk_size: int = 50,
    desc="Optimizing windows",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool], List[str]]:
    """
    Runs the optimisation to calculate the optimal velocity components

    Parameters
    ----------
    measured_spectra : np.ndarray
        measured and averaged 3D power spectra calculated with spectral.sliding_window_spectrum
        dimensions [N_windows, Nt, Ny, Nx]
    bnds_list: List[(float, float), (float, float), (float, float)]
        [(min_vel_y, max_vel_y), (min_vel_x, max_vel_x), (min_depth, max_depth)] velocity (m/s) and depth (m) bounds
        this is supplied as a list with potentially different values for each window
    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)
    window_dims: List[int, int, int]
        [dim_t, dim_y, dim_x] window dimensions
    res: float
        image resolution (m/pxl)
    fps: float
        image acquisition rate (fps)
    dof: float
        spectrum degrees of freedom
    penalty_weight: float, optional
        Defaults to 1.0. Because of the two branches of the surface spectrum (waves and turbulence-forced patterns),
        the algorithm may choose the wrong solution causing a strongly overestimated velocity magnitude, especially
        when smax > 2 * the actual velocity. The penalty_weight parameter increases the inertia of the optimiser, penalising
        solutions with a higher velocity magnitude. Setting penalty_weight > 0 will produce more stable results, but may slightly
        underestimate the velocity. Setting penalty_weight = 0 will eliminate the bias, but may produce more outliers.
        If the velocity magnitude can be predicted reasonably, setting smax < 2 * the typical velocity and setting 
        penalty_weight = 0 will provide the most accurate results.
    gravity_waves_switch: bool, optional
        if True (default), gravity waves are modelled
        if False, gravity waves are NOT modelled
    turbulence_switch: bool, optional
        if True (default), turbulence-generated patterns and/or floating particles are modelled
        if False, turbulence-generated patterns and/or floating particles are NOT modelled
    downsample: int, optional
        downsampling rate (default 1). If downsample > 1, then the spectrum is trimmed using a trimming ratio equal to
        'downsample'. Trimming removes the high-wavenumber tails of the spectrum, which corresponds to downsampling the
        images spatially.
    gauss_width: float, optional
        width of the synthetic spectrum smoothing kernel (default 1.0).
        gauss_width > 1 could be useful with very noisy spectra.
    chunk_size : int, optional
        Number of spectra to process at a time. Defaults to 50
    **kwargs : dict
        keyword arguments to pass to `scipy.optimize.differential_evolution, see also
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

    Returns
    -------
    optimal : np.ndarray
        optimised y [0] and x [1] velocity component (m/s) and depth (m) [2]
    cost : float
        Value of the cost function at the optimum. This parameter is inversely related to the quality parameter.
    quality : float
        Quality parameters (0 < q < 1), where 1 is highest quality and 0 is lowest quality.
        q is defined as q = 1 - 0.2*log10(cost_measured/cost_ideal)
        This parameter measures the similarity between the measured spectra and ideal spectra.
        While there is no direct link with results uncertainties, higher q indicates better quality data.
    status : Bool
        Boolean flag indicating the optimiser termination condition
    message : str
        termination message returned by the optimiser
    """

    def generate_args(): #, vel_indx, window_dims, res, fps, penalty_weight,
        # gravity_waves_switch, turbulence_switch, downsample, gauss_width, kwargs

        """
        Parameters
        ----------
        measured_spectra : LazySpectrumArray
            A single set of measured spectra and corresponding bounds list

        bnds_list : Tuple
            The bounds list for the measured spectra


        Returns
        -------
        tuple
            A single set of arguments for each corresponding pair in measured_spectra and bnds_list
        """
        args_list = [
            dict(
                measured_spectrum=measured_spectrum,
                bnds=bnds,
                vel_indx=vel_indx,
                window_dims=window_dims,
                res=res,
                fps=fps,
                penalty_weight=penalty_weight,
                gravity_waves_switch=gravity_waves_switch,
                turbulence_switch=turbulence_switch,
                downsample=downsample,
                gauss_width=gauss_width,
                kwargs=kwargs,
            ) for measured_spectrum, bnds in zip(spectra_sel, bnds_sel)
        ]
        return args_list

    idxs = range(len(measured_spectra))  # Pair with indices
    # iter_args = generate_args(idxs)

    results = [None] * len(idxs)  # Placeholder for results
    if CONCURRENCY is not None:
        max_workers = max(min(CONCURRENCY, os.cpu_count()), 1)  # never use more than the number of available cores
    else:
        max_workers = None
    # Initialize progress bar before submitting tasks
    progress_bar = tqdm(total=len(idxs), desc=desc)
    # for chunk_idxs, chunk_args in zip()
    for idx in idxs[::chunk_size]:
        # select and read the current data block in one go
        idx_sel = idxs[idx:idx + chunk_size]
        spectra_sel = measured_spectra[idx: idx + chunk_size]
        # find those that are non-zero
        bnds_sel = bnds_list[idx: idx + chunk_size]
        nonzero_idx = np.where(np.mean(spectra_sel, axis=(1, 2, 3)) != 0)[0]
        # push forward the progress bar by the amount less than the original length of arguments
        update = len(idx_sel) - len(nonzero_idx)
        if update > 0:
            progress_bar.update(update)
        idx_sel = np.array(idx_sel)[nonzero_idx].tolist()
        bnds_sel = np.array(bnds_sel)[nonzero_idx].tolist()
        spectra_sel = spectra_sel[nonzero_idx]

        args_sel = generate_args()
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers, initializer=silence_output) as executor:
            futures = {
                executor.submit(
                    optimize_single_spectrum_velocity_unpack, input_args
                ): idx for idx, input_args in zip(idx_sel, args_sel)
            }
            # collect futures
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()  # Store result in the correct position
                progress_bar.update(1)  # increase
    progress_bar.close()

    # wrap results together
    optimal = np.array([
        [res[0], res[1], res[2]] if res is not None else [np.nan, np.nan, np.nan] for res in results
    ])  # vy, vx, d
    cost = np.array([res[3] if res is not None else np.nan for res in results])
    quality = np.array([res[4] if res is not None else np.nan for res in results])

    return optimal, cost, quality


def quality_calc(
    x,
    measured_spectrum,
    vel_indx, 
    window_dims, 
    res, 
    fps, 
    gauss_width, 
    gravity_waves_switch, 
    turbulence_switch
)-> float:
    """
    Calculates a quality metric for the optimisation based on the resemblance between the measured spectrum and the theoretical one.
    The metric ranges from 0 (worst quality) to 1 (best quality).

    Parameters
    ----------
    x : np.ndarray
        Vector of optimised parameters (vy, vx, d).
        
    measured_spectrum : np.ndarray
        measured and averaged 3D power spectra calculated with spectral.sliding_window_spectrum
        dimensions [N_windows, Nt, Ny, Nx]

    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)

    window_dims: [int, int, int]
        [dim_t, dim_y, dim_x] window dimensions

    res: float
        image resolution (m/pxl)

    fps: float
        image acquisition rate (fps)
    
    gauss_width: float=1
        width of the synthetic spectrum smoothing kernel.
        gauss_width > 1 could be useful with very noisy spectra.

    gravity_waves_switch: bool=True
        if True, gravity waves are modelled
        if False, gravity waves are NOT modelled

    turbulence_switch: bool=True
        if True, turbulence-generated patterns and/or floating particles are modelled
        if False, turbulence-generated patterns and/or floating particles are NOT modelled
        
    Returns
    -------
    quality : float
        quality metric
        
    """
    depth = np.exp(x[2])    # guessed depth
    velocity = [x[0], x[1]]    # guessed velocity components

    # calculate the synthetic spectrum based on the guessed velocity
    synthetic_spectrum = dispersion.intensity(
        velocity, depth, vel_indx,
        window_dims, res, fps, gauss_width,
        gravity_waves_switch, turbulence_switch
    )
    cost_measured = nsp_inv(measured_spectrum, synthetic_spectrum)
    cost_ideal = nsp_inv(synthetic_spectrum, synthetic_spectrum)
    quality = 1 - 0.2*np.log10(cost_measured/cost_ideal)
    return quality


