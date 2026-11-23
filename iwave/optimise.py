import multiprocessing
import numpy as np
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy import optimize
from tqdm import tqdm
from typing import Optional, Tuple, List, Union

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
    pass_downsampling: List[int],
    gauss_width: float,
    kwargs: dict
) -> Tuple[float, float, float, float, float, bool, str]:
    """
    Returns:
        v, u, d, cost, quality, status, message
    """
    # Zero spectra were skipped during FFT, skip optimization for them too
    if not np.any(measured_spectrum):
        return np.nan, np.nan, np.nan, np.nan, np.nan, False, "Spectrum is zero"
    
    # assume very large depth for initial passes
    pass_kwargs = kwargs.copy()
    
    for n, sample in enumerate(pass_downsampling):
        if n > 0:
            pass_bnds = [
                (pass_vy-0.1,pass_vy+0.1),
                (pass_vx-0.1,pass_vx+0.1),
                (np.log(1000), np.log(1000))]
            if n == len(pass_downsampling) - 1:
                # in the last step, the penalty_weight becomes zero and depth is also
                penalty_weight = 0
                
                # Reduce population size for last step
                
                pass_kwargs["popsize"] = max(1, pass_kwargs.get("popsize", 8) // 2)
                
                pass_bnds = [
                    (pass_vy-0.1,pass_vy+0.1),
                    (pass_vx-0.1,pass_vx+0.1),
                    (np.log(bnds[2][0]), np.log(bnds[2][1]))] # log-transform depth to homogenise convergence
        else:
            pass_bnds = [bnds[0], bnds[1], (np.log(bnds[2][0]), np.log(bnds[2][1]))]

        pass_measured_spectrum, pass_res, pass_fps, pass_window_dims = dispersion.spectrum_downsample(measured_spectrum, res, fps, window_dims, pass_downsampling[n])
            
        pass_opt = optimize.differential_evolution(
            cost_function_velocity_wrapper,
            bounds=pass_bnds,
            args=(pass_measured_spectrum, vel_indx, pass_window_dims, pass_res, pass_fps, 
                penalty_weight, gravity_waves_switch, turbulence_switch, gauss_width),
            **pass_kwargs
        )
        
        # Extract step 1 results
        pass_vy = pass_opt.x[0]
        pass_vx = pass_opt.x[1]
    
    
    status = pass_opt.success
    message = pass_opt.message
    
    # Calculate quality metric
    quality = quality_calc(pass_opt.x, measured_spectrum, vel_indx, window_dims, res, fps, 
                          gauss_width, gravity_waves_switch, turbulence_switch)
    cost = np.sum(pass_opt.fun**2)
    pass_opt.x[2] = np.exp(pass_opt.x[2])  # transforms back optimised depth into linear scale
    
    vy, vx, d = pass_opt.x
    return vy, vx, d, cost, quality, status, message
    
    

def optimize_single_spectrum_velocity_unpack(kwargs):
    """Wrap all arguments for optimization in a single dictionary.
    """
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
    chunk_size: int = 50,
    gauss_width: float=1,
    desc="Optimizing windows",
    pass_downsampling: Optional[List[int]] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    gauss_width: float, optional
        width of the synthetic spectrum smoothing kernel (default 1.0).
        gauss_width > 1 could be useful with very noisy spectra.
    chunk_size : int, optional
        Number of spectra to process at a time. Defaults to 50
    pass_downsampling: list of int, optional
        List of downsampling rates. If None (default), this will be set to [2, 1], meaning that first
        the optimization will occur with a trimmed spectrum of half, and after, refined with the full (1) spectrum. 
        Trimming removes the high-wavenumber tails of the spectrum, which corresponds to downsampling the images 
        spatially.
    **kwargs : dict
        keyword arguments to pass to `scipy.optimize.differential_evolution, see also
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

    Returns
    -------
    optimal : np.ndarray[float]
        optimized y [0] and x [1] velocity component (m/s) and depth (m) [2]
    cost : np.ndarray[float]
        Value of the cost function at the optimum. This parameter is inversely related to the quality parameter.
    quality : np.ndarray[float]
        Quality parameters (0 < q < 1), where 1 is highest quality and 0 is lowest quality.
        q is defined as q = 1 - 0.2*log10(cost_measured/cost_ideal)
        This parameter measures the similarity between the measured spectra and ideal spectra.
        While there is no direct link with results uncertainties, higher q indicates better quality data.
    """

    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer.')
    idxs = range(len(measured_spectra))
    results = [None] * len(idxs)
    available_workers = os.cpu_count() or 1
    requested_workers = CONCURRENCY if CONCURRENCY is not None else available_workers
    # No batch can use more workers than it has windows. Do not start an idle
    # interpreter/JIT instance for every CPU when the selected batch is small.
    max_workers = max(1, min(
        requested_workers, available_workers, chunk_size, len(idxs) or 1,
    ))

    # Keep the pool outside the batch loop: spawned interpreters are reused
    # for this fit. The worker function, solver options and return arrays are
    # exactly main's; no HNM strategy or diagnostics are introduced here.
    with tqdm(total=len(idxs), desc=desc) as progress_bar:
        with ProcessPoolExecutor(
            mp_context=ctx, max_workers=max_workers, initializer=silence_output,
        ) as executor:
            for start in idxs[::chunk_size]:
                # Deliberately no prefetch or full-spectrum cache. Extract one
                # requested batch, finish it and release it before the next.
                selection = slice(start, start + chunk_size)
                idx_sel = idxs[selection]
                spectra_sel = measured_spectra[selection]
                bnds_sel = bnds_list[selection]
                args_sel = [
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
                        pass_downsampling=pass_downsampling,
                        gauss_width=gauss_width,
                        kwargs=kwargs,
                    )
                    for measured_spectrum, bnds in zip(spectra_sel, bnds_sel)
                ]
                futures = {
                    executor.submit(optimize_single_spectrum_velocity_unpack, item): i
                    for i, item in zip(idx_sel, args_sel)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    progress_bar.update(1)
                del futures, args_sel, spectra_sel

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


