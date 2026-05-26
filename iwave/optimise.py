import multiprocessing
import numpy as np
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from scipy import optimize
from tqdm import tqdm
from typing import Tuple, List, Union

from iwave import dispersion, LazySpectrumArray, CONCURRENCY, spectral

# Create a context with the desired start method
ctx = multiprocessing.get_context("spawn")

PARAM_ORDER = ("vy", "vx", "resolution", "log_depth", "vel_indx")


def get_active_params(
    estimate_res: bool,
    estimate_depth: bool,
    estimate_vel_indx: bool,
):
    active = ["vy", "vx"]

    if estimate_res:
        active.append("resolution")

    if estimate_depth:
        active.append("log_depth")

    if estimate_vel_indx:
        active.append("vel_indx")

    return active


def build_default_params(
    resolution: float,
    depth: float,
    vel_indx: float,
):
    return {
        "resolution": resolution,
        "log_depth": np.log(depth),
        "vel_indx": vel_indx,
    }


def vector_to_params(
    x,
    active_params,
    defaults,
):
    if len(x) != len(active_params):
        raise ValueError(
            f"Expected {len(active_params)} parameters, "
            f"got {len(x)}"
        )
    
    params = defaults.copy()

    for name, value in zip(active_params, x):
        params[name] = value

    return params


def select_bounds(
    full_bounds,
    active_params,
):
    return [full_bounds[p] for p in active_params]


def normalize_bounds(bnds):

    if len(bnds) != 5:
        raise ValueError(
            "Bounds must contain "
            "(vy, vx, resolution, depth, vel_indx)"
        )

    return {
        "vy": bnds[0],
        "vx": bnds[1],
        "resolution": bnds[2],
        "log_depth": (
            np.log(bnds[3][0]),
            np.log(bnds[3][1]),
        ),
        "vel_indx": bnds[4],
    }
        

def cost_function_velocity_depth(
    x: Tuple[float, ...],
    measured_spectrum: np.ndarray,
    vel_indx: float,
    nd_kt: np.ndarray,
    nd_ky: np.ndarray,
    nd_kx: np.ndarray,
    res: float,
    fps: float,
    penalty_weight: float,
    gravity_waves_switch: bool,
    turbulence_switch: bool,
    gauss_width: float,
    depth: float = 10.0,
    estimate_res: bool = False,
    estimate_depth: bool = False,
    estimate_vel_indx: bool = False,
) -> float: 
    """
    Creates a synthetic spectrum based on guessed parameters, 
    then compares it with the measured spectrum and returns a cost function for minimisation

    Parameters
    ----------
    x :  tuple of floats
        Optimization vector. Content depends on the estimation flags:
        - If estimate_depth=False and estimate_vel_indx=False: [velocity_y, velocity_x]
        - If estimate_depth=True and estimate_vel_indx=False: [velocity_y, velocity_x, log(depth)]
        - If estimate_depth=False and estimate_vel_indx=True: [velocity_y, velocity_x, vel_indx]
        - If estimate_depth=True and estimate_vel_indx=True: [velocity_y, velocity_x, log(depth), vel_indx]
    measured_spectrum : np.ndarray
        measured, averaged, and normalised 3D power spectrum calculated with spectral.py
    vel_indx : float
        Surface velocity to depth-averaged-velocity index used when estimate_vel_indx=False.
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
    depth : float, optional
        Fixed depth value (m) when estimate_depth=False. Ignored when estimate_depth=True.
    estimate_res : bool, optional
        If True, resolution is estimated from x after the velocity components.
    estimate_depth : bool, optional
        If True, depth is estimated from x after the velocity components.
    estimate_vel_indx : bool, optional
        If True, alpha is estimated from x after the velocity and optional depth.

    Returns
    -------
    cost_function : float
        cost function to be minimised (non-dimensional)
        the cost function is defined as the inverse of the cross-correlation between the measured spectrum and the
        synthetic spectrum calculated according to the estimated flow parameters

    """
    # maintained this structure to avoid repeated loading from dictionary
    if estimate_res:
        if estimate_depth:
            if estimate_vel_indx:
                vy, vx, resolution, log_depth, vel_indx = x
            else:
                vy, vx, resolution, log_depth = x

            depth_local = np.exp(log_depth)

        else:
            depth_local = depth

            if estimate_vel_indx:
                vy, vx, resolution, vel_indx = x
            else:
                vy, vx, resolution = x
    else:
        if estimate_depth:
            if estimate_vel_indx:
                vy, vx, log_depth, vel_indx = x
            else:
                vy, vx, log_depth = x

            depth_local = np.exp(log_depth)

        else:
            depth_local = depth

            if estimate_vel_indx:
                vy, vx, vel_indx = x
            else:
                vy, vx = x
                
        resolution = res
            
    
    # scale the wavenumber/frequency arrays to physical units
    kt = nd_kt * fps # rad/s
    ky = nd_ky / resolution # rad/m
    kx = nd_kx / resolution # rad/m

    synthetic_spectrum = dispersion.intensity(
        [vy, vx], depth_local, vel_indx,
        kt, ky, kx,
        gauss_width,
        gravity_waves_switch, turbulence_switch
    )
    cost_function = nsp_inv(measured_spectrum, synthetic_spectrum)

    # add a penalisation proportional to the non-dimensionalised velocity modulus
    vel_norm = np.sqrt(vy * vy + vx * vx)
    cost_function = cost_function * (1 + 2 * penalty_weight * vel_norm / (resolution * fps))
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
    x: Tuple[float, ...],
    *args
) -> float:
    return cost_function_velocity_depth(x, *args)
    

def optimize_single_spectrum_velocity(
    measured_spectrum: np.ndarray,
    bnds: tuple,
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    penalty_weight: float,
    gravity_waves_switch: bool,
    turbulence_switch: bool,
    downsample: int,
    gauss_width: float,
    depth: float,
    estimate_res: bool,
    estimate_depth: bool,
    estimate_vel_indx: bool,
    kwargs: dict
) -> Tuple[float, float, float, float, float, float, float, bool, str]:
    """
    Returns:
        vy, vx, res_opt, d, alpha_opt, cost, quality, status, message
    """
    # Zero spectra were skipped during FFT, skip optimization for them too
    if not np.any(measured_spectrum):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, "Spectrum is zero"
    
    if downsample > 1: # reduce dimensions of spectrum (for two-step approach)
        measured_spectrum, window_dims = dispersion.spectrum_downsample(measured_spectrum, window_dims, downsample)
    
    nd_kt, nd_ky, nd_kx = spectral.nondim_wave_numbers(window_dims)
    
    res = res * downsample if downsample > 1 else res
    
    active_params = get_active_params(
        estimate_res,
        estimate_depth,
        estimate_vel_indx,
    )

    param_bounds = select_bounds(
        bnds,
        active_params,
    )
    
    opt = optimize.differential_evolution(
        cost_function_velocity_wrapper,
        bounds=param_bounds,
        args=(measured_spectrum, vel_indx, nd_kt, nd_ky, nd_kx, res, fps, penalty_weight, gravity_waves_switch, turbulence_switch, gauss_width, depth, estimate_res, estimate_depth, estimate_vel_indx),
        **kwargs
    )
    status = opt.success
    message = opt.message
    
    defaults = build_default_params(
        res,
        depth,
        vel_indx,
    )

    params = vector_to_params(
        opt.x,
        active_params,
        defaults,
    )

    vy = params["vy"]
    vx = params["vx"]
    res_opt = params["resolution"]
    d = np.exp(params["log_depth"])
    alpha_opt = params["vel_indx"]

    if estimate_res:
        x_for_quality = [vy, vx, res_opt]
    else:
        x_for_quality = [vy, vx, res]
    if estimate_depth:
        x_for_quality.append(np.log(d))
    else:
        x_for_quality.append(d)
    if estimate_vel_indx:
        x_for_quality.append(alpha_opt)

    quality = quality_calc(
        x_for_quality,
        measured_spectrum,
        vel_indx,
        window_dims,
        res,
        fps,
        gauss_width,
        gravity_waves_switch,
        turbulence_switch,
        estimate_res,
        estimate_depth,
        estimate_vel_indx,
        depth,
    )
    cost = np.sum(opt.fun**2)
    
    return vy, vx, res_opt, d, alpha_opt, cost, quality, status, message  
    

def optimize_single_spectrum_velocity_two_steps(
    measured_spectrum: np.ndarray,
    bnds: tuple,
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    penalty_weight: float,
    gravity_waves_switch: bool,
    turbulence_switch: bool,
    two_step_downsample: int,
    gauss_width: float,
    depth: float,
    estimate_res: bool,
    estimate_depth: bool,
    estimate_vel_indx: bool,
    kwargs: dict
) -> Tuple[float, float, float, float, float, float, float, bool, str]:
    """
    Two-step optimization per window: first with downsampled spectrum, then with full spectrum.
    This is more efficient than processing all windows in step 1 then all windows in step 2.
    
    Returns:
        vy, vx, res_opt, d, alpha_opt, cost, quality, status, message
    """
    # Zero spectra were skipped during FFT, skip optimization for them too
    if not np.any(measured_spectrum):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, "Spectrum is zero"
    
    # Step 1: Optimize with downsampled spectrum (always uses fixed depth and fixed alpha in step 1)
    measured_spectrum_step1, window_dims_step1 = dispersion.spectrum_downsample(
        measured_spectrum, window_dims, two_step_downsample
    )
    
    fps_step1 = fps
    nd_kt_step1, nd_ky_step1, nd_kx_step1 = spectral.nondim_wave_numbers(window_dims_step1)
    
    if estimate_res:
        res_step1 = None
    else:
        res_step1 = res * two_step_downsample if two_step_downsample > 1 else res
    
    # Step 1 bounds: only velocity and resolution, no depth or alpha optimization
    step1_active = ["vy", "vx", "resolution"] if estimate_res else ["vy", "vx"]

    bnds_step1 = select_bounds(
        bnds,
        step1_active,
    )

    opt_step1 = optimize.differential_evolution(
        cost_function_velocity_wrapper,
        bounds=bnds_step1,
        args=(measured_spectrum_step1, vel_indx, nd_kt_step1, nd_ky_step1, nd_kx_step1, res_step1, fps_step1, 
              penalty_weight, gravity_waves_switch, turbulence_switch, gauss_width, depth, estimate_res, False, False),
        **kwargs
    )
    
    # Extract step 1 results
    vy_step1 = opt_step1.x[0]
    vx_step1 = opt_step1.x[1]
    res_step1 = opt_step1.x[2] if estimate_res else res
    
    # Step 2: Refine bounds based on step 1 result. The bounds are set to be around the step 1 solution, with a margin of 0.1 m/s.
    bnds_step2 = bnds.copy()
    bnds_step2["vy"] = (
        vy_step1 - 0.1,
        vy_step1 + 0.1,
    )
    bnds_step2["vx"] = (
        vx_step1 - 0.1,
        vx_step1 + 0.1,
    )
    # define boundaries for resolution estimation
    if estimate_res:
        if two_step_downsample > 1:
            bnds_step2["resolution"] = (
                res_step1/two_step_downsample - 0.01,
                res_step1/two_step_downsample + 0.01,
            )
        else:
            bnds_step2["resolution"] = (
                res_step1 - 0.01,
                res_step1 + 0.01,
            )
        bnds_step2["resolution"] = np.where(
            np.array(bnds_step2["resolution"]) < bnds_step1[2][0],
            bnds_step1[2][0],
            np.array(bnds_step2["resolution"])
        )

    active_params = get_active_params(
        estimate_res,
        estimate_depth,
        estimate_vel_indx,
    )

    param_bounds_step2 = select_bounds(
        bnds_step2,
        active_params,
    )
    
    
    nd_kt, nd_ky, nd_kx = spectral.nondim_wave_numbers(window_dims)
    
    # Reduce population size for step 2, only if not estimating depth and/or alpha
    kwargs_step2 = kwargs.copy()
    if not estimate_depth and not estimate_vel_indx:
        kwargs_step2["popsize"] = max(1, kwargs_step2.get("popsize", 8) // 2)
    
    # Step 2: Optimize with full spectrum and refined bounds
    opt_step2 = optimize.differential_evolution(
        cost_function_velocity_wrapper,
        bounds=param_bounds_step2,
        args=(measured_spectrum, vel_indx, nd_kt, nd_ky, nd_kx, res, fps, 
              0, gravity_waves_switch, turbulence_switch, gauss_width, depth, estimate_res, estimate_depth, estimate_vel_indx),
        **kwargs_step2
    )
    
    status = opt_step2.success
    message = opt_step2.message
    
    # Extract results and handle depth and alpha based on estimation flags
    defaults = build_default_params(
        res,
        depth,
        vel_indx,
    )

    params = vector_to_params(
        opt_step2.x,
        active_params,
        defaults,
    )

    vy = params["vy"]
    vx = params["vx"]
    res_opt = params["resolution"]
    d = np.exp(params["log_depth"])
    alpha_opt = params["vel_indx"]

    if estimate_res:
        x_for_quality = [vy, vx, res_opt]
    else:
        x_for_quality = [vy, vx, res]
    if estimate_depth:
        x_for_quality.append(np.log(d))
    else:
        x_for_quality.append(d)
    if estimate_vel_indx:
        x_for_quality.append(alpha_opt)

    quality = quality_calc(
        x_for_quality,
        measured_spectrum,
        vel_indx,
        window_dims,
        res,
        fps,
        gauss_width,
        gravity_waves_switch,
        turbulence_switch,
        estimate_res,
        estimate_depth,
        estimate_vel_indx,
        depth,
    )
    cost = np.sum(opt_step2.fun**2)
    
    return vy, vx, res_opt, d, alpha_opt, cost, quality, status, message


def optimize_single_spectrum_velocity_unpack(kwargs):
    """Wrap all arguments for optimization in a single dictionary.
    Routes to either two-step or one-step optimization based on two_step_downsample parameter.
    """
    two_step_downsample = kwargs.pop("two_step_downsample", 0)
    
    if two_step_downsample > 0:
        # Two-step optimization
        kwargs['two_step_downsample'] = two_step_downsample
        return optimize_single_spectrum_velocity_two_steps(**kwargs)
    else:
        # One-step optimization: use downsample=1
        kwargs['downsample'] = 1
        return optimize_single_spectrum_velocity(**kwargs)

def silence_output():
    """Suppress output of worker functions."""
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

def optimise_velocity(
    measured_spectra: Union[np.ndarray, LazySpectrumArray],
    bnds_list: tuple,
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    penalty_weight: float=1,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True,
    chunk_size: int = 50,
    downsample : int=1,
    gauss_width: float=1,
    depth: float=10.,
    estimate_res: bool=False,
    estimate_depth: bool=False,
    estimate_vel_indx: bool=False,
    desc="Optimizing windows",
    two_step_downsample: int = 0,
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
    downsample: int, optional
        downsampling rate (default 1). If downsample > 1, then the spectrum is trimmed using a trimming ratio equal to
        'downsample'. Trimming removes the high-wavenumber tails of the spectrum, which corresponds to downsampling the
        images spatially.
    gauss_width: float, optional
        width of the synthetic spectrum smoothing kernel (default 1.0).
        gauss_width > 1 could be useful with very noisy spectra.
    chunk_size : int, optional
        Number of spectra to process at a time. Defaults to 50
    depth : float, optional
        Fixed depth value (m) when estimate_depth=False. Ignored when estimate_depth=True. Defaults to 1.
    estimate_depth : bool, optional
        If True, depth is estimated. If False, uses provided depth parameter. Defaults to False.
    estimate_vel_indx : bool, optional
        If True, alpha is estimated during optimisation. Defaults to False.
    two_step_downsample: int, optional
        If > 0, performs a two-step optimization per chunk. First step uses this downsample factor,
        then bounds are refined and second step runs with downsample=1. Defaults to 0 (disabled).
    **kwargs : dict
        keyword arguments to pass to `scipy.optimize.differential_evolution, see also
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

    Returns
    -------
    optimal : np.ndarray[float]
        optimized y [0] and x [1] velocity component (m/s), resolution (m) [2], and depth (m) [3]
    cost : np.ndarray[float]
        Value of the cost function at the optimum. This parameter is inversely related to the quality parameter.
    quality : np.ndarray[float]
        Quality parameters (0 < q < 1), where 1 is highest quality and 0 is lowest quality.
        q is defined as q = 1 - 0.2*log10(cost_measured/cost_ideal)
        This parameter measures the similarity between the measured spectra and ideal spectra.
        While there is no direct link with results uncertainties, higher q indicates better quality data.
    """

    # def generate_args(): #, vel_indx, window_dims, res, fps, penalty_weight,
    #     # gravity_waves_switch, turbulence_switch, downsample, gauss_width, kwargs

    #     """
    #     Parameters
    #     ----------
    #     measured_spectra : LazySpectrumArray
    #         A single set of measured spectra and corresponding bounds list

    #     bnds_list : Tuple
    #         The bounds list for the measured spectra


    #     Returns
    #     -------
    #     tuple
    #         A single set of arguments for each corresponding pair in measured_spectra and bnds_list
    #     """
    #     args_list = [
    #         dict(
    #             measured_spectrum=measured_spectrum,
    #             bnds=bnds,
    #             vel_indx=vel_indx,
    #             window_dims=window_dims,
    #             res=res,
    #             fps=fps,
    #             penalty_weight=penalty_weight,
    #             gravity_waves_switch=gravity_waves_switch,
    #             turbulence_switch=turbulence_switch,
    #             downsample=downsample,
    #             gauss_width=gauss_width,
    #             kwargs=kwargs,
    #         ) for measured_spectrum, bnds in zip(spectra_sel, bnds_sel)
    #     ]
    #     return args_list

    idxs = range(len(measured_spectra))  # Pair with indices
    # iter_args = generate_args(idxs)

    results = [None] * len(idxs)  # Placeholder for results
    if CONCURRENCY is not None:
        max_workers = max(min(CONCURRENCY, os.cpu_count()), 1)  # never use more than the number of available cores
    else:
        max_workers = None
    # Initialize progress bar before submitting tasks
    progress_bar = tqdm(total=len(idxs), desc=desc)
    
    normalized_bnds_list = [
        normalize_bounds(b)
        for b in bnds_list
    ]
    
    with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers, initializer=silence_output) as executor:
    
        for idx in idxs[::chunk_size]:
            # select and read the current data block in one go
            idx_sel = idxs[idx:idx + chunk_size]
            spectra_sel = measured_spectra[idx: idx + chunk_size]
            # get bounds for this chunk (nonzero check now done in worker)
            bnds_sel = normalized_bnds_list[idx: idx + chunk_size]
            # bnds_sel = [normalize_bounds(b) for b in bnds_sel]

            # Generate args for this chunk (handles both one-step and two-step via two_step_downsample parameter)
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
                    depth=depth,
                    estimate_res=estimate_res,
                    estimate_depth=estimate_depth,
                    estimate_vel_indx=estimate_vel_indx,
                    two_step_downsample=two_step_downsample,
                    gauss_width=gauss_width,
                    kwargs=kwargs,
                ) for measured_spectrum, bnds in zip(spectra_sel, bnds_sel)
            ]

        
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
        tuple(result[:5]) if result is not None else (np.nan, np.nan, np.nan, np.nan, np.nan)
        for result in results
    ])  # vy, vx, res, d, alpha
    cost = np.array([result[5] if result is not None else np.nan for result in results])
    quality = np.array([result[6] if result is not None else np.nan for result in results])

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
    turbulence_switch,
    estimate_res: bool = False,
    estimate_depth: bool = True,
    estimate_vel_indx: bool = False,
    depth: float = None,
)-> float:
    """
    Calculates a quality metric for the optimisation based on the resemblance between the measured spectrum and the theoretical one.
    The metric ranges from 0 (worst quality) to 1 (best quality).

    Parameters
    ----------
    x : array-like
        Vector of parameters [vy, vx, d].
        
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
    
    x : array-like Vector of parameters [vy, vx, res, depth].
        
    Returns
    -------
    quality : float
        quality metric
        
    """
    velocity = [x[0], x[1]]    # guessed velocity components
    
    if estimate_res:
        res = x[2]
        if estimate_depth:
            depth = x[3]  
    else:
        if estimate_depth:
            depth = x[2]
    
    if estimate_vel_indx:
        vel_indx = x[-1]
        
    kt, ky, kx = spectral.wave_numbers(window_dims, res, fps)

    synthetic_spectrum = dispersion.intensity(
        velocity, depth, vel_indx,
        kt, ky, kx, gauss_width,
        gravity_waves_switch, turbulence_switch
    )
    cost_measured = nsp_inv(measured_spectrum, synthetic_spectrum)
    cost_ideal = nsp_inv(synthetic_spectrum, synthetic_spectrum)
    quality = 1 - 0.2*np.log10(cost_measured/cost_ideal)
    return quality


