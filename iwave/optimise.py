import multiprocessing
import numpy as np
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy import optimize
from tqdm import tqdm
from typing import Optional, Tuple, List, Union

from iwave import dispersion, LazySpectrumArray, CONCURRENCY

# Start fitting processes with fresh Python interpreters (the spawn method).
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
    """Score a trial velocity and depth by comparing model and measured power.

    x contains (vy, vx, log(depth)); velocities use m/s and depth uses metres.
    Build the model spectrum using vel_indx (depth-average / surface velocity),
    window_dims (frames, y, x), res (m/pixel), fps, gauss_width, and the two
    physical-component switches.

    Return a dimensionless inverse-overlap score: smaller means closer agreement.
    A positive penalty_weight increases it for faster trial velocities, to reduce
    some wrong high-speed fits. This can bias velocity low or depth high; zero
    removes that penalty, not every other possible bias. The penalty is applied
    here for DE, not to HNM's separate local fitting score.
    """
    
    depth = np.exp(x[2])    # Convert trial log(depth) back to depth in metres.
    velocity = [x[0], x[1]]    # Trial velocity components, in y then x order.

    # Build the model spectrum for this trial velocity and depth.
    synthetic_spectrum = dispersion.intensity(
        velocity, depth, vel_indx,
        window_dims, res, fps, gauss_width,
        gravity_waves_switch, turbulence_switch
    )
    cost_function = nsp_inv(measured_spectrum, synthetic_spectrum)
    
    # Increase the score for faster trial velocities when a penalty is requested.
    cost_function = cost_function*(1 + 2 * penalty_weight * np.linalg.norm(velocity)/(res*fps))
    return cost_function


def nsp_inv(
    measured_spectrum: np.ndarray,
    synthetic_spectrum: np.ndarray
) -> float:
    """Compare measured and model power; smaller values mean greater overlap.

    Both spectra have matching (frequency, ky, kx) dimensions. The returned score
    is sum(measured)*sum(model)/sum(measured*model). It is the inverse-overlap
    quantity used by this code, not a separate estimate of physical uncertainty.
    """
    # Multiply matching entries to measure overlap between the two spectra.
    spectra_correlation = measured_spectrum * synthetic_spectrum
    with np.errstate(divide='ignore', invalid='ignore'):
        # A smaller score means greater measured/model overlap.
        cost = np.sum(synthetic_spectrum)* np.sum(measured_spectrum)  / np.sum(spectra_correlation)
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
    """Fit one spectrum and return (vy, vx, depth, cost, quality, status, message)."""
    # Skip fitting when the measured spectrum is entirely zero.
    if not np.any(measured_spectrum):
        return np.nan, np.nan, np.nan, np.nan, np.nan, False, "Spectrum is zero"
    
    # Use a very large depth for the preliminary fitting passes.
    pass_kwargs = kwargs.copy()
    
    for n, sample in enumerate(pass_downsampling):
        if n > 0:
            pass_bnds = [
                (pass_vy-0.1,pass_vy+0.1),
                (pass_vx-0.1,pass_vx+0.1),
                (np.log(1000), np.log(1000))]
            if n == len(pass_downsampling) - 1:
                # For the final pass, remove the velocity penalty and use the requested
                # depth limits.
                penalty_weight = 0
                
                # Use fewer trial solutions in the final DE pass.
                
                pass_kwargs["popsize"] = max(1, pass_kwargs.get("popsize", 8) // 2)
                
                pass_bnds = [
                    (pass_vy-0.1,pass_vy+0.1),
                    (pass_vx-0.1,pass_vx+0.1),
                    # Fit log(depth) so depth stays positive and a wide depth range is
                    # easier to search.
                    (np.log(bnds[2][0]), np.log(bnds[2][1]))]
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
        
        # Take the fitted velocity as the starting result for the next pass.
        pass_vy = pass_opt.x[0]
        pass_vx = pass_opt.x[1]
    
    
    status = pass_opt.success
    message = pass_opt.message
    
    # Calculate how closely the final model matches the measurements.
    quality = quality_calc(pass_opt.x, measured_spectrum, vel_indx, window_dims, res, fps, 
                          gauss_width, gravity_waves_switch, turbulence_switch)
    cost = np.sum(pass_opt.fun**2)
    # Convert fitted log(depth) back to depth in metres.
    pass_opt.x[2] = np.exp(pass_opt.x[2])
    
    vy, vx, d = pass_opt.x
    return vy, vx, d, cost, quality, status, message
    
    

def optimize_single_spectrum_velocity_unpack(kwargs):
    """Unpack the task dictionary and pass its values to the single-window DE fit."""
    return optimize_single_spectrum_velocity(**kwargs)

def silence_output():
    """Run a fitting task without printing its standard output or error messages."""
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
    """Fit window spectra using the original DE interface.

    measured_spectra has (window, frequency, ky, kx) dimensions. bnds_list supplies
    one ((vy_min, vy_max), (vx_min, vx_max), (depth_min, depth_max)) set per window.
    Use m/s and metres. vel_indx is depth-averaged / surface velocity;
    window_dims is original (frames, y, x), res is m/pixel, and fps is frames/s.

    penalty_weight=1 discourages fast solutions but can bias the estimate; zero
    disables the penalty. gauss_width=1 sets model frequency width. The wave and
    turbulence switches include waves and flow-carried patterns, both by default.
    chunk_size=50 limits windows per batch. pass_downsampling defaults to [2, 1]:
    fit the central reduced spectrum first, then the full spectrum.
    Extra kwargs go to scipy.optimize.differential_evolution.

    Return (optimal, cost, quality). optimal has one (vy, vx, depth) row per window.
    Cost measures inverse overlap; smaller is better. Quality is
    1 - 0.2*log10(cost_measured/cost_ideal), with 1 for a perfect match.
    It is not limited to [0, 1] and does not estimate parameter errors.
    This FFT-only interface adds no HNM method or new diagnostic switches.
    """

    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer.')
    idxs = range(len(measured_spectra))
    results = [None] * len(idxs)
    available_workers = os.cpu_count() or 1
    requested_workers = CONCURRENCY if CONCURRENCY is not None else available_workers
    # A batch cannot use more fitting processes than it has windows.
    # Avoid starting unused Python processes when the batch is small.
    max_workers = max(1, min(
        requested_workers, available_workers, chunk_size, len(idxs) or 1,
    ))

    # Start the fitting processes once, outside the loop over batches.
    # Reuse them for this call, keeping the same fitting function and options.
    # This FFT-only version adds no HNM fitting or uncertainty calculations.
    with tqdm(total=len(idxs), desc=desc) as progress_bar:
        with ProcessPoolExecutor(
            mp_context=ctx, max_workers=max_workers, initializer=silence_output,
        ) as executor:
            for start in idxs[::chunk_size]:
                # Prepare one batch, finish its fits, and release its data before reading
                # the next.
                # Do not prepare later batches in the background or keep all spectra in
                # memory.
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
                # Fits can finish in any order. Use their saved window numbers
                # to restore image order before arranging results into rows and columns.
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    progress_bar.update(1)
                del futures, args_sel, spectra_sel

    # Combine the per-window results into output arrays.
    optimal = np.array([
        [res[0], res[1], res[2]] if res is not None else [np.nan, np.nan, np.nan] for res in results
    ])  # y velocity, x velocity, and depth.
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
    """Calculate agreement between the measured spectrum and the fitted model.

    x contains (vy, vx, log(depth)). measured_spectrum is the measured power for
    one window. vel_indx is depth-average / surface velocity; window_dims is
    (frames, y, x), res is m/pixel, and fps is the analysed frame rate.
    gauss_width and the two switches define the same model used for fitting.

    Return q = 1 - 0.2*log10(cost_measured/cost_ideal). A perfect match gives 1,
    but values are not limited to [0, 1]. A high score indicates spectral agreement,
    not a measured velocity/depth error or proof that the chosen fit is unique.
    """
    depth = np.exp(x[2])    # Convert trial log(depth) back to depth in metres.
    velocity = [x[0], x[1]]    # Trial velocity components, in y then x order.

    # Build the model spectrum from the fitted velocity and depth.
    synthetic_spectrum = dispersion.intensity(
        velocity, depth, vel_indx,
        window_dims, res, fps, gauss_width,
        gravity_waves_switch, turbulence_switch
    )
    cost_measured = nsp_inv(measured_spectrum, synthetic_spectrum)
    cost_ideal = nsp_inv(synthetic_spectrum, synthetic_spectrum)
    quality = 1 - 0.2*np.log10(cost_measured/cost_ideal)
    return quality


