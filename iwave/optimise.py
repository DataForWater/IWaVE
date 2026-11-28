import multiprocessing
import numpy as np
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from scipy import optimize
from tqdm import tqdm
from typing import Optional, Tuple, List, Union
from copy import deepcopy
from numbers import Real

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
    kwargs: dict,
    *,
    _return_search_context: bool = False,
) -> Tuple[float, float, float, float, float, bool, str]:
    """Fit one spectrum and return (vy, vx, depth, cost, quality, status, message)."""
    # Skip fitting when the measured spectrum is entirely zero.
    if not np.any(measured_spectrum):
        result = (np.nan, np.nan, np.nan, np.nan, np.nan, False, "Spectrum is zero")
        return (*result, None) if _return_search_context else result
    
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
    result = (vy, vx, d, cost, quality, status, message)
    if _return_search_context:
        context = dict(velocity_bounds=np.asarray(pass_bnds[:2], float).tolist(),
                       log_depth_bounds=list(pass_bnds[2]),
                       final_downsampling=int(pass_downsampling[-1]),
                       final_penalty_weight=float(penalty_weight))
        return (*result, context)
    return result
    
    

DE_DEFAULTS = {"gauss_width": 1.0, "penalty_weight": 1.0,
               "pass_downsampling": [2, 1]}


def normalise_method(method):
    """Check that the requested fitting method is DE or HNM, using its standard spelling."""
    if isinstance(method, str) and method in ('DE', 'HNM'):
        return method
    raise ValueError("method must be 'DE' or 'HNM'.")


def resolve_method_options(method="DE", de_options=None, hnm_options=None,
                           legacy=None, scipy_defaults=None, bnds=None):
    """Check method options once in the main program.

    Values in the new nested DE dictionary override matching defaults. For older
    DE calls, a nonempty SciPy keyword dictionary replaces those defaults instead.
    Keep these calling styles separate so their meaning is not ambiguous.
    """
    legacy = dict(legacy or {})
    method = normalise_method(method)
    if method == 'HNM':
        if de_options is not None or legacy:
            raise ValueError('HNM accepts hnm_options only, not DE/legacy optimisation settings.')
        from iwave.optimise_hnm import resolve_hnm_options
        return None, resolve_hnm_options(hnm_options, bnds=bnds)
    if hnm_options is not None:
        raise ValueError('hnm_options is only valid with HNM.')
    if de_options is not None and not isinstance(de_options, dict):
        raise ValueError("de_options must be a dictionary or None.")
    if de_options is not None and legacy:
        raise ValueError("Do not mix de_options with legacy top-level DE settings.")
    unknown = set(de_options or {}) - {*DE_DEFAULTS, "scipy_options"}
    if unknown:
        raise ValueError("Unknown de_options: " + ", ".join(map(str, unknown)))
    out = deepcopy(DE_DEFAULTS)
    out.update(deepcopy(de_options or {}))
    scipy_options = out.pop("scipy_options", {})
    if not isinstance(scipy_options, dict):
        raise ValueError("de_options['scipy_options'] must be a dictionary.")
    scipy_options = {**deepcopy(scipy_defaults or {}), **scipy_options}
    if legacy:
        for key in DE_DEFAULTS:
            if key in legacy:
                out[key] = deepcopy(legacy.pop(key))
        if not out["pass_downsampling"]:
            out["pass_downsampling"] = [2, 1]
        if legacy:
            # Keep the earlier rule: supplied DE keyword settings replace the default
            # dictionary.
            scipy_options = legacy
    for key in ("gauss_width", "penalty_weight"):
        value = out[key]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0
                or (key == "gauss_width" and value == 0)):
            raise ValueError(f"Invalid DE {key}.")
    passes = out["pass_downsampling"]
    if (not isinstance(passes, (list, tuple)) or not passes
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                   or n < 1 for n in passes)):
        raise ValueError("DE pass_downsampling must be a nonempty list of positive integers.")
    if {"func", "bounds", "args"} & set(scipy_options):
        raise ValueError("Do not override the DE objective, bounds or objective args.")
    out["pass_downsampling"] = list(passes)
    out["scipy_options"] = dict(scipy_options)
    return out, None


def optimize_single_spectrum_velocity_unpack(kwargs):
    """Send one window to the requested method, defaulting to DE.

    Keep this function at module level so a newly started Python fitting process
    can import and call it.
    """
    arguments = dict(kwargs)
    method = normalise_method(arguments.pop("method", "DE"))
    metadata = arguments.pop("_return_metadata", False)
    spectral_flag = arguments.pop('calculate_spectral_uncertainty', False)
    mode_flag = arguments.pop('calculate_mode_ambiguity', False)
    from iwave.spectral import resolve_spectral_request
    from iwave.optimise_hnm import resolve_mode_request
    spectral_options = resolve_spectral_request(spectral_flag, arguments.pop('spectral_uncertainty_options', None))
    mode_options = resolve_mode_request(mode_flag, arguments.pop('mode_ambiguity_options', None))
    need_context = spectral_options is not None or mode_options is not None
    spectral_context = arguments.pop('_spectral_context', None)
    if need_context and not metadata:
        raise ValueError('Enabled uncertainty diagnostics require return_metadata=True.')
    reject_removed_uncertainty_options(arguments)
    if method == 'HNM':
        from iwave.optimise_hnm import optimize_single_spectrum_velocity_hnm, resolve_hnm_options
        result = optimize_single_spectrum_velocity_hnm(**arguments, return_metadata=metadata)
        if need_context:
            fit_options = resolve_hnm_options(arguments.get('hnm_options'), bnds=arguments['bnds'])
            width = fit_options['gauss_width']
            record = dict(result[7])
            trust = record.get('trust_region')
            constraints = {'velocity_disk': trust} if trust is not None else None
            context = None
    else:
        result = optimize_single_spectrum_velocity(
            **arguments, _return_search_context=need_context)
        fit_options = None
        if need_context:
            context = result[7]
            result = result[:7]
            width = arguments['gauss_width']
            constraints = context
            record = dict(method='DE', solver='DE', search_context=context)
        elif metadata:
            result = (*result, None)
    if spectral_options is not None:
        from iwave.spectral import estimate_spectral_uncertainty
        bounds = np.asarray(arguments['bnds'], float)
        diagnostic = estimate_spectral_uncertainty(
            arguments['measured_spectrum'], result[:2], result[2], bool(bounds[2,0]<bounds[2,1]),
            arguments['window_dims'], arguments['res'], arguments['fps'], arguments['vel_indx'],
            spectral_context, arguments['gravity_waves_switch'], arguments['turbulence_switch'],
            width, bounds, constraints, spectral_options, optimizer_success=bool(result[5]),
            penalized_fit=context is not None and context['final_penalty_weight']!=0)
        diagnostic['solver'] = method
        if method == 'HNM' and fit_options['kernel_truncate'] != spectral_options['kernel_truncate']:
            diagnostic['flags'].append('model_kernel_mismatch')
            diagnostic['valid'] = False
            diagnostic['parameter_valid'] = dict.fromkeys(('vy','vx','depth'),False)
            diagnostic['accepted_intervals'] = dict.fromkeys(('vy','vx','depth'),None)
        record['spectral_uncertainty'] = diagnostic
        result = (*result[:7], record)
    if mode_options is not None:
        from iwave.optimise_hnm import estimate_mode_ambiguity
        record['mode_ambiguity'] = estimate_mode_ambiguity(
            arguments['measured_spectrum'],result[:2],result[2],arguments['bnds'],
            arguments['window_dims'],arguments['res'],arguments['fps'],arguments['vel_indx'],
            spectral_context,record,fit_options,arguments['gravity_waves_switch'],arguments['turbulence_switch'],
            mode_options,optimizer_success=bool(result[5]),method=method)
        result = (*result[:7],record)
    return result

def reject_removed_uncertainty_options(arguments):
    removed=set(arguments)&{'calculate_uncertainty','uncertainty_options',
                            'calculate_hough_uncertainty','hough_uncertainty_options'}
    if removed:
        raise TypeError('Removed uncertainty API: '+', '.join(sorted(removed))+
                        '. Use calculate_spectral_uncertainty or calculate_mode_ambiguity.')


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
    penalty_weight: Optional[float]=None,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True,
    chunk_size: int = 50,
    gauss_width: Optional[float]=None,
    desc="Optimizing windows",
    pass_downsampling: Optional[List[int]] = None,
    method: str = "DE",
    de_options: Optional[dict] = None,
    hnm_options: Optional[dict] = None,
    progress: bool = True,
    return_diagnostics: bool = False,
    return_metadata: bool = False,
    calculate_spectral_uncertainty: bool = False,
    calculate_mode_ambiguity: bool = False,
    spectral_uncertainty_options: Optional[dict] = None,
    mode_ambiguity_options: Optional[dict] = None,
    spectral_contexts = None,
    **kwargs
) -> tuple:
    """Fit the supplied windows and return their velocity, depth, cost and quality.

    Parameters
    ----------
    measured_spectra : array or spectrum-array object
        Power in (window, frequency, ky, kx) order. A spectrum-array object can
        prepare windows only as each batch is requested.
    bnds_list : sequence
        For each window, limits ((vy_min, vy_max), (vx_min, vx_max), (d_min, d_max)).
        Velocities are m/s and depth is metres. Equal depth limits fix depth;
        different positive limits ask the method to fit it.
    vel_indx : float
        Depth-averaged / surface velocity.
    window_dims : three integers
        Original frame count, window height and width. Frequency count in the
        stored spectrum is smaller than this original frame count.
    res, fps : floats
        Metres per pixel and frame rate of the analysed sequence.
    penalty_weight, gauss_width, pass_downsampling
        Earlier DE options: speed penalty (default 1), model width (default 1),
        and reduced/full fitting passes (default [2, 1]). A speed penalty can
        suppress some outliers but also bias the fit; zero disables it.
    gravity_waves_switch, turbulence_switch : bool
        Include waves and/or patterns carried by the flow; both default True.
    chunk_size : integer
        Maximum windows handled together, default 50.
    method : 'DE' or 'HNM'
        Choose differential evolution (default) or Hough followed by Nelder-Mead.
    de_options, hnm_options : dict, optional
        Keep method settings separate. DE accepts gauss_width, penalty_weight,
        pass_downsampling and nested scipy_options. Older DE parameters/kwargs
        cannot be combined with de_options.
        HNM uses one flat dictionary for Hough peak/vote controls, fixed/adaptive
        column selection and local fitting. All starts share a circle with
        relative_radius=0.5 and min_radius. initial_step_factor sets the first
        velocity changes; depth_initial and log_depth_step control depth starts.
        n_candidates/maxfev default to 1/160 at fixed depth or 12/640 when depth
        is unknown; maxfev is per start. Explicit values override individual keys.
        See Iwave.velocimetry and README.md for the complete option descriptions.
    progress : bool
        Show the main program's progress bar. Default True.
    calculate_spectral_uncertainty, calculate_mode_ambiguity : bool
        Enable noise bounds near the selected fit and/or comparisons of existing
        HNM solutions. Both default False and neither changes the chosen fit.
    spectral_uncertainty_options, mode_ambiguity_options : dict
        Tune the checks, but do not enable them by merely supplying a dictionary.
    spectral_contexts : sequence of dictionaries, optional
        Full normalised spectra before filtering, with averaging information.
        Required for array inputs when either check is on. Ordinary video-window
        spectra obtain it automatically during the same transform calculation.
    return_diagnostics, return_metadata : bool
        Default output is (optimal, cost, quality). Diagnostics also returns
        status and messages. Metadata additionally returns detailed per-window
        records and requires diagnostics. These flags control returned information,
        not whether the two optional checks above are enabled.
    **kwargs
        Earlier direct SciPy DE settings. Prefer de_options['scipy_options'].

    Returns
    -------
    optimal : array
        One (vy, vx, depth) row per window, in m/s and metres.
    cost, quality : arrays
        Cost is smaller for better agreement. HNM reports the squared inverse
        overlap without a speed penalty. Quality is
        1 - 0.2*log10(cost_measured/cost_ideal): a perfect match gives 1,
        but it is not restricted to [0, 1] and is not a depth error estimate.

    Noise bounds use a model of spectral noise near one selected solution. A
    failed or missing check does not mean zero error; the old fixed-score-loss
    uncertainty API is not provided.
    """

    # Each optional check needs its own True/False switch.
    from iwave.spectral import resolve_spectral_request, validate_spectral_context
    from iwave.optimise_hnm import resolve_mode_request
    reject_removed_uncertainty_options(kwargs)
    method = normalise_method(method)
    spectral_options = resolve_spectral_request(calculate_spectral_uncertainty, spectral_uncertainty_options)
    mode_options = resolve_mode_request(calculate_mode_ambiguity, mode_ambiguity_options)
    # Supplying option values does not turn either check on.
    # If a check is requested, retain the full spectrum from before filtering.
    noise_options = [o for o in (spectral_options,mode_options) if o is not None]
    if noise_options and not return_metadata:
        raise ValueError('Enabled uncertainty diagnostics require return_metadata=True.')
    if noise_options:
        shape = (int(np.ceil(window_dims[0]/2)), window_dims[1], window_dims[2])
        if spectral_contexts is not None:
            if len(spectral_contexts)!=len(measured_spectra):raise ValueError('One spectral context is required per window.')
            for item in spectral_contexts:
                for options in noise_options:validate_spectral_context(item, shape, options)
        elif len(measured_spectra) and not hasattr(measured_spectra,'get_with_spectral_context'):
            raise ValueError('Supply spectral_contexts for precomputed/array spectra; full pre-threshold normalized power and averaging metadata are required.')
        for options,key in ((spectral_options,'parameter_scales'),(mode_options,'merge_tolerances')):
            scales = options[key] if options is not None else None
            if scales is not None and len(scales)<3 and any(np.asarray(b)[2,0]<np.asarray(b)[2,1] for b in bnds_list):
                raise ValueError('Joint diagnostics require three '+key+'.')
    # Reject incompatible settings before starting any fitting processes.
    legacy = dict(kwargs)
    for key, value in (("penalty_weight", penalty_weight),
                       ("pass_downsampling", pass_downsampling),
                       ("gauss_width", gauss_width)):
        if value is not None:
            legacy[key] = value
    method = normalise_method(method)
    de, _ = resolve_method_options(
        method, de_options=de_options, hnm_options=hnm_options, legacy=legacy)
    if (isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(progress, (bool, np.bool_)):
        raise ValueError("progress must be Boolean.")
    if return_metadata and not return_diagnostics:
        raise ValueError("return_metadata=True requires return_diagnostics=True.")
    if len(bnds_list) != len(measured_spectra):
        raise ValueError("One bounds tuple is required per spectrum.")
    if method == 'HNM':
        # Check all HNM options before starting fitting processes.
        from iwave.optimise_hnm import validate_hnm_setup
        # Use temporary limits to check common options even when there are no windows.
        for bounds in bnds_list if len(bnds_list) else [((-1, 1), (-1, 1), (1, 1))]:
            validate_hnm_setup(bounds, vel_indx, window_dims, res, fps,
                               gravity_waves_switch, turbulence_switch, hnm_options)
        # Wait until each window reaches its fitting process before filling in missing
        # defaults.
        # Fixed-depth and unknown-depth windows need different default search settings.
        backend_arguments = {'hnm_options': deepcopy(hnm_options)}
    else:
        backend_arguments = dict(penalty_weight=de["penalty_weight"],
                                 pass_downsampling=de["pass_downsampling"],
                                 gauss_width=de["gauss_width"],
                                 kwargs=de["scipy_options"])

    idxs = range(len(measured_spectra))
    results = [None] * len(idxs)
    available_workers = os.cpu_count() or 1
    requested_workers = CONCURRENCY if CONCURRENCY is not None else available_workers
    # Do not use more fitting processes than the number of windows in the batch.
    # Keep a valid setting of at least one, even for empty input or an unknown CPU count.
    max_workers = max(1, min(
        requested_workers, available_workers, chunk_size, len(idxs) or 1,
    ))

    # Create the fitting processes once so later batches reuse their Python setup
    # and the calculations already prepared inside each process.
    # Prepare and finish one batch at a time; do not read later data in the background.
    with tqdm(total=len(idxs), desc=desc, disable=not progress) as progress_bar:
        with ProcessPoolExecutor(
            mp_context=ctx, max_workers=max_workers, initializer=silence_output,
        ) as executor:
            for start in idxs[::chunk_size]:
                selection = slice(start, start + chunk_size)
                idx_sel = idxs[selection]
                if noise_options and spectral_contexts is None:
                    # One FFT calculation provides both filtered power and the unfiltered
                    # diagnostic data.
                    spectra_sel, contexts_sel = measured_spectra.get_with_spectral_context(
                        selection
                    )
                else:
                    spectra_sel = measured_spectra[selection]
                    contexts_sel = (
                        spectral_contexts[selection] if noise_options
                        else [None] * len(spectra_sel)
                    )
                bnds_sel = bnds_list[selection]
                args_sel = [
                    dict(
                        measured_spectrum=measured_spectrum,
                        bnds=bnds,
                        vel_indx=vel_indx,
                        window_dims=window_dims,
                        res=res,
                        fps=fps,
                        gravity_waves_switch=gravity_waves_switch,
                        turbulence_switch=turbulence_switch,
                        method=method,
                        _return_metadata=return_metadata,
                        calculate_spectral_uncertainty=bool(calculate_spectral_uncertainty),
                        calculate_mode_ambiguity=bool(calculate_mode_ambiguity),
                        spectral_uncertainty_options=spectral_options,
                        mode_ambiguity_options=mode_options,
                        _spectral_context=spectral_context,
                        **backend_arguments,
                    )
                    for measured_spectrum, bnds, spectral_context
                    in zip(spectra_sel, bnds_sel, contexts_sel)
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
                # Release references held by the main program before preparing the next
                # batch,
                # which might need a large amount of memory.
                del futures, args_sel, spectra_sel, contexts_sel

    # Combine the per-window results into output arrays.
    optimal = np.array([
        [res[0], res[1], res[2]] if res is not None else [np.nan, np.nan, np.nan] for res in results
    # Return (vy, vx, depth), including correctly shaped empty results when needed.
    ]).reshape(-1, 3)
    cost = np.array([res[3] if res is not None else np.nan for res in results])
    quality = np.array([res[4] if res is not None else np.nan for res in results])

    if return_diagnostics:
        status = np.array([res[5] if res is not None else False for res in results], dtype=bool)
        messages = np.array([res[6] if res is not None else "Missing result" for res in results], dtype=object)
        if return_metadata:
            metadata = np.empty(len(results), dtype=object)
            metadata[:] = [res[7] if res is not None else None for res in results]
            return optimal, cost, quality, status, messages, metadata
        return optimal, cost, quality, status, messages
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


