import numpy as np
from scipy import optimize

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Tuple

from iwave import dispersion

def cost_function_velocity(
    velocity: Tuple[float, float],
    measured_spectrum: np.ndarray,
    depth: float,
    vel_indx: float,
    window_dims: Tuple[int, int, int],
    res: float,
    fps: float,
    gauss_width: float,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True,
) -> float:
    """
    Creates a synthetic spectrum based on guessed parameters, 
    then compares it with the measured spectrum and returns a cost function for minimisation

    Parameters
    ----------
    velocity :  [float, float]
        velocity_y, velocity_x
        tentative surface velocity components along y and x (m/s)

    measured_spectrum : np.ndarray
        measured, averaged, and normalised 3D power spectrum calculated with spectral.py

    depth : float
        tentative water depth (m)

    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)

    window_dims: [int, int, int]
        [dim_t, dim_y, dim_x] window dimensions

    res: float
        image resolution (m/pxl)

    fps: float
        image acquisition rate (fps)
    
    gauss_width: float
        width of the synthetic spectrum smoothing kernel

    gravity_waves_switch: bool=True
        if True, gravity waves are modelled
        if False, gravity waves are NOT modelled

    turbulence_switch: bool=True
        if True, turbulence-generated patterns and/or floating particles are modelled
        if False, turbulence-generated patterns and/or floating particles are NOT modelled

    Returns
    -------
    cost_function : float
        cost function to be minimised

    """
    
    # calculate the synthetic spectrum based on the guess velocity
    synthetic_spectrum = dispersion.intensity(
        velocity, depth, vel_indx,
        window_dims, res, fps, gauss_width,
        gravity_waves_switch, turbulence_switch
    )
    cost_function = nsp_inv(measured_spectrum, synthetic_spectrum)
    return cost_function

def cost_function_velocity_depth(
    x: Tuple[float, float, float],
    measured_spectrum: np.ndarray,
    vel_indx: float,
    window_dims: Tuple[int, int, int],
    res: float,
    fps: float,
    gauss_width: float,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True,
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

    window_dims: [int, int, int]
        [dim_t, dim_y, dim_x] window dimensions

    res: float
        image resolution (m/pxl)

    fps: float
        image acquisition rate (fps)
    
    gauss_width: float
        width of the synthetic spectrum smoothing kernel

    gravity_waves_switch: bool=True
        if True, gravity waves are modelled
        if False, gravity waves are NOT modelled

    turbulence_switch: bool=True
        if True, turbulence-generated patterns and/or floating particles are modelled
        if False, turbulence-generated patterns and/or floating particles are NOT modelled

    Returns
    -------
    cost_function : float
        cost function to be minimised

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

    synthetic_spectrum: np.ndarray
        synthetic 3D power spectrum

    Returns
    -------
    cost : float
        cost function to be minimised

    """
    # spectra_correlation = measured_spectrum * synthetic_spectrum # calculate correlation
    spectra_correlation = measured_spectrum * synthetic_spectrum /np.sum(synthetic_spectrum) # calculate correlation
    cost = 1 / np.sum(spectra_correlation) # calculate cost function

    return cost


def spectrum_preprocessing(
        measured_spectrum: np.ndarray, 
        kt: np.ndarray,
        ky: np.ndarray,
        kx: np.ndarray,
        velocity_threshold: float,
        spectrum_threshold: float=1
) -> np.ndarray:
    """
    pre-processing of the measured spectrum to improve convergence of the optimisation

    Parameters
    ----------
    measured_spectrum : np.ndarray
        measured, averaged, and normalised 3D power spectrum calculated with spectral.py
        dimensions [wi, kti, kyi, kx]

    kt: np.ndarray
        radian frequency vector (rad/s)

    ky: np.ndarray
        y-wavenumber vector (rad/m)
    
    kx: np.ndarray
        x-wavenumber vector (rad/m)

    velocity_threshold: float
        maximum threshold velocity for spectrum filtering (m/s).
    
    spectrum_threshold: float=1
        threshold parameter for spectrum filtering. 
        the spectrum with amplitude < threshold_preprocessing * mean(measured_spectrum) is filtered out.
        threshold_preprocessing < 1 yields a more severe filtering but could eliminate part of useful signal.

    Returns
    -------
    preprocessed_spectrum : np.ndarray
        pre-processed and normalised measured 3D spectrum

    """
    # spectrum normalisation: divides the spectrum at each frequency by the average across all wavenumber combinations at the same frequency
    preprocessed_spectrum = measured_spectrum / np.mean(measured_spectrum, axis=(2, 3), keepdims=True)

    # apply threshold
    threshold = spectrum_threshold * np.mean(preprocessed_spectrum, axis = 1, keepdims = True)
    preprocessed_spectrum[preprocessed_spectrum < threshold] = 0

    # set the first slice (frequency=0) to 0
    preprocessed_spectrum[:,0,:,:] = 0

    kt_threshold = dispersion_threshold(ky, kx, velocity_threshold)
    
    # set all frequencies higher than the threshold frequency to 0
    kt_reshaped = kt[:, np.newaxis, np.newaxis] # reshape kt to be broadcastable
    kt_threshold_bc = np.broadcast_to(kt_threshold, (kt.shape[0], kt_threshold.shape[1], kt_threshold.shape[2])) # broadcast kt_threshold to match the dimensions of kt
    kt_bc = np.broadcast_to(kt_reshaped, kt_threshold_bc.shape) # broadcast kt to match the dimensions of kt_threshold
    mask = np.where(kt_bc <= kt_threshold_bc, 1, 0) # create mask
    mask = np.expand_dims(mask, axis=0)

    preprocessed_spectrum = preprocessed_spectrum *mask # apply mask

    # remove NaNs
    preprocessed_spectrum = np.nan_to_num(preprocessed_spectrum)

    # normalisation
    preprocessed_spectrum = preprocessed_spectrum / np.sum(measured_spectrum, axis=(1, 2, 3), keepdims = True)
    return preprocessed_spectrum

def dispersion_threshold(
    ky, 
    kx, 
    velocity_threshold
) -> np.ndarray:
    
    """
    Calculate the frequency corresponding to the threshold velocity

    Parameters
    ----------
    ky: np.ndarray
        wavenumber array along the direction y

    kx: np.ndarray
        wavenumber array along the direction x

    velocity_threshold : float
        threshold_velocity (m/s)

    Returns
    -------
    kt_threshold : np.ndarray
        1 x N_y x N_x: threshold frequency

    """

    # create 2D wavenumber grid
    kx, ky = np.meshgrid(kx, ky)

    # transpose to 1 x N_y x N_x
    ky = np.expand_dims(ky, axis=0)
    kx = np.expand_dims(kx, axis=0)

    # wavenumber modulus
    k_mod = np.sqrt(ky ** 2 + kx ** 2)  
    
    return k_mod*velocity_threshold


def cost_function_velocity_wrapper(
    x: Tuple[float, float],
    *args
) -> float:
    return cost_function_velocity(x, *args)


def optimize_single_spectrum_velocity(
    measured_spectrum: np.ndarray,
    bnds: Tuple[Tuple[float, float], Tuple[float, float]],
    depth: float,
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    gauss_width: float,
    gravity_waves_switch: bool,
    turbulence_switch: bool,
    kwargs: dict
) -> Tuple[float, float, float]:
    opt = optimize.differential_evolution(
        cost_function_velocity_wrapper,
        bounds=bnds,
        args=(measured_spectrum, depth, vel_indx, window_dims, res, fps, gauss_width, gravity_waves_switch, turbulence_switch),
        **kwargs
    )
    return float(opt.x[0]), float(opt.x[1]), float(opt.fun)

def optimize_single_spectrum_velocity_unpack(args):
    return optimize_single_spectrum_velocity(*args)

def optimise_velocity(
    measured_spectra: np.ndarray,
    bnds: Tuple[Tuple[float, float], Tuple[float, float]],
    depth: float,
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    gauss_width: float=1,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True,
    **kwargs
) -> np.ndarray:
    """
    Runs the optimisation to calculate the optimal velocity components

    Parameters
    ----------
    measured_spectrum : np.ndarray
        measured and averaged 3D power spectra calculated with spectral.sliding_window_spectrum
        dimensions [N_windows, Nt, Ny, Nx]

    bnds : [(float, float), (float, float)]
        [(min_vel_y, max_vel_y), (min_vel_x, max_vel_x)] velocity bounds (m/s)

    depth : float
        water depth (m)

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

    **kwargs : dict
        keyword arguments to pass to `scipy.optimize.differential_evolution, see also
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

    Returns
    -------
    optimal : np.ndarray

    optimal[:,0] : float
        optimised y velocity component (m/s)

    optimal[:,1] : float
        optimised x velocity component (m/s)
        
    optimal[:,2] : float
        cost_function calculated with optimised velocity components
    """

    args_list = [
        (measured_spectrum, bnds, depth, vel_indx, window_dims, res, fps, gauss_width, gravity_waves_switch, turbulence_switch, kwargs)
        for measured_spectrum in measured_spectra
    ]

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(optimize_single_spectrum_velocity_unpack, args_list),
                total=len(args_list),
                desc="Optimizing windows"
            )
        )

    optimised_params = np.array([
        [float(result[0]), float(result[1]), float(result[2])] 
        for result in results
    ])

    return optimised_params


def cost_function_velocity_depth_wrapper(
    x: Tuple[float, float, float],
    *args
) -> float:
    return cost_function_velocity_depth(x, *args)


def optimize_single_spectrum_velocity_depth(
    measured_spectrum: np.ndarray,
    bnds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    gauss_width: float,
    gravity_waves_switch: bool,
    turbulence_switch: bool,
    kwargs: dict
) -> Tuple[float, float, float, float]:
    bnds[2] = np.log(bnds[2]) # transform the boundaries for the depth parameter to improve convergence
    opt = optimize.differential_evolution(
        cost_function_velocity_depth_wrapper,
        bounds=bnds,
        args=(measured_spectrum, vel_indx, window_dims, res, fps, gauss_width, gravity_waves_switch, turbulence_switch),
        **kwargs
    )
    opt.x[2] = np.exp(opt.x[2]) # transforms back optimised depth into linear scale
    return float(opt.x[0]), float(opt.x[1]), float(opt.x[2]), float(opt.fun)


def optimize_single_spectrum_velocity_depth_unpack(args):
    return optimize_single_spectrum_velocity_depth(*args)


def optimise_velocity_depth(
    measured_spectra: np.ndarray,
    bnds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    vel_indx: float,
    window_dims: Tuple[int, int, int], 
    res: float, 
    fps: float,
    gauss_width: float=1,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True,
    **kwargs
) -> np.ndarray:
    """
    Runs the optimisation to calculate the optimal velocity components

    Parameters
    ----------
    measured_spectra : np.ndarray
        measured and averaged 3D power spectra calculated with spectral.sliding_window_spectrum
        dimensions [N_windows, Nt, Ny, Nx]

    bnds : [(float, float), (float, float), (float, float)]
        [(min_vel_y, max_vel_y), (min_vel_x, max_vel_x), (min_depth, max_depth)] velocity (m/s) and depth (m) bounds

    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)

    window_dims : [int, int, int]
        [dim_t, dim_y, dim_x] window dimensions

    res : float
        image resolution (m/pxl)

    fps : float
        image acquisition rate (fps)
    
    gauss_width : float=1
        width of the synthetic spectrum smoothing kernel.
        gauss_width > 1 could be useful with very noisy spectra.

    gravity_waves_switch : bool=True
        if True, gravity waves are modelled
        if False, gravity waves are NOT modelled

    turbulence_switch : bool=True
        if True, turbulence-generated patterns and/or floating particles are modelled
        if False, turbulence-generated patterns and/or floating particles are NOT modelled

    **kwargs : dict
        keyword arguments to pass to `scipy.optimize.differential_evolution, see also
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

    Returns
    -------
    optimal : np.ndarray

    optimal[:,0] : float
        optimised y velocity component (m/s)

    optimal[:,1] : float
        optimised x velocity component (m/s)

    optimal[:,2] : float
        optimised depth (m)
        
    optimal[:,3] : float
        cost_function calculated with optimised velocity components
    """
    
    args_list = [
        (measured_spectrum, bnds, vel_indx, window_dims, res, fps, gauss_width, gravity_waves_switch, turbulence_switch, kwargs)
        for measured_spectrum in measured_spectra
    ]

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(optimize_single_spectrum_velocity_depth_unpack, args_list),
                total=len(args_list),
                desc="Optimizing windows"
            )
        )

    optimised_params = np.array([
        [float(result[0]), float(result[1]), float(result[2]), float(result[3])] 
        for result in results
    ])

    return optimised_params