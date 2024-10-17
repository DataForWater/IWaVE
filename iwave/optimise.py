import numpy as np
from typing import Tuple
from iwave import dispersion, spectral
from scipy import optimize


def optimise_velocity(
        measured_spectrum: np.ndarray,
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
) -> Tuple[float, float, float]:
    """
    Pre-processes the measured spectrum, 
    then runs the optimisation to calculate the optimal velocity components

    Parameters
    ----------
    measured_spectrum : np.ndarray
        measured and averaged 3D power spectrum calculated with spectral.py

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
    optimised_velocity_y : float
        optimised y velocity component (m/s)

    optimised_velocity_x : float
        optimised x velocity component (m/s)
        
    optimised_cost_function : float
        cost_function calculated with optimised velocity components
    """

    # optimisation
    opt = optimize.differential_evolution(
        cost_function_velocity,
        bounds=bnds,
        args=(measured_spectrum, depth, vel_indx, window_dims, res, fps, gauss_width, gravity_waves_switch, turbulence_switch),
        **kwargs
    )

    # optimised parameters and cost function
    optimised_velocity_y = opt.x[0]
    optimised_velocity_x = opt.x[1] 
    optimised_cost_function = opt.fun
    
    return optimised_velocity_y, optimised_velocity_x, optimised_cost_function

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


def optimise_velocity_depth(
    measured_spectrum: np.ndarray,
    bnds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    vel_indx: float,
    window_dims: Tuple[int, int, int],
    res: float,
    fps: float,
    gauss_width: float=1,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True,
    **kwargs
) -> Tuple[float, float, float, float]:
    """
    Pre-processes the measured spectrum, 
    then runs the optimisation to calculate the optimal velocity components

    Parameters
    ----------
    measured_spectrum : np.ndarray
        measured and averaged 3D power spectrum calculated with spectral.py

    bnds : [(float, float), (float, float), (float, float)]
        [(min_vel_y, max_vel_y), (min_vel_x, max_vel_x), (min_depth, max_depth)] velocity (m/s) and depth (m) bounds

    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)

    window_dims: [int, int, int]
        [dim_t, dim_y, dim_x] window dimensions
        
    kt: np.ndarray
        radian frequency vector (rad/s)

    ky: np.ndarray
        y-wavenumber vector (rad/m)
    
    kx: np.ndarray
        x-wavenumber vector (rad/m)

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
    optimised_velocity_y : float
        optimised y velocity component (m/s)

    optimised_velocity_x : float
        optimised x velocity component (m/s)
        
    optimised_depth : float
        optimised depth (m)

    optimised_cost_function : float
        cost_function calculated with optimised velocity components
    """

    # optimisation
    opt = optimize.differential_evolution(
        cost_function_velocity_depth,
        bounds=bnds,
        args=(measured_spectrum, vel_indx, window_dims, res, fps, gauss_width, gravity_waves_switch, turbulence_switch),
        **kwargs
    )
    # optimised parameters and cost function
    optimised_velocity_y = opt.x[0]
    optimised_velocity_x = opt.x[1] 
    optimised_depth = opt.x[2]
    optimised_cost_function = opt.fun
    
    return optimised_velocity_y, optimised_velocity_x, optimised_depth, optimised_cost_function


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
        velocity_y, velocity_x, depth
        tentative surface velocity components along y and x (m/s) and depth (m)

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
    
    depth = x[2]    # guessed depth
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
    spectra_correlation = measured_spectrum * synthetic_spectrum # calculate correlation
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
    preprocessed_spectrum = measured_spectrum / np.mean(measured_spectrum, axis=(1, 2), keepdims=True)

    # apply threshold
    threshold = spectrum_threshold 
    preprocessed_spectrum[preprocessed_spectrum < threshold] = 0

    # set the first slice (frequency=0) to 0
    preprocessed_spectrum[0,:,:] = 0

    # calculate the threshold frequency based on threshold velocity
    kt_gw, kt_turb = dispersion.dispersion(ky, kx, [velocity_threshold, velocity_threshold], 100, 1) # calculate frequency from velocity

    # set all frequencies higher than the threshold frequency to 0
    kt_reshaped = kt[:, np.newaxis, np.newaxis] # reshape kt to be broadcastable
    kt_turb_bc = np.broadcast_to(kt_turb, (kt.shape[0], kt_turb.shape[1], kt_turb.shape[2])) # broadcast kt_turb to match the dimensions of kt
    kt_bc = np.broadcast_to(kt_reshaped, kt_turb_bc.shape) # broadcast kt to match the dimensions of kt_turb
    preprocessed_spectrum[kt_bc > kt_turb_bc] = 0 # apply mask

    # remove NaNs
    preprocessed_spectrum = np.nan_to_num(preprocessed_spectrum)

    # normalisation
    preprocessed_spectrum = preprocessed_spectrum / np.sum(measured_spectrum)
    return preprocessed_spectrum