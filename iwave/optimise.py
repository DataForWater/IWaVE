import numpy as np
from scipy import optimize
from scipy.stats import chi2

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Tuple, List

from iwave import dispersion
# from iwave import spectral


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

    window_dims: [int, int, int]
        [dim_t, dim_y, dim_x] window dimensions

    res: float
        image resolution (m/pxl)

    fps: float
        image acquisition rate (fps)
    
    penalty_weight: float=1
        Because of the two branches of the surface spectrum (waves and turbulence-forced patterns), the algorithm 
        may choose the wrong solution causing a strongly overestimated velocity magnitude, especially 
        when smax > 2 * the actual velocity. The penalty_weight parameter increases the inertia of the optimiser, penalising
        solutions with a higher velocity magnitude. Setting penalty_weight > 0 will produce more stable results, but may slightly
        underestimate the velocity and overestimate the depth. Setting penalty_weight = 0 will eliminate the bias, 
        but may produce more outliers. If the velocity magnitude can be predicted reasonably, setting smax < 2 * the 
        typical velocity and setting penalty_weight = 0 will provide the most accurate results.

    gravity_waves_switch: bool=True
        if True, gravity waves are modelled
        if False, gravity waves are NOT modelled

    turbulence_switch: bool=True
        if True, turbulence-generated patterns and/or floating particles are modelled
        if False, turbulence-generated patterns and/or floating particles are NOT modelled

    gauss_width: float
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
    cost_function = cost_function*(1 + 2*penalty_weight*np.linalg.norm(velocity)/(res*fps))
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
        cost function to be minimised (non-dimensional)
        the cost function is defined as the inverse of the cross-correlation between the measured spectrum and the
        synthetic spectrum calculated according to the estimated flow parameters

    """
    spectra_correlation = measured_spectrum * synthetic_spectrum # calculate correlation
    cost = np.sum(synthetic_spectrum)* np.sum(measured_spectrum)  / np.sum(spectra_correlation) # calculate cost function
    
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
    
    # normalise so that the maximum at each frequency is 1    
    for i in range(preprocessed_spectrum.shape[0]):
        max_value = np.max(preprocessed_spectrum[i,:,:])
        preprocessed_spectrum[i,:,:] = preprocessed_spectrum[i,:,:]/max_value
        
    # remove NaNs
    preprocessed_spectrum = np.nan_to_num(preprocessed_spectrum)
    
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
    
    if downsample>1: # reduce dimensions of spectrum (for two-step approach)
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
    

def optimize_single_spectrum_velocity_unpack(args):
    return optimize_single_spectrum_velocity(*args)

def optimise_velocity(
    measured_spectra: np.ndarray,
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
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool], List[str]]:
    """
    Runs the optimisation to calculate the optimal velocity components

    Parameters
    ----------
    measured_spectra : np.ndarray
        measured and averaged 3D power spectra calculated with spectral.sliding_window_spectrum
        dimensions [N_windows, Nt, Ny, Nx]

    bnds_list: [(float, float), (float, float), (float, float)]
        [(min_vel_y, max_vel_y), (min_vel_x, max_vel_x), (min_depth, max_depth)] velocity (m/s) and depth (m) bounds
        this is supplied as a list with potentially different values for each window

    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)

    window_dims: [int, int, int]
        [dim_t, dim_y, dim_x] window dimensions

    res: float
        image resolution (m/pxl)

    fps: float
        image acquisition rate (fps)
        
    dof: float
        spectrum degrees of freedom
    
    penalty_weight: float=1
        Because of the two branches of the surface spectrum (waves and turbulence-forced patterns), the algorithm 
        may choose the wrong solution causing a strongly overestimated velocity magnitude, especially 
        when smax > 2 * the actual velocity. The penalty_weight parameter increases the inertia of the optimiser, penalising
        solutions with a higher velocity magnitude. Setting penalty_weight > 0 will produce more stable results, but may slightly
        underestimate the velocity. Setting penalty_weight = 0 will eliminate the bias, but may produce more outliers.
        If the velocity magnitude can be predicted reasonably, setting smax < 2 * the typical velocity and setting 
        penalty_weight = 0 will provide the most accurate results.

    gravity_waves_switch: bool=True
        if True, gravity waves are modelled
        if False, gravity waves are NOT modelled

    turbulence_switch: bool=True
        if True, turbulence-generated patterns and/or floating particles are modelled
        if False, turbulence-generated patterns and/or floating particles are NOT modelled
        
    downsample: int=1
        downsampling rate. If downsample > 1, then the spectrum is trimmed using a trimming ratio equal to 'downsample'.
        Trimming removes the high-wavenumber tails of the spectrum, which corresponds to downsampling the images spatially.

    gauss_width: float=1
        width of the synthetic spectrum smoothing kernel.
        gauss_width > 1 could be useful with very noisy spectra.

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

    args_list = [
        (measured_spectrum, bnds, vel_indx, window_dims, res, fps, penalty_weight, gravity_waves_switch, turbulence_switch, downsample, gauss_width, kwargs)
        for measured_spectrum, bnds in zip(measured_spectra, bnds_list)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(optimize_single_spectrum_velocity_unpack, args_list),
                total=len(args_list),
                desc="Optimizing windows"
            )
        )
        
    optimal = np.array([[res[0], res[1], res[2]] for res in results])  # vy, vx, d
    cost    = np.array([res[3] for res in results])
    quality = np.array([res[4] for res in results])
    status  = [res[5] for res in results]
    message = [res[6] for res in results]

    return optimal, cost, quality, status, message


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


