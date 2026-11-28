import numpy as np
import numexpr as ne
from . import const
from typing import Tuple
from iwave import spectral

def intensity(
    velocity: Tuple[float, float],
    depth: float,
    vel_indx: float,
    window_dims: Tuple[int, int, int],
    res: float,
    fps: float,
    gauss_width: float,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True
):
    """Predict a window's power spectrum for a trial velocity and depth.

    Parameters
    ----------
    velocity : pair of floats
        Trial surface velocity (vy, vx), in m/s.
    depth : float
        Water depth, in metres.
    vel_indx : float
        Depth-averaged velocity divided by surface velocity.
    window_dims : three integers
        Numbers of frames, y pixels and x pixels in the analysed window.
    res : float
        Image resolution, in metres per pixel.
    fps : float
        Frame rate of the analysed image sequence.
    gauss_width : float
        Width of the Gaussian around each predicted frequency, in rad/s,
        using exp(-((frequency - prediction)/gauss_width)**2).
    gravity_waves_switch, turbulence_switch : bool
        Include waves and/or patterns carried by the flow.

    Returns
    -------
    numpy.ndarray
        Predicted power in (frequency, ky, kx) order, ready for comparison with
        the measured spectrum.
    """

    # Set the frequency and spatial-wavenumber axes for this window.
    kt, ky, kx = spectral.wave_numbers(window_dims, res, fps)

    # Predict the frequencies of waves and flow-carried patterns for the trial velocity.
    kt_gw, kt_turb = dispersion(ky, kx, velocity, depth, vel_indx)

    # Build the predicted power spectrum around those frequencies.
    th_spectrum = theoretical_spectrum(
        kt_gw, kt_turb, kt, gauss_width,
        gravity_waves_switch, turbulence_switch
    )

    return th_spectrum



def dispersion(
        ky: np.ndarray, 
        kx: np.ndarray, 
        velocity: Tuple[float, float],
        depth: float, 
        vel_indx: float
):
    """Predict frequencies for waves and for patterns carried by the flow.

    ky and kx are the y and x wavenumber axes, in rad/m. velocity is (vy, vx)
    in m/s; depth is in metres. vel_indx is depth-averaged / surface velocity.
    Return (kt_gw, kt_turb), both in rad/s with shape (1, number_y, number_x).
    The first describes gravity-capillary waves; the second describes flow-carried
    patterns or particles.
    """

    # Make a grid containing every combination of x and y wavenumbers.
    kx, ky = np.meshgrid(kx, ky)

    # Add a length-one frequency axis so these arrays fit the spectrum dimensions.
    ky = np.expand_dims(ky, axis=0)
    kx = np.expand_dims(kx, axis=0)

    # Calculate the magnitude of the spatial wavenumber.
    k_mod = np.sqrt(ky ** 2 + kx ** 2)  

    # Calculate the terms needed for the wave-frequency prediction.
    # See Eq. (7), Dolcetti et al., 2022.
    beta = beta_calc(k_mod, depth, vel_indx)

    # See Eq. (3), Dolcetti et al., 2022.
    omega_a = omega_a_calc(ky, kx, velocity)

    # See Eq. (4), Dolcetti et al., 2022.
    omega_i = omega_i_calc(k_mod, depth)

    # Predict gravity-capillary wave frequencies: Eq. (6), Dolcetti et al., 2022.
    kt_gw = omega_gw_calc(beta, omega_a, omega_i)

    # Predict frequencies of flow-carried patterns: Eq. (3), Dolcetti et al., 2022.
    kt_turb = omega_a

    return kt_gw, kt_turb

def beta_calc(
        k_mod: np.ndarray,
        depth: float,
        vel_indx: float
):
    """Calculate how a vertical velocity gradient changes the wave frequency.

    k_mod is spatial-wavenumber magnitude in rad/m; depth is in metres.
    vel_indx is depth-averaged velocity divided by surface velocity.
    Return the dimensionless correction beta at each spatial wavenumber.
    """

    # Find the slope of the equivalent linear velocity profile:
    # vel(z) = vel_0 * (m*(z/depth) + 1 - m).
    m = 2 * (1 - vel_indx) 

    # Calculate the velocity-profile correction; set it to zero at zero wavenumber.
    # Away from zero, beta = (m / 2) * tanh(k_mod * depth) / (k_mod * depth).
    beta = np.divide(
        (m / 2) * np.tanh(k_mod * depth),
        k_mod * depth,
        out=np.zeros_like(k_mod),
        where=k_mod != 0
    )

    return beta

def omega_a_calc(
        ky: np.ndarray,
        kx: np.ndarray,
        velocity: Tuple[float, float]
):
    """Calculate the frequency change caused by movement with the flow.

    ky and kx contain spatial wavenumbers in rad/m. velocity is (vy, vx) in m/s.
    Return ky*vy + kx*vx, in rad/s, for each wavenumber pair.
    """
    omega_a = ky * velocity[0] + kx * velocity[1]

    return omega_a

def omega_i_calc(
        k_mod: np.ndarray,
        depth: float
):
    """Calculate gravity-capillary wave frequencies in still water.

    k_mod is wavenumber magnitude in rad/m and depth is in metres.
    Return the intrinsic wave frequency, in rad/s.
    """

    # Predict the gravity-capillary wave frequency in still water.
    omega_i = np.sqrt(
            (const.g + const.surf_tens / const.density * k_mod ** 2) * k_mod * np.tanh(k_mod * depth))
    
    return omega_i

def omega_gw_calc(
        beta,
        omega_a,
        omega_i,
):
    """Combine still-water frequency with flow and velocity-profile effects.

    omega_i is the still-water wave frequency; omega_a is the change due to
    movement with the flow, both in rad/s. beta describes the vertical velocity
    profile. Return the predicted moving-water wave frequency in rad/s.
    """

    kt_gw = (1 - beta) * omega_a + np.sqrt(
        (beta * omega_a)**2 + omega_i**2
    )

    return kt_gw


def theoretical_spectrum(
    kt_gw: np.ndarray,
    kt_turb: np.ndarray,
    kt: np.ndarray,
    gauss_width: float,
    gravity_waves_switch: bool=True,
    turbulence_switch: bool=True
):
    """Build power around the predicted wave and flow-carried frequencies.

    kt_gw and kt_turb give the two predictions in rad/s; kt is the measured
    frequency axis. gauss_width sets Gaussian width using
    exp(-((kt - prediction)/gauss_width)**2). Each switch enables its own part.
    Where both are enabled, keep the larger power value, not their sum.
    Return power in (frequency, ky, kx) order.
    """

    # Repeat the predicted wave frequency along the measured frequency axis.
    kt_gw = np.tile(kt_gw, (len(kt), 1, 1))
    
    # Repeat the predicted frequency of flow-carried patterns along that same axis.
    kt_turb = np.tile(kt_turb, (len(kt), 1, 1))  

    # Repeat the measured frequency axis across all spatial positions in the spectrum.
    kt = np.expand_dims(kt, axis=(1, 2))
    kt = np.tile(kt, (1, kt_gw.shape[-2], kt_gw.shape[-1]))

    # Build the gravity-wave part of the predicted spectrum.
    th_spectrum_gw = gauss_spectrum_calc(kt_gw, kt, gauss_width, gravity_waves_switch)
    
    # Build the part for patterns or particles carried by the flow.
    th_spectrum_turb = gauss_spectrum_calc(kt_turb, kt, gauss_width, turbulence_switch)

    # At each spectrum position, keep the larger of the two predicted powers.
    th_spectrum = np.maximum(th_spectrum_gw, th_spectrum_turb)

    return th_spectrum



def gauss_spectrum_calc(
        kt_theory: np.ndarray,
        kt: np.ndarray,
        gauss_width: float,
        switch: bool = True,
):
    """Build one Gaussian-shaped part of the predicted power spectrum.

    kt_theory gives its predicted frequency; kt gives the measured frequencies.
    Both use rad/s. gauss_width appears in
    exp(-((kt - kt_theory)/gauss_width)**2). If switch is False, return zeros.
    Otherwise return the power at each supplied frequency and spatial wavenumber.
    """
    # Give the predicted frequency line a Gaussian width, rather than zero thickness.
    if switch:
        dkt = kt - kt_theory
        gauss_spectrum = ne.evaluate('exp(-dkt**2 / gauss_width ** 2)')
    else:
        gauss_spectrum = np.zeros(kt.shape)

    return gauss_spectrum



def spectrum_downsample(
        measured_spectrum: np.ndarray,
        res: float,
        fps: float,
        window_dims : Tuple[int, int, int],
        downsample: int,
) -> Tuple[np.ndarray, float, float, Tuple[int, int, int]]:
    """Keep the central spatial-wavenumber region for a reduced fitting pass.

    measured_spectrum is in (frequency, ky, kx) order. downsample sets how much
    to reduce its two spatial dimensions: values above one remove the outer,
    high-wavenumber parts. This is a crop of the spectrum, not averaging pixels.
    Return the cropped spectrum, res, fps, and the adjusted window_dims.
    res (m/pixel) and fps are returned unchanged; window_dims contains
    (frame count, y pixels, x pixels) for the reduced calculation.
    """
    kt_old, ky_old, kx_old = spectral.wave_numbers(window_dims, res, fps)
    res = res*downsample
    fps = fps
    window_dims = [window_dims[0], np.int64(np.ceil(window_dims[1]/downsample)), np.int64(np.ceil(window_dims[2]/downsample))]
    kt_new, ky_new, kx_new = spectral.wave_numbers(window_dims, res, fps)
    # Choose matching start and end positions on each wavenumber axis.
    # Later calculations must reconstruct axes aligned with the original spectrum.
    kx_indx_first = np.where(np.isclose(kx_old, np.min(kx_new), atol = 1e-06))[0][0]
    kx_indx_last = np.where(np.isclose(kx_old, np.max(kx_new), atol = 1e-06))[0][0]
    kx_indx = np.arange(kx_indx_first, kx_indx_last + 1)
    ky_indx_first = np.where(np.isclose(ky_old, np.min(ky_new), atol = 1e-05))[0][0]
    ky_indx_last = np.where(np.isclose(ky_old, np.max(ky_new), atol = 1e-05))[0][0]
    ky_indx = np.arange(ky_indx_first, ky_indx_last + 1)
    kt_indx_first = np.where(np.isclose(kt_old, np.min(kt_new), atol = 1e-05))[0][0]
    kt_indx_last = np.where(np.isclose(kt_old, np.max(kt_new), atol = 1e-05))[0][0]
    kt_indx = np.arange(kt_indx_first, kt_indx_last + 1)
    if len(kx_indx) != len(kx_new) or len(ky_indx) != len(ky_new) or len(kt_indx) != len(kt_new):
        raise ValueError("The dimensions of the trimmed array do not match the target after resampling")
    # Keep only the selected central part of the spectrum.
    trimmed_spectrum = measured_spectrum[np.ix_(kt_indx,ky_indx,kx_indx)]
    return trimmed_spectrum, res, fps, window_dims
