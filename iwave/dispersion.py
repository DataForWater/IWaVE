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
    """
    Create synthetic 3D spectrum based on tentative velocity and depth.

    Parameters
    ----------
    velocity : [float, float]
        velocity_y x velocity_x
        tentative surface velocity components along y and x (m/s)

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
    Intensities : np.ndarray
        synthetic 3D power spectrum to be compared with measured spectrum

    """

    # calculate the wavenumber/frequency arrays
    kt, ky, kx = spectral.wave_numbers(window_dims, res, fps)

    # calculate theoretical dispersion relation of gravity waves and turbulence-forced waves
    kt_gw, kt_turb = dispersion(ky, kx, velocity, depth, vel_indx)

    # calculate the theoretical 3D spectrum intensity
    th_spectrum = theoretical_spectrum(
        kt_gw, kt_turb, kt, gauss_width,
        gravity_waves_switch, turbulence_switch
    )

    th_spectrum = th_spectrum / np.sum(th_spectrum)

    return th_spectrum

def dispersion(
        ky: np.ndarray, 
        kx: np.ndarray, 
        velocity: Tuple[float, float],
        depth: float, 
        vel_indx: float
):
    """
    Calculate the frequency of gravity waves and floating particles according to tentative velocity and depth

    Parameters
    ----------
    ky: np.ndarray
        wavenumber array along the direction y

    kx: np.ndarray
        wavenumber array along the direction x

    velocity : [float, float]
        velocity_y x velocity_x
        tentative surface velocity components along y and x (m/s)

    depth : float
        tentative water depth (m)

    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)

    Returns
    -------
    kt_gw : np.ndarray
        1 x N_y x N_x: theoretical frequency of gravity waves for each [k_y, k_x] combination
    
    kt_turb : np.ndarray
        1 x N_y x N_x: theoretical frequency of turbulence-generated waves and/or floating particles for each [k_y, k_x] combination

    """

    # create 2D wavenumber grid
    kx, ky = np.meshgrid(kx, ky)

    # transpose to 1 x N_y x N_x
    ky = np.expand_dims(ky, axis=0)
    kx = np.expand_dims(kx, axis=0)

    # wavenumber modulus
    k_mod = np.sqrt(ky ** 2 + kx ** 2)  

    # calculate the main terms of the dispersion relation
    # Eq. (7), Dolcetti et al,. 2022
    beta = beta_calc(k_mod, depth, vel_indx)

    # Eq. (3), Dolcetti et al., 2022
    omega_a = omega_a_calc(ky, kx, velocity)

    # Eq. (4), Dolcetti et al., 2022
    omega_i = omega_i_calc(k_mod, depth)

    # calculate the frequency of gravity-capillary waves, Eq. (6), Dolcetti et al., 2022
    kt_gw = omega_gw_calc(beta, omega_a, omega_i)

    #calculate the frequency of turbulence-generated waves and/or floating particles, Eq. (3), Dolcetti et al., 2022
    kt_turb = omega_a

    return kt_gw, kt_turb

def beta_calc(
        k_mod: np.ndarray,
        depth: float,
        vel_indx: float
):
    """
    Calculate the beta term that represents the effect of the velocity variation along the vertical direction

    Parameters
    ----------
    k_mod: np.ndarray
        1 x N_y x N_x: wavenumber modulus

    depth : float
        tentative water depth (m)

    vel_indx : float
        surface velocity to depth-averaged-velocity index (-)

    Returns
    -------
    beta : np.ndarray
        1 x N_y x N_x: shear velocity term for the dispersion relation of gravity waves

    """

    # calculate shear rate coefficient of the equivalent linear velocity profile:
    # vel(z) = vel_0 * ( m*(z/depth) + 1 - m)
    m = 2 * (1 - vel_indx) 

    # calculate beta forcing beta = 0 where k_mod = 0
    # beta = (m / 2) * tanh(k_mod * depth) / (k_mod * depth)
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
    """
    Calculate the omega_a term of the dispersion relation, i.e., the frequency of turbulence-generated waves and/or floating particles

    Parameters
    ----------
    ky: np.ndarray
        1 x N_y x N_x: wavenumber y-component

    kx: np.ndarray
        1 x N_y x N_x: wavenumber x-component

    velocity : [float, float]
        [velocity_y, velocity_x] tentative velocity components (m/s)

    Returns
    -------
    omega_a : np.ndarray
        1 x N_y x N_x: frequency of turbulence-generated waves and/or floating particles (rad/s)

    """
    omega_a = ky * velocity[0] + kx * velocity[1]

    return omega_a

def omega_i_calc(
        k_mod: np.ndarray,
        depth: float
):
    """
    Calculate the omega_i term of the dispersion relation, i.e., the frequency of gravity-capillary waves in still water

    Parameters
    ----------
    k_mod: np.ndarray
        1 x N_y x N_x: wavenumber modulus

    depth : float
        tentative water depth (m)

    Returns
    -------
    omega_i : np.ndarray
        1 x N_y x N_x: frequency of gravity-capillary waves in still water (rad/s)

    """

    # calculate the frequency of capillary-gravity waves
    omega_i = np.sqrt(
            (const.g + const.surf_tens / const.density * k_mod ** 2) * k_mod * np.tanh(k_mod * depth))
    
    return omega_i

def omega_gw_calc(
        beta,
        omega_a,
        omega_i,
):
    """
    Calculate the frequency of gravity-capillary waves considering the effects of the flow velocity

    Parameters
    ----------
    omega_i: np.ndarray
        N_y x N_x: frequency in still water
        
    omega_a: np.ndarray
        N_y x N_x: flow velocity effect

    beta: np.ndarray
        N_y x N_x: velocity-profile effect

    Returns
    -------
    kt_gw : np.ndarray
        N_y x N_x: frequency of gravity-capillary waves with flow velocity (rad/s)

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
    """
    Assemble theoretical 3D spectrum with Gaussian width.

    Parameters
    ----------
    kt_gw: np.ndarray
        1 x N_y x N_x
        frequency of gravity-capillary waves (rad/s)

    kt_turb: np.ndarray
        1 x N_y x N_x
        frequency of turbulence-generated waves and/or floating particles (rad/s)

    kt : np.ndarray
        frequency array (rad/s)
        
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
    theor_spectrum : np.ndarray
        synthetic 3D power spectrum to be compared with measured spectrum

    """

    # build 3D kt_gw matrix with dimensions N_t x N_y x N_x
    kt_gw = np.tile(kt_gw, (len(kt), 1, 1))
    
    # build 3D kt_turb matrix with dimensions N_t x N_y x N_x
    kt_turb = np.tile(kt_turb, (len(kt), 1, 1))  

    # build 3D kt matrix with dimensions N_t x N_y x N_x
    kt = np.expand_dims(kt, axis=(1, 2))
    kt = np.tile(kt, (1, kt_gw.shape[-2], kt_gw.shape[-1]))

    # build 3D spectrum of gravity waves
    th_spectrum_gw = gauss_spectrum_calc(kt_gw, kt, gauss_width, gravity_waves_switch)
    
    # build 3D spectrum of turbulence-generated waves and/or floating particles
    th_spectrum_turb = gauss_spectrum_calc(kt_turb, kt, gauss_width, turbulence_switch)

    # assemble spectra
    th_spectrum = np.maximum(th_spectrum_gw, th_spectrum_turb)

    return th_spectrum

def gauss_spectrum_calc(
        kt_theory: np.ndarray,
        kt: np.ndarray,
        gauss_width: float,
        switch: bool = True,
):
    """
    creates branch of the theoretical 3D spectrum with Gaussian width based on input theoretical standard deviation

    Parameters
    ----------
    kt_theory: np.ndarray
        1 x N_y x N_x
        theoretical frequency of (gravity waves/turbulence waves) (rad/s)

    kt : np.ndarray
        frequency array (rad/s)

    gauss_width: float
        width of the synthetic spectrum smoothing kernel

    switch: bool=True
        if False, returns empty spectrum

    Returns
    -------
    gauss_spectrum : np.ndarray
        synthetic 3D power spectrum to be compared with measured spectrum

    """
    # builds 3D spectrum intensity with Gaussian smoothing around the theoretical dispersion relation
    if switch:
        dkt = kt - kt_theory
        gauss_spectrum = ne.evaluate('exp(-dkt**2 / gauss_width ** 2)')
        # gauss_spectrum = np.exp(-(kt - kt_theory) ** 2 / gauss_width ** 2)
    else:
        gauss_spectrum = np.zeros(kt.shape)

    return gauss_spectrum
