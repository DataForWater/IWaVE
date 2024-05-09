import numpy as np
from . import const

import numpy as np


def intensity(
        kt,
        kx,
        ky,
        v_x,
        v_y,
        d,
        vel_indx,
        gauss_width,
        gravity_waves_switch=True,
        turbulence_switch=True
):
    """Simulate intensity in wave spectrum using v_x, v_y and depth as parameters."""

    m = 2 * (1 - vel_indx)  # calculate shear rate based on velocity index
    kti = np.tile(kt, (
        len(ky), len(kx), 1))  # builds a 3D Kt vector to quantify the frequency for each point of the 3D spectrum

    # calculates theoretical dispersion relation of gravity waves and turbulence-forced waves
    kt_gw, kt_turb = dispersion(kx, ky, d, m, v_x, v_y)
    kt_gw = np.tile(kt_gw[:, :, np.newaxis], (1, 1, len(kt)))  # transforms into 3D
    kt_turb = np.tile(kt_turb[:, :, np.newaxis], (1, 1, len(kt)))  # transforms into 3D
    # theoretical spectrum intensity for gravity waves
    if gravity_waves_switch:
        theory_spectrum_gw = np.exp(-(kti - kt_gw) ** 2 / gauss_width ** 2)
    else:
        theory_spectrum_gw = np.zeros(kti.shape)
    # theoretical spectrum intensity for turbulence generate waves
    if turbulence_switch:
        theory_spectrum_turb = np.exp(-(kti - kt_turb) ** 2 / gauss_width ** 2)
    else:
        theory_spectrum_turb = np.zeros(kti.shape)
    return np.maximum(theory_spectrum_gw, theory_spectrum_turb)  # assembles the two spectra


def dispersion(kx, ky, d, m, v_x, v_y):
    """Calculate frequency of gravity-capillary waves and turbulence-forced waves.

     Calculation of frequency according to velocity v_x and v_y, depth d, and shear factor m.
     """

    kxi, kyi = np.meshgrid(kx, ky)
    ki = np.sqrt(kxi ** 2 + kyi ** 2)  # wavenumber modulus

    # frequency of gravity-capillary waves
    # Kt_gw = np.where(K !=0, (Kx*v_x + Ky*v_y)*(1-(m/2)*np.tanh(K*d)/(K*d)) +\
    #    np.sqrt((g+surftens/dens*K**2)*K*np.tanh(K*d)+(m*(Kx*v_x + Ky*v_y)*np.tanh(K*d)/(2*K*d))**2), 0)

    # separate terms that go to Inf when K=0
    term1 = np.divide(
        (m / 2) * np.tanh(ki * d),
        ki * d,
        out=np.zeros_like(ki),
        where=ki != 0
    )
    term2 = np.divide(
        (kxi * v_x + kyi * v_y) * np.tanh(ki * d),
        2 * ki * d,
        out=np.zeros_like(ki),
        where=ki != 0
    )
    # TODO split out in smaller parts
    kt_gw = np.where(
        ki != 0,
        (kxi * v_x + kyi * v_y) * (1 - term1) + np.sqrt(
            (const.g + const.surf_tens / const.density * ki ** 2) * ki * np.tanh(ki * d) + term2 ** 2),
        0
    )

    # frequency of turbulence-generated waves
    kt_turb = kxi * v_x + kyi * v_y

    # Kt_gw[K==0] = 0    # force to 0 at K=0
    # Kt_turb[K==0] = 0  # force to 0 at K=0

    return kt_gw, kt_turb
