import numpy as np

def intensity(kt, kx, ky, v_x, v_y, d, vel_indx, gauss_width, gravity_waves_switch, turbulence_switch):
  """Simulate intensity in wave spectrum using v_x, v_y and depth as parameters."""
  
    m = 2*(1 - vel_indx)  # calculate shear rate based on velocity index

    Kt = np.tile(kt,(len(ky),len(kx),1))  # builds a 3D Kt vector to quantify the frequency for each point of the 3D spectrum

    # calculates theoretical dispersion relation of gravity waves and turbulence-forced waves
    Kt_gw, Kt_turb = dispersion_relation_funct(kx, ky, d, m, v_x, v_y)
  
    Kt_gw = np.tile(Kt_gw[:,:,np.newaxis], (1, 1, len(kt)))   # transforms into 3D
    Kt_turb = np.tile(Kt_turb[:,:,np.newaxis], (1, 1, len(kt)))   # transforms into 3D

    if gravity_waves_switch:  # creates the theoretical spectrum intensity for gravity waves
        theory_spectrum_gw = np.exp(-(Kt - Kt_gw)**2 / gauss_width**2)
    else:
        theory_spectrum_gw = np.zeros(Kt.shape)

    if turbulence_switch:  # creates the theoretical spectrum intensity for turbulence generate waves
        theory_spectrum_turb = np.exp(-(Kt - Kt_turb)**2 / gauss_width**2)
    else:
        theory_spectrum_turb = np.zeros(Kt.shape)

    intensities = np.maximum(theory_spectrum_gw, theory_spectrum_turb)  # assembles the two spectra

    return intensities
    
