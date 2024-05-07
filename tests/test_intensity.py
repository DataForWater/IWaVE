import numpy as np
import matplotlib.pyplot as plt
from iwave import intensity
from iwave import dispersion_relation_funct

"""test the calculation of the theoretical spectrum intensity."""

# image size (pxl)
img_size = [300, 300]
# pixel size (m)
pxl_size = 0.02
# frame rate (fps)
frame_rate = 20
# number of frames (#)
frames_num = 200

# parameters
gravity_waves_switch = 1 # if 0, then gravity waves are swtiched off
turbulence_switch = 1    # if 0, then turbulence-forced waves are switched off
gauss_width = 1          # width of the Gaussian smoothing kernel
vel_indx = 0.85          # surface to depth-averaged velocity index
m = 2*(1-vel_indx)       # velocity shear based on vel_indx

# free parameters for optimisation
v_x = 0.89
v_y = 0
d = 0.3

# wavenumber and frequency arrays
ks = 2*np.pi/pxl_size
Deltakx = ks/img_size[1]
Deltaky = ks/img_size[0]
kx = np.arange(0, ks, Deltakx)
ky = np.arange(0, ks, Deltaky)
kx = np.fft.fftshift(kx)
ky = np.fft.fftshift(ky)
I = np.where(kx==0)
kx[:I[0][0]] = kx[:I[0][0]] - kx[I[0][0]-1] - Deltakx
I = np.where(ky==0)
ky[:I[0][0]] = ky[:I[0][0]] - ky[I[0][0]-1] - Deltaky

Deltaf = frame_rate/frames_num
f = np.arange(0, frame_rate, Deltaf)
f = f[:int(np.ceil(len(f)/2))]
kt = 2*np.pi*f

intensities = intensity(kt, kx, ky, v_x, v_y, d, vel_indx, gauss_width, gravity_waves_switch, turbulence_switch)

# comparison with theoretical dispersion relation
Kt_gw, Kt_turb = dispersion_relation_funct(kx, ky, d, m, v_x, v_y)

fg, ax = plt.subplots(1, 2, figsize=(12, 4))
cax1 = ax[0].pcolor(kx, kt, np.squeeze(intensities[np.where(ky==0)[0][0], :, :]).T)
ax[0].plot(kx, Kt_gw[np.where(ky==0)[0][0], :], 'k', linewidth=1)
ax[0].plot(kx, Kt_turb[np.where(ky==0)[0][0], :], 'k', linewidth=1)
ax[0].set_xlabel('kx')
ax[0].set_ylabel('kt')
ax[0].set_ylim([0, frame_rate])
ax[0].set_xlim([-100, 100])
fg.colorbar(cax1, ax=ax[0])

cax2 = ax[1].pcolor(ky, kt, np.squeeze(intensities[:, np.where(kx==0)[0][0], :]).T)
ax[1].plot(ky, Kt_gw[:, np.where(ky==0)[0][0]], 'k', linewidth=1)
ax[1].plot(ky, Kt_turb[:, np.where(ky==0)[0][0]], 'k', linewidth=1)
ax[1].set_xlabel('ky')
ax[1].set_ylabel('kt')
ax[1].set_ylim([0, frame_rate])
ax[1].set_xlim([-100, 100])
fg.colorbar(cax2, ax=ax[1])

plt.show()
