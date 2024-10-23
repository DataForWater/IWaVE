
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import spectral


def plot_spectrum(windows: np.ndarray):
    windows = windows[-1]

    """
    for nn in range(0, 100):
        plt.imshow(windows[nn])
        plt.show()
    """
    
    spectrum = spectral._numpy_fourier_transform(windows)

    plt.imshow(np.log(spectrum[:, :, 32]))
    plt.colorbar()
    plt.show()

    kt, ky, kx = spectral.wave_numbers(windows.shape, 0.02, 20)
    X, Y = np.meshgrid(kt, kx)
    
    Z = spectrum[:,:,1]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)

    plt.show()

    plt.imshow(np.log(spectrum[:, :, 32]))
    plt.colorbar()
    plt.show()

    print(spectrum.shape)

def img_windows_norm(img_windows):
    img_windows = img_windows - img_windows.mean(axis=0)
    img_windows = img_windows / img_windows.std(axis=0)
    img_windows[np.isinf(img_windows)] = 0
    img_windows[np.isnan(img_windows)] = 0

    return img_windows

def plots():
    plot_results(optimised_parameter, kt, ky, kx)
    """
    for iw = selected_windows:
        kt_gw, kt_turb = dispersion(ky, kx, optimised_velocity[iw], depth[iw], vel_indx)
        plot(kx, kt, measured_spectrum[iw, ky==0, :, :]) # plot x-t spectrum cross-section
        hold on; plot(kx, kt_gw[ky==0, :]) # plot gravity waves theoretical relation based on optimised parameters
        hold on; plot(kx, kt_turb[ky==0, :]) # plot turbulent waves theoretical relation based on optimised parameters

        plot(kx, kt, measured_spectrum[iw, :, kx==0, :]) # plot y-t spectrum cross-section
        hold on; plot(kx, kt_gw[:, kx==0]) # plot gravity waves theoretical relation based on optimised parameters
        hold on; plot(kx, kt_turb[:, kx==0]) # plot turbulent waves theoretical relation based on optimised parameters
        """

"""
###################################################################
#fn_windows = '/home/sp/git_d4w/IWaVE/examples/sheaf/windows.bin'
fn_windows = '/home/sp/pCloudDrive/Docs/d4w/iwave/windows_0000_0200.bin'
###################################################################

with open(fn_windows, "rb") as f:
    windows = np.load(f)

plot_spectrum(windows)
"""