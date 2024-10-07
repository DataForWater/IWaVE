
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import spectral


def plot_spectrum(windows: np.ndarray):
    windows = windows[-1]
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

    print("ok")


def img_windows_norm(img_windows):
    img_windows = img_windows - img_windows.mean(axis=0)
    img_windows = img_windows / img_windows.std(axis=0)
    img_windows[np.isinf(img_windows)] = 0
    img_windows[np.isnan(img_windows)] = 0

    return img_windows

###################################################################
fn_windows = '/home/sp/git_d4w/IWaVE/examples/sheaf/windows.bin'
###################################################################

with open(fn_windows, "rb") as f:
    windows = np.load(f)

plot_spectrum(windows)