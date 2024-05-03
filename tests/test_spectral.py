import pytest
import numpy as np
from iwave import spectral
import matplotlib.pyplot as plt


def test_get_wave_numbers(img_windows, res=0.02, fps=25):
    # feed in only the first image window, and of that, only the first frame
    kt, ky, kx = spectral._get_wave_numbers(img_windows[0].shape, res=res, fps=fps)
    print(ky.shape)


def test_numpy_fft(img_windows):
    # normalize images
    img_windows = img_windows - img_windows.mean(axis=0)
    img_windows = img_windows / img_windows.std(axis=0)
    img_windows[np.isinf(img_windows)] = 0
    img_windows[np.isnan(img_windows)] = 0
    spectrum = spectral.numpy_fourier_transform(img_windows[-1])

    # make too large components equal to zero (?)
    # test_min_val = 1 * spectrum.mean(axis=0)repmat(mean(kOmegaSpectrum(:,:, 2: ceil(
    #     end / 2)), 3), ...
    # [1, 1, size(kOmegaSpectrum, 3)]))
    # spectrum[spectrum < min_spectrum] = 0.
    # check = np.abs(np.fft.ifft(spectrum, axis=0))**2
    # kt, ky, kx = spectral._get_wave_numbers(
    #     img_windows[0].shape,
    #     0.02, 20
    # )
    # plt.imshow(np.log(check[50]))
    # plt.colorbar()
    # plt.show()
    # print(spectrum.shape)

