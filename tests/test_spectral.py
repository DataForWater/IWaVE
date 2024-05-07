import numpy as np
from iwave import spectral


def test_get_wave_numbers(img_windows, res=0.02, fps=25):
    # feed in only the first image window, and of that, only the first frame
    kt, ky, kx = spectral._get_wave_numbers(img_windows[0].shape, res=res, fps=fps)
    print(ky.shape)


def test_numpy_fft(img_windows_norm):
    windows = img_windows_norm[-1]
    spectrum = spectral.numpy_fourier_transform(windows)
    assert spectrum.shape == (
        int(np.ceil(len(windows) / 2)),
        windows.shape[1],
        windows.shape[2]
    )
    # TODO: code below for plotting, may be moved to API
    # kt, ky, kx = spectral._get_wave_numbers(
    #     img_windows[0].shape,
    #     0.02, 20
    # )
    # plt.imshow(np.log(check[50]))
    # plt.colorbar()
    # plt.show()
    # print(spectrum.shape)


def test_numba_fft(img_windows_norm):
    spectrum = spectral.numba_fourier_transform(img_windows_norm[-1])
    spectrum_numpy = spectral.numpy_fourier_transform(img_windows_norm[-1])
    # test if the answer is (very) close to the answer of the numpy version
    assert np.allclose(spectrum_numpy, spectrum)


def test_numba_fft_multi(img_windows_norm):
    spectra = spectral.numba_fourier_transform_multi(img_windows_norm)
    # test if all image windows give the same result as the numpy version of the spectrum code
    for windows, spectrum in zip(img_windows_norm, spectra):
        spectrum_numpy = spectral.numpy_fourier_transform(windows)
        assert np.allclose(spectrum_numpy, spectrum)

