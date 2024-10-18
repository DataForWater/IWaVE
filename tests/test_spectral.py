import numpy as np
import pytest

from iwave import spectral


def test_get_wave_numbers(img_windows, res=0.02, fps=25):
    """Check shape of wave numbers, conditional on shape of image."""
    # feed in only the first image window, and of that, only the first frame
    kt, ky, kx = spectral.wave_numbers(img_windows[0].shape, res=res, fps=fps)
    assert len(kt) == int(np.ceil(len(img_windows[0])/2))
    assert len(ky) == img_windows.shape[-2]
    assert len(kx) == img_windows.shape[-1]


def test_numpy_fft(img_windows_norm):
    """Check shape of spectrum."""
    windows = img_windows_norm[-1]
    spectrum = spectral._numpy_fourier_transform(windows)
    assert spectrum.shape == (
        int(np.ceil(len(windows) / 2)),
        windows.shape[1],
        windows.shape[2]
    )
    # TODO: code below for plotting, may be moved to API
    # import matplotlib.pyplot as plt
    # kt, ky, kx = spectral._get_wave_numbers(
    #     windows.shape,
    #     0.02, 20
    # )
    # plt.imshow(np.log(spectrum[:, :, 32]))
    # plt.colorbar()
    # plt.show()
    # print(spectrum.shape)


def test_numba_fft(img_windows_norm):
    """Compare spectrum derived with numba versus derived with numpy."""
    spectrum = spectral._numba_fourier_transform(img_windows_norm[-1])
    spectrum_numpy = spectral._numpy_fourier_transform(img_windows_norm[-1])
    # test if the answer is (very) close to the answer of the numpy version
    assert np.allclose(spectrum_numpy, spectrum)


def test_numba_fft_multi(img_windows_norm):
    """Compare multi-spectrum derived with numba versus derived with numpy."""
    spectra = spectral._numba_fourier_transform_multi(img_windows_norm)
    # test if all image windows give the same result as the numpy version of the spectrum code
    for windows, spectrum in zip(img_windows_norm, spectra):
        spectrum_numpy = spectral._numpy_fourier_transform(windows)
        assert np.allclose(spectrum_numpy, spectrum)


@pytest.mark.parametrize(
    "engine", ["numpy", "numba"]
)
def test_sliding_window_spectrum(img_windows_norm, engine):
    """Check shape of average of multiple spectra over several time slices."""
    spectrum = spectral.sliding_window_spectrum(img_windows_norm, 20, 10, engine=engine)
    # test if the spectra have the desired size
    assert spectrum.shape[-3] == int(np.ceil(20 / 2))