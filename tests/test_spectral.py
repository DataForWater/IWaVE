import numpy as np
import pytest

from iwave import spectral


def test_get_wave_numbers(img_windows, res=0.02, fps=25):
    """Check frequency and wavenumber counts for the image dimensions."""
    # Use the first frame of the first window.
    kt, ky, kx = spectral.wave_numbers(img_windows[0].shape, res=res, fps=fps)
    assert len(kt) == int(np.ceil(len(img_windows[0])/2))
    assert len(ky) == img_windows.shape[-2]
    assert len(kx) == img_windows.shape[-1]


def test_numpy_fft(img_windows_norm):
    """Check the dimensions of the calculated spectrum."""
    windows = img_windows_norm[-1]
    spectrum = spectral._numpy_fourier_transform(windows)
    assert spectrum.shape == (
        int(np.ceil(len(windows)/2)),
        windows.shape[1],
        windows.shape[2]
    )


def test_numba_fft(img_windows_norm):
    """Compare the Numba result with the NumPy reference."""
    spectrum = spectral._numba_fourier_transform(img_windows_norm[-1])
    spectrum_numpy = spectral._numpy_fourier_transform(img_windows_norm[-1])
    # Compare the values with the NumPy reference calculation.
    assert np.allclose(spectrum_numpy, spectrum)


def test_numba_fft_multi(img_windows_norm):
    """Compare a batch of Numba spectra with their NumPy reference values."""
    spectra = spectral._numba_fourier_transform_multi(img_windows_norm)
    # Compare every window with the NumPy reference spectrum.
    for windows, spectrum in zip(img_windows_norm, spectra):
        spectrum_numpy = spectral._numpy_fourier_transform(windows)
        assert np.allclose(spectrum_numpy, spectrum)


def test_sliding_window_spectrum(img_windows_norm):
    """Check the dimensions after averaging spectra from several time segments."""
    spectrum = spectral.sliding_window_spectrum(img_windows_norm, 20, 10)
    # Check the expected spectrum dimensions.
    assert spectrum.shape[-3] == int(np.ceil(20 / 2))