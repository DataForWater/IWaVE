import numpy as np
import pytest

from iwave import spectral, LazySpectrumArray


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
        int(np.ceil(len(windows)/2)),
        windows.shape[1],
        windows.shape[2]
    )


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


def test_sliding_window_spectrum(img_windows_norm):
    """Compare segment averaging with the original NumPy transform."""
    imgs = img_windows_norm[:2]
    spectrum = spectral.sliding_window_spectrum(imgs, 20, 10)
    starts = range(0, imgs.shape[1] - 20 + 1, 10)
    expected = sum(
        np.stack([spectral._numpy_fourier_transform(w[start:start+20]) for w in imgs])
        for start in starts
    ) / len(starts)
    assert spectrum.shape[-3] == 10
    assert np.allclose(spectrum, expected)


def test_real_fft_matches_numpy_for_odd_and_even_lengths():
    """Keep original signs, scaling and bins, including reversed input views."""
    for length in (15, 16):
        imgs = np.random.default_rng(4).normal(size=(2, length, 7, 10))[:, :, ::-1, :]
        imgs[0] = 0
        imgs.setflags(write=False)
        expected = np.stack([spectral._numpy_fourier_transform(w) for w in imgs])
        actual = spectral.spectral_imgs(imgs)
        assert actual.shape == (2, (length + 1)//2, 7, 10)
        assert np.allclose(actual, expected)


def test_lazy_spectra_reuse_scratch_without_overwriting_results():
    """A final short batch reuses scratch but cannot change previous outputs."""
    windows = np.random.default_rng(5).normal(size=(3, 32, 6, 8))
    kt, ky, kx = spectral.wave_numbers((16, 6, 8), .1, 20.)
    lazy = LazySpectrumArray(windows, 16, 8, kt, kx, ky)
    first = lazy[:2]
    scratch = lazy._fft_workspace._buffer
    saved = first.copy()
    last = lazy[2:]
    assert lazy._fft_workspace._buffer is scratch
    assert last.shape == (1, 8, 6, 8)
    assert np.array_equal(first, saved)
    assert not np.shares_memory(first, scratch)