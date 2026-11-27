import numpy as np
import pytest

from iwave import spectral, LazySpectrumArray


def test_fused_spectral_mask_matches_original():
    """Preserve the mean threshold, zero-frequency removal and speed mask."""
    data = np.arange(80., dtype=float).reshape(2, 4, 2, 5) / 7
    kt, ky, kx = np.arange(4.), np.array([-1., 0.]), np.arange(5.) - 2
    expected = data.copy()
    expected[expected < expected.mean(axis=1, keepdims=True)] = 0
    expected[:, 0] = 0
    expected *= kt[:, None, None] <= spectral.dispersion_threshold(ky, kx, 1.)
    actual = spectral.preprocess_normalized_spectrum(data, kt, ky, kx, 1.)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, np.arange(80.).reshape(2, 4, 2, 5) / 7)


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


def test_uint8_rectangles_use_bounded_inputs_without_changing_spectra():
    import pickle
    import numba as nb
    from iwave import window, LazyWindowArray
    imgs = np.random.default_rng(6).integers(0, 256, (33, 8, 30), dtype=np.uint8)
    imgs[:, :4, :5] = 0
    saved = imgs.copy(); imgs.setflags(write=False)
    wx, wy = window.sliding_window_idx(imgs[0], (4,5), (0,0))
    windows = LazyWindowArray(imgs, window.multi_sliding_window_array, wx, wy, 'time')
    kt, ky, kx = spectral.wave_numbers((15,4,5), .1, 20.)
    lazy = LazySpectrumArray(windows, 15, 7, kt, kx, ky)
    expected = LazySpectrumArray(windows[:], 15, 7, kt, kx, ky)[:]
    actual = lazy[:]; retained = actual.copy()
    np.testing.assert_array_equal(actual, expected)
    scratch = lazy._fft_workspace._input_buffer
    assert scratch.shape == (min(len(windows), nb.get_num_threads()),33,4,5)
    np.testing.assert_array_equal(lazy[-1], expected[-1])
    np.testing.assert_array_equal(lazy[::-1], expected[::-1])
    np.testing.assert_array_equal(actual, retained)
    np.testing.assert_array_equal(imgs, saved)
    assert lazy[:0].shape[0] == 0
    assert lazy._fft_workspace._input_buffer is scratch
    assert not np.shares_memory(actual, scratch)
    restored = pickle.loads(pickle.dumps(lazy))
    assert restored._fft_workspace._input_buffer is None
    np.testing.assert_array_equal(restored[:], expected)


def test_other_dtypes_and_normalisation_modes_keep_original_path():
    from iwave import window, LazyWindowArray
    original = np.resize(np.array([-1.,1.]), (16,8,12))
    wx, wy = window.sliding_window_idx(original[0], (4,4), (0,0))
    kt, ky, kx = spectral.wave_numbers((8,4,4), .1, 20.)
    for imgs in (original, (original+1).astype(np.uint16), (original+1).astype(np.uint8)):
        for norm in ('time','xy',None):
            windows = LazyWindowArray(imgs, window.multi_sliding_window_array, wx, wy, norm)
            lazy = LazySpectrumArray(windows, 8, 4, kt, kx, ky)
            expected = LazySpectrumArray(windows[:], 8, 4, kt, kx, ky)[:]
            np.testing.assert_array_equal(lazy[:], expected)
            if norm != 'time' or imgs.dtype != np.uint8:
                assert lazy._fft_workspace._input_buffer is None