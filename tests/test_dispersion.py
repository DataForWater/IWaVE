import numpy as np
from iwave import dispersion, spectral

def test_dispersion(img_windows_norm):
    """Check the dimensions of predicted wave frequencies."""
    kt, ky, kx = spectral.wave_numbers(img_windows_norm.shape, res=0.01, fps=25)
    kt_gw, kt_turb = dispersion.dispersion(ky, kx, velocity=[1, 0], depth=1, vel_indx=1)
    # Check that wave frequencies have one value per spatial wavenumber pair.
    assert kt_gw.shape == (
        (1, img_windows_norm.shape[-2], img_windows_norm.shape[-1])
    )

def test_intensity(img_windows_norm):
    """Check the dimensions of model power."""
    kt, ky, kx = spectral.wave_numbers(img_windows_norm.shape, res=0.01, fps=25)
    kt_gw, kt_turb = dispersion.dispersion(ky, kx, velocity=[1, 0], depth=1, vel_indx=1)
    th_spectrum = dispersion.theoretical_spectrum(kt_gw, kt_turb, kt, gauss_width=1, gravity_waves_switch=True, turbulence_switch=True)
    measured_spectrum = spectral._numpy_fourier_transform(img_windows_norm)
    # Check that the model and measured power arrays have matching dimensions.
    assert th_spectrum.shape == (
        (int(np.ceil(measured_spectrum.shape[-3] / 2)), measured_spectrum.shape[-2], measured_spectrum.shape[-1])
    )
        
    
def test_downsample(img_windows_norm):
    """Check the dimensions of model power."""
    img_single_window = img_windows_norm[0]
    full_spectrum = spectral._numpy_fourier_transform(img_single_window)
    trimmed_spectrum, res, fps, window_dims = dispersion.spectrum_downsample(full_spectrum, res=0.01, fps=25, window_dims=img_single_window.shape, downsample=2)
    downsampled_img = img_single_window[:, ::2, ::2]
    downsampled_spectrum = spectral._numpy_fourier_transform(downsampled_img)
    assert trimmed_spectrum.shape == downsampled_spectrum.shape


def test_theoretical_spectrum_matches_tiled_branches():
    """Check that avoiding repeated coordinate arrays preserves both Gaussian parts exactly."""
    kt = np.arange(7, dtype=float)
    gw = np.array([[[1., 2.], [3., 4.]]])
    turb = np.array([[[4., 3.], [2., 1.]]])
    frequency = np.tile(kt[:, None, None], (1, 2, 2))
    for wave, floating in ((True, True), (True, False), (False, True), (False, False)):
        expected = np.maximum(
            dispersion.gauss_spectrum_calc(np.tile(gw, (7, 1, 1)), frequency, .7, wave),
            dispersion.gauss_spectrum_calc(np.tile(turb, (7, 1, 1)), frequency, .7, floating))
        actual = dispersion.theoretical_spectrum(gw, turb, kt, .7, wave, floating)
        np.testing.assert_array_equal(actual, expected)