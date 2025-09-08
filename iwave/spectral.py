import numba as nb
import numpy as np


@nb.njit(cache=True)
def fft_x(x):
    return np.fft.fft(x, axis=-1)


@nb.njit(cache=True)
def fft_y(x):
    return np.fft.fft(x, axis=-2)


@nb.njit(cache=True)
def ifft_(x):
    return np.fft.ifft(x, axis=0)


@nb.njit(cache=True)
def fftshift_(x):
    return np.fft.fftshift(x)


def wave_numbers(
    window_dims: tuple,
    res: float,
    fps: float
):
    """
    get t, y, x wave numbers

    Parameters
    ----------
    windows : np.ndarray
        time x Y x X windows with intensities
    res : float
        resolution in xy direction
    fps : float
        frames per second

    Returns
    -------
    kt, ky, kx: np.ndarray
        wave numbers of time, y and x

    """
    ks = 2 * np.pi / res  # this assumes the resolution is the same in x and
    # y-direction: TODO make variable for both directions
    kts = 2* np.pi * fps # change frequency units to rad/s
    dkt = kts / window_dims[-3]
    dky = ks / window_dims[-2]
    dkx = ks / window_dims[-1]
    # omega wave numbers (time dim)
    kt = np.arange(0, kts, dkt)
    kt = kt[0:np.int64(np.ceil(len(kt) / 2))]

    # determine wave numbers in x-direction
    kx = np.arange(0, ks, dkx)
    # kx = 0:dkx: (ks - dkx)
    ky = np.arange(0, ks, dky)

    # apply fftshift on determined wave numbers
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    idx_x0 = np.where(kx == 0)[0][0]
    kx[0: idx_x0] = kx[0:idx_x0] - kx[idx_x0 - 1] - dkx
    idx_y0 = np.where(ky == 0)[0][0]
    ky[0: idx_y0] = ky[0:idx_y0] - ky[idx_y0 - 1] - dky

    return kt, ky, kx


@nb.njit(parallel=True, cache=True, nogil=True)
def _numba_fourier_transform(
    windows: np.ndarray
) -> np.ndarray:
    """
    Perform 3D spectral analysis with numba jitted code.

    Parameters
    ----------
    windows : np.ndarray
        time x Y x X windows with intensities

    Returns
    -------
    spectrum : np.ndarray
        3D power spectrum of 3D fourier transform

    """
    spectrum_2d = np.empty(windows.shape, dtype=np.complex128)
    for n in nb.prange(windows.shape[0]):
        spectrum_2d[n] = fftshift_(
        fft_y(
            fft_x(windows[n])
        )
    )
    spectrum_3d = ifft_(spectrum_2d)
    # return spectrum_3d
    power = np.abs(spectrum_3d) ** 2
    # abbreviate to positive omega
    return power[:int(np.ceil(len(power)/2))]


@nb.njit(parallel=True, cache=True, nogil=True)
def _numba_fourier_transform_multi(
    imgs: np.ndarray
) -> np.ndarray:
    """
    Perform 3D spectral analysis for multiple windows at once with numba jitted code.

    Parameters
    ----------
    imgs : np.ndarray
        n x time x Y x X windows with intensities

    Returns
    -------
    spectra : np.ndarray
        n x 3D power spectrum of 3D fourier transform of all imgs

    """
    spectra = np.empty((imgs.shape[0], int(np.ceil(imgs.shape[1]/2)), imgs.shape[2], imgs.shape[3]), dtype=np.float64)
    for m in nb.prange(imgs.shape[0]):
        spectrum_3d = np.empty(imgs[m].shape, dtype=np.complex128)
        for n in nb.prange(imgs.shape[1]):
            spectrum_3d[n] = fftshift_(
                fft_y(
                    fft_x(imgs[m, n])
                )
            )
        for y in nb.prange(spectrum_3d.shape[1]):
            for x in nb.prange(spectrum_3d.shape[2]):
                spectrum_3d[:, y, x] = ifft_(spectrum_3d[:, y, x])
        # return spectrum_3d
        power = np.abs(spectrum_3d) ** 2
        # abbreviate to positive omega
        spectra[m] = power[:int(np.ceil(len(power)/2))]
    return spectra


def _numpy_fourier_transform(
        windows: np.ndarray,
        norm: bool = False
) -> np.ndarray:
    """
    Pure numpy implementation of 3D spectral analysis

    Parameters
    ----------
    windows : np.ndarray
        time x Y x X windows with intensities
    norm : bool
        normalize spectrum (default: False)

    Returns
    -------
    power : np.ndarray
        3D power spectrum of 3D fourier transform

    """
    spectrum_2d = np.fft.fftshift(
        np.fft.fft(np.fft.fft(windows, axis=-2), axis=-1)
    )
    spectrum_3d = np.fft.ifft(spectrum_2d, axis=0)
    spectrum = np.abs(spectrum_3d) ** 2

    # abbreviate to positive omega
    spectrum = spectrum[:int(np.ceil(len(spectrum)/2))]

    if norm:
        spectrum_norm = spectrum / np.expand_dims(
            spectrum.mean(axis=-1).mean(axis=-1),
            axis=(-1, -2)
        )
        return spectrum_norm
    return spectrum


def spectral_imgs(
    imgs: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Perform 3D spectral analysis.

    Parameters
    ----------
    imgs : np.ndarray
        [n * t * Y * X] 4-D array containing image [n] sequences [t], split in
        subwindows of Y * X pixels
    engine : str, optional
        "numpy" or "numba", compute method to use, typically numba (default) is
        a lot faster. Numpy function is easier to read.
    kwargs : dict with additional keyword arguments for processing

    Returns
    -------
    spectra : np.ndarray
        wave spectra for all image window sequences

    """
    n, t, y, x = imgs.shape
    t_out = int(np.ceil(t / 2))  # Reduced time dimension
    spectra =  _numba_fourier_transform_multi(imgs, **kwargs)
    return spectra


def sliding_window_spectrum(
    imgs: np.ndarray,
    win_t: int,
    overlap: int,
    **kwargs
) -> np.ndarray:
    """
    Splits the video into shorter segments and calculates the average 3D spectrum.

    Parameters
    ----------
    imgs : np.ndarray
        [n * t * Y * X] 4-D array containing image [n] sequences [t], split in 
        subwindows of Y * X pixels
    win_t : int
        number of frames per segment
    overlap : int
        overlap (frames)
    kwargs : dict with additional keyword arguments for processing

    Returns
    -------
    spectra : np.ndarray
        average wave spectra for all image window sequences

    """
    
    # Check for division by zero
    if win_t == overlap:
        raise ValueError("win_t and overlap should not be equal.")
    
    # number of segments
    num_segments = (imgs.shape[1] - win_t) // (win_t - overlap)+ 1
    
    # sum of individual segments
    spectrum_sum = sum(
        spectral_imgs(
            imgs[:, segment_t0:(segment_t0 + win_t), :, :],
            **kwargs
        ) for segment_t0 in range(0, imgs.shape[1] - win_t + 1, win_t - overlap)
    )
    
    # renormalisation
    spectra = spectrum_sum / num_segments
    return spectra


def spectrum_preprocessing(
        measured_spectrum: np.ndarray,
        kt: np.ndarray,
        ky: np.ndarray,
        kx: np.ndarray,
        velocity_threshold: float,
        spectrum_threshold: float = 1
) -> np.ndarray:
    """
    pre-processing of the measured spectrum to improve convergence of the optimisation

    Parameters
    ----------
    measured_spectrum : np.ndarray
        measured, averaged, and normalised 3D power spectrum calculated with spectral.py
        dimensions [wi, kti, kyi, kx]

    kt : np.ndarray
        radian frequency vector (rad/s)

    ky : np.ndarray
        y-wavenumber vector (rad/m)

    kx : np.ndarray
        x-wavenumber vector (rad/m)

    velocity_threshold : float
        maximum threshold velocity for spectrum filtering (m/s).

    spectrum_threshold : float, optional
        threshold parameter for spectrum filtering (default 1.0).
        the spectrum with amplitude < threshold_preprocessing * mean(measured_spectrum) is filtered out.
        threshold_preprocessing < 1 yields a more severe filtering but could eliminate part of useful signal.

    Returns
    -------
    preprocessed_spectrum : np.ndarray
        pre-processed and normalised measured 3D spectrum

    """
    # spectrum normalisation: divides the spectrum at each frequency by the average across all wavenumber combinations at the same frequency
    with np.errstate(divide='ignore', invalid='ignore'):
        preprocessed_spectrum = measured_spectrum / np.mean(measured_spectrum, axis=(2, 3), keepdims=True)

    # apply threshold
    threshold = spectrum_threshold * np.mean(preprocessed_spectrum, axis=1, keepdims=True)
    preprocessed_spectrum[preprocessed_spectrum < threshold] = 0

    # set the first slice (frequency=0) to 0
    preprocessed_spectrum[:, 0, :, :] = 0

    kt_threshold = dispersion_threshold(ky, kx, velocity_threshold)

    # set all frequencies higher than the threshold frequency to 0
    kt_reshaped = kt[:, np.newaxis, np.newaxis]  # reshape kt to be broadcastable
    kt_threshold_bc = np.broadcast_to(kt_threshold, (kt.shape[0], kt_threshold.shape[1], kt_threshold.shape[
        2]))  # broadcast kt_threshold to match the dimensions of kt
    kt_bc = np.broadcast_to(kt_reshaped, kt_threshold_bc.shape)  # broadcast kt to match the dimensions of kt_threshold
    mask = np.where(kt_bc <= kt_threshold_bc, 1, 0)  # create mask
    mask = np.expand_dims(mask, axis=0)

    preprocessed_spectrum = preprocessed_spectrum * mask  # apply mask

    # remove NaNs
    preprocessed_spectrum = np.nan_to_num(preprocessed_spectrum)
    return preprocessed_spectrum


def dispersion_threshold(
        ky,
        kx,
        velocity_threshold
) -> np.ndarray:
    """
    Calculate the frequency corresponding to the threshold velocity

    Parameters
    ----------
    ky: np.ndarray
        wavenumber array along the direction y

    kx: np.ndarray
        wavenumber array along the direction x

    velocity_threshold : float
        threshold_velocity (m/s)

    Returns
    -------
    kt_threshold : np.ndarray
        1 x N_y x N_x: threshold frequency

    """

    # create 2D wavenumber grid
    kx, ky = np.meshgrid(kx, ky)

    # transpose to 1 x N_y x N_x
    ky = np.expand_dims(ky, axis=0)
    kx = np.expand_dims(kx, axis=0)

    # wavenumber modulus
    k_mod = np.sqrt(ky ** 2 + kx ** 2)

    return k_mod * velocity_threshold

