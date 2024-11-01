import numba as nb
import numpy as np
from typing import Literal


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
    engine: Literal["numpy", "numba"] = "numba",
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
    if engine == "numpy":
        return np.array([_numpy_fourier_transform(windows, **kwargs) for windows in imgs])
    elif engine == "numba":
        return _numba_fourier_transform_multi(imgs, **kwargs)
    else:
        raise ValueError(f'engine "{engine}" does not exist. Choose "numba" (default) or "numpy"')


def sliding_window_spectrum(
    imgs: np.ndarray,
    win_t: int,
    overlap: int,
    engine: Literal["numpy", "numba"] = "numba",
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
    engine : str, optional
        "numpy" or "numba", compute method to use, typically numba (default) 
        is a lot faster. Numpy function is easier to read.
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
    num_segments = imgs.shape[1] // (win_t - overlap)
    
    # sum of individual segments
    spectrum_sum = sum(spectral_imgs(imgs[:, segment_t0:(segment_t0 + win_t), :, :], engine, **kwargs)
                       for segment_t0 in range(0, imgs.shape[1] - win_t + 1, win_t - overlap))
    
    # renormalisation
    spectra = spectrum_sum / num_segments
    return spectra