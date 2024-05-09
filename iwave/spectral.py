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


def _get_wave_numbers(window_dims, res, fps):
    """
    get t, y, x wave numbers

    Parameters
    ----------
    windows : np.ndarray
        time x Y x X windows with intensities
    res :

    Returns
    -------

    """
    kts = 2 * np.pi * fps  #
    ks = 2 * np.pi / res  # this assumes the resolution is the same in x and
    # y-direction: TODO make variable for both directions
    dkt = kts / window_dims[0]
    dky = ks / window_dims[1]
    dkx = ks / window_dims[2]
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
def numba_fourier_transform(windows):
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
def numba_fourier_transform_multi(imgs):
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


def numpy_fourier_transform(windows, norm=False):
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
    power = np.abs(spectrum_3d) ** 2

    # abbreviate to positive omega
    power = power[:int(np.ceil(len(power)/2))]

    if norm:
        power_norm = power / np.expand_dims(
            power.mean(axis=-1).mean(axis=-1),
            axis=(-1, -2)
        )
        return power_norm
    return power
