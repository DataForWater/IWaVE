import numba as nb
import numpy as np

@nb.njit(cache=True)
def fft_(x):
    return np.fft.fft(x, axis=-1)

@nb.njit(cache=True)
def fft_ax_(x):
    return np.fft.fft(x, axis=-2)

# base functions
@nb.njit(cache=True)
def fft2_(x):
    return np.fft.fft2(x)


@nb.njit(cache=True)
def fft2__(x, fsize):
    return np.fft.fft2(x, fsize)


@nb.njit(cache=True)
def ifft2_(x):
    return np.fft.ifft2(x)

@nb.njit(cache=True)
def ifft_(x, axis=0):
    return np.fft.ifft(x, axis=axis)

@nb.njit(cache=True)
def fftshift_(x, axes=0):
    return np.fft.fftshift(x, axes)

@nb.njit(cache=True)
def fftn_(x):
    return np.fft.fftn(x)

@nb.njit(cache=True)
def conj_(x):
    return np.conj(x)


def _get_wave_numbers(window_dims, res, fps):
    """
    get t, y, x wave numbers

    Parameters
    ----------
    windows : np.ndarray
        time x Y x X windows with intensities
    res

    Returns
    -------

    """
    ks = 2 * np.pi / res  # this assumes the resolution is the same in x and
    # y-direction: TODO make variable for both directions
    dkt = fps / window_dims[0]
    dky = ks / window_dims[1]
    dkx = ks / window_dims[2]
    # omega wave numbers (time dim)
    kt = np.arange(0, fps, dkt)
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
    Numba implementation of 3D spectral analysis

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
    spectrum_2d = np.zeros(windows.shape, dtype=np.complex128)
    for n in nb.prange(windows.shape[0]):
        spectrum_2d[n] = fftshift_(
        fft_ax_(
            fft_(windows[n])
        )
    )
    spectrum_3d = ifft_(spectrum_2d, axis=0)
    # return spectrum_3d
    power = spectrum_3d.real ** 2

    # abbreviate to positive omega
    return power[:int(np.ceil(len(power)/2))]


    # # Initialize an empty array to store the Fourier transform
    # fourier_transform = np.zeros_like(windows, dtype=np.complex128)
    # # Apply the 2D Fourier transform to each frame
    # for i in nb.prange(windows.shape[0]):
    #     fourier_transform[i] = fft2_(windows[i])
    #
    # # 1D Fourier transform along the time dimension
    # for axis in range(1, 3):
    #     fourier_transform = ifft_(fourier_transform, axis=0)
    # # ...
    # raise NotImplementedError


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
    power = spectrum_3d.real ** 2

    # abbreviate to positive omega
    power = power[:int(np.ceil(len(power)/2))]

    if norm:
        power_norm = power / np.expand_dims(
            power.mean(axis=-1).mean(axis=-1),
            axis=(-1, -2)
        )
        return power_norm
    return power
