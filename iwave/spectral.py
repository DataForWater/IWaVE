"""Full-grid image power spectra with bounded reusable real-FFT scratch.

Preprocessing preserves the original spatial-FFT / temporal-IFFT power, but
computes the real temporal transform first. The separable transforms commute;
using its positive-time sign avoids an explicit conjugation pass. Keep the
original 1/T**2 power scale and ceil(T/2) bins (no even-length Nyquist bin).

Each lazy array retains bounded FFT/input scratch, NOT cached power spectra.
Uint8 time-normalised rectangles are gathered into one full-clip slot per
concurrent FFT task, rather than a batch-sized image array. Every temporal
segment is summed in order; returned spectra remain independent of scratch.
Only preprocessing is implemented here; fitting stays in the main optimiser.
"""
from threading import Lock
import numba as nb
import numpy as np
from rocket_fft import r2c, c2c
from iwave import window


# Low-level Rocket-FFT calls take explicit integer axis arrays and output
# buffers. Keep them outside the hot loop; only time is reduced to a half-grid.
_FFT_TIME_AXES = np.array([0], dtype=np.int64)
_FFT_SPACE_AXES = np.array([1, 2], dtype=np.int64)


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
    # omega wave numbers (time dim)    
    kt = np.fft.fftfreq(window_dims[-3], d=1/fps) * 2 * np.pi  # rad/s
    kt = kt[:int(np.ceil(len(kt)/2))]  # abbreviate to positive omega

    # determine wave numbers in x- and y-direction
    # this assumes the resolution is the same in x and
    # y-direction: TODO make variable for both directions
    kx = np.fft.fftfreq(window_dims[-1], d=res) * 2 * np.pi  # rad/m
    ky = np.fft.fftfreq(window_dims[-2], d=res) * 2 * np.pi  # rad/m

    # apply fftshift on determined wave numbers
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    return kt, ky, kx


@nb.njit(cache=True, nogil=True)
def _window_power(img, z, output, win_t, step):
    """Original transform/sign/segment arithmetic, shared by both input paths."""
    nt, ny, nx = img.shape
    nf = (win_t + 1)//2
    segments = (nt - win_t)//step + 1
    for segment in range(segments):
        start = segment*step
        source = img[start:start + win_t]
        if not np.any(source):
            if segment == 0:
                output[:] = 0.
            continue
        # Positive-time RFFT replaces the IFFT; retain its 1/T**2 power scale.
        r2c(source, z, _FFT_TIME_AXES, False, 1.0, 1)
        c2c(z, z, _FFT_SPACE_AXES, True, 1.0, 1)
        for f in range(nf):
            for y in range(ny):
                yy = (y + (ny + 1)//2) % ny
                for x in range(nx):
                    xx = (x + (nx + 1)//2) % nx
                    v = z[f, yy, xx]
                    power = (v.real*v.real + v.imag*v.imag)/(win_t*win_t)
                    if segment == 0:
                        output[f, y, x] = power
                    else:
                        output[f, y, x] += power
    output /= segments


@nb.njit(parallel=True, cache=True, nogil=True)
def _workspace_power(imgs, work, output, win_t, step):
    """Reuse disjoint complex slots, with one FFT thread per slot."""
    n = len(imgs)
    slots = min(n, work.shape[0])
    for slot in nb.prange(slots):
        for m in range(slot, n, slots):
            _window_power(imgs[m], work[slot], output[m], win_t, step)


@nb.njit(parallel=True, cache=True, nogil=True)
def _workspace_rectangular_power(imgs, origins, clips, work, output, win_t, step):
    """Normalise the WHOLE clip in bounded input slots, then average segments."""
    n = len(origins)
    slots = min(n, clips.shape[0], work.shape[0])
    for slot in nb.prange(slots):
        for m in range(slot, n, slots):
            window._gather_normalize_time_uint8(imgs, origins[m], clips[slot])
            _window_power(clips[slot, :, ::-1, :], work[slot], output[m], win_t, step)


class _FFTWorkspace:
    """Transient, per-owner FFT/input scratch; never a cached spectrum.

    Keep the largest concurrency needed for the current transform geometry.
    A shorter final batch reuses slots. A larger thread/batch request grows
    the buffer; changed FFT geometry replaces it instead of retaining a cache
    of shapes. Native Rocket/PocketFFT may still allocate internal scratch.
    """
    def __init__(self):
        self._buffer = None
        self._input_buffer = None
        self._lock = Lock()

    def __getstate__(self):
        # Scratch and locks are not scientific state. Pickled IWaVE/lazy arrays
        # remain portable and allocate fresh scratch after deserialization.
        return {}

    def __setstate__(self, state):
        self.__init__()

    def _power(self, imgs, win_t, overlap):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('The same FFT workspace cannot be used concurrently.')
        try:
            n, _, ny, nx = imgs.shape
            result = np.empty((n, (win_t + 1)//2, ny, nx), dtype=np.float64)
            if not n:
                return result
            slots = min(n, nb.get_num_threads())
            geometry = (win_t//2 + 1, ny, nx)
            if (self._buffer is None or self._buffer.shape[1:] != geometry
                    or self._buffer.shape[0] < slots):
                # Drop the previous allocation BEFORE reserving the new one.
                self._buffer = None
                self._buffer = np.empty((slots, *geometry), dtype=np.complex128)
            _workspace_power(imgs, self._buffer, result, win_t, win_t - overlap)
            # No returned array/view references scratch. Callers can retain
            # this power after every later batch overwrites the scratch.
            return result
        finally:
            self._lock.release()

    def _power_rectangular(self, imgs, origins, window_shape, win_t, overlap):
        """Private uint8/time-normalised path; no batch-sized image array."""
        for value in (win_t, overlap):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError('Temporal length and overlap must be integers.')
        if not 0 <= overlap < win_t <= len(imgs) or min(window_shape) < 1:
            raise ValueError('Need positive spatial sizes and 0 <= overlap < win_t <= frames.')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('The same FFT workspace cannot be used concurrently.')
        try:
            n = len(origins)
            ny, nx = window_shape
            result = np.empty((n, (win_t + 1)//2, ny, nx), dtype=np.float64)
            if not n:
                return result
            slots = min(n, nb.get_num_threads())
            geometry = (win_t//2 + 1, ny, nx)
            if (self._buffer is None or self._buffer.shape[1:] != geometry
                    or self._buffer.shape[0] < slots):
                self._buffer = None
                self._buffer = np.empty((slots, *geometry), dtype=np.complex128)
            clip_shape = (len(imgs), ny, nx)
            if (self._input_buffer is None or self._input_buffer.shape[1:] != clip_shape
                    or self._input_buffer.shape[0] < slots):
                self._input_buffer = None
                self._input_buffer = np.empty((slots, *clip_shape), dtype=np.float64)
            _workspace_rectangular_power(imgs, origins, self._input_buffer,
                self._buffer, result, win_t, win_t-overlap)
            return result
        finally:
            self._lock.release()


def _numba_fourier_transform_multi(imgs):
    """Keep the original batched entry point, now using real-input scratch."""
    return sliding_window_spectrum(imgs, imgs.shape[1], 0)


def _numba_fourier_transform(windows):
    """Single-window wrapper around the same batched real-input kernel."""
    return _numba_fourier_transform_multi(windows[None])[0]


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


def spectral_imgs(imgs: np.ndarray) -> np.ndarray:
    """Return full spatial power for real (window, time, y, x) images.

    Output shape is (window, ceil(time/2), y, x). Input is not modified.
    This convenience entry point transforms the complete clip as one segment;
    sliding_window_spectrum averages shorter, possibly overlapping segments.
    """
    if imgs.ndim != 4 or np.iscomplexobj(imgs):
        raise ValueError('FFT input must be real (window,time,y,x) images.')
    return _numba_fourier_transform_multi(imgs)


def sliding_window_spectrum(
    imgs: np.ndarray,
    win_t: int,
    overlap: int,
    *,
    _workspace=None,
) -> np.ndarray:
    """Average full-grid power over the complete temporal segments of imgs.

    Parameters
    ----------
    imgs : np.ndarray
        Real (window, time, y, x) images, already normalised if required.
    win_t : int
        Frames per FFT segment.
    overlap : int
        Shared frames between adjacent segments; 0 <= overlap < win_t.

    Returns
    -------
    np.ndarray
        Owned float64 power, shape (window, ceil(win_t/2), y, x).

    Notes
    -----
    Segments are summed in order, then divided by their count. Normalisation
    of the image clip and the later power thresholding remain separate steps.
    The private _workspace argument lets LazySpectrumArray reuse complex
    buffers between batches. Standalone calls create temporary scratch.
    """
    if imgs.ndim != 4 or np.iscomplexobj(imgs):
        raise ValueError('FFT input must be real (window,time,y,x) images.')
    for value in (win_t, overlap):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('Temporal length and overlap must be integers.')
    if win_t < 1 or not 0 <= overlap < win_t or imgs.shape[1] < win_t:
        raise ValueError('Need 0 <= overlap < win_t <= available frames.')
    if min(imgs.shape[-2:]) < 1:
        raise ValueError('Spatial FFT dimensions must be positive.')
    # Normal lazy windows are already float64: no copy. Converting other real
    # public inputs avoids low-level FFT dtype mismatches, without changing
    # image ownership or forcing a copy of reversed/strided float64 views.
    imgs = np.asarray(imgs, dtype=np.float64)
    workspace = _FFTWorkspace() if _workspace is None else _workspace
    return workspace._power(imgs, win_t, overlap)


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
    normalized = normalize_power(measured_spectrum)
    return preprocess_normalized_spectrum(normalized, kt, ky, kx,
                                          velocity_threshold, spectrum_threshold, copy=False)


def normalize_power(power, *, copy=True):
    """Divide each frequency slice by its spatial mean, once.

    copy=False is used for owned FFT power. Do not clean zero-slice NaNs here:
    the original threshold rule uses them in the subsequent frequency mean.
    """
    result = np.array(power, dtype=np.float64, copy=True) if copy else power
    denominator = result.mean(axis=(-2, -1), keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(result, denominator, out=result)
    return result


@nb.njit(parallel=True, cache=True, nogil=True, fastmath=False)
def _apply_spectral_mask(result, threshold, support):
    """Fuse elementwise passes; keep the original NumPy mean reduction."""
    largest = np.finfo(np.float64).max
    for window in nb.prange(result.shape[0]):
        for f in range(result.shape[1]):
            for y in range(result.shape[2]):
                for x in range(result.shape[3]):
                    value = result[window, f, y, x]
                    if value < threshold[window, 0, y, x]:
                        value = 0.
                    if f == 0:
                        value = 0.
                    # Multiplication, rather than assignment, preserves -0.
                    if not support[f, y, x]:
                        value = value*0.
                    if np.isnan(value):
                        value = 0.
                    elif value == np.inf:
                        value = largest
                    elif value == -np.inf:
                        value = -largest
                    result[window, f, y, x] = value


def preprocess_normalized_spectrum(normalized, kt, ky, kx,
                                   velocity_threshold, spectrum_threshold=1., *, copy=True):
    """Apply the original thresholds to already spatial-normalised power.

    The public wrapper protects caller input. LazySpectrumArray owns its
    power buffer and uses copy=False to avoid another spectral allocation.
    Threshold order and NaN cleanup are unchanged from main.
    """
    result = normalized.copy() if copy else normalized
    threshold = spectrum_threshold * result.mean(axis=1, keepdims=True)
    if (result.ndim == 4 and result.dtype == np.float64
            and result.flags.c_contiguous and result.flags.writeable
            and result.shape[1] > 0 and threshold.dtype == np.float64
            and threshold.shape == (result.shape[0], 1, *result.shape[-2:])
            and all(type(axis) is np.ndarray and axis.ndim == 1
                    for axis in (kt, ky, kx))
            and (len(kt), len(ky), len(kx)) == result.shape[1:]
            and np.isfinite(result).all()):
        support = kt[:, None, None] <= dispersion_threshold(ky, kx, velocity_threshold)
        if support.shape == result.shape[1:]:
            _apply_spectral_mask(result, threshold, support)
            return result
    result[result < threshold] = 0
    result[:, 0] = 0
    result *= kt[:, None, None] <= dispersion_threshold(ky, kx, velocity_threshold)
    return np.nan_to_num(result, copy=False)


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
