"""Calculate full-spatial-grid power spectra using reusable temporary arrays.

FFT converts image changes into frequency/wavenumber content. Calculate the
time transform first, using real-valued images, then the spatial transforms.
Use the same positive time sign and 1/T**2 power scale as the old inverse-time
calculation. Keep ceil(T/2) frequencies: omit the highest-frequency end sample for an
even frame count, as the original code does.

For ordinary uint8 time-normalised rectangles, prepare one full-clip window
per simultaneously processed window, not an image array for the whole requested batch.
Reuse these arrays for subsequent windows, but never overwrite a returned
spectrum. Add time segments in order before averaging their powers.
Fitting is performed separately by optimise.py.
"""
from threading import Lock
import numba as nb
import numpy as np
from rocket_fft import r2c, c2c
from iwave import window


# Rocket-FFT needs arrays specifying the transform axes and where results go.
# Create those once, outside the repeated calculations. Only time uses a half-length grid.
_FFT_TIME_AXES = np.array([0], dtype=np.int64)
_FFT_SPACE_AXES = np.array([1, 2], dtype=np.int64)


def wave_numbers(
    window_dims: tuple,
    res: float,
    fps: float
):
    """Set the frequency and spatial-wavenumber axes for a window.

    window_dims gives (frame count, y pixels, x pixels); res is metres per pixel
    in both spatial directions, and fps is frames per second. Return kt in rad/s
    and ky, kx in rad/m. Retain the nonnegative time frequencies used by fitting;
    move zero spatial wavenumber to the middle of ky and kx.
    """
    # Frequency axis for changes over time.
    # Angular frequency, in rad/s.
    kt = np.fft.fftfreq(window_dims[-3], d=1/fps) * 2 * np.pi
    # Keep only the nonnegative frequency samples used by the fitting code.
    kt = kt[:int(np.ceil(len(kt)/2))]

    # Set spatial wavenumbers along x and y.
    # Use the same pixel size in both directions.
    # TODO: allow different x and y pixel sizes.
    kx = np.fft.fftfreq(window_dims[-1], d=res) * 2 * np.pi  # Angular wavenumber, in rad/m.
    ky = np.fft.fftfreq(window_dims[-2], d=res) * 2 * np.pi  # Angular wavenumber, in rad/m.

    # Move zero spatial wavenumber to the middle of the axes.
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    return kt, ky, kx


@nb.njit(cache=True, nogil=True)
def _window_power(img, z, output, win_t, step):
    """Transform and average one window using the same signs and scaling for both input paths."""
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
        # Use a real-input transform with the positive time sign to match the old inverse
        # FFT.
        # Keep its original power scale, 1/T**2.
        # forward=False selects that sign. The last argument gives each transform one
        # thread;
        # Numba already runs different windows in parallel using separate temporary arrays.
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
    # Transform each time segment, then average its power, not its complex FFT values.
    # A zero segment still counts in the divisor, as in the original calculation.
    output /= segments


@nb.njit(parallel=True, cache=True, nogil=True)
def _workspace_power(imgs, work, output, win_t, step):
    """Use separate transform arrays for parallel windows, with one CPU thread per FFT."""
    n = len(imgs)
    slots = min(n, work.shape[0])
    for slot in nb.prange(slots):
        for m in range(slot, n, slots):
            _window_power(imgs[m], work[slot], output[m], win_t, step)


@nb.njit(parallel=True, cache=True, nogil=True)
def _workspace_rectangular_power(imgs, origins, clips, work, output, win_t, step):
    """Normalise each window using its whole clip, then average the powers of its time segments."""
    n = len(origins)
    slots = min(n, clips.shape[0], work.shape[0])
    for slot in nb.prange(slots):
        for m in range(slot, n, slots):
            window._gather_normalize_time_uint8(imgs, origins[m], clips[slot])
            _window_power(clips[slot, :, ::-1, :], work[slot], output[m], win_t, step)


class _FFTWorkspace:
    """Keep temporary input and FFT arrays for reuse by one spectrum-array object.

    Keep enough temporary arrays for the largest simultaneous request at the current shape.
    A shorter request reuses that capacity; a larger one grows it. A shape change
    replaces the old arrays rather than keeping arrays for every previous shape.
    The FFT library may also allocate its own temporary memory. No returned
    spectrum is stored here.
    """
    def __init__(self):
        self._buffer = None
        self._input_buffer = None
        self._lock = Lock()

    def __getstate__(self):
        # Do not save temporary arrays or locks when copying this object between processes.
        # Recreate them when needed; only retain the information required for the
        # calculation.
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
                # Release the old arrays before requesting larger replacements.
                self._buffer = None
                self._buffer = np.empty((slots, *geometry), dtype=np.complex128)
            _workspace_power(imgs, self._buffer, result, win_t, win_t - overlap)
            # Returned power does not share memory with the reusable temporary arrays.
            # It stays valid when a later batch overwrites those temporary arrays.
            return result
        finally:
            self._lock.release()

    def _power_rectangular(self, imgs, origins, window_shape, win_t, overlap):
        """Prepare 8-bit (uint8) image windows in reusable input arrays.

        Each parallel calculation needs one full-clip window at a time, rather than
        an input-image array holding the whole requested batch.
        """
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
            # Use no more temporary input arrays than the batch size or Numba thread count.
            # Each array holds one full-clip window, so all overlapping segments use the
            # same
            # time averages. Reuse that array when processing another window;
            # do not reuse statistics calculated from a different window.
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
    """Calculate window-batch spectra using the shared real-input transform code."""
    return sliding_window_spectrum(imgs, imgs.shape[1], 0)


def _numba_fourier_transform(windows):
    """Calculate one window by calling the same code used for a batch."""
    return _numba_fourier_transform_multi(windows[None])[0]


def _numpy_fourier_transform(
        windows: np.ndarray,
        norm: bool = False
) -> np.ndarray:
    """Calculate one window's power spectrum using NumPy, for reference.

    windows has (time, y, x) dimensions. Return power in (frequency, ky, kx)
    order, retaining the package's transform signs, scaling and frequency range.
    """
    spectrum_2d = np.fft.fftshift(
        np.fft.fft(np.fft.fft(windows, axis=-2), axis=-1)
    )
    spectrum_3d = np.fft.ifft(spectrum_2d, axis=0)
    spectrum = np.abs(spectrum_3d) ** 2

    # Keep only the nonnegative frequency samples used by the fitting code.
    spectrum = spectrum[:int(np.ceil(len(spectrum)/2))]

    if norm:
        spectrum_norm = spectrum / np.expand_dims(
            spectrum.mean(axis=-1).mean(axis=-1),
            axis=(-1, -2)
        )
        return spectrum_norm
    return spectrum


def spectral_imgs(imgs: np.ndarray) -> np.ndarray:
    """Return spectra for real images in (window, time, y, x) order.

    Leave input images unchanged. Output dimensions are
    (window, ceil(time/2), y, x). This call treats the whole supplied sequence as
    one time segment; use sliding_window_spectrum to average shorter segments.
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
    """Average powers from all complete time segments of the image windows.

    imgs has (window, time, y, x) dimensions and is already brightness-normalised
    if required. win_t is frames per segment; overlap is the number of shared
    frames, with 0 <= overlap < win_t. Add segment powers in order, then divide
    by the count. Return a separate float64 array with dimensions
    (window, ceil(win_t/2), y, x).

    Image normalisation before this step and spectrum filtering afterwards are
    separate operations. A supplied private _workspace reuses temporary arrays;
    standalone calls create their own.
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
    # Ordinary windows already use float64, so no conversion is needed.
    # Convert other real inputs to the type required by the low-level transform.
    # Keep existing float64 image views without copying or changing their source data.
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
    """Prepare measured power for fitting, without changing the input spectrum.

    measured_spectrum has (window, frequency, ky, kx) dimensions. kt is angular
    frequency in rad/s; ky and kx are spatial wavenumbers in rad/m.
    First divide each frequency slice by its spatial mean power. At each spatial
    wavenumber pair, spectrum_threshold multiplies the mean over frequency to
    set the cutoff; increasing it removes more weak power when the mean is finite.
    Then remove zero frequency and values outside the velocity_threshold support
    (velocity in m/s). Finally clean undefined numerical values.
    Return the processed array with the same dimensions.
    """
    normalized = normalize_power(measured_spectrum)
    return preprocess_normalized_spectrum(normalized, kt, ky, kx,
                                          velocity_threshold, spectrum_threshold, copy=False)


def normalize_power(power, *, copy=True):
    """Divide each frequency slice by its mean over spatial wavenumbers.

    copy=False may change the supplied power array, so use it only for a private
    array. Leave undefined values from zero slices in place for now: cleaning
    them before the next mean/threshold calculation would change that rule.
    """
    result = np.array(power, dtype=np.float64, copy=True) if copy else power
    denominator = result.mean(axis=(-2, -1), keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(result, denominator, out=result)
    return result


def preprocess_normalized_spectrum(normalized, kt, ky, kx,
                                   velocity_threshold, spectrum_threshold=1., *, copy=True):
    """Apply fitting filters to power already divided by its spatial mean.

    The public wrapper protects the input. The spectrum-array object can use
    copy=False because the power array is private to its request. Keep the
    existing order of thresholding and cleanup of undefined values.
    """
    result = normalized.copy() if copy else normalized
    threshold = spectrum_threshold * result.mean(axis=1, keepdims=True)
    result[result < threshold] = 0
    result[:, 0] = 0
    result *= kt[:, None, None] <= dispersion_threshold(ky, kx, velocity_threshold)
    return np.nan_to_num(result, copy=False)


def dispersion_threshold(
        ky,
        kx,
        velocity_threshold
) -> np.ndarray:
    """Calculate the largest frequency retained at each spatial wavenumber pair.

    ky and kx are in rad/m and velocity_threshold is in m/s.
    Return velocity_threshold*sqrt(ky**2 + kx**2), in rad/s, on a (1, y, x) grid.
    """

    # Make a grid containing every combination of x and y wavenumbers.
    kx, ky = np.meshgrid(kx, ky)

    # Add a length-one frequency axis so these arrays fit the spectrum dimensions.
    ky = np.expand_dims(ky, axis=0)
    kx = np.expand_dims(kx, axis=0)

    # Calculate the magnitude of the spatial wavenumber.
    k_mod = np.sqrt(ky ** 2 + kx ** 2)

    return k_mod * velocity_threshold
