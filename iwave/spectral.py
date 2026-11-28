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
Later sections calculate the shared model score and optional noise effects
near the fitted velocity/depth. optimise.py controls fitting itself.
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


# Model-spectrum comparison used during fitting.
import math
from iwave import dispersion


@nb.njit(cache=True, nogil=True, fastmath=False, error_model='numpy')
def _overlap_sums(data, omega, first, second, two_branches, width,
                  radius, dw, dense, statistics):
    """Calculate model weights and their sums in one loop, without full-size temporary arrays.

    Keep the same Gaussian model, whether using every frequency or just ranges
    around predicted lines. Where those ranges overlap, count each frequency
    once. The larger of two Gaussian values can be calculated from the smaller
    squared distance, so a second exponential is unnecessary. This does not add
    any filtering or change which peaks are used.

    Run sums in one thread with normal floating-point rules. The order differs
    from NumPy's sums, so the last digits can differ through rounding.
    """
    numerator = denominator = model_square = 0.
    n, columns = data.shape
    if dense:
        for i in range(n):
            frequency = omega[i]
            for j in range(columns):
                z = (frequency - first[j]) / width
                distance = z*z
                if two_branches:
                    z = (frequency - second[j]) / width
                    distance = min(distance, z*z)
                weight = math.exp(-distance)
                numerator += data[i, j] * weight
                denominator += weight
                if statistics:
                    model_square += weight * weight
    else:
        lower = omega[0] - (radius + 2)*dw
        upper = omega[-1] + (radius + 2)*dw
        for branch in range(2 if two_branches else 1):
            centres = first if branch == 0 else second
            for j in range(columns):
                centre = centres[j]
                if centre < lower or centre > upper:
                    continue
                # Limit frequency indices before converting them to integers, even for
                # distant predictions.
                base = int(math.floor(min(max((centre-omega[0])/dw,
                                             -radius-2), n+radius+2)))
                previous = 0
                if branch:
                    previous = int(math.floor(min(max((first[j]-omega[0])/dw,
                                                     -radius-2), n+radius+2)))
                for offset in range(-radius, radius + 2):
                    i = base + offset
                    if i < 0 or i >= n:
                        continue
                    if branch and previous-radius <= i <= previous+radius+1:
                        continue
                    frequency = omega[0] + dw*i
                    z = (frequency - first[j]) / width
                    distance = z*z
                    if two_branches:
                        z = (frequency - second[j]) / width
                        distance = min(distance, z*z)
                    weight = math.exp(-distance)
                    numerator += data[i, j] * weight
                    denominator += weight
                    if statistics:
                        model_square += weight * weight
    return numerator, denominator, model_square


class SpectralOverlapObjective:
    """Compare measured power with the larger of the two Gaussian model values.

    Minimise -sum(measured*model)/sum(model), without a velocity penalty.
    Normally calculate frequencies close to either predicted line: keep at least
    kernel_truncate*gauss_width on each side. Omitted weights are at most
    exp(-kernel_truncate**2). Count overlapping ranges once, using max rather
    than adding the two model values. If a predicted line lies just outside the
    observed frequency range, keep the part of its tail still visible in the data.
    Do not mirror either data or model at the edge.
    """
    def __init__(self, spectrum, axes, depth, alpha, wave, turb, options):
        self.omega, ky, kx = axes
        a = np.asarray(spectrum)
        expected = (len(self.omega), len(ky), len(kx))
        if (a.shape != expected or not np.issubdtype(a.dtype, np.number)
                or np.iscomplexobj(a) or not np.isfinite(a).all()
                or np.any(a < 0)):
            raise ValueError('Local fitting needs a finite nonnegative full spectrum matching (omega,ky,kx).')
        # Put the two spatial axes into a single list of (ky, kx) pairs.
        # Keep both signs and every available nonnegative frequency for each pair.
        self.data = np.asarray(a, float).reshape(len(self.omega), -1)
        self.spatial_shape = a.shape[1:]
        self.x, self.y = (v.ravel() for v in np.meshgrid(kx, ky))
        self.k = np.hypot(self.x, self.y)
        self.alpha = alpha
        self.reference_depth = float(depth)
        self.depth = None
        self._set_depth(depth)
        self.wave, self.turb = bool(wave), bool(turb)
        self.width = options['gauss_width']
        self.dw = float(self.omega[1] - self.omega[0])
        extent = options['kernel_truncate'] * self.width / self.dw
        self.radius = int(math.ceil(min(extent, len(self.omega))))
        self.calls = 0
        self.zero = not np.any(self.data)
        # Use the full frequency calculation when a narrower calculation would need more
        # memory.
        # Keep the existing choice between these two calculation paths.
        # The full calculation uses every frequency; the shorter one omits only values
        # far into the Gaussian tails. Neither path drops spatial columns.
        self.dense = len(self.omega) <= (self.wave + self.turb) * (2*self.radius + 2)
        self.offsets = (None if self.dense else np.arange(-self.radius, self.radius + 2)[:, None])

    def normalized_weights(self, parameters):
        """Return model weights using the same frequency ranges as the fitting score.

        This does not run a fit or increase the count of fitting evaluations.
        """
        q=np.asarray(parameters,float)
        if q.shape not in ((2,),(3,)) or not np.isfinite(q).all():return None
        with np.errstate(over='ignore',under='ignore'):
            depth=float(np.exp(q[2])) if q.size==3 else self.reference_depth
        if not np.isfinite(depth) or depth<=0:return None
        self._set_depth(depth)
        with np.errstate(over='ignore',invalid='ignore'):
            a=self.y*q[0]+self.x*q[1];centres=[]
            if self.turb:centres.append(a)
            if self.wave:centres.append((1-self.beta)*a+np.hypot(self.beta*a,self.intrinsic))
        if not centres or not all(np.isfinite(c).all() for c in centres):return None
        n=len(self.omega);index=np.arange(n)[:,None]
        omega=self.omega[:,None] if self.dense else self.omega[0]+self.dw*index
        weight=np.exp(-((omega-centres[0])/self.width)**2)
        for centre in centres[1:]:
            np.maximum(weight,np.exp(-((omega-centre)/self.width)**2),out=weight)
        if not self.dense:
            support=np.zeros(weight.shape,bool)
            for centre in centres:
                base=np.floor(np.clip((centre-self.omega[0])/self.dw,-self.radius-2,n+self.radius+2)).astype(np.int64)
                active=(centre>=self.omega[0]-(self.radius+2)*self.dw)&(centre<=self.omega[-1]+(self.radius+2)*self.dw)
                support|=(index>=base-self.radius)&(index<=base+self.radius+1)&active
            weight*=support
        mass=float(weight.sum())
        return weight.reshape((n,*self.spatial_shape))/mass if np.isfinite(mass) and mass>0 else None

    def _set_depth(self, depth):
        # Reuse stored coordinates and measurements; recalculate only terms that depend on
        # depth.
        if depth != self.depth:
            self.beta = dispersion.beta_calc(self.k, depth, self.alpha)
            self.intrinsic = dispersion.omega_i_calc(self.k, depth)
            self.depth = float(depth)

    def __call__(self, parameters):
        self.calls += 1
        return self._evaluate(parameters)

    def statistics(self, parameters):
        # This last score check does not use up the Nelder-Mead fitting budget.
        return self._evaluate(parameters, statistics=True)

    def _evaluate(self, velocity, statistics=False):
        velocity = np.asarray(velocity, float)
        if velocity.shape not in ((2,), (3,)) or not np.isfinite(velocity).all():
            return (np.inf, 0., 0., 0.) if statistics else np.inf
        depth = self.reference_depth
        if velocity.size == 3:
            with np.errstate(over='ignore', under='ignore'):
                depth = float(np.exp(velocity[2]))
            if not np.isfinite(depth) or depth <= 0:
                return (np.inf, 0., 0., 0.) if statistics else np.inf
        self._set_depth(depth)
        if self.zero:
            return (np.inf, 0., 0., 0.) if statistics else np.inf
        with np.errstate(over='ignore', invalid='ignore'):
            a = self.y * velocity[0] + self.x * velocity[1]
            centres = []
            if self.turb:
                centres.append(a)
            if self.wave:
                centres.append((1-self.beta)*a + np.hypot(self.beta*a, self.intrinsic))
        if not all(np.isfinite(c).all() for c in centres):
            return (np.inf, 0., 0., 0.) if statistics else np.inf
        # Keep the same validation, saved depth terms, and predicted wave frequencies.
        # Only combine the repeated array operations into one loop.
        # Do not change the separate diagnostic weights or the count of fitting evaluations.
        numerator, denominator, model_square = _overlap_sums(
            self.data, self.omega, centres[0], centres[-1], len(centres) == 2,
            self.width, self.radius, self.dw, self.dense, statistics,
        )
        if denominator <= 0 or not np.isfinite(numerator + denominator):
            return (np.inf, 0., 0., 0.) if statistics else np.inf
        value = -numerator / denominator
        return (value, denominator, model_square, numerator) if statistics else value


# Optional check of how spectral noise affects the selected estimate.
# It needs the full spectrum before filtering and describes variation near this fit.
# It is not a complete error budget and does not cover jumps to other solutions.
from copy import deepcopy
from numbers import Real
from time import perf_counter
from scipy.stats import norm
from iwave.io import estimate_objective_curvature, resolve_curvature_options, _constraints

SPECTRAL_UNCERTAINTY_DEFAULTS = dict(confidence=.95, variance_estimator='plugin', n_effective=None,
    fd_step=.05, stability_rtol=.25, curvature_rtol=1e-7, curvature_atol=1e-10,
    stationarity_tol=.1, parameter_scales=None, check_region=True,
    quadratic_rtol=.5, kernel_truncate=5.)


def boolean(value, name):
    if not isinstance(value,(bool,np.bool_)):
        raise ValueError(name+' must be Boolean.')
    return bool(value)


def resolve_spectral_request(enabled=False, options=None):
    enabled=boolean(enabled,'calculate_spectral_uncertainty')
    if options is not None and not isinstance(options,dict):
        raise ValueError('spectral_uncertainty_options must be a dictionary or None.')
    if set(options or {})-set(SPECTRAL_UNCERTAINTY_DEFAULTS):
        raise ValueError('Unknown spectral_uncertainty_options: '+', '.join(sorted(set(options)-set(SPECTRAL_UNCERTAINTY_DEFAULTS))))
    out=deepcopy(SPECTRAL_UNCERTAINTY_DEFAULTS);out.update(deepcopy(options or {}))
    for key in ('confidence','fd_step','stability_rtol','curvature_rtol','curvature_atol','stationarity_tol','quadratic_rtol','kernel_truncate'):
        x=out[key]
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x) or x<=0:
            raise ValueError('Invalid spectral_uncertainty_options['+repr(key)+'].')
        out[key]=float(x)
    if out['confidence']>=1:raise ValueError('confidence must be between 0 and 1.')
    if out['kernel_truncate']<5:raise ValueError('Spectral uncertainty requires kernel_truncate >= 5 for negligible Gaussian-tail error.')
    out['check_region']=boolean(out['check_region'],'check_region')
    if out['variance_estimator'] not in ('plugin','gamma_unbiased'):
        raise ValueError("variance_estimator must be 'plugin' or 'gamma_unbiased'.")
    if out['n_effective'] is not None:
        x=out['n_effective']
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x) or x<=0:
            raise ValueError('n_effective must be positive and finite.')
        out['n_effective']=float(x)
    numerical={k:out[k] for k in ('fd_step','stability_rtol','curvature_rtol','curvature_atol','parameter_scales','kernel_truncate')}
    checked=resolve_curvature_options(numerical)
    for k in numerical:out[k]=checked[k]
    return out if enabled else None


def effective_averages(n, length, overlap):
    for value,name in ((n,'n_averages'),(length,'time_size'),(overlap,'time_overlap')):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
            raise ValueError(name+' must be an integer.')
    if n<1 or length<1 or not 0<=overlap<length:raise ValueError('Invalid temporal averaging metadata.')
    # Overlapping time segments share frames, so their spectra are not independent.
    # Use their squared overlap fractions to adjust the noise of the average.
    # This is an approximation, not a measured calibration of physical errors.
    hop=length-overlap;denominator=1.
    for lag in range(1,min(n-1,(length-1)//hop)+1):
        rho=(length-lag*hop)/length
        denominator+=2*(1-lag/n)*rho*rho
    return float(n/denominator)


def validate_spectral_context(context, shape, options):
    if not isinstance(context,dict) or 'normalized_spectrum' not in context:
        raise ValueError('Spectral uncertainty requires a complete pre-threshold normalized_spectrum and averaging metadata; use LazySpectrumArray or spectral_contexts.')
    full=np.asarray(context['normalized_spectrum'])
    if full.shape!=tuple(shape) or not np.issubdtype(full.dtype,np.number) or np.iscomplexobj(full) or not np.isfinite(full).all() or np.any(full<0):
        raise ValueError('Invalid complete normalized_spectrum.')
    means=full.mean(axis=(-2,-1))
    if not np.all(np.isclose(means,1.,rtol=5e-7,atol=1e-10)|np.all(full==0,axis=(-2,-1))):
        raise ValueError('normalized_spectrum must have spatial mean 1 (or an all-zero slice) at each frequency.')
    if options['n_effective'] is not None:k=options['n_effective'];source='options_override'
    elif context.get('n_effective') is not None:
        k=context['n_effective'];source='context_effective_count'
    else:
        try:k=effective_averages(context['n_averages'],context['time_size'],context['time_overlap'])
        except KeyError as exc:raise ValueError('Missing averaging metadata; supply n_effective or n_averages/time_size/time_overlap.') from exc
        source='rectangular_welch_approximation'
    if isinstance(k,(bool,np.bool_)) or not isinstance(k,Real) or not np.isfinite(k) or k<=0:
        raise ValueError('Effective averaging count must be positive and finite.')
    return full,float(k),source


def normalization_jacobian(gradient, full, active):
    den=full.mean(axis=(-2,-1),keepdims=True);safe=np.where(den>0,den,1.)
    # Each frequency slice is divided by a shared spatial mean.
    # Account for the resulting link between its values when estimating noise effects.
    # Keep the fitting mask fixed; do not apply the threshold again to changed data.
    a=gradient*active[None];n=full.shape[-1]*full.shape[-2]
    return (a/safe[None]-(a*full[None]).sum(axis=(-2,-1),keepdims=True)/(n*safe[None]**2))*(den>0)[None]


def prepare_spectral_data(spectrum,spectral_context,options):
    """Check the full normalised spectrum and the fixed set of values used for fitting.

    The noise calculation needs pre-filter power, not only the surviving values.
    """
    original=np.asarray(spectrum)
    if not np.issubdtype(original.dtype,np.number) or np.iscomplexobj(original):
        raise ValueError('The processed spectrum must contain real numeric power.')
    M=np.asarray(original,float)
    if M.ndim!=3 or not np.isfinite(M).all() or np.any(M<0):raise ValueError('Invalid fitted spectrum.')
    full,k,source=validate_spectral_context(spectral_context,M.shape,options)
    active=M>0;den=full.mean(axis=(-2,-1),keepdims=True)
    expected=np.divide(full,den,out=np.zeros_like(full,dtype=float),where=den>0)*active
    if not np.allclose(expected,M,rtol=1e-7,atol=1e-10):
        raise ValueError('Complete normalized spectrum and active support do not reproduce fitted spectrum.')
    return M,full,active,k,source


def estimate_spectral_uncertainty(spectrum, velocity, depth, depth_estimated,
        window_dims,res,fps,vel_indx,spectral_context,gravity_waves_switch=True,
        turbulence_switch=True,gauss_width=1.,bounds=None,search_constraints=None,
        options=None,optimizer_success=None,penalized_fit=False):
    """Estimate a bound for each parameter near the fit, under the assumed noise.

    Do not refit or alter the estimate. intervals holds trial bounds;
    accepted_intervals retains only parameters that pass the checks.
    Supplied depth has no estimated bound. Failed checks mean missing bounds,
    not zero error.
    """
    started=perf_counter();opts=resolve_spectral_request(True,options)
    free=boolean(depth_estimated,'depth_estimated');p=3 if free else 2
    M,full,active,k,source=prepare_spectral_data(spectrum,spectral_context,opts)
    names=['vy','vx']+(['depth'] if free else []);coordinates=['vy','vx']+(['log_depth'] if free else [])
    z=float(norm.ppf((1+opts['confidence'])/2))
    out=dict(kind='spectral_noise_sensitivity',confidence_level=opts['confidence'],
        calibrated_confidence=False,conditional_on_selected_basin=True,
        between_basin_uncertainty_included=False,valid=False,flags=[],parameter_names=coordinates,
        parameter_valid={name:False for name in names},parameter_flags={name:[] for name in names},
        intervals=dict(vy=None,vx=None,depth=None),accepted_intervals=dict(vy=None,vx=None,depth=None),
        standard_errors=None,standard_errors_physical=None,covariance=None,correlation=None,hessian=None,
        parameter_units=['m/s','m/s']+(['log(d/1m)'] if free else []),
        n_effective=k,spectral_dof=2*k,effective_averages_source=source,
        depth_estimated=free,depth_status='estimated' if free else 'fixed_not_estimated',
        marginal_standardized_threshold=z,threshold_kind='noise-scaled marginal Gaussian delta',
        variance_estimator=opts['variance_estimator'],threshold_selection_conditioned=True,
        raw_bin_independence_assumed=True,region_checks=[],score_thresholds={},options=opts)
    def flag(name,parameters=None):
        if name not in out['flags']:out['flags'].append(name)
        for parameter in names if parameters is None else parameters:
            if name not in out['parameter_flags'][parameter]:out['parameter_flags'][parameter].append(name)
    def finish():
        for name in names:
            good=not out['parameter_flags'][name] and out['intervals'][name] is not None
            out['parameter_valid'][name]=good
            if good:out['accepted_intervals'][name]=out['intervals'][name]
        out['valid']=all(out['parameter_valid'].values())
        out['elapsed_seconds']=perf_counter()-started
        return out
    if optimizer_success is not None and not bool(optimizer_success):flag('optimizer_failed')
    if penalized_fit:flag('penalized_fit')
    if not np.isfinite([*velocity,depth]).all() or depth<=0:flag('invalid_fit');return finish()
    numerical={key:opts[key] for key in ('fd_step','stability_rtol','curvature_rtol','curvature_atol','parameter_scales','kernel_truncate')}
    curv=estimate_objective_curvature(M,velocity,depth,free,window_dims,res,fps,vel_indx,
        gravity_waves_switch,turbulence_switch,gauss_width,bounds,search_constraints,numerical)
    out['curvature']=curv;out['hessian']=curv['hessian']
    for reason in curv['flags']:flag(reason)
    if curv['hessian'] is None or curv.get('rank')!=p or 'negative_curvature' in curv['flags']:
        flag('unresolved_curvature');return finish()
    # Make small parameter changes to estimate how the score and model weights vary.
    # Do not change the fitted result. When fitting depth, make these changes in log(depth).
    # Convert the resulting depth-bound endpoints back to metres.
    H=np.asarray(curv['hessian']);inverse=np.linalg.inv(H)
    theta=np.r_[velocity,np.log(depth)] if free else np.asarray(velocity,float)
    steps=.5*np.asarray(curv['difference_steps'])
    def weights(q):
        T=dispersion.intensity(q[:2],np.exp(q[2]) if free else depth,vel_indx,
                window_dims,res,fps,gauss_width,gravity_waves_switch,turbulence_switch)
        return T/T.sum()
    w=weights(theta);derivatives=[]
    for i in range(p):
        e=np.zeros(p);e[i]=steps[i]
        derivatives.append((weights(theta+e)-weights(theta-e))/(2*steps[i]))
    dw=np.asarray(derivatives);score=float(np.sum(w*M));g=np.einsum('pijk,ijk->p',dw,M)
    if not np.isfinite(score) or score<=0:flag('invalid_overlap');return finish()
    J=-dw/score+g[:,None,None,None]*w[None]/score**2
    J=normalization_jacobian(J,full,active)
    variance=full**2/(k if opts['variance_estimator']=='plugin' else k+1)
    V=np.einsum('pijk,qijk,ijk->pq',J,J,variance)
    # Use the local change in score to translate spectral-noise effects into parameter
    # errors.
    # The covariance matrix records those error sizes and how they vary together.
    # This applies near the chosen fit only; it does not cover other possible solutions.
    C=inverse@V@inverse.T;C=(C+C.T)*.5
    if not np.isfinite(C).all() or np.any(np.diag(C)<=0):flag('unresolved_noise_variance');return finish()
    se=np.sqrt(np.diag(C));gradient=np.asarray(curv['gradient'])
    displacement=inverse@gradient
    stationarity=float(np.max(abs(displacement)/se))
    out.update(standard_errors=se.tolist(),
        standard_errors_physical=dict(vy=float(se[0]),vx=float(se[1]),depth=float(depth*se[2]) if free else None),
        covariance=C.tolist(),
        correlation=(C/se[:,None]/se[None,:]).tolist(),gradient_noise_covariance=V.tolist(),
        stationarity_in_standard_errors=stationarity,
        evaluations=curv['evaluations']+2*p+1)
    if stationarity>opts['stationarity_tol']:flag('not_stationary')
    for i,name in enumerate(names):
        limits=theta[i]+np.array([-1,1])*z*se[i]
        if name=='depth':
            with np.errstate(over='ignore',under='ignore'):limits=np.exp(limits)
        if not np.isfinite(limits).all() or (name=='depth' and limits[0]<=0):flag('unbounded_interval',[name])
        else:out['intervals'][name]=limits.tolist()
    # Reject bounds that cross allowed limits or extend beyond where the local
    # curved approximation of the score is adequate. A rejected bound means missing
    # information, not zero error; other parameters may still have usable bounds.
    _,disk,feasible=_constraints(bounds,search_constraints,free)
    for i,name in enumerate(names):
        for sign in (-1,1):
            delta=sign*z*C[:,i]/se[i];point=theta+delta
            prediction=float(.5*delta@H@delta);inside=feasible(point)
            out['score_thresholds'][name]=dict(log_loss=prediction,relative_overlap_loss=float(-np.expm1(-prediction)))
            check=dict(parameter=name,sign=sign,feasible=inside,predicted_log_loss=prediction,
                       effective_overlap_loss=float(-np.expm1(-prediction)),actual_log_loss=None)
            if not inside:flag('constraint_limited',[name])
            elif opts['check_region']:
                s=float(np.sum(weights(point)*M));loss=-np.log(s/score) if s>0 else np.inf
                out['evaluations']+=1
                check['actual_log_loss']=float(loss) if np.isfinite(loss) else None
                if not np.isfinite(loss) or abs(loss-prediction)>opts['quadratic_rtol']*max(abs(prediction),1e-12):
                    flag('nonquadratic_region',[name])
                if np.isfinite(loss) and loss < -max(1e-12,opts['stationarity_tol']*prediction):flag('not_stationary')
            out['region_checks'].append(check)
    if disk is not None:
        radius=z*np.sqrt(max(0.,float(np.linalg.eigvalsh(C[:2,:2])[-1])))
        if np.linalg.norm(theta[:2]-disk[0])+radius>disk[1]:flag('possible_disk_intersection')
    return finish()
