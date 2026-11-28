"""Calculate measured spectra, compare models, and estimate nearby noise effects.

First prepare the power spectrum used for fitting. Later functions calculate
model agreement and how assumed spectral noise changes velocity/depth near
the chosen solution. They do not search for a new solution.
"""
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


@nb.njit(parallel=True, cache=True, nogil=True)
def _numba_fourier_transform(
    windows: np.ndarray
) -> np.ndarray:
    """Calculate one window's power spectrum with Numba-compiled code.

    windows has (time, y, x) dimensions. Return power in (frequency, ky, kx)
    order, using the package's spatial FFT and inverse-time FFT convention.
    """
    spectrum_2d = np.empty(windows.shape, dtype=np.complex128)
    for n in nb.prange(windows.shape[0]):
        spectrum_2d[n] = fftshift_(
        fft_y(
            fft_x(windows[n])
        )
    )
    spectrum_3d = ifft_(spectrum_2d)
    power = np.abs(spectrum_3d) ** 2
    # Keep only the nonnegative frequency samples used by the fitting code.
    return power[:int(np.ceil(len(power)/2))]


@nb.njit(parallel=True, cache=True, nogil=True)
def _numba_fourier_transform_multi(
    imgs: np.ndarray
) -> np.ndarray:
    """Calculate power spectra for several windows using Numba-compiled code.

    imgs has (window, time, y, x) dimensions. Return one (frequency, ky, kx)
    power array per window, skipping transform work for entirely zero windows.
    """
    spectra = np.empty((imgs.shape[0], int(np.ceil(imgs.shape[1]/2)), imgs.shape[2], imgs.shape[3]), dtype=np.float64)
    for m in nb.prange(imgs.shape[0]):
        # Skip the transform for windows containing only zeros.
        if not np.any(imgs[m]):
            spectra[m] = 0.0
            continue
        
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
        power = np.abs(spectrum_3d) ** 2
        # Keep only the nonnegative frequency samples used by the fitting code.
        spectra[m] = power[:int(np.ceil(len(power)/2))]
    return spectra


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


def spectral_imgs(
    imgs: np.ndarray,
    **kwargs
) -> np.ndarray:
    """Calculate power spectra for an image-window batch.

    imgs has (window, time, y, x) dimensions. This HNM-only version uses the
    original Numba transform path; there is no engine selector. kwargs are
    forwarded to that calculation. Return one spectrum per window.
    _numpy_fourier_transform provides the separate NumPy reference calculation.
    """
    n, t, y, x = imgs.shape
    t_out = int(np.ceil(t / 2))  # Number of retained frequency samples.
    spectra =  _numba_fourier_transform_multi(imgs, **kwargs)
    return spectra


def sliding_window_spectrum(
    imgs: np.ndarray,
    win_t: int,
    overlap: int,
    **kwargs
) -> np.ndarray:
    """Average the spectra from shorter, possibly overlapping video segments.

    imgs has (window, time, y, x) dimensions. win_t is the number of frames per
    segment; overlap is their shared frame count. Pass kwargs to the transform.
    Use complete segments only, adding their powers before dividing by the count.
    Return power in (window, frequency, ky, kx) order.
    """
    
    # Reject an overlap that would give a zero step between time segments.
    if win_t == overlap:
        raise ValueError("win_t and overlap should not be equal.")
    
    # Count complete time segments.
    num_segments = (imgs.shape[1] - win_t) // (win_t - overlap)+ 1
    
    # Add the power spectrum from each segment.
    spectrum_sum = sum(
        spectral_imgs(
            imgs[:, segment_t0:(segment_t0 + win_t), :, :],
            **kwargs
        ) for segment_t0 in range(0, imgs.shape[1] - win_t + 1, win_t - overlap)
    )
    
    # Divide by the number of segments to get average power.
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
    # For each frequency, divide power by its mean over all spatial wavenumbers.
    with np.errstate(divide='ignore', invalid='ignore'):
        preprocessed_spectrum = measured_spectrum / np.mean(measured_spectrum, axis=(2, 3), keepdims=True)

    # Remove power below the chosen relative threshold.
    threshold = spectrum_threshold * np.mean(preprocessed_spectrum, axis=1, keepdims=True)
    preprocessed_spectrum[preprocessed_spectrum < threshold] = 0

    # Remove the zero-frequency slice, which describes values constant in time.
    preprocessed_spectrum[:, 0, :, :] = 0

    kt_threshold = dispersion_threshold(ky, kx, velocity_threshold)

    # Find which spectrum values are allowed by the velocity limit.
    kt_bc = kt[:, None, None]  
    mask = kt_bc <= kt_threshold 

    preprocessed_spectrum *= mask  # Set excluded spectrum values to zero.
    
    # Replace undefined numerical values using the existing cleanup rule.
    preprocessed_spectrum = np.nan_to_num(preprocessed_spectrum)
    return preprocessed_spectrum


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
