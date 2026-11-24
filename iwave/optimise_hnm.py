"""Hough/Nelder-Mead velocity and optional log-depth estimation.

Hough voting supplies the initial velocity. Nelder-Mead refines it within a
fixed velocity disk centred on that estimate, intersected with the global bounds.
When depth is unknown, Hough uses one reference depth and only NM estimates it.
The final section optionally compares already-found endpoints using shared
spectral noise; it does not start another search or assign basin probabilities.
"""
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
import math
from numbers import Real
import numba as nb
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter
from scipy.optimize import minimize
from scipy.signal import find_peaks
from iwave import dispersion, spectral
from iwave.spectral import SpectralOverlapObjective as _RidgeObjective

from copy import deepcopy

K_MIN_BASE_BINS = 2.0  # Radial cutoff, in spatial Fourier bins.
# Hough proposal controls, exposed directly as entries in hnm_options.
DEFAULTS = dict(velocity_step=.01, vote_sigma=.015, k_min_factor=1.,
                peak_height=.25, peak_prominence=.30, scale_percentile=99.5,
                min_ridges=12, max_accumulator_mb=None)
# Fixed design conventions, not separately fitted parameters.
RADIAL_STRATA, ANGULAR_STRATA, CALIBRATION_COLUMNS = 4, 12, 32


# Adaptive sampling and NM settings share the public hnm_options dictionary.
ADAPTIVE_DEFAULTS = dict(
    batch_size=64, min_columns=64, stable_batches=2, convergence_tol=.01,
    exploration_every=4, min_direction_ratio=.05,
)
# Baseline refinement settings for fixed depth.
REFINEMENT_DEFAULTS = dict(
    gauss_width=1., kernel_truncate=5., maxfev=160, xatol=1e-4,
    fatol=1e-6, initial_step_factor=2., n_candidates=1,
    candidate_separation=.1, depth_initial=None, log_depth_step=.1,
)
# Used only for variable-depth fits, before applying explicit user settings.
DEPTH_SEARCH_DEFAULTS = dict(n_candidates=12, maxfev=640)
HNM_DEFAULTS = {
    **DEFAULTS,
    **ADAPTIVE_DEFAULTS,
    **REFINEMENT_DEFAULTS,
    "sampling": "fixed",
    "filling_ratio": 1.0,
    "refine": True,
    "relative_radius": 0.5,
    "min_radius": None,  # None uses twice the larger actual grid spacing.
}


@dataclass
class Result:
    vy: float
    vx: float
    depth: float
    success: bool
    message: str
    pairs: np.ndarray                  # (iy, ix), in processing order
    metadata: dict
    cost: float = np.nan
    quality: float = np.nan

    def as_iwave_tuple(self):
        """Return velocity, physical depth, squared NSP cost and fit quality."""
        return self.vy, self.vx, self.depth, self.cost, self.quality, self.success, self.message



def resolve_hnm_options(options=None, *, bnds=None):
    """Resolve depth-dependent defaults, then apply and validate user settings.

    Without bounds the fixed-depth defaults apply. Public entry points supply
    the bounds before resolving options; an explicit setting always wins.
    """
    if options is not None and not isinstance(options, dict):
        raise ValueError('hnm_options must be a dictionary or None.')
    for old_key in ('local_options', 'sampling_options'):
        if old_key in (options or {}):
            raise ValueError(f'Move {old_key} entries directly into hnm_options.')
    if 'initial_step' in (options or {}):
        raise ValueError('Use initial_step_factor; the step is factor * velocity_step.')
    unknown = set(options or {}) - set(HNM_DEFAULTS)
    if unknown:
        raise ValueError('Unknown hnm_options: ' + ', '.join(map(str, unknown)))
    out = deepcopy(HNM_DEFAULTS)
    if bnds is not None:
        bounds = _checked_depth_bounds(bnds)
        if bounds[2, 0] < bounds[2, 1]:
            out.update(DEPTH_SEARCH_DEFAULTS)
    out.update(deepcopy(options or {}))
    if out['sampling'] not in ('fixed', 'adaptive'):
        raise ValueError("sampling must be 'fixed' or 'adaptive'.")
    budget(out['filling_ratio'], 1, 1, 1)
    out.update(resolve_options({key: out[key] for key in DEFAULTS}))
    out.update(resolve_sampling_options({key: out[key] for key in ADAPTIVE_DEFAULTS}))
    if not isinstance(out['refine'], (bool, np.bool_)):
        raise ValueError('refine must be Boolean.')
    out['refine'] = bool(out['refine'])
    positive = ('gauss_width', 'kernel_truncate', 'xatol', 'fatol', 'initial_step_factor', 'log_depth_step')
    for key in (*positive, 'candidate_separation', 'relative_radius', 'min_radius', 'depth_initial'):
        value = out[key]
        if key in ('min_radius', 'depth_initial') and value is None:
            continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0
                or (key in (*positive, 'min_radius', 'depth_initial') and value == 0)):
            raise ValueError(f'Invalid hnm_options[{key!r}].')
        out[key] = float(value)
        if not np.isfinite(out[key]):
            raise ValueError(f'hnm_options[{key!r}] exceeds float precision.')
    for key, minimum in (('maxfev', 3), ('n_candidates', 1)):
        value = out[key]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or value < minimum):
            raise ValueError(f'{key} must be an integer >= {minimum}.')
        out[key] = int(value)
    if not np.isfinite(out['initial_step_factor'] * out['velocity_step']):
        raise ValueError('initial_step_factor * velocity_step must be finite.')
    return out


def validate_hnm_setup(bnds, vel_indx, window_dims, res, fps,
                        gravity_waves_switch=True, turbulence_switch=True,
                        hnm_options=None):
    """Validate shared geometry without reading spectra or allocating votes."""
    options = resolve_hnm_options(hnm_options, bnds=bnds)
    bounds, proposal_bounds, _, _ = _depth_setup(bnds, options, gravity_waves_switch)
    validated = validate_setup(proposal_bounds, vel_indx, window_dims, res, fps,
                                gravity_waves_switch, turbulence_switch,
                                {key: options[key] for key in DEFAULTS})
    # Expose the true bounds to callers, not Hough's fixed reference-depth copy.
    return validated[0], bounds, *validated[2:]


def optimize_single_spectrum_velocity_hnm(
    measured_spectrum, bnds, vel_indx, window_dims, res, fps,
    gravity_waves_switch=True, turbulence_switch=True,
    hnm_options=None, return_metadata=False,
):
    """Estimate one velocity and optionally return fit/search audit metadata."""
    started = perf_counter()
    options = resolve_hnm_options(hnm_options, bnds=bnds)
    validate_hnm_setup(bnds, vel_indx, window_dims, res, fps,
                       gravity_waves_switch, turbulence_switch, options)
    bounds, proposal_bounds, reference_depth, free_depth = _depth_setup(
        bnds, options, gravity_waves_switch)
    arguments = dict(measured_spectrum=measured_spectrum, bnds=proposal_bounds, vel_indx=vel_indx,
                     window_dims=window_dims, res=res, fps=fps,
                     gravity_waves_switch=gravity_waves_switch, turbulence_switch=turbulence_switch,
                     filling_ratio=options['filling_ratio'],
                     detector_options={key: options[key] for key in DEFAULTS})
    if options['sampling'] == 'adaptive':
        proposal, state = adaptive_estimate(
            **arguments,
            sampling_options={key: options[key] for key in ADAPTIVE_DEFAULTS},
            _return_state=True,
        )
    else:
        proposal, state = fixed_estimate(**arguments, _return_state=True)
    result = proposal
    if options['refine']:
        result = _refine_estimate(measured_spectrum, bnds, vel_indx, window_dims, res, fps,
                                  gravity_waves_switch, turbulence_switch, options,
                                  proposal, state, started)
    else:
        result.metadata.update(sampling=result.metadata.get('method'), method='HNM', solver='HNM',
                               refinement_enabled=False, refinement_attempted=False, refined=False,
                               local_runs=[], n_local_runs=0, objective_evaluations=0,
                               nm_evaluations=0, rejected_proposals=0, trust_region=None,
                               at_trust_boundary=False,
                               at_global_boundary=bool(state.best is not None and state.at_edge),
                               at_boundary=bool(state.best is not None and state.at_edge))
        result.metadata.update(depth_estimation_enabled=False, depth_estimated=False, depth_mode='fixed',
                               reference_depth=reference_depth, depth_bounds=bounds[2].tolist(),
                               at_depth_boundary=False, diagnostic_evaluations=0)
        diagnostics_time = perf_counter()
        if np.isfinite([result.vy, result.vx, result.depth]).all():
            axes = spectral.wave_numbers(window_dims, res, fps)
            objective = _RidgeObjective(measured_spectrum, axes, result.depth, vel_indx,
                                        gravity_waves_switch, turbulence_switch, options)
            result.cost, result.quality, stats = _fit_metrics(objective, [result.vy, result.vx])
            result.metadata.update(stats, diagnostic_evaluations=1)
        result.metadata['timings'].update(diagnostics=perf_counter()-diagnostics_time,
                                          total=perf_counter()-started)
    output = result.as_iwave_tuple()
    return (*output, dict(result.metadata, selected_pairs=result.pairs)) if return_metadata else output


# Hough sampling, peak extraction and voting.


def resolve_options(options=None):
    if options is not None and not isinstance(options, dict):
        raise ValueError('detector_options must be a dictionary or None.')
    unknown = set(options or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown detector_options: '+', '.join(map(str, unknown)))
    out = {**DEFAULTS, **(options or {})}
    for key in ('velocity_step','vote_sigma','k_min_factor','peak_height',
                'peak_prominence','scale_percentile'):
        x = out[key]
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x):
            raise ValueError(f'{key} must be a finite real scalar.')
    if out['velocity_step']<=0 or min(out['vote_sigma'],out['k_min_factor'],out['peak_height'])<0:
        raise ValueError('Require velocity_step>0; sigma, cutoff factor and height >=0.')
    if out['peak_prominence']<=0 or not 0<out['scale_percentile']<=100:
        raise ValueError('Require prominence>0 and percentile in (0,100].')
    if isinstance(out['min_ridges'],bool) or not isinstance(out['min_ridges'],(int,np.integer)) or out['min_ridges']<3:
        raise ValueError('min_ridges must be an integer >=3.')
    cap = out['max_accumulator_mb']
    if cap is not None and (isinstance(cap,(bool,np.bool_)) or not isinstance(cap,Real) or not np.isfinite(cap) or cap<=0):
        raise ValueError('max_accumulator_mb must be None or finite and positive.')
    return out


def budget(filling_ratio, ny, nx, n_eligible):
    if isinstance(filling_ratio,(bool,np.bool_)) or not isinstance(filling_ratio,Real) or not np.isfinite(filling_ratio) or not 0<filling_ratio<=1:
        raise ValueError('filling_ratio must be in (0,1].')
    # The budget counts spatial Fourier columns, not frequency bins.
    return min(int(math.ceil(float(filling_ratio)*ny*nx)), n_eligible)


def eligible_pairs(ky,kx,factor=1.):
    X,Y=np.meshgrid(kx,ky)
    cutoff=K_MIN_BASE_BINS*factor*min(abs(kx[1]-kx[0]),abs(ky[1]-ky[0]))
    tol=16*np.finfo(float).eps*max(float(np.max(np.hypot(X,Y))),cutoff,1.)
    keep=(np.hypot(X,Y)>=cutoff-tol)&((X!=0)|(Y!=0))
    return np.argwhere(keep),float(cutoff)


@lru_cache(maxsize=8)
def _geometry_order(ky_tuple,kx_tuple,factor):
    """A-priori stratified greedy log-det design, cached for repeated windows.

    Equal-population radial groups and signed angular sectors prevent collapsing
    onto only the high-k boundary. Within each cell maximise log-det information
    gain for regressors (kx,ky), with equal nominal frequency errors.
    This is an explicitly defined heuristic, NOT a globally optimal design.
    """
    ky,kx=np.array(ky_tuple),np.array(kx_tuple)
    pairs,_=eligible_pairs(ky,kx,factor)
    if not len(pairs):
        return pairs
    v=np.column_stack((kx[pairs[:,1]],ky[pairs[:,0]]))
    k=np.linalg.norm(v,axis=1); v=v/k.max()
    radial=np.empty(len(pairs),int)
    rank=np.argsort(k,kind='stable')
    radial[rank]=np.minimum(np.arange(len(rank))*RADIAL_STRATA//len(rank),RADIAL_STRATA-1)
    angle=np.minimum(((np.arctan2(v[:,1],v[:,0])+np.pi)/(2*np.pi)*ANGULAR_STRATA).astype(int),ANGULAR_STRATA-1)
    angular_order=[0,6,3,9,1,7,4,10,2,8,5,11]
    cells=[np.flatnonzero((radial==r)&(angle==a)) for a in angular_order for r in (3,0,2,1)]
    used=np.zeros(len(pairs),bool); order=[]; G=np.eye(2)*1e-3
    while len(order)<len(pairs):
        for cell in cells:
            candidates=cell[~used[cell]]
            if not len(candidates):
                continue
            inv=np.linalg.inv(G)
            gain=np.einsum('ij,jk,ik->i',v[candidates],inv,v[candidates])
            chosen=int(candidates[np.argmax(gain)])
            used[chosen]=True; order.append(chosen)
            G+=np.outer(v[chosen],v[chosen])
    answer=pairs[np.asarray(order)]
    answer.setflags(write=False)
    return answer


def fixed_plan(ky,kx,filling_ratio=1.,k_min_factor=1.):
    ky,kx=np.asarray(ky),np.asarray(kx)
    if ky.ndim!=1 or kx.ndim!=1 or min(len(ky),len(kx))<2:
        raise ValueError('Need one-dimensional kx/ky arrays with >=2 samples.')
    if not np.isfinite(ky).all() or not np.isfinite(kx).all():
        raise ValueError('Axes must be finite.')
    if not np.allclose(np.diff(ky),np.diff(ky)[0]) or not np.allclose(np.diff(kx),np.diff(kx)[0]) or ky[1]<=ky[0] or kx[1]<=kx[0]:
        raise ValueError('Axes must be increasing uniformly sampled Fourier axes.')
    if not np.isfinite(k_min_factor) or k_min_factor<0:
        raise ValueError('k_min_factor must be finite and >=0.')
    order=_geometry_order(tuple(ky),tuple(kx),float(k_min_factor))
    return order[:budget(filling_ratio,len(ky),len(kx),len(order))].copy()


def validate_setup(bnds, vel_indx, window_dims, res, fps,
                   gravity_waves_switch=True, turbulence_switch=True,
                   detector_options=None, check_grid=True):
    """Validate geometry/resources without reading data or allocating a grid.

    Shared by the parent process and direct single-spectrum calls.
    """
    options = resolve_options(detector_options)
    try:
        dims = np.asarray(window_dims, dtype=float)
        b = np.asarray(bnds, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid window dimensions or bounds.') from error
    if (dims.shape != (3,) or not np.isfinite(dims).all()
            or np.any(dims < 2) or np.any(dims != np.floor(dims))):
        raise ValueError('window_dims must be integer (Nt,Ny,Nx), each >=2.')
    for name, value in (('res', res), ('fps', fps), ('vel_indx', vel_indx)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f'{name} must be a finite real scalar.')
    if min(res, fps) <= 0 or not .5 < vel_indx <= 1:
        raise ValueError('Require positive res/fps and 0.5 < vel_indx <= 1.')
    for switch in (gravity_waves_switch, turbulence_switch):
        if not isinstance(switch, (bool, np.bool_)):
            raise ValueError('Branch switches must be Boolean.')
    if not gravity_waves_switch and not turbulence_switch:
        raise ValueError('Enable at least one physical branch.')
    if (b.shape != (3, 2) or not np.isfinite(b).all() or b[2, 0] <= 0
            or b[2, 0] != b[2, 1] or np.any(b[:2, 1] <= b[:2, 0])):
        raise ValueError('Hough needs increasing vy/vx bounds and fixed positive depth.')
    if not check_grid:
        axes = spectral.wave_numbers(tuple(dims.astype(int)), res, fps)
        if len(axes[0]) < 3:
            raise ValueError('Need at least three nonnegative-frequency bins.')
        return options, b, axes, None, 0.
    requested = np.ceil((b[:2, 1] - b[:2, 0]) / options['velocity_step']) + 1
    if not np.isfinite(requested).all() or np.any(requested > np.iinfo(np.intp).max):
        raise ValueError('Requested grid exceeds host array-addressing limits.')
    counts = tuple(int(n) for n in requested)
    estimated_mb = float(4 * 8 * counts[0] * counts[1] / 2**20)
    cap = options['max_accumulator_mb']
    if cap is not None and estimated_mb > cap:
        raise MemoryError(f'Estimated accumulator workspace {estimated_mb:.1f} MiB exceeds optional limit {cap}.')
    axes = spectral.wave_numbers(tuple(dims.astype(int)), res, fps)
    if len(axes[0]) < 3:
        raise ValueError('Need at least three nonnegative-frequency bins.')
    return options, b, axes, counts, estimated_mb


@nb.njit(cache=True, nogil=True, fastmath=False, error_model='numpy')
def _accumulate_votes(acc, points, beta, intrinsic, vy, vx, dy, dx, wave, turb):
    """Rasterise already detected/interpolated peaks, without changing votes.

    SciPy smoothing/find_peaks and physical dispersion arrays stay outside.
    Retain peak order, branch order, the sensitive-axis choice and the two-bin
    linear split. Different 'other' coordinates touch disjoint cells, so both
    adjacent deposits can be made together without changing addition order at
    any cell. Serial updates avoid atomics, races and a per-thread grid copy.
    """
    for p in range(len(points)):
        x, y, w, weight = points[p]
        b, oi = beta[p], intrinsic[p]
        swap = abs(y*dy) < abs(x*dx)
        for branch in range(int(turb) + int(wave)):
            if turb and branch == 0:
                a = w
            else:
                a = ((1-b)*w - np.sqrt(b*b*w*w + (1-2*b)*oi*oi))/(1-2*b)
            count = len(vy) if swap else len(vx)
            limit = len(vx) if swap else len(vy)
            for other in range(count):
                pos = (((a-y*vy[other])/x-vx[0])/dx if swap
                       else ((a-x*vx[other])/y-vy[0])/dy)
                # Outside this interval neither adjacent cell is in the grid.
                # Check before int conversion; no clipped edge vote is created.
                if not -1. <= pos < limit:
                    continue
                low = int(np.floor(pos))
                frac = pos-low
                if 0 <= low < limit:
                    if swap:
                        acc[other, low] += weight*(1-frac)
                    else:
                        acc[low, other] += weight*(1-frac)
                if 0 <= low+1 < limit:
                    if swap:
                        acc[other, low+1] += weight*frac
                    else:
                        acc[low+1, other] += weight*frac


class State:
    """Accumulate Hough votes and retain the sampler's per-window state."""

    def __init__(self, spectrum, bnds, vel_indx, window_dims, res, fps,
                 gravity_waves_switch=True,turbulence_switch=True,
                 filling_ratio=1.,detector_options=None):
        self.t0 = perf_counter()
        self.options, b, axes, counts, self.estimated_grid_mb = validate_setup(
            bnds, vel_indx, window_dims, res, fps,
            gravity_waves_switch, turbulence_switch, detector_options)
        self.times=dict(planning=0.,calibration=0.,proxy=0.,selection=0.,extraction=0.,voting=0.,scoring=0.)
        self.wave,self.turb=bool(gravity_waves_switch),bool(turbulence_switch)
        self.alpha=float(vel_indx)
        self.depth=float(b[2,0]); self.bnds=b.copy()
        self.omega,self.ky,self.kx=axes
        self.spectrum=np.asarray(spectrum)
        if self.spectrum.shape!=(len(self.omega),len(self.ky),len(self.kx)) or len(self.omega)<3 or not np.issubdtype(self.spectrum.dtype,np.number) or np.iscomplexobj(self.spectrum):
            raise ValueError('Need a real spectrum matching package (omega,ky,kx) axes.')
        t=perf_counter()
        self.eligible,self.cutoff=eligible_pairs(self.ky,self.kx,self.options['k_min_factor'])
        self.M=budget(filling_ratio,len(self.ky),len(self.kx),len(self.eligible))
        self.plan=fixed_plan(self.ky,self.kx,filling_ratio,self.options['k_min_factor'])
        self.times['planning']+=perf_counter()-t
        self.filling_ratio=filling_ratio
        # One scale for all selected columns preserves their relative weights.
        # Calibration reads only the fixed spatial plan, within the same budget.
        C = self.M
        self.calibration_pairs=self.plan[:C]
        t=perf_counter()
        calibration=self.columns(self.calibration_pairs)
        self.scale=float(np.percentile(calibration,self.options['scale_percentile'])) if calibration.size else 0.
        if self.scale<=0 and calibration.size:
            self.scale=float(np.max(calibration))
        self.scale_fallback=False
        # A zero calibration set must not trigger reads outside the sampling plan.
        if self.scale<=0:
            self.scale=1.; self.scale_fallback=True
        self.times['calibration']+=perf_counter()-t
        self.vy=np.linspace(*b[0],int(counts[0])); self.vx=np.linspace(*b[1],int(counts[1]))
        self.dy=self.vy[1]-self.vy[0]; self.dx=self.vx[1]-self.vx[0]
        self.acc=np.zeros((len(self.vy),len(self.vx)))
        self.visited=np.zeros((len(self.ky),len(self.kx)),bool)
        self.used=[]; self.point_blocks=[]
        self.best=None; self.at_edge=True; self.n_points=0
        self._candidate_field=None
        self.X,self.Y=np.meshgrid(self.kx,self.ky)
        self.kmax=max(float(np.max(np.hypot(self.X,self.Y))),1e-12)
        self.G=np.eye(2)*1e-3
        self.history=[]

    def columns(self,pairs):
        """Read the requested (ky, kx) columns with all frequency bins."""
        a=self.spectrum[:,pairs[:,0],pairs[:,1]]
        if not np.isfinite(a).all() or np.any(a<0):
            raise ValueError('Inspected spectrum columns must be finite and nonnegative.')
        return np.asarray(a,float)

    def process(self,pairs):
        """Extract frequency peaks and add both physical branches to the votes."""
        pairs=np.asarray(pairs,int).reshape(-1,2)
        if not len(pairs):
            return
        if self.visited[pairs[:,0],pairs[:,1]].any() or len(np.unique(pairs,axis=0))!=len(pairs):
            raise ValueError('Sampler attempted duplicate columns.')
        if len(self.used)+len(pairs)>self.M:
            raise ValueError('Sampler exceeded the spatial-column budget.')
        t=perf_counter(); columns=self.columns(pairs)/self.scale; rows=[]
        # Smooth only along frequency before detecting ridge peaks.
        columns = gaussian_filter1d(
            columns,
            sigma=2.0,          # standard deviation in frequency bins
            axis=0,            # columns has shape (n_frequencies, n_pairs)
            mode="reflect",
        )
        for column,(iy,ix) in zip(columns.T,pairs):
            peaks,props=find_peaks(column,height=self.options['peak_height'],prominence=self.options['peak_prominence'])
            for p,weight in zip(peaks,props['prominences']):
                # Parabolic interpolation locates the peak between bins.
                l,c,r=column[p-1:p+2]; denominator=l-2*c+r
                off=0. if denominator==0 else .5*(l-r)/denominator
                w=self.omega[p]+np.clip(off,-.5,.5)*(self.omega[1]-self.omega[0])
                rows.append((self.kx[ix],self.ky[iy],w,min(float(weight),1.)))
        points=np.asarray(rows,float).reshape(-1,4)
        self.times['extraction']+=perf_counter()-t
        self.visited[pairs[:,0],pairs[:,1]]=True
        self.used.extend(map(tuple,pairs.tolist())); self.point_blocks.append(points)
        self.n_points+=len(points)
        # Any new votes invalidate the one-state scoring/candidate hand-off.
        self._candidate_field=None
        t=perf_counter()
        if len(points):
            k=np.hypot(points[:,0],points[:,1]); beta=dispersion.beta_calc(k,self.depth,self.alpha)
            intrinsic=dispersion.omega_i_calc(k,self.depth)
            _accumulate_votes(self.acc, points, beta, intrinsic, self.vy, self.vx,
                              self.dy, self.dx, self.wave, self.turb)
        self.times['voting']+=perf_counter()-t

    def score(self):
        """Locate the vote maximum and update adaptive directional information."""
        t=perf_counter()
        sigma=self.options['vote_sigma']
        # Release an earlier adaptive score before allocating the next field.
        self._candidate_field=None
        smoothed=gaussian_filter(self.acc,(sigma/self.dy,sigma/self.dx)) if sigma else self.acc
        self.G=np.eye(2)*1e-3
        if self.n_points>=self.options['min_ridges'] and np.max(smoothed)>0:
            iy,ix=np.unravel_index(np.argmax(smoothed),smoothed.shape)
            self.best=np.array([self.vy[iy],self.vx[ix]])
            self.at_edge=iy in (0,len(self.vy)-1) or ix in (0,len(self.vx)-1)
            points=np.concatenate(self.point_blocks)
            x,y,w,weights=points.T
            a=y*self.best[0]+x*self.best[1]
            residual=np.full(len(points),np.inf)
            if self.turb: residual=np.minimum(residual,abs(w-a))
            if self.wave:
                k=np.hypot(x,y); b=dispersion.beta_calc(k,self.depth,self.alpha); oi=dispersion.omega_i_calc(k,self.depth)
                residual=np.minimum(residual,abs(w-((1-b)*a+np.sqrt((b*a)**2+oi**2))))
            inside=residual<=2*(self.omega[1]-self.omega[0])
            features=np.column_stack((x[inside],y[inside]))/self.kmax
            self.G+=(features.T*weights[inside])@features
        else:
            self.best=None; self.at_edge=True
        # Candidate selection consumes this exact field once; do not smooth it
        # a second time. This is bounded per-state scratch, not a window cache.
        self._candidate_field=smoothed if self.best is not None else None
        self.history.append(None if self.best is None else self.best.tolist())
        self.times['scoring']+=perf_counter()-t
        return None if self.best is None else self.best.copy()

    def result(self,method,reason,extra=None):
        points=np.concatenate(self.point_blocks) if self.point_blocks else np.empty((0,4))
        rank=np.linalg.matrix_rank(points[:,:2]) if len(points) else 0
        ok=self.best is not None and not self.at_edge and rank==2
        vy,vx=(self.best if self.best is not None and rank==2 else [np.nan,np.nan])
        if not ok:
            reason+='; '+('boundary maximum' if self.best is not None and self.at_edge else 'insufficient independent ridge evidence')
        times={**self.times,'total':perf_counter()-self.t0}
        metadata=dict(method=method,budget=self.M,n_eligible=len(self.eligible),
            n_columns=len(self.used),n_ridges=self.n_points,filling_ratio=self.filling_ratio,
            actual_fraction=len(self.used)/(len(self.ky)*len(self.kx)),
            k_min_rad_m=self.cutoff,k_min_base_bins=K_MIN_BASE_BINS,
            intensity_scale=self.scale,calibration_pairs=self.calibration_pairs.tolist(),
            scale_fallback=self.scale_fallback,grid_shape=self.acc.shape,
            grid_workspace_estimate_mb=self.estimated_grid_mb,timings=times,
            estimate_history_vy_vx=self.history,
            information_ratio=float(np.linalg.eigvalsh(self.G)[0]/np.linalg.eigvalsh(self.G)[-1]),
            **(extra or {}))
        return Result(float(vy),float(vx),self.depth,bool(ok),reason,np.asarray(self.used,int).reshape(-1,2),metadata)


# Fixed spatial sampling.

def fixed_estimate(measured_spectrum,bnds,vel_indx,window_dims,res,fps,
             filling_ratio=1.,detector_options=None,
             gravity_waves_switch=True,turbulence_switch=True, _return_state=False):
    state=State(measured_spectrum,bnds,vel_indx,window_dims,res,fps,
                gravity_waves_switch,turbulence_switch,filling_ratio,detector_options)
    state.process(state.plan)
    state.score()
    result = state.result('fixed','fixed spatial-column budget processed')
    return (result, state) if _return_state else result

# Adaptive sampling combines integrated power with directional coverage.

def resolve_sampling_options(options=None):
    if options is not None and not isinstance(options,dict):
        raise ValueError('sampling_options must be a dict or None.')
    unknown=set(options or {})-set(ADAPTIVE_DEFAULTS)
    if unknown:
        raise ValueError('Unknown sampling_options: '+', '.join(map(str,unknown)))
    out={**ADAPTIVE_DEFAULTS,**(options or {})}
    for key in ('batch_size','min_columns','stable_batches','exploration_every'):
        if isinstance(out[key],(bool,np.bool_)) or not isinstance(out[key],(int,np.integer)) or out[key]<1:
            raise ValueError(f'{key} must be an integer >=1.')
    for key in ('convergence_tol','min_direction_ratio'):
        if isinstance(out[key], (bool, np.bool_)) or not isinstance(out[key], Real) or not np.isfinite(out[key]) or out[key]<0:
            raise ValueError(f'{key} must be finite and >=0.')
    if out['min_direction_ratio']>1:
        raise ValueError('min_direction_ratio must be <=1.')
    return out


def adaptive_estimate(measured_spectrum,bnds,vel_indx,window_dims,res,fps,
             filling_ratio=1.,detector_options=None,sampling_options=None,
             gravity_waves_switch=True,turbulence_switch=True, _return_state=False):
    opt=resolve_sampling_options(sampling_options)
    state=State(measured_spectrum,bnds,vel_indx,window_dims,res,fps,
                gravity_waves_switch,turbulence_switch,filling_ratio,detector_options)
    if state.M<2:
        result = state.result('adaptive','budget cannot constrain two velocity components')
        return (result, state) if _return_state else result
    t=perf_counter()
    # Integrated power ranks columns without fitting a dispersion model.
    if not np.isfinite(state.spectrum).all() or np.any(state.spectrum<0):
        raise ValueError('Adaptive power proxy requires a finite nonnegative full spectrum.')
    energy=np.sum(state.spectrum,axis=0,dtype=float)
    candidates=state.eligible
    powers=energy[candidates[:,0],candidates[:,1]]
    if not np.isfinite(powers).all():
        raise ValueError('Integrated spectral power overflowed.')
    state.times['proxy']+=perf_counter()-t
    if not len(powers) or np.max(powers)<=0:
        result = state.result('adaptive','zero integrated spectral power')
        return (result, state) if _return_state else result
    strength=np.sqrt(np.maximum(powers/np.max(powers),1e-3))
    features=np.column_stack((state.kx[candidates[:,1]],state.ky[candidates[:,0]]))/state.kmax
    eligible_index={tuple(pair):i for i,pair in enumerate(candidates.tolist())}
    priority=[int(np.argmax(powers))]
    priority.extend(eligible_index[tuple(p)] for p in state.calibration_pairs.tolist()
                    if eligible_index[tuple(p)]!=priority[0])
    priority_position=0; geometry_position=0
    previous=None; stable=0; reason='maximum spatial-column budget reached'
    while len(state.used)<state.M:
        t=perf_counter(); selected=[]; taken=state.visited.copy(); G=state.G.copy()
        count=min(opt['batch_size'],state.M-len(state.used))
        for _ in range(count):
            if priority_position<len(priority):
                idx=priority[priority_position]; priority_position+=1
            elif (len(state.used)+len(selected))%opt['exploration_every']==0:
                while geometry_position<len(state.plan) and taken[tuple(state.plan[geometry_position])]:
                    geometry_position+=1
                idx=eligible_index[tuple(state.plan[geometry_position])]
                geometry_position+=1
            else:
                inv=np.linalg.inv(G)
                gain=np.log1p(np.einsum('ij,jk,ik->i',features,inv,features))
                score=strength*gain
                score[taken[candidates[:,0],candidates[:,1]]]=-np.inf
                idx=int(np.argmax(score))
            pair=candidates[idx]
            if taken[tuple(pair)]:
                raise RuntimeError('Adaptive scheduling selected a duplicate.')
            selected.append(pair); taken[tuple(pair)]=True
            G+=strength[idx]*np.outer(features[idx],features[idx])
        state.times['selection']+=perf_counter()-t
        state.process(selected)
        current=state.score()
        eigenvalues=np.linalg.eigvalsh(state.G)
        informed=eigenvalues[-1]>1e-3 and eigenvalues[0]/eigenvalues[-1]>=opt['min_direction_ratio']
        if (current is not None and previous is not None and not state.at_edge
                and np.linalg.norm(current-previous)<=opt['convergence_tol']+1e-12 and informed):
            stable+=1
        else:
            stable=0
        previous=current
        # Ensure the common calibration prefix was actually counted/processed.
        calibration_done=priority_position>=len(priority)
        if (len(state.used)>=min(opt['min_columns'],state.M) and calibration_done
                and stable>=opt['stable_batches']):
            reason='heuristic parameter stability reached'
            break
    result = state.result('adaptive',reason,dict(sampling_options=opt,
        stable_batches_observed=stable,first_pair_power_rank=1))
    return (result, state) if _return_state else result


# Local forward-model refinement.

class _TrustRegionObjective:
    """Hard velocity disk and parameter-box gate; no in-region penalty.

    Infeasible proposals return infinity without evaluating the spectrum. This
    is a constraint, not a velocity penalty. A feasible initial simplex keeps
    finite vertices available for Nelder-Mead's inward contractions.
    """
    def __init__(self, objective, centre, radius, bounds):
        self.objective = objective
        self.centre = np.asarray(centre, float).copy()
        self.radius = float(radius)
        self.bounds = np.asarray(bounds, float).copy()
        self.calls = self.rejected = 0

    def feasible(self, velocity):
        v = np.asarray(velocity, float)
        return bool(v.shape == (len(self.bounds),) and np.isfinite(v).all()
                    and np.all(v >= self.bounds[:, 0])
                    and np.all(v <= self.bounds[:, 1])
                    and np.hypot(*(v[:2]-self.centre)) <= self.radius)

    def __call__(self, velocity):
        self.calls += 1
        if not self.feasible(velocity):
            self.rejected += 1
            return np.inf
        return self.objective(velocity)


def _trust_simplex(seed, centre, radius, bounds, step):
    """Three distinct feasible vertices; pull boundary starts slightly inward."""
    seed = np.asarray(seed, float).copy()
    centre = np.asarray(centre, float)
    distance = np.hypot(*(seed-centre))
    if distance > radius:
        raise ValueError('NM seed lies outside the primary Hough trust region.')
    if distance > .95*radius:
        seed = centre + (.95*radius/distance)*(seed-centre)
    simplex = np.tile(seed, (3, 1))
    for axis in range(2):
        other = (seed[1-axis]-centre[1-axis])/radius
        half = radius*np.sqrt(max(0., 1.-other*other))
        lower = max(bounds[axis, 0], centre[axis]-half)
        upper = min(bounds[axis, 1], centre[axis]+half)
        above, below = upper-seed[axis], seed[axis]-lower
        direction, room = (1., above) if above >= below else (-1., below)
        simplex[axis+1, axis] += direction*min(step, .5*room)
    if np.linalg.matrix_rank(simplex[1:]-simplex[0]) != 2:
        raise ValueError('Trust-region simplex is numerically degenerate; check bounds/min_radius.')
    return simplex


def _hough_candidates(state, count, separation, centre, radius):
    # Consume the scoring hand-off even on early return, so the extra field is
    # not retained during local fitting. A repeated call recomputes as before.
    field=state._candidate_field
    state._candidate_field=None
    if state.best is None:
        return []
    candidates = [state.best.copy()]
    if count == 1:
        return candidates
    sigma = state.options['vote_sigma']
    if field is None:
        field = gaussian_filter(state.acc, (sigma/state.dy, sigma/state.dx)) if sigma else state.acc
    peaks = (field == maximum_filter(field, size=3, mode='nearest')) & (field > 0)
    indices = np.argwhere(peaks)
    # Necessary bounding-square test only: retain the original scalar hypot
    # disk test below, including its boundary rounding. Filter before sorting,
    # preserving row-major peak order and stable ties among retained maxima.
    inside = ((np.abs(state.vy[indices[:,0]]-centre[0]) <= radius)
              & (np.abs(state.vx[indices[:,1]]-centre[1]) <= radius))
    indices = indices[inside]
    order = np.argsort(-field[indices[:,0],indices[:,1]], kind='stable')
    for index in order:
        iy, ix = indices[index]
        v = np.array([state.vy[iy], state.vx[ix]])
        # Restarts stay inside the disk of the primary Hough maximum.
        if np.hypot(*(v-centre)) > radius:
            continue
        if all(np.linalg.norm(v-p) > 1e-12 and np.linalg.norm(v-p) >= separation for p in candidates):
            candidates.append(v)
            if len(candidates) == count:
                break
    return candidates[:count]


def _boundary_flags(velocity, centre, radius, bounds, xatol):
    distance = float(np.hypot(*(velocity-centre)))
    radial_tolerance = min(.01*radius, max(xatol, 1e-10*radius))
    trust_boundary = bool(radius-distance <= radial_tolerance)
    tolerance = np.minimum(.01*(bounds[:, 1]-bounds[:, 0]),
                           np.maximum(xatol, 1e-10*(bounds[:, 1]-bounds[:, 0])))
    global_boundary = bool(np.any(velocity-bounds[:, 0] <= tolerance)
                           or np.any(bounds[:, 1]-velocity <= tolerance))
    return distance, trust_boundary, global_boundary


def _checked_depth_bounds(bnds):
    """Validate the shape and depth interval before selecting defaults."""
    try:
        bounds = np.asarray(bnds, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid velocity/depth bounds.') from error
    if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or bounds[2, 0] <= 0 or bounds[2, 1] < bounds[2, 0]):
        raise ValueError('Depth bounds must be finite, positive and ordered.')
    return bounds


def _depth_setup(bnds, options, wave):
    """Keep Hough at one reference depth; return the original NM depth bounds."""
    bounds = _checked_depth_bounds(bnds)
    free = bool(bounds[2, 0] < bounds[2, 1])
    if free and not options['refine']:
        raise ValueError('Depth estimation requires refine=True.')
    if free and not wave:
        raise ValueError('Depth estimation requires the gravity-wave branch; turbulence alone is depth-independent.')
    if free and options['maxfev'] < 4:
        raise ValueError('Joint velocity/depth NM needs maxfev >= 4.')
    reference = float(bounds[2, 0])
    if free:
        logs = np.log(bounds[2])
        if logs[0] >= logs[1]:
            raise ValueError('Depth bounds are indistinguishable in log coordinates.')
        reference = options['depth_initial']
        if reference is None:
            reference = float(np.exp(.5*logs[0] + .5*logs[1]))
        if not bounds[2, 0] <= reference <= bounds[2, 1]:
            raise ValueError('depth_initial must lie within the depth bounds.')
    proposal_bounds = bounds.copy()
    proposal_bounds[2] = reference
    return bounds, proposal_bounds, reference, free


def _joint_simplex(seed, centre, radius, bounds, step, log_depth, log_bounds, log_step):
    """Add one feasible log-depth vertex to the velocity triangle."""
    triangle = _trust_simplex(seed, centre, radius, bounds, step)
    simplex = np.empty((4, 3))
    simplex[:3, :2] = triangle
    simplex[:3, 2] = log_depth
    simplex[3] = simplex[0]
    above, below = log_bounds[1]-log_depth, log_depth-log_bounds[0]
    direction, room = (1., above) if above >= below else (-1., below)
    simplex[3, 2] += direction*min(log_step, .5*room)
    if np.linalg.matrix_rank(simplex[1:]-simplex[0]) != 3:
        raise ValueError('Joint simplex is degenerate; check depth bounds/log_depth_step.')
    return simplex


def _depth_boundary(log_depth, log_bounds, xatol):
    span = log_bounds[1]-log_bounds[0]
    tolerance = min(.01*span, max(xatol, 1e-10*span))
    return bool(log_depth-log_bounds[0] <= tolerance or log_bounds[1]-log_depth <= tolerance)


def _fit_metrics(objective, parameters):
    """Recover the existing NSP-based cost/quality from final model statistics.

    The NM scalar supplies measured/model overlap per unit model mass, but
    quality additionally needs the model self-overlap. Accumulate it once at
    the final guess, using the same narrow-band model as the objective.
    """
    value, model_mass, model_square, overlap = objective.statistics(parameters)
    measured_mass = float(np.sum(objective.data))
    stats = dict(fit_objective=float(value), measured_mass=measured_mass,
                 model_mass=float(model_mass), model_square=float(model_square),
                 measured_model_overlap=float(overlap))
    values = np.array([measured_mass, model_mass, model_square, overlap])
    if not np.isfinite(values).all() or np.any(values <= 0):
        return np.nan, np.nan, stats
    log_cost = np.log(measured_mass) + np.log(model_mass) - np.log(overlap)
    log_ideal = 2*np.log(model_mass) - np.log(model_square)
    quality = 1 - .2*(log_cost-log_ideal)/np.log(10.)
    with np.errstate(over='ignore'):
        cost = float(np.exp(2*log_cost))
        stats.update(nsp_cost=float(np.exp(log_cost)), ideal_nsp_cost=float(np.exp(log_ideal)))
    # DE returns its final objective squared. HNM has no penalty factor.
    return cost, float(quality), stats


def _refine_estimate(spectrum, bnds, alpha, window_dims, res, fps, wave, turb,
                     options, proposal, state, started):
    bounds, proposal_bounds, reference_depth, free_depth = _depth_setup(bnds, options, wave)
    metadata = deepcopy(proposal.metadata)
    metadata.update(sampling=metadata.get('method'), method='HNM', solver='HNM',
                    refinement_enabled=True, refinement_attempted=False, refined=False,
                    proposal_vy_vx=[proposal.vy, proposal.vx], proposal_success=proposal.success,
                    initial_step=options['initial_step_factor'] * options['velocity_step'],
                    initial_step_factor=options['initial_step_factor'],
                    hough_velocity_step=options['velocity_step'], local_runs=[], n_local_runs=0,
                    objective_evaluations=0, nm_evaluations=0, rejected_proposals=0,
                    diagnostic_evaluations=0, depth_estimation_enabled=free_depth,
                    depth_estimated=False,
                    depth_mode='joint_log' if free_depth else 'fixed',
                    reference_depth=reference_depth, depth_bounds=bounds[2].tolist(),
                    parameter_names=['vy','vx','log_depth'] if free_depth else ['vy','vx'],
                    trust_region=None, at_trust_boundary=False, at_global_boundary=False,
                    at_depth_boundary=False, at_boundary=False)
    # There is no defensible trust centre if the detector produced no candidate.
    if state.best is None or not np.isfinite([proposal.vy, proposal.vx]).all():
        metadata['timings'].update(objective_setup=0., local_fit=0., diagnostics=0., total=perf_counter()-started)
        depth = np.nan if free_depth else reference_depth
        return Result(proposal.vy, proposal.vx, depth, False,
                      'HNM: no usable Hough proposal; refinement skipped', proposal.pairs, metadata)
    core = {key: options[key] for key in DEFAULTS}
    _, _, axes, _, _ = validate_setup(proposal_bounds, alpha, window_dims, res, fps,
                                      wave, turb, core, check_grid=False)
    initial_step = options['initial_step_factor'] * options['velocity_step']
    centre = state.best.copy()
    minimum = options['min_radius']
    minimum_source = 'explicit'
    if minimum is None:
        minimum = 2.*max(state.dy, state.dx)
        minimum_source = 'twice_actual_max_grid_spacing'
    radius = max(options['relative_radius']*float(np.hypot(*centre)), minimum)
    if not np.isfinite(radius) or radius <= 8*np.spacing(max(1., float(np.max(np.abs(centre))))):
        raise ValueError('Trust radius must be finite and larger than velocity floating-point resolution.')
    box = np.column_stack((np.maximum(bounds[:2, 0], centre-radius),
                           np.minimum(bounds[:2, 1], centre+radius)))
    metadata['trust_region'] = dict(centre_vy_vx=centre.tolist(), radius=float(radius),
                                   relative_radius=options['relative_radius'],
                                   min_radius=float(minimum), min_radius_source=minimum_source,
                                   clipped_box=box.tolist())
    log_bounds = np.log(bounds[2])
    search_bounds = np.vstack((bounds[:2], log_bounds)) if free_depth else bounds[:2]
    if free_depth:
        box = np.vstack((box, log_bounds))
    setup_time = perf_counter()
    objective = _RidgeObjective(spectrum, axes, reference_depth, alpha, wave, turb, options)
    constrained = _TrustRegionObjective(objective, centre, radius, search_bounds)
    setup_time = perf_counter()-setup_time
    seeds = _hough_candidates(state, options['n_candidates'], options['candidate_separation'], centre, radius)
    runs, best, best_value, best_result = [], None, np.inf, None
    selected_run_index = None
    fitting_time = perf_counter()
    if not objective.zero:
        for candidate in seeds:
            if free_depth:
                simplex = _joint_simplex(candidate, centre, radius, bounds[:2], initial_step,
                                         np.log(reference_depth), log_bounds, options['log_depth_step'])
            else:
                simplex = _trust_simplex(candidate, centre, radius, bounds[:2], initial_step)
            if not all(constrained.feasible(vertex) for vertex in simplex):
                raise ValueError('Could not construct a feasible trust-region simplex.')
            before_calls, before_rejected = objective.calls, constrained.rejected
            result = minimize(constrained, simplex[0], method='Nelder-Mead', bounds=box,
                              options=dict(maxfev=options['maxfev'], xatol=options['xatol'],
                                           fatol=options['fatol'], initial_simplex=simplex))
            distance, at_trust, at_global = _boundary_flags(result.x[:2], centre, radius, bounds[:2], options['xatol'])
            at_depth = free_depth and _depth_boundary(result.x[2], log_bounds, options['xatol'])
            depth = float(np.clip(np.exp(result.x[2]), *bounds[2])) if free_depth else reference_depth
            record = dict(candidate_velocity=candidate.tolist(), initial_velocity=simplex[0,:2].tolist(),
                          initial_simplex=simplex.tolist(),
                          velocity=np.asarray(result.x[:2]).tolist(), depth=depth,
                          log_depth=float(result.x[2]) if free_depth else float(np.log(reference_depth)),
                          objective_value=float(result.fun), feasible=bool(constrained.feasible(result.x)),
                          nfev=int(result.nfev), nit=int(result.nit),
                          objective_evaluations=objective.calls-before_calls,
                          rejected_proposals=constrained.rejected-before_rejected,
                          correction_norm=distance, at_trust_boundary=at_trust, at_global_boundary=at_global,
                          at_depth_boundary=bool(at_depth), converged=bool(result.success), message=str(result.message))
            runs.append(record)
            if record['feasible'] and np.isfinite(result.fun) and result.fun < best_value:
                best, best_value, best_result = result.x.copy(), float(result.fun), result
                selected_run_index = len(runs)-1
    fitting_time = perf_counter()-fitting_time
    distance, at_trust, at_global, at_depth = None, False, False, False
    cost = quality = np.nan
    diagnostics_time = 0.
    if best is not None:
        distance, at_trust, at_global = _boundary_flags(best[:2], centre, radius, bounds[:2], options['xatol'])
        at_depth = free_depth and _depth_boundary(best[2], log_bounds, options['xatol'])
        depth = float(np.clip(np.exp(best[2]), *bounds[2])) if free_depth else reference_depth
        diagnostics_time = perf_counter()
        cost, quality, stats = _fit_metrics(objective, best)
        diagnostics_time = perf_counter()-diagnostics_time
        metadata.update(stats, diagnostic_evaluations=1)
    ok = bool(best_result is not None and best_result.success and best_value < 0
              and not (at_trust or at_global or at_depth))
    if best is None:
        best = np.array([np.nan, np.nan])
        depth = np.nan if free_depth else reference_depth
        reason = 'zero spectrum' if objective.zero else 'no finite feasible local fit'
    else:
        reason = str(best_result.message)
        if at_trust:
            reason += '; trust-region boundary reached (not expanded)'
        if at_global:
            reason += '; global velocity boundary reached'
        if at_depth:
            reason += '; depth boundary reached'
    metadata.update(refinement_attempted=bool(runs), refined=best_result is not None,
                    depth_estimated=bool(free_depth and best_result is not None),
                    local_runs=runs, n_local_runs=len(runs), selected_run_index=selected_run_index,
                    objective_evaluations=objective.calls,
                    nm_evaluations=constrained.calls, rejected_proposals=constrained.rejected,
                    objective_columns=int(objective.data.shape[1]),
                    objective_frequency_bins=int(objective.data.shape[0]),
                    objective_evaluator='dense_broadcast' if objective.dense else 'narrow_band',
                    kernel_tail_weight_bound=float(np.exp(-options['kernel_truncate']*options['kernel_truncate'])),
                    correction_norm=distance, at_trust_boundary=at_trust, at_global_boundary=at_global,
                    at_depth_boundary=bool(at_depth), at_boundary=bool(at_trust or at_global or at_depth))
    metadata['timings'].update(objective_setup=setup_time, local_fit=fitting_time,
                               diagnostics=diagnostics_time, total=perf_counter()-started)
    return Result(float(best[0]), float(best[1]), float(depth), ok,
                  'HNM: '+reason, proposal.pairs, metadata, cost, quality)


# === Optional comparisons of existing HNM endpoints; no extra searches ===
# These comparisons describe observed endpoints only, never global coverage.
from scipy.stats import norm
from iwave.spectral import SpectralOverlapObjective
from iwave.spectral import (boolean,resolve_spectral_request,
    prepare_spectral_data,normalization_jacobian)
from iwave.io import _constraints

MODE_AMBIGUITY_DEFAULTS=dict(confidence=.95,variance_estimator='plugin',n_effective=None,merge_tolerances=None)


def resolve_mode_request(enabled=False,options=None):
    enabled=boolean(enabled,'calculate_mode_ambiguity')
    if options is not None and not isinstance(options,dict):raise ValueError('mode_ambiguity_options must be a dictionary or None.')
    if set(options or {})-set(MODE_AMBIGUITY_DEFAULTS):raise ValueError('Unknown mode_ambiguity_options.')
    out=deepcopy(MODE_AMBIGUITY_DEFAULTS);out.update(deepcopy(options or {}))
    noise=resolve_spectral_request(True,{k:out[k] for k in ('confidence','variance_estimator','n_effective')})
    for key in ('confidence','variance_estimator','n_effective'):out[key]=noise[key]
    if out['merge_tolerances'] is not None:
        a=np.asarray(out['merge_tolerances'])
        if a.shape not in ((2,),(3,)) or not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a) or not np.isfinite(a).all() or np.any(a<=0):
            raise ValueError('merge_tolerances needs two or three positive finite numbers.')
        out['merge_tolerances']=a.astype(float).tolist()
    return out if enabled else None


def _score_statistics(weights,M,full,active,variance):
    if weights is None:return None
    overlap=float(np.sum(weights*M))
    if not np.isfinite(overlap) or overlap<=0:return None
    gradient=normalization_jacobian((-weights/overlap)[None],full,active)[0]
    noise_variance=float(np.sum(gradient*gradient*variance))
    if not np.isfinite(noise_variance):return None
    return dict(overlap=overlap,log_score=float(-np.log(overlap)),gradient=gradient,variance=noise_variance)


def _contrast(reference,alternative,variance,multiplier):
    """Stable shared-data contrast variance, including normalization covariance."""
    gap=float(alternative['log_score']-reference['log_score'])
    difference=alternative['gradient']-reference['gradient']
    v=float(np.sum(difference*difference*variance))
    covariance=float(np.sum(reference['gradient']*alternative['gradient']*variance))
    out=dict(valid=False,score_gap=gap,reference_variance=reference['variance'],
        alternative_variance=alternative['variance'],score_covariance=covariance,
        difference_variance=v,standard_error=None,standardized_gap=None,
        noise_threshold=None,competitive=None,reason=None)
    if not np.isfinite([gap,v,covariance]).all() or v<0:
        out['reason']='nonfinite_score_contrast';return out
    tolerance=32*np.finfo(float).eps*max(1.,abs(reference['log_score']),abs(alternative['log_score']))
    if v==0:
        if abs(gap)<=tolerance:
            out.update(valid=True,standard_error=0.,noise_threshold=0.,competitive=True,
                       reason='indistinguishable_zero_contrast')
        else:out['reason']='unresolved_zero_noise_contrast'
        return out
    se=float(np.sqrt(v));threshold=float(multiplier*se)
    out.update(valid=True,standard_error=se,standardized_gap=float(gap/se),noise_threshold=threshold,
               competitive=bool(gap<=threshold),reason='within_noise_scale' if gap<=threshold else 'score_gap_resolved')
    if gap < -tolerance:out['reason']='alternative_has_better_score'
    return out


def estimate_mode_ambiguity(spectrum,velocity,depth,bounds,window_dims,res,fps,vel_indx,
        spectral_context,metadata,fit_options,gravity_waves_switch=True,turbulence_switch=True,
        options=None,optimizer_success=None,method='HNM'):
    """Compare distinct interior endpoints from existing NM starts.

    A false ambiguous flag applies ONLY to the explored endpoints. Selection and
    multiple-comparison effects are not calibrated. The estimator is untouched.
    """
    started=perf_counter();opts=resolve_mode_request(True,options)
    M,full,active,k,source=prepare_spectral_data(spectrum,spectral_context,opts)
    b=np.asarray(bounds,float)
    if b.shape!=(3,2) or not np.isfinite(b).all() or np.any(b[:,0]>b[:,1]) or b[2,0]<=0:
        raise ValueError('bounds must be ordered (vy,vx,positive depth) intervals.')
    free=bool(b[2,0]<b[2,1]);p=3 if free else 2
    runs=metadata.get('local_runs',[]) if isinstance(metadata,dict) else []
    if not isinstance(runs,list):raise ValueError('local_runs must be a list.')
    out=dict(kind='spectral_mode_ambiguity',status='not_assessed',assessed=False,valid=False,
        reason=None,ambiguous=None,flags=[],options=opts,solver=method,
        message=('Ambiguity cannot be assessed if n_candidates = 1.'
                 if method=='HNM' and isinstance(fit_options,dict)
                 and fit_options.get('n_candidates')==1 else None),
        confidence_level=opts['confidence'],threshold_multiplier=float(norm.ppf((1+opts['confidence'])/2)),
        calibrated_confidence=False,selection_adjusted=False,global_uniqueness_claim=False,
        parameter_names=['vy','vx']+(['log_depth'] if free else []),
        parameter_units=['m/s','m/s']+(['log(d/1m)'] if free else []),
        n_effective=k,effective_averages_source=source,variance_estimator=opts['variance_estimator'],
        n_starts_requested=fit_options.get('n_candidates') if isinstance(fit_options,dict) else None,
        n_starts_attempted=len(runs),n_valid_runs=None,n_distinct_modes=None,
        excluded_runs=[],modes=[],comparisons=[],merge_tolerances=None,
        extra_searches=0,model_evaluations=0,
        scope='Existing converged, feasible, interior endpoints only; no search outside the existing region.')
    def finish(reason=None):
        if reason is not None:out['reason']=reason
        out['elapsed_seconds']=perf_counter()-started
        return out
    if method!='HNM':return finish('solver_has_no_reusable_multistart')
    if not isinstance(metadata,dict):return finish('no_refinement_metadata')
    if not metadata.get('refinement_enabled',False):return finish('refinement_disabled')
    if not optimizer_success or metadata.get('at_boundary',False):return finish('selected_fit_invalid')
    if len(runs)<2:return finish('single_start' if len(runs)==1 else 'no_local_runs')
    if not np.isfinite(depth) or depth<=0:return finish('selected_fit_invalid')
    q0=np.r_[velocity,np.log(depth)] if free else np.asarray(velocity,float)
    if q0.shape!=(p,) or not np.isfinite(q0).all():return finish('selected_fit_invalid')
    constraints={'velocity_disk':metadata['trust_region']} if metadata.get('trust_region') is not None else None
    box,disk,feasible=_constraints(bounds,constraints,free)
    out['search_bounds']=box.tolist();out['velocity_disk']=deepcopy(metadata.get('trust_region'))
    if not feasible(q0):return finish('selected_fit_outside_search_region')
    tolerances=opts['merge_tolerances']
    if tolerances is None:tolerances=np.full(p,10*fit_options['xatol'])
    elif len(tolerances)<p:raise ValueError('Joint mode ambiguity requires three merge_tolerances.')
    else:tolerances=np.asarray(tolerances[:p],float)
    out['merge_tolerances']=np.asarray(tolerances).tolist()
    qualified=[]
    for i,run in enumerate(runs):
        reason=None;q=None
        if not isinstance(run,dict):reason='invalid_run_record'
        elif not run.get('converged',False):reason='optimizer_failed'
        elif any(run.get(key,False) for key in ('at_trust_boundary','at_global_boundary','at_depth_boundary')):reason='boundary_limited'
        elif not run.get('feasible',False):reason='infeasible_endpoint'
        else:
            try:
                d=float(run['depth']);v=np.asarray(run['velocity'],float);value=float(run['objective_value'])
                q=np.r_[v,np.log(d)] if free and d>0 else v
                if v.shape!=(2,) or q.shape!=(p,) or not np.isfinite(q).all() or d<=0 or not np.isfinite(value) or value>=0:
                    reason='invalid_endpoint'
                elif not free and d!=depth:reason='inconsistent_fixed_depth'
                elif not feasible(q):reason='infeasible_endpoint'
            except (KeyError,TypeError,ValueError):reason='invalid_run_record'
        if reason is not None:out['excluded_runs'].append(dict(run_index=i,reason=reason))
        else:qualified.append(dict(run_index=i,point=q,value=value,depth=d))
    out['n_valid_runs']=len(qualified)
    if len(qualified)<2:return finish('insufficient_valid_runs')
    selected=metadata.get('selected_run_index')
    chosen=[r for r in qualified if r['run_index']==selected]
    if not chosen or np.max(abs(chosen[0]['point']-q0)/tolerances)>1:
        return finish('selected_run_unavailable')
    # Complete-link merging avoids chaining many nearly adjacent endpoints.
    groups=[dict(point=q0.copy(),depth=float(depth),members=[chosen[0]],selected=True)]
    for run in sorted(qualified,key=lambda r:(r['value'],r['run_index'])):
        if run['run_index']==selected:continue
        group=next((g for g in groups if all(np.max(abs(run['point']-member['point'])/tolerances)<=1 for member in g['members'])),None)
        if group is None:groups.append(dict(point=run['point'],depth=run['depth'],members=[run],selected=False))
        else:group['members'].append(run)
    out['n_distinct_modes']=len(groups)
    for i,g in enumerate(groups):
        out['modes'].append(dict(mode_index=i,selected=g['selected'],velocity=g['point'][:2].tolist(),depth=g['depth'],
            parameters=g['point'].tolist(),run_indices=sorted(r['run_index'] for r in g['members']),
            n_runs=len(g['members']),
            fitted_objective_value=float(chosen[0]['value'] if g['selected'] else g['members'][0]['value']),
            log_score=float(-np.log(-(chosen[0]['value'] if g['selected'] else g['members'][0]['value'])))))
    if len(groups)==1:
        out.update(status='assessed',assessed=True,valid=True,ambiguous=False)
        return finish('one_observed_basin')
    variance=full**2/(k if opts['variance_estimator']=='plugin' else k+1)
    evaluator=SpectralOverlapObjective(M,spectral.wave_numbers(window_dims,res,fps),depth,vel_indx,
        gravity_waves_switch,turbulence_switch,fit_options)
    reference=_score_statistics(evaluator.normalized_weights(q0),M,full,active,variance)
    out['model_evaluations']+=1
    if reference is None:return finish('invalid_reference_score')
    out['modes'][0]['log_score']=reference['log_score']
    for i,g in enumerate(groups[1:],1):
        alternative=_score_statistics(evaluator.normalized_weights(g['point']),M,full,active,variance)
        out['model_evaluations']+=1
        if alternative is None:
            comparison=dict(mode_index=i,valid=False,competitive=None,reason='invalid_alternative_score')
        else:
            out['modes'][i]['log_score']=alternative['log_score']
            comparison=dict(_contrast(reference,alternative,variance,out['threshold_multiplier']),mode_index=i)
            if comparison['reason']=='alternative_has_better_score':out['flags'].append('alternative_has_better_score')
        out['comparisons'].append(comparison)
    valid=all(c['valid'] for c in out['comparisons'])
    competitive=any(c['competitive'] is True for c in out['comparisons'])
    out.update(status='assessed' if valid else 'partial',assessed=valid,valid=valid,
               ambiguous=True if competitive else False if valid else None)
    return finish('competitive_alternative' if competitive else 'alternatives_resolved_on_noise_scale' if valid else 'invalid_score_comparison')
