"""Fixed-depth Hough/Nelder-Mead velocity estimation.

Hough voting supplies the initial velocity. Nelder-Mead refines it within a
fixed disk centred on that estimate, intersected with the global bounds.
"""
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
import math
from numbers import Real
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter
from scipy.optimize import minimize
from scipy.signal import find_peaks
from iwave import dispersion, spectral

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
REFINEMENT_DEFAULTS = dict(
    gauss_width=1., kernel_truncate=5., maxfev=160, xatol=1e-4,
    fatol=1e-6, initial_step_factor=2., n_candidates=1,
    candidate_separation=.1,
)
HNM_DEFAULTS = {
    **DEFAULTS,
    **ADAPTIVE_DEFAULTS,
    **REFINEMENT_DEFAULTS,
    "sampling": "fixed",
    "filling_ratio": 1.0,
    "refine": True,
    "relative_radius": 0.8,
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

    def as_iwave_tuple(self):
        """Return the common optimizer fields; HNM leaves cost/quality unset."""
        return self.vy, self.vx, self.depth, np.nan, np.nan, self.success, self.message



def resolve_hnm_options(options=None):
    """Copy and validate the flat HNM settings before starting workers."""
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
    out.update(deepcopy(options or {}))
    if out['sampling'] not in ('fixed', 'adaptive'):
        raise ValueError("sampling must be 'fixed' or 'adaptive'.")
    budget(out['filling_ratio'], 1, 1, 1)
    out.update(resolve_options({key: out[key] for key in DEFAULTS}))
    out.update(resolve_sampling_options({key: out[key] for key in ADAPTIVE_DEFAULTS}))
    if not isinstance(out['refine'], (bool, np.bool_)):
        raise ValueError('refine must be Boolean.')
    out['refine'] = bool(out['refine'])
    positive = ('gauss_width', 'kernel_truncate', 'xatol', 'fatol', 'initial_step_factor')
    for key in (*positive, 'candidate_separation', 'relative_radius', 'min_radius'):
        value = out[key]
        if key == 'min_radius' and value is None:
            continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0
                or (key in (*positive, 'min_radius') and value == 0)):
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
    options = resolve_hnm_options(hnm_options)
    return validate_setup(bnds, vel_indx, window_dims, res, fps,
                          gravity_waves_switch, turbulence_switch,
                          {key: options[key] for key in DEFAULTS})


def optimize_single_spectrum_velocity_hnm(
    measured_spectrum, bnds, vel_indx, window_dims, res, fps,
    gravity_waves_switch=True, turbulence_switch=True,
    hnm_options=None, return_metadata=False,
):
    """Estimate one velocity and optionally append per-window diagnostics."""
    started = perf_counter()
    options = resolve_hnm_options(hnm_options)
    validate_hnm_setup(bnds, vel_indx, window_dims, res, fps,
                       gravity_waves_switch, turbulence_switch, options)
    arguments = dict(measured_spectrum=measured_spectrum, bnds=bnds, vel_indx=vel_indx,
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
        result.metadata['timings']['total'] = perf_counter()-started
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
        t=perf_counter()
        if len(points):
            k=np.hypot(points[:,0],points[:,1]); beta=dispersion.beta_calc(k,self.depth,self.alpha)
            intrinsic=dispersion.omega_i_calc(k,self.depth)
            for (x,y,w,weight),b,oi in zip(points,beta,intrinsic):
                branches=[]
                if self.turb: branches.append(w)
                if self.wave: branches.append(((1-b)*w-np.sqrt(b*b*w*w+(1-2*b)*oi*oi))/(1-2*b))
                # Each branch constrains ky*vy + kx*vx to one value.
                for a in branches:
                    # Rasterise along the more sensitive axis and split adjacent bins.
                    swap=abs(y*self.dy)<abs(x*self.dx)
                    pos=((a-y*self.vy)/x-self.vx[0])/self.dx if swap else ((a-x*self.vx)/y-self.vy[0])/self.dy
                    low=np.floor(pos).astype(np.int64); frac=pos-low
                    for shift,weights in ((0,1-frac),(1,frac)):
                        dest=low+shift; limit=len(self.vx) if swap else len(self.vy)
                        valid=(dest>=0)&(dest<limit); other=np.flatnonzero(valid)
                        index=(other,dest[valid]) if swap else (dest[valid],other)
                        np.add.at(self.acc,index,weight*weights[valid])
        self.times['voting']+=perf_counter()-t

    def score(self):
        """Locate the vote maximum and update adaptive directional information."""
        t=perf_counter()
        sigma=self.options['vote_sigma']
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

class _RidgeObjective:
    """Original max-of-Gaussians forward score, evaluated in narrow bands.

    Minimises -sum(M*T)/sum(T), with no velocity penalty. A band extends at
    least kernel_truncate*gauss_width from each predicted branch. Outside the
    union, omitted kernel weights are <= exp(-kernel_truncate**2).
    Overlapping branch bands are counted once and use max(), not addition.
    Predictions just outside the observed frequency interval retain their
    observable Gaussian tails; neither model support nor data are reflected.
    """
    def __init__(self, spectrum, axes, depth, alpha, wave, turb, options):
        self.omega, ky, kx = axes
        a = np.asarray(spectrum)
        expected = (len(self.omega), len(ky), len(kx))
        if (a.shape != expected or not np.issubdtype(a.dtype, np.number)
                or np.iscomplexobj(a) or not np.isfinite(a).all()
                or np.any(a < 0)):
            raise ValueError('Local fitting needs a finite nonnegative full spectrum matching (omega,ky,kx).')
        self.data = np.asarray(a, float).reshape(len(self.omega), -1)
        self.x, self.y = (v.ravel() for v in np.meshgrid(kx, ky))
        k = np.hypot(self.x, self.y)
        self.beta = dispersion.beta_calc(k, depth, alpha)
        self.intrinsic = dispersion.omega_i_calc(k, depth)
        self.wave, self.turb = bool(wave), bool(turb)
        self.width = options['gauss_width']
        self.dw = float(self.omega[1] - self.omega[0])
        extent = options['kernel_truncate'] * self.width / self.dw
        self.radius = int(math.ceil(min(extent, len(self.omega))))
        self.calls = 0
        self.zero = not np.any(self.data)
        # Avoid making a "narrow" workspace larger than a dense broadcast.
        self.dense = len(self.omega) <= (self.wave + self.turb) * (2*self.radius + 2)
        self.offsets = (None if self.dense else np.arange(-self.radius, self.radius + 2)[:, None])

    def __call__(self, velocity):
        self.calls += 1
        if self.zero:
            return np.inf
        with np.errstate(over='ignore', invalid='ignore'):
            a = self.y * velocity[0] + self.x * velocity[1]
            centres = []
            if self.turb:
                centres.append(a)
            if self.wave:
                centres.append((1-self.beta)*a + np.hypot(self.beta*a, self.intrinsic))
        if not all(np.isfinite(c).all() for c in centres):
            return np.inf
        if self.dense:
            weight = np.exp(-((self.omega[:, None] - centres[0]) / self.width)**2)
            if len(centres) == 2:
                np.maximum(weight, np.exp(-((self.omega[:, None] - centres[1]) / self.width)**2), out=weight)
            denominator = float(weight.sum())
            numerator = float(np.einsum('ij,ij->', self.data, weight))
        else:
            n = len(self.omega)
            # Clipping bases here only prevents unsafe float-to-int casts for
            # distant predictions. Such branches contribute no observed bins.
            bases = [np.floor(np.clip((c-self.omega[0])/self.dw,
                     -self.radius-2, n+self.radius+2)).astype(np.int64) for c in centres]
            numerator = denominator = 0.
            for branch, centre in enumerate(centres):
                active = np.flatnonzero((centre >= self.omega[0]-(self.radius+2)*self.dw)
                                        & (centre <= self.omega[-1]+(self.radius+2)*self.dw))
                if not len(active):
                    continue
                index = bases[branch][active] + self.offsets
                valid = (index >= 0) & (index < n)
                if branch:
                    previous = bases[0][active]
                    valid &= ~((index >= previous-self.radius) & (index <= previous+self.radius+1))
                frequency = self.omega[0] + self.dw*index
                weight = np.exp(-((frequency-centres[0][active])/self.width)**2)
                if len(centres) == 2:
                    np.maximum(weight, np.exp(-((frequency-centres[1][active])/self.width)**2), out=weight)
                weight *= valid
                sampled = self.data[np.clip(index, 0, n-1), active]
                numerator += float(np.einsum('ij,ij->', sampled, weight))
                denominator += float(weight.sum())
        if denominator <= 0 or not np.isfinite(numerator + denominator):
            return np.inf
        return -numerator / denominator


class _TrustRegionObjective:
    """Hard disk/box feasibility gate; never changes the in-region objective.

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
        return bool(v.shape == (2,) and np.isfinite(v).all()
                    and np.all(v >= self.bounds[:, 0])
                    and np.all(v <= self.bounds[:, 1])
                    and np.hypot(*(v-self.centre)) <= self.radius)

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
    if state.best is None:
        return []
    candidates = [state.best.copy()]
    if count == 1:
        return candidates
    sigma = state.options['vote_sigma']
    field = gaussian_filter(state.acc, (sigma/state.dy, sigma/state.dx)) if sigma else state.acc
    peaks = (field == maximum_filter(field, size=3, mode='nearest')) & (field > 0)
    indices = np.argwhere(peaks)
    order = np.argsort(-field[peaks], kind='stable')
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


def _refine_estimate(spectrum, bnds, alpha, window_dims, res, fps, wave, turb,
                     options, proposal, state, started):
    metadata = deepcopy(proposal.metadata)
    metadata.update(sampling=metadata.get('method'), method='HNM', solver='HNM',
                    refinement_enabled=True, refinement_attempted=False, refined=False,
                    proposal_vy_vx=[proposal.vy, proposal.vx], proposal_success=proposal.success,
                    initial_step=options['initial_step_factor'] * options['velocity_step'],
                    initial_step_factor=options['initial_step_factor'],
                    hough_velocity_step=options['velocity_step'], local_runs=[], n_local_runs=0,
                    objective_evaluations=0, nm_evaluations=0, rejected_proposals=0,
                    trust_region=None, at_trust_boundary=False, at_global_boundary=False,
                    at_boundary=False)
    # There is no defensible trust centre if the detector produced no candidate.
    if state.best is None or not np.isfinite([proposal.vy, proposal.vx]).all():
        metadata['timings'].update(objective_setup=0., local_fit=0., total=perf_counter()-started)
        return Result(proposal.vy, proposal.vx, proposal.depth, False,
                      'HNM: no usable Hough proposal; refinement skipped', proposal.pairs, metadata)
    core = {key: options[key] for key in DEFAULTS}
    _, bounds, axes, _, _ = validate_setup(bnds, alpha, window_dims, res, fps,
                                         wave, turb, core, check_grid=False)
    # The requested Hough spacing defines the initial NM scale. Feasibility
    # may shorten individual edges; it does not change the trust radius.
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
    setup_time = perf_counter()
    objective = _RidgeObjective(spectrum, axes, bounds[2, 0], alpha, wave, turb, options)
    constrained = _TrustRegionObjective(objective, centre, radius, bounds[:2])
    setup_time = perf_counter()-setup_time
    seeds = _hough_candidates(state, options['n_candidates'], options['candidate_separation'], centre, radius)
    runs, best, best_value, best_result = [], None, np.inf, None
    fitting_time = perf_counter()
    if not objective.zero:
        for candidate in seeds:
            simplex = _trust_simplex(candidate, centre, radius, bounds[:2], initial_step)
            if not all(constrained.feasible(vertex) for vertex in simplex):
                raise ValueError('Could not construct a feasible trust-region simplex.')
            before_calls, before_rejected = objective.calls, constrained.rejected
            result = minimize(constrained, simplex[0], method='Nelder-Mead', bounds=box,
                              options=dict(maxfev=options['maxfev'], xatol=options['xatol'],
                                           fatol=options['fatol'], initial_simplex=simplex))
            distance, at_trust, at_global = _boundary_flags(result.x, centre, radius, bounds[:2], options['xatol'])
            record = dict(candidate_velocity=candidate.tolist(), initial_velocity=simplex[0].tolist(),
                          initial_simplex=simplex.tolist(),
                          velocity=np.asarray(result.x).tolist(), nfev=int(result.nfev), nit=int(result.nit),
                          objective_evaluations=objective.calls-before_calls,
                          rejected_proposals=constrained.rejected-before_rejected,
                          correction_norm=distance, at_trust_boundary=at_trust, at_global_boundary=at_global,
                          converged=bool(result.success), message=str(result.message))
            runs.append(record)
            if constrained.feasible(result.x) and np.isfinite(result.fun) and result.fun < best_value:
                best, best_value, best_result = result.x.copy(), float(result.fun), result
    fitting_time = perf_counter()-fitting_time
    distance, at_trust, at_global = (None, False, False)
    if best is not None:
        distance, at_trust, at_global = _boundary_flags(best, centre, radius, bounds[:2], options['xatol'])
    ok = bool(best_result is not None and best_result.success and best_value < 0 and not (at_trust or at_global))
    if best is None:
        best = np.array([np.nan, np.nan])
        reason = 'zero spectrum' if objective.zero else 'no finite feasible local fit'
    else:
        reason = str(best_result.message)
        if at_trust:
            reason += '; trust-region boundary reached (not expanded)'
        if at_global:
            reason += '; global velocity boundary reached'
    metadata.update(refinement_attempted=bool(runs), refined=best_result is not None,
                    local_runs=runs, n_local_runs=len(runs), objective_evaluations=objective.calls,
                    nm_evaluations=constrained.calls, rejected_proposals=constrained.rejected,
                    objective_columns=int(objective.data.shape[1]),
                    objective_frequency_bins=int(objective.data.shape[0]),
                    objective_evaluator='dense_broadcast' if objective.dense else 'narrow_band',
                    kernel_tail_weight_bound=float(np.exp(-options['kernel_truncate']*options['kernel_truncate'])),
                    correction_norm=distance, at_trust_boundary=at_trust, at_global_boundary=at_global,
                    at_boundary=bool(at_trust or at_global))
    metadata['timings'].update(objective_setup=setup_time, local_fit=fitting_time, total=perf_counter()-started)
    return Result(float(best[0]), float(best[1]), float(bounds[2, 0]), ok,
                  'HNM: '+reason, proposal.pairs, metadata)
