"""Selectable fixed-depth Hough backend for the existing IWaVE dispatcher.
"""
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
import math
from numbers import Real
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from iwave import dispersion, spectral

from copy import deepcopy

K_MIN_BASE_BINS = 2.0 # minimum wavenumber parameter TODO: identify default
DEFAULTS = dict(velocity_step=.01, vote_sigma=.015, k_min_factor=1.,
                peak_height=.25, peak_prominence=.20, scale_percentile=99.5,
                min_ridges=12, max_accumulator_mb=None)
# Fixed design conventions, not separately fitted parameters.
RADIAL_STRATA, ANGULAR_STRATA, CALIBRATION_COLUMNS = 4, 12, 32


HOUGH_DEFAULTS = {
    **DEFAULTS,
    "sampling": "fixed",
    "filling_ratio": 1.0,
    "sampling_options": None,
}


ADAPTIVE_DEFAULTS=dict(batch_size=64,min_columns=64,stable_batches=2,
    convergence_tol=.01,exploration_every=4,min_direction_ratio=.05)


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
        """Only a compatibility adapter; placeholders do NOT calculate scores."""
        return self.vy, self.vx, self.depth, np.nan, np.nan, self.success, self.message



def resolve_hough_options(options=None):
    """Return a fresh complete settings dictionary; reject silent misspellings."""
    if options is not None and not isinstance(options, dict):
        raise ValueError("hough_options must be a dictionary or None.")
    unknown = set(options or {}) - set(HOUGH_DEFAULTS)
    if unknown:
        raise ValueError("Unknown hough_options: " + ", ".join(map(str, unknown)))
    out = deepcopy(HOUGH_DEFAULTS)
    out.update(deepcopy(options or {}))
    if out["sampling"] not in ("fixed", "adaptive"):
        raise ValueError("sampling must be 'fixed' or 'adaptive'.")
    # Validate the ratio independently of dimensions without allocating data.
    budget(out["filling_ratio"], 1, 1, 1)
    core = resolve_options({key: out[key] for key in DEFAULTS})
    out.update(core)
    if out["sampling"] == "adaptive":
        out["sampling_options"] = resolve_sampling_options(out["sampling_options"])
    elif out["sampling_options"] is not None:
        raise ValueError("sampling_options is only valid with sampling='adaptive'.")
    return out


def validate_hough_setup(bnds, vel_indx, window_dims, res, fps,
                         gravity_waves_switch=True, turbulence_switch=True,
                         hough_options=None):
    """Parent-safe validation: no spectrum read, peak search or grid allocation."""
    options = resolve_hough_options(hough_options)
    return validate_setup(
        bnds, vel_indx, window_dims, res, fps,
        gravity_waves_switch, turbulence_switch,
        {key: options[key] for key in DEFAULTS})


def optimize_single_spectrum_velocity_hough(
    measured_spectrum, bnds, vel_indx, window_dims, res, fps,
    gravity_waves_switch=True, turbulence_switch=True,
    hough_options=None, return_metadata=False,
):
    """Return the original seven worker slots, optionally plus audit metadata.

    Cost and quality are NaN placeholders, never calculated. Metadata includes
    processed (iy,ix) pairs and numerical-stability history, not uncertainty.
    """
    options = resolve_hough_options(hough_options)
    arguments = dict(
        measured_spectrum=measured_spectrum, bnds=bnds, vel_indx=vel_indx,
        window_dims=window_dims, res=res, fps=fps,
        gravity_waves_switch=gravity_waves_switch,
        turbulence_switch=turbulence_switch,
        filling_ratio=options["filling_ratio"],
        hough_options={key: options[key] for key in DEFAULTS},
    )
    if options["sampling"] == "adaptive":
        result = adaptive_estimate(
            **arguments, sampling_options=options["sampling_options"])
    else:
        result = fixed_estimate(**arguments)
    output = result.as_iwave_tuple()
    if return_metadata:
        metadata = dict(result.metadata, selected_pairs=result.pairs)
        return (*output, metadata)
    return output


## HOUGH CORE


def resolve_options(options=None):
    if options is not None and not isinstance(options, dict):
        raise ValueError('hough_options must be a dictionary or None.')
    unknown = set(options or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown hough_options: '+', '.join(map(str, unknown)))
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
    # User's window_size means spatial count Ny*Nx, never Nt or one side length.
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
                   hough_options=None):
    """Validate geometry/resources without reading data or allocating a grid.

    Shared by the parent process and direct single-spectrum calls.
    """
    options = resolve_options(hough_options)
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
    def __init__(self, spectrum, bnds, vel_indx, window_dims, res, fps,
                 gravity_waves_switch=True,turbulence_switch=True,
                 filling_ratio=1.,hough_options=None):
        self.t0 = perf_counter()
        self.options, b, axes, counts, self.estimated_grid_mb = validate_setup(
            bnds, vel_indx, window_dims, res, fps,
            gravity_waves_switch, turbulence_switch, hough_options)
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
        # Common a-priori calibration prefix, INCLUDED in the processing budget.
        # Adaptive can still start voting at its highest-power column, since
        # calibration uses at most a quarter of the budget, capped at 32 columns.
        C=min(CALIBRATION_COLUMNS,max(1,self.M//4)) if self.M else 0
        self.calibration_pairs=self.plan[:C]
        t=perf_counter()
        calibration=self.columns(self.calibration_pairs)
        self.scale=float(np.percentile(calibration,self.options['scale_percentile'])) if calibration.size else 0.
        if self.scale<=0 and calibration.size:
            self.scale=float(np.max(calibration))
        self.scale_fallback=False
        # Do not use unseen columns to repair a zero prefix. Fixed scaling at 1
        # keeps the fixed version independent of all unselected amplitudes.
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
        a=self.spectrum[:,pairs[:,0],pairs[:,1]]
        if not np.isfinite(a).all() or np.any(a<0):
            raise ValueError('Inspected spectrum columns must be finite and nonnegative.')
        return np.asarray(a,float)

    def process(self,pairs):
        pairs=np.asarray(pairs,int).reshape(-1,2)
        if not len(pairs):
            return
        if self.visited[pairs[:,0],pairs[:,1]].any() or len(np.unique(pairs,axis=0))!=len(pairs):
            raise ValueError('Sampler attempted duplicate columns.')
        if len(self.used)+len(pairs)>self.M:
            raise ValueError('Sampler exceeded the spatial-column budget.')
        t=perf_counter(); columns=self.columns(pairs)/self.scale; rows=[]
        for column,(iy,ix) in zip(columns.T,pairs):
            peaks,props=find_peaks(column,height=self.options['peak_height'],prominence=self.options['peak_prominence'])
            for p,weight in zip(peaks,props['prominences']):
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
                for a in branches:
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


# FIXED SAMPLING

def fixed_estimate(measured_spectrum,bnds,vel_indx,window_dims,res,fps,
             filling_ratio=1.,hough_options=None,
             gravity_waves_switch=True,turbulence_switch=True):
    state=State(measured_spectrum,bnds,vel_indx,window_dims,res,fps,
                gravity_waves_switch,turbulence_switch,filling_ratio,hough_options)
    state.process(state.plan)
    state.score()
    return state.result('fixed','fixed spatial-column budget processed')

## ADAPTIVE SAMPLING
# the adaptive approach searches for the best columns to use depending on integrated spectrum along kt

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
             filling_ratio=1.,hough_options=None,sampling_options=None,
             gravity_waves_switch=True,turbulence_switch=True):
    opt=resolve_sampling_options(sampling_options)
    state=State(measured_spectrum,bnds,vel_indx,window_dims,res,fps,
                gravity_waves_switch,turbulence_switch,filling_ratio,hough_options)
    if state.M<2:
        return state.result('adaptive','budget cannot constrain two velocity components')
    t=perf_counter()
    # O(Nt*Ny*Nx) cheap proxy. No local maxima, peak locations or candidate
    # dispersion templates are evaluated in constructing this map.
    if not np.isfinite(state.spectrum).all() or np.any(state.spectrum<0):
        raise ValueError('Adaptive power proxy requires a finite nonnegative full spectrum.')
    energy=np.sum(state.spectrum,axis=0,dtype=float)
    candidates=state.eligible
    powers=energy[candidates[:,0],candidates[:,1]]
    if not np.isfinite(powers).all():
        raise ValueError('Integrated spectral power overflowed.')
    state.times['proxy']+=perf_counter()-t
    if not len(powers) or np.max(powers)<=0:
        return state.result('adaptive','zero integrated spectral power')
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
    return state.result('adaptive',reason,dict(sampling_options=opt,
        stable_batches_observed=stable,first_pair_power_rank=1))
