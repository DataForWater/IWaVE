"""IWaVE main api."""

import cv2
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
from iwave import window, spectral, io, optimise, dispersion, LazySpectrumArray, LazyWindowArray
from iwave.io import SpectralUncertainty, ModeAmbiguity


repr_template = """
Resolution [m]: {}
Window size (y, x): {}
Overlap (y, x): {}
Size of time slices: {}
Overlap in time slices: {}
Number of images: {}
Frames per second: {}
""".format

# optim kwargs for differential evolution algorithm
OPTIM_KWARGS_SADE = {
    "strategy" : 'best1bin', 
    "popsize": 8,
    "maxiter": int(1e05),
    "workers": 1,
    "init" : 'sobol',
    "atol" : 1e-12
}

@dataclass(eq=False)
class VelocimetryStatus:
    """Window diagnostics for the latest fit, with a derived overall summary.
    Status if False if estimates lie on the bounds of the search space.
    None arrays mean that no fit has run. An empty completed fit is unsuccessful.
    The summary is derived so it stays consistent with the window mask.
    """

    windows: Optional[np.ndarray] = field(default=None, repr=False)
    windows_message: Optional[np.ndarray] = field(default=None, repr=False)
    method: Optional[str] = None

    def __post_init__(self):
        if self.windows is None and self.windows_message is None:
            return
        if self.windows is None or self.windows_message is None:
            raise ValueError('Window status and messages must be supplied together.')
        self.windows = np.asarray(self.windows, dtype=bool)
        self.windows_message = np.asarray(self.windows_message, dtype=object)
        if self.windows.shape != self.windows_message.shape:
            raise ValueError('Window status and messages must have the same shape.')

    @property
    def overall(self):
        if self.windows is None:
            return None
        return bool(self.windows.size and np.all(self.windows))

    @property
    def overall_message(self):
        if self.windows is None:
            return None
        suffix = f' ({self.method})' if self.method is not None else ''
        return (f'{np.count_nonzero(self.windows)}/{self.windows.size} windows '
                f'reported success{suffix}.')

    @property
    def global_message(self):
        """Alias for overall_message; Python reserves the name global."""
        return self.overall_message

    def __bool__(self):
        # Preserve the usual `if iw.status:` check without exposing array truth.
        return bool(self.overall)

    def __repr__(self):
        return (f'VelocimetryStatus(overall={self.overall!r}, '
                f'overall_message={self.overall_message!r})')


class Iwave(object):
    def __init__(
        self,
        resolution: float,
        window_size: Tuple[int, int] = (64, 64),
        overlap: Tuple[int, int] = (0, 0),
        time_size: int = 128,
        time_overlap: int = 0,
        fps: Optional[float] = None,
        imgs: Optional[np.ndarray] = None,
        norm: Literal["time", "xy"] = "time",
        spectrum_threshold: Optional[float] = 1.0,
        smax: float = 4.0,
        window_chunk_size: Optional[int] = 50,
    ):
        """Initialize an Iwave instance.

        Parameters
        ----------
        resolution : float
            Physical resolution of the images that will be provided.
        window_size : Tuple[int, int], optional
            Size (pixels y, pixels x) of interrogation windows over which velocities are estimated.
        overlap : Tuple[int, int], optional
            Overlap in space (y, x) used to select windows from images or frames.
        time_size : int, optional
            Amount of frames in time used for one spectral analysis. Must be <= amount of frames available.
        time_overlap : int, optional
            Amount of overlap in frames, used to establish time slices.
        fps : float, optional
            Frames per second, can be set at the start, otherwise inherited from read video, or imposed with image set.
        imgs : Optional[np.ndarray], optional
            Array of images used for analysis. If not provided, defaults to None.
        norm : Literal["time", "xy"]
            Normalization to apply over subwindowed images, either over time ("time") or space ("xy").
        spectrum_threshold : float, optional
            cut-off threshold for spectrum intensities, anything below this value is set to zero. Defaults to 1.0.
        smax : float, optional
            Maximum velocity expected in the scene. Defaults to 4 m/s
        window_chunk_size : int, optional
            Number of windows to process at a time. Defaults to 50.
        """
        self.window_chunk_size = window_chunk_size
        self.resolution = resolution
        # ensures that window dimensions are even. this is to facilitate dimension reduction of the spectra.
        # this is currently working only for a downsampling rate of 2
        # TODO: generalise to any downsampling value
        self.window_size: tuple[int, int] = (
            window_size[0] if window_size[0] % 2 == 0 else window_size[0] + 1,
            window_size[1] if window_size[1] % 2 == 0 else window_size[1] + 1,
        )
        self.overlap = tuple(int(o) for o in overlap)
        self.time_size = time_size
        self.time_overlap = time_overlap
        self.norm = norm
        self.spectrum_threshold = spectrum_threshold
        self.smax = smax
        self.fps = fps
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = None
        self.win_x = None
        self.win_y = None
        self.vy = None  # y velocity component (m/s)
        self.vx = None  # x velocity component (m/s)
        self.d = None  # water depth (m)
        self.depth_estimated = None  # set after fitting: True only when depth was reconstructed
        self.cost = None  # cost function value (float)
        self.quality = None  # quality parameter (0 < q < 1), where 1 is highest quality and 0 is lowest quality
        self.status = VelocimetryStatus()
        self.method = None
        self.method_options = None
        self.sampling_metadata = None  # per-window audit records; None entries for DE
        self.spectral_uncertainty = None
        self.mode_ambiguity = None
        self.spectral_uncertainty_options = None
        self.mode_ambiguity_options = None
        self.calculate_spectral_uncertainty = False
        self.calculate_mode_ambiguity = False
                
    # Compatibility views of the structured status; no duplicate arrays.
    @property
    def status_windows(self):
        return self.status.windows

    @property
    def message_windows(self):
        return self.status.windows_message

    @property
    def message(self):
        return self.status.overall_message

    def __repr__(self):
        if self.imgs is not None:
            no_imgs = len(self.imgs)
        else:
            no_imgs = None

        return repr_template(
            self.resolution,
            self.window_size,
            self.overlap,
            self.time_size,
            self.time_overlap,
            no_imgs,
            self.fps
        )

    @property
    def imgs(self):
        """Return image set."""
        return self._imgs

    @imgs.setter
    def imgs(self, images):
        """Set images and derived properties subwindows, x and y axes, wave number axes and derived spectra."""
        if images is not None:
            if images.ndim != 3:
                raise ValueError(f"Provided image array must have 3 dimensions. Provided dimensions are {images.ndim}: {images.shape}")
        self._imgs = images
        if images is not None:
            # TODO: check if image set is large enough for the given dimension of subwindowing and time windowing
            # subwindow images and get axes. This always necessary, so in-scope methods only.
            # self._get_subwindow(images)
            # self._get_subwindow(da.from_array(images, chunks=(len(images), -1, -1)))
            self._get_win_xy(images[0])  # set sampling indexes per interrogation window
            self._get_x_y_axes(images)  # set envisaged axes
            # set the wave numbers from image fps
            self._get_wave_numbers()
            self._windows = LazyWindowArray(
                self.imgs,
                sliding_window_func=window.multi_sliding_window_array,
                win_x=self.win_x,
                win_y=self.win_y,
                norm=self.norm
            )
            self._spectrum = LazySpectrumArray(
                windows=self.windows,
                time_size=self.time_size,
                time_overlap=self.time_overlap,
                kt=self.kt,
                kx=self.kx,
                ky=self.ky,
                smax=self.smax,
                threshold=self.spectrum_threshold
            )

    @property
    def spectrum(self):
        """Return images represented in subwindows."""
        if not hasattr(self, "_spectrum"):
            if self.imgs is None:
                return None
            else:
                self._spectrum = LazySpectrumArray(
                    windows=self.windows,
                    time_size=self.time_size,
                    time_overlap=self.time_overlap,
                    kt=self.kt,
                    kx=self.kx,
                    ky=self.ky,
                    smax=self.smax,
                    threshold=self.spectrum_threshold
                )
        return self._spectrum

    @spectrum.setter
    def spectrum(self, _spectrum):
        """Set images represented in subwindows directly, only used for testing purposes."""
        self._spectrum = _spectrum

    @property
    def windows(self):
        """Return images represented in subwindows."""
        if not hasattr(self, "_windows"):
            if self.imgs is None:
                return None
            else:
                self._windows = LazyWindowArray(
                    self.imgs,
                    sliding_window_func=window.multi_sliding_window_array,
                    win_x=self.win_x,
                    win_y=self.win_y,
                    norm=self.norm
                )
        return self._windows

    @property
    def x(self):
        """Return x-axis of velocimetry field."""
        return self._x

    @x.setter
    def x(self, _x):
        self._x = _x

    @property
    def y(self):
        """Return y-axis of velocimetry field."""
        return self._y

    @y.setter
    def y(self, _y):
        self._y = _y

    @property
    def win_x(self):
        return self._win_x

    @win_x.setter
    def win_x(self, win_x):
        self._win_x = win_x

    @property
    def win_y(self):
        return self._win_y

    @win_y.setter
    def win_y(self, win_y):
        self._win_y = win_y

    @property
    def spectrum_dims(self) -> tuple[int, int, int]:
        """Return expected dimensions of the spectrum derived from image windows."""
        return (self.time_size, *self.window_size)

    def _get_win_xy(self, img_sample):
        win_x, win_y = window.sliding_window_idx(
            img_sample,
            window_size=self.window_size,
            overlap=self.overlap,
        )
        self.win_x = win_x
        self.win_y = win_y

    def _get_wave_numbers(self):
        """Prepare and set wave number axes."""
        self.kt, self.ky, self.kx = spectral.wave_numbers(
            self.spectrum_dims,
            self.resolution, self.fps
        )

    def _get_x_y_axes(self, images: np.ndarray):
        """Prepare and set x and y axes of velocity grid."""
        x, y = window.get_rect_coordinates(
            dim_sizes=images.shape[-2:],
            window_sizes=self.window_size,
            overlap=self.overlap,
        )
        self.x = x
        self.y = y

    def plot_spectrum(
        self,
        window_idx: int,
        dim: Literal["x", "y", "time"],
        slice: Optional[int] = None,
        log: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs
    ):
        """Plot 2D slice of spectrum of selected subwindow.

        Parameters
        ----------
        window_idx : int
            Index of the spectrum window to plot.
        dim : {"x", "y", "time"}
            Dimension along which to plot the spectrum.
        slice : int, optional
            Index of the slice to plot in the specified dimension. If not provided, the middle index is used.
        log : bool, optional
            If True (default), spectrum is plotted on log scale.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. New axes will be generated if not provided.
        kwargs
            Additional keyword arguments to pass to the plotting function pcolormesh.
            See :py:func:`matplotlib.pyplot.pcolormesh` for options.
        """
        spectrum_sel = self.spectrum[window_idx]
        p = io.plot_spectrum(spectrum_sel, self.kt, self.ky, self.kx, dim, slice, ax=ax, log=log, **kwargs)
        return p
    
    def plot_spectrum_fitted(
        self,
        window_idx: int,
        dim: Literal["x", "y", "time"],
        slice: Optional[int] = None,
        log: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs
        ):
        """Plot 2D slice of spectrum of selected subwindow.

        Parameters
        ----------
        window_idx : int
            Index of the spectrum window to plot.
        dim: {"x", "y", "time"}
            Dimension along which to plot the spectrum.
        slice : int, optional
            Index of the slice to plot in the specified dimension. If not provided, the middle index is used.
        log : bool, optional
            If True (default), spectrum is plotted on log scale.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. New axes will be generated if not provided.
        kwargs
            Additional keyword arguments to pass to the plotting function pcolormesh.
            See :py:func:`matplotlib.pyplot.pcolormesh` for options.
        """
        spectrum_sel = self.spectrum[window_idx]
        kt_waves_theory, kt_advected_theory = dispersion.dispersion(
            self.ky,
            self.kx,
            (self.vy.flatten()[window_idx], self.vx.flatten()[window_idx]),
            depth=1,
            vel_indx=0.85
        )
        p = io.plot_spectrum_fitted(
            spectrum_sel,
            kt_waves_theory,
            kt_advected_theory,
            self.kt,
            self.ky,
            self.kx,
            dim,
            slice,
            ax=ax,
            log=log,
            **kwargs
        )
        return p

    def read_imgs(self, path: str, fps: float, wildcard: str = None):
        """Read frames stored as images on disk from path and wildcard.

        Parameters
        ----------
        path : str
            The directory path where the image frames are located.
        fps : float
           frames per second (must be explicitly set if it cannot be read from the video)
        wildcard : str, optional
            The pattern to match filenames. Defaults to None, meaning all files in the directory will be read.
        """
        self.fps = fps
        self.imgs = io.get_imgs(path=path, wildcard=wildcard)

    def read_video(self, file: str, start_frame: int = 0, end_frame: int = 4, stride: int = 1, fps: Optional[float] = None):
        """Read video from start to end frame.

        Parameters
        ----------
        file : str
            Path to the video file.
        start_frame : int, optional
            The starting frame number from which to begin reading the video.
        end_frame : int, optional
            The ending frame number until which to read the video.
        fps : float, optional
            frames per second (must be explicitly set if it cannot be read from the video)
        stride : int, optional
            lower the sampling rate by this factor. Default 1.
        """
        # set the FPS from the video metadata
        cap = cv2.VideoCapture(file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is not None:
            self.fps = fps
        if self.fps is None:
            raise ValueError("FPS could not be read from video metadata. Please provide fps explicitly with `fps=<value>`.")

        cap.release()
        del cap
        # Retrieve images
        self.imgs = io.get_video(fn=file, start_frame=start_frame, end_frame=end_frame, stride=stride)



    def save_frames(self, dst: str):
        raise NotImplementedError

    def save_windows(self):
        raise NotImplementedError


    def velocimetry(
        self,
        alpha: float = 0.85,
        depth: float = 1.,  # If depth = 0, then the water depth is estimated.
        pass_downsampling: Optional[List[int]] = None,
        dmin: Optional[float] = 0.01,
        dmax: Optional[float] = 3.0,
        gravity_waves_switch: Optional[bool] = True,
        turbulence_switch: Optional[bool] = True,
        penalty_weight: Optional[float] = None,
        method: Literal["DE", "HNM"] = "DE",
        de_options: Optional[dict] = None,
        hnm_options: Optional[dict] = None,
        progress: bool = True,
        gauss_width: Optional[float] = None,
        calculate_spectral_uncertainty: bool = False,
        calculate_mode_ambiguity: bool = False,
        spectral_uncertainty_options: Optional[dict] = None,
        mode_ambiguity_options: Optional[dict] = None,
        spectral_contexts = None,
        **opt_kwargs
    ):
        """
        Estimate and set the velocity components u and v on the instance from the subwindowed spectra.

        Differential evolution remains the default, with its existing effective
        defaults. Select method='HNM' for Hough voting followed by
        default-on radial-trust-region Nelder-Mead. hnm_options['refine']=False
        disables refinement for fixed-depth fits. With depth=0, NM jointly fits
        velocity and log(depth). HNM also reports the existing NSP-based quality.

        Parameters
        ----------
        alpha : float, optional
            depth-average to surface velocity ratio [-], default 0.85
        depth : float, optional
            depth of the water column [m], default 1. If set to 0. it will be estimated.
        pass_downsampling : List[int], optional
            List of downsampling factors to use during the passes of the multi-steps optimisation. Defaults to [2, 1].
        dmin : float, optional
            Minimum depth expected in the scene. Defaults to 0.01 m
        dmax : float, optional
            Maximum depth expected in the scene. Defaults to 3 m
        gravity_waves_switch: bool, optional
            If True, gravity waves are modelled. If False, gravity waves are NOT modelled. Default True. 
            Setting gravity_waves_swtich = False may improve performance if floating tracers dominate the scene and waves are minimal.
        turbulence_switch: bool=True
            If True, turbulence-generated patterns and/or floating particles are modelled. If False, 
            turbulence-generated patterns and/or floating particles are NOT modelled. Default True.
            Setting turbulence_switch = False may improve performance if water waves dominate the scene, or if tracers
            dynamics are not representative of the actual flow velocity (e.g., due to air resistance, surface tension, etc.)
        penalty_weight : float, optional
            Parameter to reduce the risk of outliers by penalising solutions with high velocity modulus.
            Inactive if set to 0. Defaults to 1. 
            Outliers can be frequent if smax > 2 * flow velocity. Increase penalty_weight only if reducing smax 
            is not possible, since setting penalty_weight > 0 may introduce a bias.
        
        method : {'DE', 'HNM'}
            Backend selection; default is DE.
        de_options : dict, optional
            DE-only gauss_width, penalty_weight, pass_downsampling and nested
            scipy_options. Nested overrides merge with the existing defaults.
        hnm_options : dict, optional
            HNM settings: sampling='fixed'/'adaptive', filling_ratio=1.0,
            refine=True, relative_radius=0.5, min_radius=None.
            Fixed-depth n_candidates=1; variable-depth n_candidates=12.
            Hough controls: velocity_step=0.01 m/s,
            vote_sigma=0.015 m/s, k_min_factor=1.0, peak_height=0.25,
            peak_prominence=0.30, scale_percentile=99.5, min_ridges=12,
            and max_accumulator_mb=None.
            All detector, sampling and NM settings share this flat dictionary.
            initial_step_factor=2 sets the initial NM step to factor*velocity_step.
            None min_radius uses twice the actual maximum Hough grid spacing.
            All NM restarts share a disk centred on the primary Hough estimate.
            depth=0 enables joint velocity/log-depth fitting within dmin/dmax.
            depth_initial=None uses their geometric midpoint as Hough's reference
            and NM's depth start; a positive in-bounds value overrides it.
            log_depth_step=0.1 sets the fourth simplex vertex in natural-log units.
            xatol applies to velocity coordinates and log(depth).
            When estimating depth, n_candidates defaults to 12 and maxfev to 640;
            fixed-depth defaults remain 1 and 160. Explicit overrides win for
            each key independently. maxfev is the budget per candidate.
            Depth estimation requires refine=True and gravity_waves_switch=True.
            No HNM penalty is implemented.
        progress : bool, optional
            Enable the parent progress bar (default True). False is quiet.
        gauss_width : float, optional
            Legacy DE Gaussian width; default 1.0. penalty_weight,
            pass_downsampling and opt_kwargs also remain supported for DE.
            Do not mix legacy DE settings with de_options or use them on HNM.

        calculate_spectral_uncertainty : bool, default False
            Noise-scaled local sensitivity for velocity and estimated depth,
            using full normalized power before thresholding and the number of
            temporal averages. Bounds: spectral_uncertainty.vx_bounds[iy,ix],
            vy_bounds and depth_bounds. Full records: .details[iy,ix].
        calculate_mode_ambiguity : bool, default False
            Compare distinct converged interior endpoints from existing HNM
            starts using shared-data spectral noise. No additional searches.
            Results: mode_ambiguity.ambiguous_solution[iy,ix] and
            mode_ambiguity.additional_estimates[iy,ix]. Single starts, DE and
            invalid selected fits have unknown ambiguity (None), not evidence
            of uniqueness. Assessment reasons stay under .details[iy,ix].
        spectral_uncertainty_options, mode_ambiguity_options : dict, optional
            Tuning only; neither dictionary activates work. Both default to
            confidence=.95, variance_estimator='plugin', n_effective=None.
            Mode merging defaults to ten NM xatol units in each coordinate;
            merge_tolerances can override these numerical duplicate tolerances.
        spectral_contexts : sequence of dictionaries, optional
            Required for cached/array spectra when either diagnostic is on.
            Each contains complete pre-threshold normalized_spectrum and either
            n_effective or n_averages/time_size/time_overlap. Normal lazy video
            spectra supply this automatically without a second FFT pass.

        Notes
        -----
        The two new diagnostics are independent and never change fit parameters.
        The former fixed-loss uncertainty API and attributes have been removed.
        status.windows and status.windows_message contain per-window outcomes.
        status.overall is true only if every window succeeds; overall_message
        gives the count. bool(status) is equivalent to bool(status.overall).
        status_windows, message_windows and message remain read-only aliases.
        method_options records resolved settings. sampling_metadata contains
        HNM selected pairs and fitting diagnostics, or None on DE unless the
        diagnostic is enabled. Native uncertainty objects expose vx_bounds,
        vy_bounds, depth_bounds, valid and ambiguous_solution directly; detailed
        records are at .details[row, column]. Mode alternatives are available at
        mode_ambiguity.additional_estimates[row, column]. No user-side wrapper
        is required. Bounds have shape (rows, columns, 2); missing endpoints are
        NaN, and unassessed ambiguity is None, not False. Disabled diagnostics
        remain None. Spectral intervals remain conditional local
        noise sensitivity. Mode flags apply only to the endpoints explored;
        their thresholds are not selection-adjusted confidence tests.

        See also
        --------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

        """
        if self.spectrum is None:
            raise AttributeError(
                "No images available. Please set images first `iw.imgs = imgs`, where `imgs` is a numpy array "
                "of images, or read a video file using `iw.read_video(file)`")
        optimise.reject_removed_uncertainty_options(opt_kwargs)
        # Preserve old DE calls; keep method-specific settings separate.
        legacy = dict(opt_kwargs)
        for key, value in (("penalty_weight", penalty_weight),
                           ("pass_downsampling", pass_downsampling),
                           ("gauss_width", gauss_width)):
            if value is not None:
                legacy[key] = value
        method = optimise.normalise_method(method)
        from iwave.spectral import resolve_spectral_request
        from iwave.optimise_hnm import resolve_mode_request
        spectral_options = resolve_spectral_request(calculate_spectral_uncertainty, spectral_uncertainty_options)
        mode_options = resolve_mode_request(calculate_mode_ambiguity, mode_ambiguity_options)
        # set search bounds to -/+ maximum velocity for both directions, and depth if applicable
        if depth == 0:  # If depth = 0, then the water depth is estimated.
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (dmin, dmax)]
        else:
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (depth, depth)]
        
        # Resolve defaults only after the actual depth interval is known.
        de, hnm = optimise.resolve_method_options(
            method, de_options=de_options, hnm_options=hnm_options,
            legacy=legacy, scipy_defaults=OPTIM_KWARGS_SADE, bnds=bounds)

        # Report this once in the parent, not once per worker/window. The
        # local sensitivity can still be useful, but one start cannot compare
        # different solutions. The same statement is stored on mode_ambiguity.
        mode_message = None
        if mode_options is not None and method == 'HNM' and hnm['n_candidates'] == 1:
            import warnings
            mode_message = 'Ambiguity cannot be assessed if n_candidates = 1.'
            warnings.warn(mode_message, UserWarning, stacklevel=2)

        # bounds = [(-self.smax, self.smax), (-self.smax, self.smax)]
        # if depth == 0:
        #     # also estimate water depth, set bounds
        #     bounds.append((dmin, dmax))
        # Create a list of bounds for each window.
        bounds_list = [tuple(bounds) for _ in range(len(self.spectrum))]
        
        # Optimize velocities and depth for each window in the spectrum
        output, cost, quality, status, messages, metadata = optimise.optimise_velocity(
            measured_spectra=self.spectrum,
            bnds_list=bounds_list,
            vel_indx=alpha,
            window_dims=self.spectrum_dims,
            res=self.resolution,
            fps=self.fps,
            gravity_waves_switch=gravity_waves_switch,
            turbulence_switch=turbulence_switch,
            chunk_size=self.window_chunk_size,
            desc="Optimizing windows",
            method=method,
            de_options=de,
            hnm_options=hnm,
            progress=progress,
            return_diagnostics=True,
            return_metadata=True,
            calculate_spectral_uncertainty=bool(calculate_spectral_uncertainty),
            calculate_mode_ambiguity=bool(calculate_mode_ambiguity),
            spectral_uncertainty_options=spectral_options,
            mode_ambiguity_options=mode_options,
            spectral_contexts=spectral_contexts,
        )
        self.vy = output[:, 0].reshape(len(self.y), len(self.x))
        self.vx = output[:, 1].reshape(len(self.y), len(self.x))
        self.d = output[:, 2].reshape(len(self.y), len(self.x))
        self.cost = cost.reshape(len(self.y), len(self.x))
        self.quality = quality.reshape(len(self.y), len(self.x))
        self.status = VelocimetryStatus(
            windows=status.reshape(len(self.y), len(self.x)),
            windows_message=messages.reshape(len(self.y), len(self.x)),
            method=method,
        )
        self.sampling_metadata = metadata.reshape(len(self.y), len(self.x))
        self.method = method
        self.method_options = de if de is not None else hnm
        self.calculate_spectral_uncertainty = bool(calculate_spectral_uncertainty)
        self.calculate_mode_ambiguity = bool(calculate_mode_ambiguity)
        self.spectral_uncertainty_options = spectral_options
        self.mode_ambiguity_options = mode_options
        # Record what was actually fitted, not merely whether depth=0 was
        # requested. An equal lower/upper depth bound still means fixed depth.
        self.depth_estimated = bool(bounds[2][0] < bounds[2][1])
        shape = self.vx.shape
        self.spectral_uncertainty = SpectralUncertainty(shape) if spectral_options is not None else None
        self.mode_ambiguity = ModeAmbiguity(shape) if mode_options is not None else None
        if self.mode_ambiguity is not None:
            self.mode_ambiguity.message = mode_message
        # Assemble the native public results here. Workers retain their compact
        # numerical audit dictionaries under sampling_metadata and .details.
        # This adds no refits, no Hessian evaluations and no extra FFT passes.
        for index in np.ndindex(shape):
            metadata = self.sampling_metadata[index]
            ambiguous = None
            if self.mode_ambiguity is not None:
                self.mode_ambiguity._set_window(index, metadata['mode_ambiguity'])
                ambiguous = self.mode_ambiguity.ambiguous_solution[index]
            if self.spectral_uncertainty is not None:
                self.spectral_uncertainty._set_window(index, metadata['spectral_uncertainty'], ambiguous)

    def plot_transect(self, *, row=None, column=None, include_depth=None,
                      show_uncertainty=True, show_modes=True, distance_units='pixels',
                      axes=None, label=None, color=None):
        """Plot one row or column, automatically including reconstructed depth and quality.

        Examples
        --------
        iw.plot_transect(column=0)  # all rows of window column zero
        iw.plot_transect(row=2)     # all columns of window row two
        fig, axes = iw.plot_transect(column=0, include_depth=False)

        Supply exactly one zero-based WINDOW row/column index. The horizontal
        axis is x for a row, y for a column, in pixels by default. Use
        distance_units='m' to multiply image-window centres by resolution. Bounds are accepted local spectral
        sensitivity, not calibrated total uncertainty. Other observed modes
        appear separately, without invented bounds. Unassessed ambiguity has
        no special marker; the result flag remains None, not False.

        Depth appears by default only when the latest fit reconstructed it.
        It does not depend on enabling the uncertainty diagnostic. False hides
        depth; True requests it but rejects fixed-depth fits. The method returns
        (figure, axes), ending with a quality panel, without calling show().
        Reuse axes and set label/color to
        overlay strategies with the same panel count. No fitting is performed.

        See iwave.io.plot_transect for full parameter/marker explanations.
        """
        from iwave.io import plot_transect
        return plot_transect(self, row=row, column=column, include_depth=include_depth,
                             show_uncertainty=show_uncertainty, show_modes=show_modes,
                             distance_units=distance_units, axes=axes, label=label, color=color)

    def plot_velocimetry(self, ax: Optional[matplotlib.axes.Axes] = None, *,
                         validate_vectors=True, distance_units='pixels',
                         color='C0', ambiguous_color='tab:red', **kwargs):
        """Plot velocity arrows; hide failed fits and colour observed ambiguity.

        validate_vectors=True (default) skips failed fits. False includes their
        finite vectors. Nonfinite vectors are always skipped. Pixel positions
        are the default; distance_units='m' converts positions using resolution.
        Arrow components remain velocities in m/s in BOTH coordinate systems;
        this does not turn them into pixel velocities. Matplotlib autoscaling
        controls displayed arrow length unless you specify scale explicitly.

        Known ambiguous solutions use ambiguous_color (red by default); unknown
        ambiguity is not coloured as if a rival had been detected. color is the
        single base Matplotlib colour for all other vectors. ax/other quiver
        keywords keep their usual meaning. Returns one Matplotlib Quiver object.
        No fitting or uncertainty computation is performed.
        """
        return io.plot_velocimetry(self, ax=ax, validate_vectors=validate_vectors,
                                   distance_units=distance_units, color=color,
                                   ambiguous_color=ambiguous_color, **kwargs)
