"""IWaVE main api."""

import cv2
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
from iwave import window, spectral, io, optimise, dispersion, LazySpectrumArray, LazyWindowArray


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
        self.cost = None  # cost function value (float)
        self.quality = None  # quality parameter (0 < q < 1), where 1 is highest quality and 0 is lowest quality
        self.status = VelocimetryStatus()
        self.method = None
        self.method_options = None
        self.sampling_metadata = None  # per-window audit records; None entries for DE
                
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

        Notes
        -----
        status.windows and status.windows_message contain per-window outcomes.
        status.overall is true only if every window succeeds; overall_message
        gives the count. bool(status) is equivalent to bool(status.overall).
        status_windows, message_windows and message remain read-only aliases.
        method_options records resolved settings. sampling_metadata contains
        HNM selected pairs and fitting diagnostics, or None on DE. These are
        numerical diagnostics, not velocity uncertainty estimates.

        See also
        --------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

        """
        if self.spectrum is None:
            raise AttributeError(
                "No images available. Please set images first `iw.imgs = imgs`, where `imgs` is a numpy array "
                "of images, or read a video file using `iw.read_video(file)`")
        # Preserve old DE calls; keep method-specific settings separate.
        legacy = dict(opt_kwargs)
        for key, value in (("penalty_weight", penalty_weight),
                           ("pass_downsampling", pass_downsampling),
                           ("gauss_width", gauss_width)):
            if value is not None:
                legacy[key] = value
        method = optimise.normalise_method(method)
        # set search bounds to -/+ maximum velocity for both directions, and depth if applicable
        if depth == 0:  # If depth = 0, then the water depth is estimated.
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (dmin, dmax)]
        else:
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (depth, depth)]
        
        # Resolve defaults only after the actual depth interval is known.
        de, hnm = optimise.resolve_method_options(
            method, de_options=de_options, hnm_options=hnm_options,
            legacy=legacy, scipy_defaults=OPTIM_KWARGS_SADE, bnds=bounds)

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
        
    
    def plot_velocimetry(self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs):
        """Plot the estimated velocity components u and v on the axes instance."""
        if ax is None:
            ax = plt.axes()
        p = ax.quiver(self.x, self.y, self.vx, self.vy, **kwargs)
        return p
