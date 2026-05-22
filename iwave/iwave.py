"""IWaVE main api."""

import cv2
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Tuple, Literal
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
        norm: Optional[Literal["time", "xy"]] = "time",
        spectrum_threshold: Optional[float] = 1.0,
        smax: Optional[float] = 5.0,
        dmin: Optional[float] = 0.01,
        dmax: Optional[float] = 10.0,
        alphamin: Optional[float] = 0.5,
        alphamax: Optional[float] = 1.0,
        gravity_waves_switch: Optional[bool]=True,
        turbulence_switch: Optional[bool]=True,
        window_chunk_size: Optional[int] = 50,
        first_pass_downsample: Optional[int]=2,
        penalty_weight: Optional[float]=1,
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
        dmin : float, optional
            Minimum depth expected in the scene. Defaults to 0.01 m
        dmax : float, optional
            Maximum depth expected in the scene. Defaults to 3 m
        alphamin : float, optional
            Minimum alpha expected in the scene. Defaults to 0.5.
        alphamax : float, optional
            Maximum alpha expected in the scene. Defaults to 1.0.
        gravity_waves_switch: bool, optional
            If True, gravity waves are modelled. If False, gravity waves are NOT modelled. Default True. 
            Setting gravity_waves_swtich = False may improve performance if floating tracers dominate the scene and waves are minimal.
        turbulence_switch: bool=True
            If True, turbulence-generated patterns and/or floating particles are modelled. If False, 
            turbulence-generated patterns and/or floating particles are NOT modelled. Default True.
            Setting turbulence_switch = False may improve performance if water waves dominate the scene, or if tracers
            dynamics are not representative of the actual flow velocity (e.g., due to air resistance, surface tension, etc.)
        window_chunk_size : int, optional
            Number of windows to process at a time. Defaults to 50.
        first_pass_downsample : int, optional
            Downsampling factor to use during the first pass of the two-steps optimisation. Defaults to 2.
        penalty_weight : float, optional
            Parameter to reduce the risk of outliers by penalising solutions with high velocity modulus.
            Inactive if set to 0. Defaults to 1. 
            Outliers can be frequent if smax > 2 * flow velocity. Increase penalty_weight only if reducing smax 
            is not possible, since setting penalty_weight > 0 may introduce a bias.
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
        self.dmin = dmin
        self.dmax = dmax
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.fps = fps
        self.gravity_waves_switch = gravity_waves_switch
        self.turbulence_switch = turbulence_switch
        self.first_pass_downsample = first_pass_downsample
        self.penalty_weight = penalty_weight
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = None
        self.win_x = None
        self.win_y = None
        self.vy = None  # y velocity component (m/s)
        self.vx = None  # x velocity component (m/s)
        self.d = None  # water depth (m)
        self.alpha = None  # depth-average to surface velocity ratio (-)
        self.cost = None  # cost function value (float)
        self.quality = None  # quality parameter (0 < q < 1), where 1 is highest quality and 0 is lowest quality
        self.status = None  # Boolean flag indicating if the optimizer exited successfully
        self.message = None  # termination message returned by the optimiser
                
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
        vel_indx = self.alpha.flatten()[window_idx] if self.alpha is not None else 0.85
        kt_waves_theory, kt_advected_theory = dispersion.dispersion(
            self.ky,
            self.kx,
            (self.vy.flatten()[window_idx], self.vx.flatten()[window_idx]),
            depth=10,
            vel_indx=vel_indx
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

    def read_video(self, file: str, start_frame: int = 0, end_frame: int = 4, stride: int = 1):
        """Read video from start to end frame.

        Parameters
        ----------
        file : str
            Path to the video file.
        start_frame : int, optional
            The starting frame number from which to begin reading the video.
        end_frame : int, optional
            The ending frame number until which to read the video.
        stride : int, optional
            lower the sampling rate by this factor. Default 1.
        """
        # set the FPS from the video metadata
        cap = cv2.VideoCapture(file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)

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
        depth: float = 10.,
        twosteps: bool = True,
        **opt_kwargs
    ):
        """
        Estimate and set the velocity components u and v on the instance from the subwindowed spectra.

        The optimisation is performed using the differential evolution algorithm of `scipy.optimize`. You can pass
        arguments of this function to the optimiser. Default arguments are set as a starting point.

        If you set `twosteps=True`, the optimisation is performed twice per chunk, with a reduced spectrum in the first step,
        and a refined step with full spectrum.

        Parameters
        ----------
        alpha : float, optional
            depth-average to surface velocity ratio [-], default 0.85. Set to 0 to estimate alpha.
        depth : float, optional
            Depth of the water column [m]. Set to 0 to estimate depth, or provide a positive value to use fixed depth.
            Default 10 (fixed).
        twosteps : bool, optional
            If True (default), performs two-step optimization with downsampling in step 1 and refinement in step 2.
        **opt_kwargs
            Additional keyword arguments passed to scipy.optimize.differential_evolution.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

        """
        if self.spectrum is None:
            raise AttributeError(
                "No images available. Please set images first `iw.imgs = imgs`, where `imgs` is a numpy array "
                "of images, or read a video file using `iw.read_video(file)`")
        # ensure defaults are set if nothing is provided
        if not opt_kwargs:
            opt_kwargs = OPTIM_KWARGS_SADE
        
        # Determine if depth and/or alpha should be estimated
        estimate_depth = (depth == 0)
        estimate_alpha = (alpha == 0)
        
        if estimate_depth and estimate_alpha:
            print("Warning: estimating both depth and alpha simultaneously may lead to wrong parameter estimates.")
        
        if (twosteps == False) & (self.penalty_weight != 0):
                print(f"Velocity and especially depth estimations with the 1 step approach are biased when penalty_weight is not zero. It is recommended to use the two steps approach and/or set first_pass_downsample=1, or alternatively to set penalty_weight = 0 and decreasing smax to reduce the number of outliers.")
        
        if twosteps:
            if self.first_pass_downsample > 0:
                print(f"Optimization in two steps: step 1 will use spectrum downsampling factor {self.first_pass_downsample}, followed by step 2 with full spectral resolution.")
            else:
                print("Optimization in two steps with full spectral resolution.")
        else:
            print("Optimization in one step with full spectral resolution.")

        if estimate_alpha:
            print(f"Alpha estimation is active. Search bounds will be {(self.alphamin, self.alphamax)}.")

        bounds = (
            (-self.smax, self.smax),   # vy
            (-self.smax, self.smax),   # vx
            (self.dmin, self.dmax),    # depth
            (self.alphamin, self.alphamax),  # vel_indx
        )

        bounds_list = [bounds for _ in range(len(self.spectrum))]
        
        # Determine the two-step downsample factor
        two_step_downsample = self.first_pass_downsample if twosteps else 0
        
        output, cost, quality = optimise.optimise_velocity(
            measured_spectra=self.spectrum,
            bnds_list=bounds_list,
            vel_indx=alpha if not estimate_alpha else 0.85,
            window_dims=self.spectrum_dims,
            res=self.resolution,
            fps=self.fps,
            penalty_weight=self.penalty_weight,
            gravity_waves_switch=self.gravity_waves_switch,
            turbulence_switch=self.turbulence_switch,
            chunk_size=self.window_chunk_size,
            downsample=1,
            gauss_width=1,  # TODO: figure out defaults
            depth=depth if not estimate_depth else 10.0,  # Pass actual depth value if fixed, else default
            estimate_depth=estimate_depth,
            estimate_vel_indx=estimate_alpha,
            two_step_downsample=two_step_downsample,
            desc="Optimizing windows",
            **opt_kwargs
        )
        self.vy = output[:, 0].reshape(len(self.y), len(self.x))
        self.vx = output[:, 1].reshape(len(self.y), len(self.x))
        self.d = output[:, 2].reshape(len(self.y), len(self.x))
        self.alpha = output[:, 3].reshape(len(self.y), len(self.x))
        self.cost = cost.reshape(len(self.y), len(self.x))
        self.quality = quality.reshape(len(self.y), len(self.x))
        self.status = True
        self.message = "Optimization terminated successfully."
        
    
    def plot_velocimetry(self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs):
        """Plot the estimated velocity components u and v on the axes instance."""
        if ax is None:
            ax = plt.axes()
        p = ax.quiver(self.x, self.y, self.vx, self.vy, **kwargs)
        return p
