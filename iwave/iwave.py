"""IWaVE main api."""

import cv2
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Tuple, Literal
from iwave import window, spectral, io, optimise, dispersion


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
        smax: Optional[float] = 4.0,
        dmin: Optional[float] = 0.01,
        dmax: Optional[float] = 3.0,
        penalty_weight: Optional[float]=1,
        gravity_waves_switch: Optional[bool]=True,
        turbulence_switch: Optional[bool]=True,
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
        smax : float, optional
            Maximum velocity expected in the scene. Defaults to 4 m/s
        dmin : float, optional
            Minimum depth expected in the scene. Defaults to 0.01 m
        dmax : float, optional
            Maximum depth expected in the scene. Defaults to 3 m
        penalty_weight : float, optional
            Parameter to reduce the risk of outliers by penalising solutions with high velocity modulus.
            Inactive if set to 0. Defaults to 1. 
            Outliers can be frequent if smax > 2 * flow velocity. Increase penalty_weight only if reducing smax 
            is not possible, since setting penalty_weight > 0 may introduce a bias.
        gravity_waves_switch: bool, optional
            If True, gravity waves are modelled. If False, gravity waves are NOT modelled. Default True. 
            Setting gravity_waves_swtich = False may improve performance if floating tracers dominate the scene and waves are minimal.
        turbulence_switch: bool=True
            If True, turbulence-generated patterns and/or floating particles are modelled. If False, 
            turbulence-generated patterns and/or floating particles are NOT modelled. Default True.
            Setting turbulence_switch = False may improve performance if water waves dominate the scene, or if tracers
            dynamics are not representative of the actual flow velocity (e.g., due to air resistance, surface tension, etc.)
        """
        self.resolution = resolution
        # ensures that window dimensions are even. this is to facilitate dimension reduction of the spectra.
        # this is currently working only for a downsampling rate of 2
        # TODO: generalise to any downsampling value
        self.window_size = tuple((dim if dim % 2 == 0 else dim + 1) for dim in window_size) 
        self.overlap = overlap
        self.time_size = time_size
        self.time_overlap = time_overlap
        self.norm = norm
        self.smax = smax
        self.dmin = dmin
        self.dmax = dmax
        self.fps = fps
        self.penalty_weight = penalty_weight
        self.gravity_waves_switch = gravity_waves_switch
        self.turbulence_switch = turbulence_switch
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = None
        self.vy = None  # y velocity component (m/s)
        self.vx = None  # x velocity component (m/s)
        self.d = None  # water depth (m)
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
            self._get_subwindow(images)
            self._get_x_y_axes(images)

    @property
    def spectrum(self):
        """Return images represented in subwindows."""
        return self._spectrum

    @spectrum.setter
    def spectrum(self, _spectrum):
        self._spectrum = _spectrum


    @property
    def windows(self):
        """Return images represented in subwindows."""
        return self._windows

    @windows.setter
    def windows(self, win):
        self._windows = win

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
    def spectrum_dims(self):
        """Return expected dimensions of the spectrum derived from image windows."""
        return (self.time_size, *self.window_size)

    def _get_subwindow(self, images: np.ndarray):
        """Create and set windows following provided parameters."""
        # get the x and y coordinates per window
        # TODO: define windows based on window size and number of windows per dimension instead of overlap
        win_x, win_y = window.sliding_window_idx(
            images[0],
            window_size=self.window_size,
            overlap=self.overlap,
        )
        # apply the coordinates on all images
        windows = window.multi_sliding_window_array(
            images,
            win_x,
            win_y,
            swap_time_dim=True
        )
        if self.norm == "xy":
            self.windows = window.normalize(windows, mode="xy")
        elif self.norm == "time":
            self.windows = window.normalize(windows, mode="time")
        else:
            self.windows = windows

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

    def get_spectra(self, threshold: float = 1.):
        """Generate and set spectra of all extracted windows."""
        spectrum = spectral.sliding_window_spectrum(
            self.windows,
            self.time_size,
            self.time_overlap,
            engine="numba"
        )
        # set the wave numbers
        self._get_wave_numbers()
        
        # preprocess
        self.spectrum = optimise.spectrum_preprocessing(
            spectrum,
            self.kt,
            self.ky,
            self.kx,
            self.smax*3,
            spectrum_threshold=threshold
        )

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

    def read_video(self, file: str, start_frame: int = 0, end_frame: int = 4):
        """
        Parameters
        ----------
        file : str
            Path to the video file.
        start_frame : int, optional
            The starting frame number from which to begin reading the video.
        end_frame : int, optional
            The ending frame number until which to read the video.
        Returns
        -------
        numpy.ndarray
            An array of grayscale images from the video between the specified frames.
        """
        # set the FPS
        cap = cv2.VideoCapture(file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()
        del cap
        # Retrieve images
        self.imgs = io.get_video(fn=file, start_frame=start_frame, end_frame=end_frame)
        # get the frame rate from the video


    def save_frames(self, dst: str):
        raise NotImplementedError

    def save_windows(self):
        raise NotImplementedError


    def velocimetry(
        self,
        alpha: float = 0.85,
        depth: float = 1.,  # If depth = 0, then the water depth is estimated.
        twosteps: bool = False,
        **opt_kwargs # If True, the calculations are initially performed on a spectrum with reduced dimensions,
                            # and subsequently refined during a second step using the whole spectrum. This will reduce 
                            # computational time for large problems, but may reduce accuracy.
    ):
        """
        Estimate and set the velocity components u and v on the instance from the subwindowed spectra.

        The optimisation is performed using the differential evolution algorithm of `scipy.optimize`. You can pass
        arguments of this function to the optimiser. Default arguments are set as a starting point.

        If you set `twosteps=True`, the optimisation is performed twice, with a reduced spectrum in the first step
        without optimizing depth, and a refined step with full spectrum and optimizing depth.

        Parameters
        ----------
        alpha : float, optional
            depth-average to surface velocity ratio [-], default 0.85
        depth : float, optional
            depth of the water column [m], default 1. If set to 0. it will be estimated.
        twosteps : bool, optional
            if set, perform the optimisation twice, with a reduced spectrum in the first step without optimizing depth
            and full spectrum in the second step with optimizing depth. Default False.

        See also
        --------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

        """
        # ensure defaults are set if nothing is provided
        if not opt_kwargs:
            opt_kwargs = OPTIM_KWARGS_SADE
        # set search bounds to -/+ maximum velocity for both directions
        if depth == 0:  # If depth = 0, then the water depth is estimated.
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (self.dmin, self.dmax)]
            if twosteps == False:
                self.penalty_weight = 0
                print(f"Depth estimation with the 1 step approach is inaccurate when penalty_weight is not zero. Now setting penalty_weight = 0. Consider reducing smax if results are incorrect, or use the two-steps approach.")
        else:
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (depth, depth)]
        # Create a list of bounds for each window. This is to enable narrowing the bounds locally during multiple passages.
        bounds_list = [bounds for _ in self.spectrum]
        
        # TODO: remove img_size from needed inputs. This can be derived from the window size and time_size
        img_size = (self.time_size, self.spectrum.shape[-2], self.spectrum.shape[-1])

        if twosteps == True:
            print(f"Step 1:")
            bounds_firststep = bounds_list
            if depth==0: # for the first step, neglect water depth effects by assuming a large depth
                for i in range(len(bounds_list)):
                    bounds_firststep[i] = [bounds[0], bounds[1], (10, 10)]
            output_step1, _, _, _, _ = optimise.optimise_velocity(
                self.spectrum,
                bounds_firststep,
                alpha,
                img_size,
                self.resolution,
                self.fps,
                self.penalty_weight,  
                self.gravity_waves_switch, 
                self.turbulence_switch, 
                downsample = 2, # for the first step, reduce the data size by 2
                gauss_width=1,  # TODO: figure out defaults
                **opt_kwargs
            )
            print(f"Step 2:")
            # re-initialise the problem using narrower bounds between 90% and 110% of the first step solution
            vy_step1 = output_step1[:, 0]
            vx_step1 = output_step1[:, 1]
            for i in range(len(bounds_list)):
                bounds_list[i] = [(vy_step1[i]-0.1*np.abs(vy_step1[i]), vy_step1[i]+0.1*np.abs(vy_step1[i])), 
                    (vx_step1[i]-0.1*np.abs(vx_step1[i]), vx_step1[i]+0.1*np.abs(vx_step1[i])), 
                        (bounds[2][0], bounds[2][1])]
            opt_kwargs["popsize"] = max(1, opt_kwargs["popsize"] // 2) # reduce the population size for the second step
            self.penalty_weight = 0 # set penalty_weight = 0 for the second step
        output, cost, quality, status, message = optimise.optimise_velocity(
            self.spectrum,
            bounds_list,
            alpha,
            img_size,
            self.resolution,
            self.fps,
            self.penalty_weight,  
            self.gravity_waves_switch, 
            self.turbulence_switch, 
            downsample = 1,
            gauss_width=1,  # TODO: figure out defaults
            **opt_kwargs
        )
        self.vy = output[:, 0].reshape(len(self.y), len(self.x))
        self.vx = output[:, 1].reshape(len(self.y), len(self.x))
        self.d = output[:, 2].reshape(len(self.y), len(self.x))
        self.cost = cost
        self.quality = quality
        self.status = status
        self.message = message
        
    
    def plot_velocimetry(self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs):
        """Plot the estimated velocity components u and v on the axes instance."""
        if ax is None:
            ax = plt.axes()
        p = ax.quiver(self.x, self.y, self.vx, self.vy, **kwargs)
        return p
