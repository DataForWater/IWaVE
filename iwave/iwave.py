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

OPTIM_KWARGS = {
    "popsize": 10,
    "maxiter": 10000,
    "workers": 1
}

class Iwave(object):
    def __init__(
        self,
        resolution: float,
        window_size: Tuple[int, int] = (128, 128),
        overlap: Tuple[int, int] = (0, 0),
        time_size: int = 128,
        time_overlap: int = 0,
        fps: Optional[float] = None,
        imgs: Optional[np.ndarray] = None,
        norm: Optional[Literal["time", "xy"]] = "time",
        smax: Optional[float] = 4.0
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
        fps : float, optional
            Frames per second, can be set at the start, otherwise inherited from read video, or imposed with image set.
        time_overlap : int, optional
            Amount of overlap in frames, used to establish time slices.
        imgs : Optional[np.ndarray], optional
            Array of images used for analysis. If not provided, defaults to None.
        norm : Literal["time", "xy"]
            Normalization to apply over subwindowed images, either over time ("time") or space ("xy").
        smax : float, optional
            Maximum velocity expected in the scene. Defaults to 4 m/s
        """
        self.resolution = resolution
        self.window_size = window_size
        self.overlap = overlap
        self.time_size = time_size
        self.time_overlap = time_overlap
        self.norm = norm
        self.smax = smax
        self.fps = fps
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = None
        self.u = None
        self.v = None

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
        kt_waves_theory, kt_advected_theory = dispersion.dispersion(
            self.ky,
            self.kx,
            (self.v.flatten()[window_idx], self.u.flatten()[window_idx]),
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
        alpha=0.85,
        depth=1.,
    ):
        # set search bounds to -/+ maximym velocity for both directions
        bounds = [(-self.smax, self.smax), (-self.smax, self.smax)]
        # TODO: remove img_size from needed inputs. This can be derived from the window size and time_size
        img_size = (self.time_size, self.spectrum.shape[-2], self.spectrum.shape[-1])
        optimal = optimise.optimise_velocity(
            self.spectrum,
            bounds,
            depth,
            alpha,
            img_size,
            self.resolution,
            self.fps,
            gauss_width=1,  # TODO: figure out defaults
            gravity_waves_switch=True,  # TODO: figure out defaults
            turbulence_switch=True,  # TODO: figure out defaults
            **OPTIM_KWARGS
        )
        self.u = optimal[:, 1].reshape(len(self.y), len(self.x))
        self.v = optimal[:, 0].reshape(len(self.y), len(self.x))

    def plot_velocimetry(self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs):
        if ax is None:
            ax = plt.axes()
        p = plt.quiver(self.x, self.y, self.u, self.v, **kwargs)
        return p


if __name__ == '__main__':
    ############################################################################
    frames_path = '/home/sp/pCloudDrive/Docs/d4w/iwave/transformed'
    video_path = '/home/sp/pCloudDrive/Docs/d4w/iwave/vid/Fersina_20230630.avi'
    ############################################################################
    
    # Initialize
    iwave = Iwave()

    # Use video
    frames = iwave.frames_from_video(video_path, start_frame=0, end_frame=4)

    # or use frames
    #frames = iwave.read_frames(frames_path)

    # Normalize frames
    frames = iwave.img_normalization(frames)

    # Subwidnows
    subwins = iwave.subwindows(frames, [64, 64], [32, 32])

    # 3D Spectrum 
    iwave.get_spectra(subwins, engine="numpy")
    
    print('ok')
    