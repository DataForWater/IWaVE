"""Main Iwave interface for loading images, fitting flow, and plotting results."""

import cv2
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

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

# Default settings for the differential-evolution fitting method.
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
        norm: Literal["time", "xy"] = "time",
        spectrum_threshold: Optional[float] = 1.0,
        smax: float = 4.0,
        window_chunk_size: Optional[int] = 50,
    ):
        """Set image geometry, spectrum settings, and how many windows to process together.

        Parameters
        ----------
        resolution : float
            Image resolution in metres per pixel.
        window_size : pair of integers
            Window height and width in pixels, in (y, x) order.
        overlap : pair of integers
            Shared pixels between neighbouring windows, in (y, x) order.
        time_size : integer
            Frames in each time segment. Must not exceed the loaded frame count.
        time_overlap : integer
            Shared frames between consecutive segments; less than time_size.
        fps : float, optional
            Frame rate of the analysed sequence. Can instead be read from a video
            or supplied when loading images.
        imgs : numpy.ndarray, optional
            Images in (frame, y, x) order; otherwise load them later.
        norm : 'time', 'xy' or None
            Subtract a mean and divide by a standard deviation, either at each pixel
            over time or across each frame. None skips this adjustment. See README.md for details.
        spectrum_threshold : float
            Multiplier for the mean over frequency at each spatial wavenumber.
            Larger values discard more weak normalised power. It is not an absolute
            brightness threshold. Default 1.0.
        smax : float
            Maximum expected velocity component in m/s. Default 4.
        window_chunk_size : integer
            Maximum windows extracted and fitted together. Default 50. Smaller batches
            can reduce memory use. The FFT keeps the complete spatial-wavenumber grid;
            the source video still remains loaded.
        """
        self.window_chunk_size = window_chunk_size
        self.resolution = resolution
        # Use even window widths and heights so the reduced-size fitting pass can halve
        # them.
        # The current reduction method uses a factor of two.
        # TODO: support other reduction factors.
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
        # Set these fields before assigning images, because image assignment calculates
        # them.
        # Otherwise their newly calculated values would immediately be erased.
        self.win_x = None
        self.win_y = None
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = None
        self.vy = None  # Velocity along image y (m/s).
        self.vx = None  # Velocity along image x (m/s).
        self.d = None  # Water depth (m).
        self.cost = None  # Final fitting score; smaller values indicate closer agreement.
        # Agreement score: a perfect match gives 1. It is not limited to [0, 1] or an
        # error bar.
        self.quality = None
        # Overall completion flag; this older interface does not report each window's fit
        # status.
        self.status = None
        self.message = None  # Overall completion message from this older fitting interface.
                
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
        """Return the loaded image sequence in (frame, y, x) order."""
        return self._imgs

    @imgs.setter
    def imgs(self, images):
        """Set the images and prepare their window positions and spectrum axes.

        Actual window images and spectra are calculated only when requested.
        """
        if images is not None:
            if images.ndim != 3:
                raise ValueError(f"Provided image array must have 3 dimensions. Provided dimensions are {images.ndim}: {images.shape}")
        self._imgs = images
        if images is not None:
            # TODO: check that enough frames and pixels are available for the requested
            # windows.
            # Prepare the window positions and coordinate axes.
            # Store how to extract windows, without making every window and spectrum yet.
            # Later requests calculate only the selected windows.
            # The loaded video frames themselves still remain in memory.
            self._get_win_xy(images[0])  # Record which pixels belong to each window.
            self._get_x_y_axes(images)  # Set the image-coordinate axes for the results.
            # Set spectrum axes from the frame rate and image resolution.
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
        """Return stored spectra, or an object that calculates them when indexed."""
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
        """Supply already calculated spectra instead of deriving them from loaded images."""
        self._spectrum = _spectrum

    @property
    def windows(self):
        """Return the windows object, which extracts image windows when indexed."""
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
        """Return the x positions of window centres in image pixels."""
        return self._x

    @x.setter
    def x(self, _x):
        self._x = _x

    @property
    def y(self):
        """Return the y positions of window centres in image pixels."""
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
        """Return the frame count, height and width used to calculate one spectrum.

        The stored spectrum has fewer frequency samples than this frame count.
        """
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
        """Set frequency and spatial-wavenumber axes from the image geometry and frame rate."""
        self.kt, self.ky, self.kx = spectral.wave_numbers(
            self.spectrum_dims,
            self.resolution, self.fps
        )

    def _get_x_y_axes(self, images: np.ndarray):
        """Set the x/y image positions for the grid of fitted windows."""
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
        """Plot one slice through the measured spectrum for a selected window.

        window_idx is its number, counted from zero. dim ('x', 'y' or 'time') chooses
        the dimension held fixed; slice gives its index, or uses the middle by default.
        log=True shows power on a logarithmic colour scale. ax can reuse an existing
        Matplotlib plot area. Other keyword options go to pcolormesh.
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
        """Plot a window's measured spectrum with the older predicted-line overlay.

        window_idx, dim, slice, log, ax and plotting keywords work as in plot_spectrum.
        Important: this convenience method uses depth=1 m and alpha=0.85 for its lines,
        not the depth/alpha from the fitted result. To draw the actual fitted depth,
        use the explicit dispersion + io.plot_spectrum_fitted example in README.md.
        """
        spectrum_sel = self.spectrum[window_idx]
        # This older plotting helper uses the fixed depth and alpha written below,
        # not the values from the last fit. For another depth or alpha, follow the
        # explicit dispersion + io.plot_spectrum_fitted example in README.md.
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
        """Load image files and set their frame rate.

        path is the folder; wildcard optionally selects matching filenames.
        fps must describe the loaded image sequence, since still images do not
        supply a video frame rate.
        """
        self.fps = fps
        self.imgs = io.get_imgs(path=path, wildcard=wildcard)

    def read_video(self, file: str, start_frame: int = 0, end_frame: int = 4, stride: int = 1, fps: Optional[float] = None):
        """Load a range of video frames, with optional frame skipping.

        file is the video path; start_frame is included and end_frame is excluded.
        stride keeps every nth frame. fps defaults to the file metadata when omitted.
        It is not automatically divided by stride: supply the original fps/stride
        explicitly when analysing a skipped-frame sequence.
        """
        # Read the frame rate from the video file.
        cap = cv2.VideoCapture(file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is not None:
            self.fps = fps
        if self.fps is None:
            raise ValueError("FPS could not be read from video metadata. Please provide fps explicitly with `fps=<value>`.")

        cap.release()
        del cap
        # Load the image files.
        self.imgs = io.get_video(fn=file, start_frame=start_frame, end_frame=end_frame, stride=stride)



    def warmup(self):
        """Prepare the image-normalisation and FFT code using small test arrays.

        Numba compiles some calculations on first use. Call this optionally before
        timing to account for that setup separately; the setup time is not removed.
        This does not start fitting processes or prepare the real video.
        Supplied images, spectra and fitted results stay unchanged. Return the seconds
        spent on normalisation, FFT, and the complete warmup.
        """
        from time import perf_counter

        started = perf_counter()
        # Numba prepares compiled code for each array type and memory layout.
        # Use small test arrays with the same reversed-y layout as real spectrum
        # calculations.
        dummy = np.arange(8 * 4 * 4, dtype=np.float64).reshape(1, 8, 4, 4)
        window.normalize_owned_windows(dummy, self.norm, np.ones(1, dtype=bool))
        normalized = perf_counter()
        spectral.spectral_imgs(dummy[:, :, ::-1, :])
        if self.norm == 'time':
            # Also prepare code for uint8 images and cropped images that share larger
            # image memory.
            # These small test arrays do not use or change the supplied video or its
            # temporary arrays.
            clip = np.arange(8*4*4, dtype=np.uint8).reshape(8, 4, 4)
            for source in (clip, clip[:, :, :3]):
                spectral._FFTWorkspace()._power_rectangular(
                    source, np.array([[0, 0]], dtype=np.intp), source.shape[1:], 8, 0,
                )
        finished = perf_counter()
        return dict(
            normalization_seconds=normalized - started,
            fft_seconds=finished - normalized,
            total_seconds=finished - started,
        )

    def save_frames(self, dst: str):
        raise NotImplementedError

    def save_windows(self):
        raise NotImplementedError


    def velocimetry(
        self,
        alpha: float = 0.85,
        # A depth of zero asks the fitting method to estimate depth as well as velocity.
        depth: float = 1.,
        pass_downsampling: Optional[List[int]] = None,
        dmin: Optional[float] = 0.01,
        dmax: Optional[float] = 3.0,
        gravity_waves_switch: Optional[bool] = True,
        turbulence_switch: Optional[bool] = True,
        penalty_weight: Optional[float] = 1,
        **opt_kwargs # A reduced first pass uses fewer spatial wavenumbers.
                            # The final pass then uses the full spectrum.
                            # This can reduce fitting time for large windows.
    ):
        """Fit velocity and optional depth with differential evolution, then save results.

        alpha is depth-averaged / surface velocity (default 0.85). depth is a supplied
        depth in metres, or zero to estimate it within dmin/dmax (default 0.01 to 3 m).
        pass_downsampling=[2, 1] fits a reduced central spectrum first, then the full
        spectrum. It is a list of reduction factors, not a twosteps switch.

        The wave and turbulence switches include gravity waves and flow-carried
        patterns respectively. Disable a component only when that model is unsuitable
        for the images. penalty_weight (default 1) discourages fast solutions but can
        bias velocity low or depth high; zero disables the penalty. Restricting smax
        can reduce wrong high-speed fits without adding that penalty.

        Extra keyword arguments go to scipy.optimize.differential_evolution.
        This FFT-only interface does not accept HNM dictionaries or the new diagnostic
        switches. Save vx, vy, depth, cost and quality on the instance; the older status
        is an overall completion flag, not a per-window optimiser-success array.
        See README.md for defaults and the SciPy documentation for solver keywords.
        """
        if self.spectrum is None:
            raise AttributeError(
                "No images available. Please set images first `iw.imgs = imgs`, where `imgs` is a numpy array "
                "of images, or read a video file using `iw.read_video(file)`")
        # Use the standard DE settings when the user has not supplied any.
        if not opt_kwargs:
            opt_kwargs = OPTIM_KWARGS_SADE
        if not pass_downsampling:
            pass_downsampling = [2, 1]
        if len(pass_downsampling)==1 and penalty_weight != 0:
            print(
                f"Velocity and especially depth estimations with the 1 step approach are biased when `penalty_weight` "
                f"is not zero. It is recommended to use the two steps approach and/or set `first_pass_downsample=1`, "
                f"or alternatively to set `penalty_weight = 0` and `smax` to reduce the number of outliers."
            )
        # Choose whether this DE pass uses a reduced spectrum.
        if len(pass_downsampling) > 1:
            print(
                    f"Optimization in {len(pass_downsampling)} steps with spectrum downsampling factor {pass_downsampling}"
                )
        else:
            print("Optimization in one step with full resolution.")
        # Limit each velocity component to +/-smax; also set depth limits if it is unknown.
        # A depth of zero asks the fitting method to estimate depth as well as velocity.
        if depth == 0:
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (dmin, dmax)]
        else:
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (depth, depth)]
        
        # Make a separate list of parameter limits for each window.
        bounds_list = [tuple(bounds) for _ in range(len(self.spectrum))]
        
        # Fit the velocity components and, if requested, depth for each window.
        output, cost, quality = optimise.optimise_velocity(
            measured_spectra=self.spectrum,
            bnds_list=bounds_list,
            vel_indx=alpha,
            window_dims=self.spectrum_dims,
            res=self.resolution,
            fps=self.fps,
            penalty_weight=penalty_weight,
            gravity_waves_switch=gravity_waves_switch,
            turbulence_switch=turbulence_switch,
            chunk_size=self.window_chunk_size,
            gauss_width=1,  # TODO: review the default options.
            pass_downsampling=pass_downsampling,
            desc="Optimizing windows",
            **opt_kwargs
        )
        # Fitting processes return (vy, vx, depth); store each under its named attribute.
        # Results are already back in their original window order, even if the fits
        # finished in a different order. Arrange them into image rows and columns.
        self.vy = output[:, 0].reshape(len(self.y), len(self.x))
        self.vx = output[:, 1].reshape(len(self.y), len(self.x))
        self.d = output[:, 2].reshape(len(self.y), len(self.x))
        self.cost = cost.reshape(len(self.y), len(self.x))
        self.quality = quality.reshape(len(self.y), len(self.x))
        self.status = True
        self.message = "Optimization terminated successfully."
        
    
    def plot_velocimetry(self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs):
        """Draw arrows for the fitted velocity components using the supplied Matplotlib options."""
        if ax is None:
            ax = plt.axes()
        p = ax.quiver(self.x, self.y, self.vx, self.vy, **kwargs)
        return p
