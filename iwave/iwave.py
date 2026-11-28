"""Main Iwave interface for loading images, fitting flow, and plotting results."""

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

# Default settings for the differential-evolution fitting method.
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
    """Store success and messages for each window, plus an overall summary.

    A result at the edge of its allowed parameter range is marked unsuccessful.
    The arrays are None before fitting. No windows means no successful fit.
    Calculate the summary from the window flags so it cannot contradict them.
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
        """Return overall_message under an alternative attribute name."""
        return self.overall_message

    def __bool__(self):
        # Allow `if iw.status:` to use the overall result instead of testing an entire
        # array.
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
            Number of windows processed together. Default 50. Smaller batches can
            reduce memory use; the source video still remains loaded.
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
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = None
        self.win_x = None
        self.win_y = None
        self.vy = None  # Velocity along image y (m/s).
        self.vx = None  # Velocity along image x (m/s).
        self.d = None  # Water depth (m).
        # Set after fitting: True means depth was estimated rather than supplied.
        self.depth_estimated = None
        self.cost = None  # Final fitting score; smaller values indicate closer agreement.
        # Agreement score: a perfect match gives 1. It is not limited to [0, 1] or an
        # error bar.
        self.quality = None
        self.status = VelocimetryStatus()
        self.method = None
        self.method_options = None
        # Extra information for each HNM fit; entries are None for ordinary DE fits.
        self.sampling_metadata = None
        self.spectral_uncertainty = None
        self.mode_ambiguity = None
        self.spectral_uncertainty_options = None
        self.mode_ambiguity_options = None
        self.calculate_spectral_uncertainty = False
        self.calculate_mode_ambiguity = False
                
    # Give access to the same status arrays under their earlier attribute names.
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
        """Fit each window's velocity and optional depth, then store the results.

        DE (differential evolution) is the default. HNM first uses measured frequency
        peaks to suggest velocities, then uses Nelder-Mead to improve those guesses
        against the full spectrum. The second step stays inside one circle around
        the main Hough estimate and within the overall parameter limits.

        Parameters
        ----------
        alpha : float
            Depth-averaged / surface velocity. Default 0.85. HNM requires 0.5 < alpha <= 1.
        depth : float
            Supplied depth in metres, default 1. Set zero to estimate depth within
            dmin/dmax instead (defaults 0.01 and 3 m).
        pass_downsampling : list of integers
            DE fitting passes, default [2, 1]: use a reduced central part of the
            spectrum first, then the full spectrum. This does not subsample the FFT.
        gravity_waves_switch, turbulence_switch : bool
            Include gravity waves and/or flow-carried patterns. Both default True.
            Disable a component only when it does not describe the images being used.
        penalty_weight : float
            DE-only penalty discouraging fast solutions, default 1. Zero disables it.
            It can reduce some wrong high-speed fits but can also bias velocity low
            or depth high. A sensible smax limit can avoid needing a strong penalty.
        method : 'DE' or 'HNM'
            Choose the fitting method; default DE.
        de_options : dict, optional
            DE gauss_width, penalty_weight, pass_downsampling and scipy_options.
            The nested scipy_options values override matching defaults. Older
            top-level DE settings and opt_kwargs still work on DE only, but do not
            mix that style with de_options.
        hnm_options : dict, optional
            One flat dictionary for Hough, column selection and Nelder-Mead settings.
            sampling='fixed' uses a predetermined column set; 'adaptive' chooses more
            columns as it goes. filling_ratio sets the maximum fraction examined.
            Refinement is on by default; refine=False is allowed for supplied depth.
            relative_radius=0.5 and min_radius set the common search-circle radius.
            None min_radius uses twice the larger actual velocity-grid spacing.

            Hough controls are velocity_step=0.01 m/s, vote_sigma=0.015 m/s,
            k_min_factor=1.0, peak_height=0.25, peak_prominence=0.30,
            scale_percentile=99.5, min_ridges=12 and max_accumulator_mb=None.
            See README.md for the meaning of each and the adaptive-selection options.

            Use n_candidates starts, each with its own maxfev evaluation budget.
            Defaults are 1/160 for supplied depth and 12/640 for unknown depth.
            User values override each default independently. All starts use the same
            circle; failure does not move or enlarge it. There is no HNM speed penalty.
            initial_step_factor=2 multiplies velocity_step to set initial trial spacing.

            With depth=0, fit velocity and log(depth) together to keep depth positive.
            depth_initial chooses the starting/reference depth; None uses the geometric
            mean of the limits. Hough uses only that reference depth. Nelder-Mead can
            vary it, with log_depth_step=0.1 setting the initial depth change.
            xatol measures changes in velocity coordinates and log(depth).
            Unknown depth requires refine=True and gravity_waves_switch=True.
        progress : bool
            Show one progress bar in the main program. Default True.
        gauss_width : float
            Earlier top-level DE model-width option, default 1.0. Use de_options in
            new code; do not pass this DE option to HNM.
        calculate_spectral_uncertainty : bool
            Default False. Estimate how the chosen velocity/depth would vary under
            the assumed spectral noise. Use complete normalised power before filtering
            and information about the averaging. No new fit is run.
        calculate_mode_ambiguity : bool
            Default False. Compare different successful HNM solutions already found
            away from the parameter limits. No extra search is run. One start, DE,
            or an invalid selected fit cannot establish whether another solution exists.
        spectral_uncertainty_options, mode_ambiguity_options : dict, optional
            Tune those checks; dictionaries alone do not turn them on. Both use
            confidence=.95, variance_estimator='plugin', n_effective=None by default.
            Near-duplicate solutions are grouped using ten xatol units per coordinate;
            merge_tolerances can supply other grouping distances.
        spectral_contexts : sequence of dicts, optional
            For previously calculated array spectra, supply one full pre-filter
            normalised_spectrum per window and either n_effective or
            n_averages/time_size/time_overlap. Normal video-window requests provide
            this information automatically without repeating the FFT.

        Stored results
        --------------
        Use vx, vy and d for the selected values; cost and quality describe agreement.
        status.windows and status.windows_message give outcomes for each window.
        status.overall and bool(status) are True only when every window succeeds.
        overall_message gives a count; the earlier status_windows, message_windows
        and message names still give read-only access to the same information.
        method_options records the settings used. sampling_metadata keeps detailed
        HNM records, or requested DE diagnostic records; otherwise DE entries are None.

        spectral_uncertainty gives vx_bounds, vy_bounds and depth_bounds with shape
        (rows, columns, 2), plus valid and ambiguous_solution flags. NaN means a bound
        is unavailable, not zero error. Full calculations/reasons are in .details.
        mode_ambiguity.additional_estimates lists other solutions; its competition
        flag is True, False or None. None means unknown and is not equivalent to False.
        Disabled checks remain None. The earlier fixed-score-loss uncertainty API
        has been removed.

        These checks never change the fit. Their bounds describe nearby variation
        under the assumed model, not every physical error or changes to a different
        solution. Comparisons cover only solutions actually found, and their stated
        thresholds are not independently calibrated for searching many solutions.

        See the README option tables and SciPy's differential_evolution documentation
        for the full tuning reference.
        """
        if self.spectrum is None:
            raise AttributeError(
                "No images available. Please set images first `iw.imgs = imgs`, where `imgs` is a numpy array "
                "of images, or read a video file using `iw.read_video(file)`")
        optimise.reject_removed_uncertainty_options(opt_kwargs)
        # Keep earlier DE calling styles working, with separate options for each method.
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
        # Limit each velocity component to +/-smax; also set depth limits if it is unknown.
        # A depth of zero asks the fitting method to estimate depth as well as velocity.
        if depth == 0:
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (dmin, dmax)]
        else:
            bounds = [(-self.smax, self.smax), (-self.smax, self.smax), (depth, depth)]
        
        # Choose depth-dependent defaults after the actual depth limits are available.
        de, hnm = optimise.resolve_method_options(
            method, de_options=de_options, hnm_options=hnm_options,
            legacy=legacy, scipy_defaults=OPTIM_KWARGS_SADE, bnds=bounds)

        # Show this warning once in the main program, not separately for every window.
        # Local uncertainty can still be calculated, but one starting guess cannot
        # compare different solutions. Store the same explanation with the mode results.
        mode_message = None
        if mode_options is not None and method == 'HNM' and hnm['n_candidates'] == 1:
            import warnings
            mode_message = 'Ambiguity cannot be assessed if n_candidates = 1.'
            warnings.warn(mode_message, UserWarning, stacklevel=2)

        # Make a separate list of parameter limits for each window.
        bounds_list = [tuple(bounds) for _ in range(len(self.spectrum))]
        
        # Fit the velocity components and, if requested, depth for each window.
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
        # Fitting processes return (vy, vx, depth); store each under its named attribute.
        # Results are already back in their original window order, even if the fits
        # finished in a different order. Arrange them into image rows and columns.
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
        # Record whether depth really was free to change in this fit.
        # Equal lower and upper depth limits still mean that depth was fixed.
        self.depth_estimated = bool(bounds[2][0] < bounds[2][1])
        shape = self.vx.shape
        self.spectral_uncertainty = SpectralUncertainty(shape) if spectral_options is not None else None
        self.mode_ambiguity = ModeAmbiguity(shape) if mode_options is not None else None
        if self.mode_ambiguity is not None:
            self.mode_ambiguity.message = mode_message
        # Collect the results for direct use through Iwave attributes.
        # Keep the more detailed calculations in sampling_metadata and .details.
        # This step only organises existing results; it does not repeat fitting or
        # transforms.
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
        """Plot existing results along one row or column of windows.

        Use column=0 for all rows in the first column, or row=2 for all columns in
        the third row. Supply exactly one index, counted from zero, not pixels.
        The horizontal axis follows x for a row and y for a column. Positions use
        pixels unless distance_units='m' requests conversion by image resolution.

        Depth appears automatically only when it was fitted, independently of whether
        uncertainty was requested. include_depth=False hides it; True requests it
        but rejects supplied-depth fits. The final panel always shows quality.
        Accepted bounds describe noise near the selected fit, not all errors.
        Other solutions appear separately without calculated bounds of their own.
        No warning symbol for competition means unknown or no competitor observed,
        not proof that the solution is unique.

        Return (figure, axes) without calling show(). Reuse axes with the same panel
        count, and set label/color, to compare methods. No fitting runs here.
        See io.plot_transect for the full list of options and symbol meanings.
        """
        from iwave.io import plot_transect
        return plot_transect(self, row=row, column=column, include_depth=include_depth,
                             show_uncertainty=show_uncertainty, show_modes=show_modes,
                             distance_units=distance_units, axes=axes, label=label, color=color)

    def plot_velocimetry(self, ax: Optional[matplotlib.axes.Axes] = None, *,
                         validate_vectors=True, distance_units='pixels',
                         color='C0', ambiguous_color='tab:red', **kwargs):
        """Plot existing velocity estimates as arrows at the window centres.

        validate_vectors=True omits failed fits; False also allows finite failed
        fits. Undefined velocities are always omitted. Pixels are the default
        position units; distance_units='m' converts positions using image resolution.
        The velocity components remain in m/s. Arrow length also depends on the
        Matplotlib scale setting; the coordinates do not turn velocity into pixels/s.

        Use ambiguous_color for known competitive alternatives, not unknown cases.
        color supplies the base colour. Both must be single Matplotlib colours.
        ax and other quiver keywords retain their usual meanings. Return a Quiver
        object. This only displays existing results; it performs no new calculations.
        """
        return io.plot_velocimetry(self, ax=ax, validate_vectors=validate_vectors,
                                   distance_units=distance_units, color=color,
                                   ambiguous_color=ambiguous_color, **kwargs)
