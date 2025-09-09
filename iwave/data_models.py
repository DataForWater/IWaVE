"""Lazy data models for memory intensive processing."""
import numpy as np

from typing import Optional, Union

from iwave import window, spectral


class LazyWindowArray:
    def __init__(self, imgs, sliding_window_func, win_x, win_y, norm):
        """Initialize a lazy array-like object for subwindows.

        Parameters
        ----------
        imgs : np.ndarray
            Images (3D numpy array)
        sliding_window_func : callable
            Function that extracts subwindows
        win_x : np.ndarray
            Sampling indexes for window extraction in x direction
        win_y : np.ndarray
            Sampling indexes for window extraction in y direction
        norm : {'xy', 'time', None}
            Whether to normalize the windows and in what way
        """
        self.imgs = imgs
        self.sliding_window_func = sliding_window_func
        self.win_x = win_x
        self.win_y = win_y
        self.norm = norm  # Optional: normalization parameter

    def __repr__(self):
        return f"Interrogation windows dimensions (windows, time, y, x): {self.shape}"

    def __getitem__(self, idx):
        """
        On-the-fly generation of subwindows for the requested index or slice.
        """
        if isinstance(idx, slice):
            # Generate windows for a range of indices
            start, stop, step = idx.indices(len(self.win_x))
            win_x_sel = self.win_x[start:stop:step]
            win_y_sel = self.win_y[start:stop:step]
            # indexed_imgs = self.imgs[start:stop:step]
        elif isinstance(idx, Union[int, np.integer]):
            # Generate windows for a single index
            win_x_sel = self.win_x[idx : idx + 1]
            win_y_sel = self.win_y[idx : idx + 1]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Only int or slice is supported.")

        # Use the sliding_window_func to generate the windows
        windows = self.sliding_window_func(self.imgs, win_x_sel, win_y_sel, swap_time_dim=True).astype(np.float64)

        # Apply normalization if needed
        if self.norm is not None:
            # find relevant windows with mean != 0. Only those require normalization
            nonzero_idx = np.where(np.mean(windows, axis=(1, 2, 3)) != 0)[0]
            if self.norm == "xy":
                windows[nonzero_idx] = window.normalize(windows[nonzero_idx], mode="xy")
            elif self.norm == "time":
                windows[nonzero_idx] = window.normalize(windows[nonzero_idx], mode="time")
        if isinstance(idx, slice):
            return windows
        elif isinstance(idx, int):
            return windows[0]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Only int or slice is supported.")

    def __len__(self):
        """
        Return the total number of windows (or equivalent slices).
        """
        return len(self.win_x)

    @property
    def shape(self):
        return len(self), self.imgs.shape[0], self.win_x.shape[1], self.win_x.shape[2]


class LazySpectrumArray(object):
    def __init__(
        self,
        windows: LazyWindowArray,
        time_size: int,
        time_overlap: int,
        kt: np.ndarray,
        kx: np.ndarray,
        ky: np.ndarray,
        smax: Optional[float] = 4.0,
        threshold: Optional[float] = 1.0
    ):
        self.windows = windows
        self.time_size = time_size
        self.time_overlap = time_overlap
        self.kt = kt
        self.kx = kx
        self.ky = ky
        self.smax = smax
        self.threshold = threshold

    def __repr__(self):
        """Return string representation showing spectrum dimensions."""
        return f"Spectrum dimensions (windows, ft, fy, fx): {self.shape}"

    def __getitem__(self, idx):
        """
        Generate on-the-fly spectra for subwindows for the requested index or slice.
        """

        # Use the sliding_window_func to generate the windows
        if isinstance(idx, slice):
            windows_sel = self.windows[idx]
        else:
            windows_sel = self.windows[idx : idx + 1]
        # now apply spectra on windows
        spectrum = spectral.sliding_window_spectrum(
            windows_sel,
            self.time_size,
            self.time_overlap,
        )

        # preprocess
        spectrum = spectral.spectrum_preprocessing(
            spectrum,
            self.kt,
            self.ky,
            self.kx,
            self.smax*3,
            spectrum_threshold=self.threshold
        )
        if isinstance(idx, slice):
            return spectrum
        elif isinstance(idx, Union[int, np.integer]):
            return spectrum[0]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Only int or slice is supported.")

    def __len__(self):
        """
        Return the total number of windows of spectram
        """
        return len(self.windows)

    @property
    def shape(self):
        return len(self), self.kt.size, self.ky.size, self.kx.size
