"""Calculate image windows and spectra only when they are requested.

The video stays in memory, but a request copies and processes only the selected
windows. This avoids creating every window and spectrum at once.
"""
import numpy as np

from typing import Optional, Union

from iwave import window, spectral


class LazyWindowArray:
    def __init__(self, imgs, sliding_window_func, win_x, win_y, norm):
        """Describe how to extract windows from the supplied images.

        imgs contains frames in (time, y, x) order. win_x and win_y give the pixel
        positions for each window; sliding_window_func performs the extraction.
        With norm='time', adjust each pixel using its average and standard deviation
        over the whole clip. With norm='xy', adjust each frame using its spatial
        average and standard deviation. None leaves image brightness unchanged.
        """
        self.imgs = imgs
        self.sliding_window_func = sliding_window_func
        self.win_x = win_x
        self.win_y = win_y
        # Choose whether to adjust image brightness over time, over each frame, or not at
        # all.
        self.norm = norm

    def __repr__(self):
        return f"Interrogation windows dimensions (windows, time, y, x): {self.shape}"

    def __getitem__(self, idx):
        """Create only the windows selected by an integer index or slice."""
        if isinstance(idx, slice):
            # Select the requested range of windows.
            start, stop, step = idx.indices(len(self.win_x))
            win_x_sel = self.win_x[start:stop:step]
            win_y_sel = self.win_y[start:stop:step]
        elif isinstance(idx, Union[int, np.integer]):
            # Select one window.
            win_x_sel = self.win_x[idx : idx + 1]
            win_y_sel = self.win_y[idx : idx + 1]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Only int or slice is supported.")

        # Copy the requested parts of the images using the supplied extraction function.
        windows = self.sliding_window_func(self.imgs, win_x_sel, win_y_sel, swap_time_dim=True).astype(np.float64)

        # Calculate brightness averages and standard deviations from the full loaded video.
        # Doing this separately for each shorter segment would give different spectra.
        # Adjust brightness only when requested.
        if self.norm is not None:
            # Keep windows with zero mean unchanged, as in the original calculation.
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
        """Return the number of windows described by this object."""
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
        """Describe the number and dimensions of the spectra available."""
        return f"Spectrum dimensions (windows, ft, fy, fx): {self.shape}"

    def __getitem__(self, idx):
        """Calculate spectra only for the selected windows."""

        # Copy the requested parts of the images using the supplied extraction function.
        if isinstance(idx, slice):
            windows_sel = self.windows[idx]
        else:
            windows_sel = self.windows[idx : idx + 1]
        # Reverse image y so positive y points upwards when calculating the spectrum.
        spectrum = spectral.sliding_window_spectrum(
            windows_sel[:,:,::-1,:],  # Reverse the local image y direction.
            self.time_size,
            self.time_overlap,
        )

        # Normalise the spectrum and remove values excluded by the filtering rules.
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

    def get_with_spectral_context(self, idx):
        """Calculate spectra once and also return the information needed for noise checks.

        Keep the full normalised power before the fitting filters discard values.
        Each returned dictionary also describes the time averaging. These dictionaries
        are made for this request; the object does not keep a copy of all spectra.
        Ordinary indexing still returns processed power only.
        """
        windows_sel = self.windows[idx] if isinstance(idx, slice) else self.windows[idx:idx+1]
        length = int(self.time_size)
        overlap = int(self.time_overlap)
        if length < 1 or not 0 <= overlap < length or windows_sel.shape[1] < length:
            raise ValueError('Invalid temporal segmentation for spectral uncertainty.')
        raw = spectral.sliding_window_spectrum(windows_sel[:,:,::-1,:], length, overlap)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = raw / np.mean(raw, axis=(2,3), keepdims=True)
        normalized = np.nan_to_num(normalized)
        # Keep a separate spectrum before filtering. The uncertainty calculation needs
        # values that fitting will discard, and must account for the shared normalisation.
        processed = spectral.spectrum_preprocessing(raw, self.kt, self.ky, self.kx,
                              self.smax*3, spectrum_threshold=self.threshold)
        # Count only complete time segments. Leftover frames do not add another spectrum.
        # The effect of overlapping segments on uncertainty is handled later.
        count = (windows_sel.shape[1]-length)//(length-overlap)+1
        contexts = [dict(normalized_spectrum=normalized[i], n_averages=count,
                         time_size=length, time_overlap=overlap,
                         source='lazy_full_normalized_before_threshold') for i in range(len(processed))]
        return (processed, contexts) if isinstance(idx, slice) else (processed[0], contexts[0])

    def __len__(self):
        """Return the number of spectra described by this object."""
        return len(self.windows)

    @property
    def shape(self):
        return len(self), self.kt.size, self.ky.size, self.kx.size
