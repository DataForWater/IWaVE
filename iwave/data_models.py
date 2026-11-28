"""Calculate a requested batch of windows or spectra when indexed.

float64 means ordinary 64-bit floating-point numbers.
Keep the source video, window positions and reusable temporary arrays, but
not all extracted windows or spectra. Every returned spectrum is independent
of temporary arrays that later requests can overwrite.
"""
import operator
from typing import Optional

import numpy as np
from iwave import window, spectral


def _selection(idx, length):
    """Treat a single index as a one-window slice during calculation.

    This lets single-window and multi-window requests follow the same steps.
    Remove that extra window dimension only when returning a single result.
    """
    if isinstance(idx, slice):
        return idx, False
    try:
        i = operator.index(idx)
    except TypeError as exc:
        raise TypeError('Only an integer or slice is supported.') from exc
    if i < 0:
        i += length
    if not 0 <= i < length:
        raise IndexError('Window index out of range.')
    return slice(i, i + 1), True


class LazyWindowArray:
    """Copy selected windows into a separate 64-bit floating-point array (float64).

    Use (window, time, y, x) order. Changing this array must not change the video.
    """

    def __init__(self, imgs, sliding_window_func, win_x, win_y, norm):
        self.imgs = imgs
        self.sliding_window_func = sliding_window_func
        self.win_x, self.win_y = win_x, win_y
        if norm not in ('xy', 'time', None):
            raise ValueError("norm must be 'xy', 'time' or None.")
        self.norm = norm
        # Ordinary rectangular windows can be copied directly from their image positions.
        # This avoids collecting each frame separately and joining the copies afterwards.
        # Check the window layout once, rather than for every batch.
        # Keep the general extraction method for custom functions or irregular pixel
        # selections.
        self._origins = (
            window.rectangular_window_origins(win_x, win_y, imgs.shape)
            if sliding_window_func is window.multi_sliding_window_array else None
        )

    def __repr__(self):
        return f'Interrogation windows dimensions (windows, time, y, x): {self.shape}'

    def __getitem__(self, idx):
        selected, scalar = _selection(idx, len(self))
        if self._origins is not None:
            windows = window.extract_rectangular_windows(
                self.imgs, self._origins[selected], self.win_x.shape[-2:]
            )
        else:
            # A custom extraction function may return pixels sharing the original video
            # memory.
            # Make a separate copy before changing their brightness values.
            windows = np.array(
                self.sliding_window_func(
                    self.imgs, self.win_x[selected], self.win_y[selected],
                    swap_time_dim=True,
                ),
                dtype=np.float64, order='C', copy=True,
            )
        if self.norm is not None and len(windows):
            # Preserve the original rule: do not normalise windows with a zero mean.
            # Compute time averages over the full clip, not separately for each FFT segment.
            # Only the way arrays are created and stored has changed.
            nonzero = windows.mean(axis=(1, 2, 3)) != 0
            window.normalize_owned_windows(windows, self.norm, nonzero)
        return windows[0] if scalar else windows

    def __len__(self):
        return len(self.win_x)

    @property
    def shape(self):
        return len(self), self.imgs.shape[0], *self.win_x.shape[-2:]


class LazySpectrumArray:
    """Calculate the requested processed spectra, keeping every spatial wavenumber.

    Only temporary transform/input arrays are reused between requests, not
    returned results. A short final request must not overwrite earlier spectra.
    This FFT-only class adds no fitting or uncertainty interface.
    """

    def __init__(self, windows, time_size: int, time_overlap: int,
                 kt: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                 smax: Optional[float] = 4., threshold: Optional[float] = 1.):
        self.windows = windows
        self.time_size, self.time_overlap = time_size, time_overlap
        self.kt, self.kx, self.ky = kt, kx, ky
        self.smax, self.threshold = smax, threshold
        # Each spectrum-array object keeps its own reusable temporary arrays.
        # Later requests may reuse those arrays but must not overwrite returned spectra.
        # Create these temporary arrays only when first needed.
        self._fft_workspace = spectral._FFTWorkspace()

    def __repr__(self):
        return f'Spectrum dimensions (windows, ft, fy, fx): {self.shape}'

    def _power(self, selected):
        windows = self.windows
        if (type(windows) is LazyWindowArray and windows._origins is not None
                and (type(windows.imgs) is np.ndarray or isinstance(windows.imgs, np.memmap))
                and windows.norm == 'time' and windows.imgs.dtype == np.uint8):
            return self._fft_workspace._power_rectangular(
                windows.imgs, windows._origins[selected], windows.win_x.shape[-2:],
                self.time_size, self.time_overlap,
            )
        # Keep the general extraction rules for unusual indexing, custom functions,
        # signed or floating-point images, and other normalisation modes.
        owned = windows[selected]
        return spectral.sliding_window_spectrum(
            owned[:, :, ::-1, :], self.time_size, self.time_overlap,
            _workspace=self._fft_workspace,
        )

    def __getitem__(self, idx):
        selected, scalar = _selection(idx, len(self))
        power = self._power(selected)
        # Only this request uses this power array, so it can be changed directly.
        # That avoids making another full-sized copy during normalisation and filtering.
        # This normalisation is different from adjusting image brightness:
        # divide each frequency slice by its mean power across spatial wavenumbers.
        normalized = spectral.normalize_power(power, copy=False)
        processed = spectral.preprocess_normalized_spectrum(
            normalized, self.kt, self.ky, self.kx, self.smax * 3,
            self.threshold, copy=False,
        )
        return processed[0] if scalar else processed

    def __len__(self):
        return len(self.windows)

    @property
    def shape(self):
        return len(self), self.kt.size, self.ky.size, self.kx.size
