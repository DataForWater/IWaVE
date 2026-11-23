"""Lazy preprocessing of one requested window batch at a time.

Image windows and power spectra are never cached here. Only the small metadata
needed to extract rectangles and the bounded complex FFT workspace are retained.
Every returned spectrum owns data independent of that FFT workspace.
"""
import operator
from typing import Optional

import numpy as np
from iwave import window, spectral


def _selection(idx, length):
    """Convert an integer to a one-window slice; leave slices unchanged.

    Retaining a window axis through preprocessing gives integers and slices
    the same numerical path. Remove it only when returning an integer result.
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
    """Extract requested windows into owned (window, time, y, x) float64 data."""

    def __init__(self, imgs, sliding_window_func, win_x, win_y, norm):
        self.imgs = imgs
        self.sliding_window_func = sliding_window_func
        self.win_x, self.win_y = win_x, win_y
        if norm not in ('xy', 'time', None):
            raise ValueError("norm must be 'xy', 'time' or None.")
        self.norm = norm
        # Standard rectangular windows can be copied with basic slicing instead
        # of a frame-by-frame advanced-index gather followed by np.stack.
        # Detect this once, not on every batch. Custom callbacks and arbitrary
        # pixel-index windows retain their original extraction path.
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
            # A custom callback may return a view into the original images.
            # Own the buffer before normalising it in place.
            windows = np.array(
                self.sliding_window_func(
                    self.imgs, self.win_x[selected], self.win_y[selected],
                    swap_time_dim=True,
                ),
                dtype=np.float64, order='C', copy=True,
            )
        if self.norm is not None and len(windows):
            # Preserve the original per-window mean!=0 gate, even for signed
            # inputs. Normalisation still spans the entire clip, NOT each FFT
            # segment. Only its allocation/layout strategy has changed.
            nonzero = windows.mean(axis=(1, 2, 3)) != 0
            window.normalize_owned_windows(windows, self.norm, nonzero)
        return windows[0] if scalar else windows

    def __len__(self):
        return len(self.win_x)

    @property
    def shape(self):
        return len(self), self.imgs.shape[0], *self.win_x.shape[-2:]


class LazySpectrumArray:
    """Produce the original full-grid processed power on demand.

    Only complex FFT scratch is retained between requests, never outputs.
    A short final batch reuses it without overwriting a spectrum from an
    earlier batch. No fitting method or uncertainty interface is added here.
    """

    def __init__(self, windows, time_size: int, time_overlap: int,
                 kt: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                 smax: Optional[float] = 4., threshold: Optional[float] = 1.):
        self.windows = windows
        self.time_size, self.time_overlap = time_size, time_overlap
        self.kt, self.kx, self.ky = kt, kx, ky
        self.smax, self.threshold = smax, threshold
        self._fft_workspace = spectral._FFTWorkspace()  # Allocates on first use.

    def __repr__(self):
        return f'Spectrum dimensions (windows, ft, fy, fx): {self.shape}'

    def __getitem__(self, idx):
        selected, scalar = _selection(idx, len(self))
        windows = self.windows[selected]
        power = spectral.sliding_window_spectrum(
            windows[:, :, ::-1, :], self.time_size, self.time_overlap,
            _workspace=self._fft_workspace,
        )
        # Keep main's y reversal/sign convention. The FFT result owns its data,
        # so release the large image batch before normalisation/thresholding.
        del windows
        # This buffer is private to this request: normalise and threshold it
        # in place instead of allocating additional full spectral cubes.
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
