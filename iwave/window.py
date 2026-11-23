"""Window geometry and image normalisation.

The public indexing helpers remain available. LazyWindowArray additionally
uses owned rectangular copies and blocked temporal normalisation to avoid
frame-stack and variance temporaries for each requested window batch.
"""
from typing import Tuple, Literal

import numpy as np
import numba as nb


def sliding_window_idx(
    image: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Create y and x indices per interrogation window.

    Parameters
    ----------
    image : np.ndarray
        black-white image or template
    window_size : tuple
        size of interrogation window (y, x)
    overlap : tuple
        overlap of pixels of interrogation windows (y, x)

    Returns
    -------
    win_x : np.ndarray (w * y * x)
        x-indices of interrogation windows (w)
    win_y : np.ndarray (w * y * x)
        y-indices of interrogation windows (w)
    """

    x, y = get_rect_coordinates(
        image.shape,
        window_size,
        overlap,
        center_on_field=False
    )
    xi, yi = np.meshgrid(x, y)
    xi = (xi - window_size[1] // 2).astype(int)
    yi = (yi - window_size[0] // 2).astype(int)
    xi, yi = np.reshape(xi, (-1, 1, 1)), np.reshape(yi, (-1, 1, 1))

    win_x, win_y = np.meshgrid(
        np.arange(0, window_size[1]),
        np.arange(0, window_size[0])
    )
    win_x = win_x[np.newaxis, :, :] + xi
    win_y = win_y[np.newaxis, :, :] + yi
    return win_x, win_y


def sliding_window_array(
    image: np.ndarray,
    win_x: np.ndarray,
    win_y: np.ndarray,
) -> np.ndarray:

    return image[win_y, win_x]


def multi_sliding_window_array(
    imgs: np.ndarray,
    win_x: np.ndarray,
    win_y: np.ndarray,
    swap_time_dim=False,
) -> np.ndarray:
    """Get multiple interrogation windows from a stack of images lazily."""
    windows = np.stack(
        [
            sliding_window_array(
                img,
                win_x,
                win_y
            ) for img in imgs
        ]
    )

    if swap_time_dim:
        return np.swapaxes(windows, 0, 1)

    return windows


def get_axis_shape(
    dim_size: int,
    window_size: int,
    overlap: int,
) -> int:
    """
    get shape of image axis given its dimension size

    Parameters
    ----------
    dim_size : int
        size of axis [pix]
    window_size : int
        size of interrogation window over axis dimension [pix]
    overlap : int
        size of overlap [pix]

    Returns
    -------
    int, amount of interrogation windows over provided axis

    """
    axis_shape = (dim_size - window_size) // (window_size - overlap) + 1
    return axis_shape


def get_array_shape(
    dim_sizes: Tuple[int, int],
    window_sizes: Tuple[int, int],
    overlaps: Tuple[int, int]
):
    """
    Get the resulting shape of velocimetry results as a tuple of dimension sizes

    Parameters
    ----------
    dim_sizes : [int, int]
        sizes of axes [pix]
    window_sizes : [int, int]
        sizes of interrogation windows [pix]
    overlaps : [int, int]
        sizes of overlaps [pix]

    Returns
    -------
    shape of array returned from velocimetry analysis

    """
    array_shape = tuple(
        get_axis_shape(dim_size, window_size, overlap) for (
            dim_size,
            window_size,
            overlap
        ) in zip(
            dim_sizes,
            window_sizes,
            overlaps
        )
    )
    return array_shape


def get_axis_coords(
    dim_size: int,
    window_size: int,
    overlap: int,
    center_on_field: bool = False,
):
    """
    Get axis coordinates for one axis with provided dimensions and window size
    parameters. Overlap for windows can be provided.

    Parameters
    ----------
    dim_size : int
        size of axis [pix]
    window_size : int
        size of interrogation window over axis dimension [pix]
    overlap : int
        size of overlap [pix]
    center_on_field : bool, optional
        take the center of the window as coordinate (default, False)

    Returns
    -------
    x- or y-coordinates of resulting velocimetry grid

    """
    # get the amount of expected coordinates
    ax_shape = get_axis_shape(dim_size, window_size, overlap)
    coords = np.arange(ax_shape) * (window_size - overlap) + (window_size) / 2.0
    if center_on_field is True:
        coords_shape = get_axis_shape(
            dim_size=dim_size,
            window_size=window_size,
            overlap=overlap
        )
        coords += (dim_size - 1 - ((coords_shape - 1) * (window_size - overlap) + (
                window_size - 1))
        ) // 2
    return coords


def get_rect_coordinates(
    dim_sizes: Tuple[int, int],
    window_sizes: Tuple[int, int],
    overlap: Tuple[int, int],
    center_on_field: bool = False,
):
    """
    create meshgrid coordinates (x, y) of velocimetry results. Overlap can be
    provided in case each interrogation window is to overlap with the 
    neighbouring interrogation window.

    Parameters
    ----------
    dim_sizes : [int, int]
        sizes of axes [pix]
    window_sizes : [int, int]
        sizes of interrogation windows [pix]
    overlaps : [int, int]
        sizes of overlaps [pix]
    center_on_field : bool, optional
        take the center of the window as coordinate (default, False)

    Returns
    -------
    xi, yi: np.ndarray (2D), np.ndarray (2D)
        x- and y-coordinates in meshgrid form

    """
    y = get_axis_coords(
        dim_sizes[0],
        window_sizes[0],
        overlap[0],
        center_on_field=center_on_field
    )
    x = get_axis_coords(
        dim_sizes[1],
        window_sizes[1],
        overlap[1],
        center_on_field=center_on_field
    )

    return x, y


def normalize(
    imgs: np.ndarray,
    mode: Literal["xy", "time"] = "time",
    *,
    copy: bool = True,
):
    """Normalise images along time or within each spatial image.

    Parameters
    ----------
    imgs : np.ndarray
        Image stack with trailing (time, y, x) dimensions; a leading window
        axis is allowed. Time statistics span the complete supplied clip.
    mode : {'xy', 'time'}
        Axes over which to subtract the mean and divide by population std.
    copy : bool, optional
        True (default) protects caller data with a float64 copy. False is for
        owned writable floating buffers, allowing subtraction/division in place.

    Returns
    -------
    np.ndarray
        Normalised images with unchanged shape; non-finite results are cleaned
        as before, including zero-contrast pixels.
    """
    # copy=False is only used for a private, floating-point extraction buffer.
    # The public default does not mutate images supplied by a caller.
    imgs = np.array(imgs, dtype=np.float64, copy=True) if copy else np.asarray(imgs)
    if not np.issubdtype(imgs.dtype, np.floating) or not imgs.flags.writeable:
        raise ValueError('In-place normalization needs a writable floating array.')
    if imgs.ndim < 3:
        raise ValueError('Images must have time, y and x dimensions.')
    if mode == 'xy':
        axes = (-2, -1)
    elif mode == 'time':
        axes = -3
    else:
        raise ValueError(f'mode must be "xy" or "time", but is "{mode}"')
    # Compute both statistics BEFORE overwriting the intensities. Keep ddof=0
    # and the original whole-clip time interval, not individual FFT segments.
    std = imgs.std(axis=axes, keepdims=True)
    mean = imgs.mean(axis=axes, keepdims=True)
    np.subtract(imgs, mean, out=imgs)
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(imgs, std, out=imgs)
    return np.nan_to_num(imgs, copy=False)


def rectangular_window_origins(win_x, win_y, image_shape):
    """Return rectangle origins, or None for arbitrary pixel-index windows.

    Checking once when constructing a lazy array retains the general indexing
    API without repeating a whole grid comparison for every window batch.
    """
    x, y = np.asarray(win_x), np.asarray(win_y)
    if x.ndim != 3 or x.shape != y.shape:
        return None
    ny, nx = x.shape[-2:]
    x0, y0 = x[:, 0, 0], y[:, 0, 0]
    if (np.any(x0 < 0) or np.any(y0 < 0)
            or np.any(x0 + nx > image_shape[-1])
            or np.any(y0 + ny > image_shape[-2])):
        return None
    if (not np.array_equal(x, np.broadcast_to(x0[:, None, None] + np.arange(nx), x.shape))
            or not np.array_equal(y, np.broadcast_to(y0[:, None, None] + np.arange(ny)[:, None], y.shape))):
        return None
    return np.column_stack((y0, x0)).astype(np.intp)


def extract_rectangular_windows(imgs, origins, window_shape):
    """Copy only one requested batch into owned (window,time,y,x) float64 data.

    A loop over rectangles avoids frame-by-frame advanced-index gathers and
    np.stack temporaries. The source may be a non-contiguous ROI or memmap;
    it is never modified and no full-video window view is materialised.
    """
    ny, nx = window_shape
    result = np.empty((len(origins), len(imgs), ny, nx), dtype=np.float64)
    for i, (y0, x0) in enumerate(origins):
        result[i] = imgs[:, y0:y0 + ny, x0:x0 + nx]
    return result


@nb.njit(parallel=True, cache=True, nogil=True, error_model='numpy')
def _normalize_time_owned(windows, nonzero):
    """Stable two-pass variance in small spatial blocks, without a video temp.

    Time is traversed in the original order; nearby pixels are processed
    together for contiguous memory access. Parallel blocks work even when the
    spatial-window batch contains fewer windows than there are CPU cores.
    No fastmath or E[x*x]-E[x]**2 approximation is used.
    """
    n, nt, ny, nx = windows.shape
    # Small per-task mean/variance arrays replace NumPy std's clip-sized
    # centred temporary. Adjacent pixels stay together for cache locality.
    pixels, block = ny*nx, 128
    blocks = (pixels + block - 1)//block
    for task in nb.prange(n*blocks):
        m = task//blocks
        if not nonzero[m]:
            continue
        start = (task % blocks)*block
        size = min(block, pixels - start)
        data = windows[m].reshape(nt, pixels)
        mean, variance = np.zeros(size), np.zeros(size)
        for t in range(nt):
            for j in range(size):
                mean[j] += data[t, start+j]
        mean /= nt
        for t in range(nt):
            for j in range(size):
                delta = data[t, start+j] - mean[j]
                variance[j] += delta*delta
        # Reuse the variance accumulator for population standard deviation.
        for j in range(size):
            variance[j] = np.sqrt(variance[j]/nt)
        for t in range(nt):
            for j in range(size):
                value = (data[t, start+j] - mean[j])/variance[j]
                # Match nan_to_num, including zero-contrast windows. The
                # numpy error model above makes division by zero non-raising.
                if np.isnan(value):
                    value = 0.
                elif value == np.inf:
                    value = np.finfo(np.float64).max
                elif value == -np.inf:
                    value = -np.finfo(np.float64).max
                data[t, start+j] = value


def normalize_owned_windows(windows, mode, nonzero):
    """Internal batch normalizer; only private float64 C buffers are allowed."""
    if (windows.ndim != 4 or windows.dtype != np.float64
            or not windows.flags.c_contiguous or not windows.flags.writeable):
        raise ValueError('Owned normalization requires writable contiguous float64 windows.')
    if not len(windows) or mode is None:
        return windows
    if mode == 'time':
        _normalize_time_owned(windows, nonzero)
    elif mode == 'xy':
        if nonzero.all():
            normalize(windows, mode, copy=False)
        else:
            for i in np.flatnonzero(nonzero):
                normalize(windows[i], mode, copy=False)
    else:
        raise ValueError("mode must be 'time', 'xy' or None.")
    return windows

