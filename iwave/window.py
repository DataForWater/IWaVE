"""Choose image windows and adjust their brightness for spectrum calculation.

Public functions still accept explicit pixel indices. For normal rectangles,
copy only the selected regions, and calculate time averages in small pixel
groups to avoid extra full-video arrays.
"""
from typing import Tuple, Literal

import numpy as np
import numba as nb


def sliding_window_idx(
    image: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Find which pixels belong to each rectangular image window.

    image provides the image shape. window_size is (height, width) in pixels;
    overlap gives shared pixels between neighbours in (y, x) order.
    Return win_x and win_y, each shaped (window, y, x), containing pixel indices.
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
    """Extract only the requested image windows from a stack of frames."""
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
    """Count how many windows fit along one image dimension.

    dim_size, window_size and overlap are lengths in pixels. overlap describes
    shared pixels between adjacent windows. Return the number of valid positions.
    """
    axis_shape = (dim_size - window_size) // (window_size - overlap) + 1
    return axis_shape


def get_array_shape(
    dim_sizes: Tuple[int, int],
    window_sizes: Tuple[int, int],
    overlaps: Tuple[int, int]
):
    """Return the rows and columns in the grid of fitted windows.

    dim_sizes, window_sizes and overlaps are (y, x) pairs in pixels.
    Count valid window positions along each image direction.
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
    """Return window-centre positions along one image axis, in pixels.

    dim_size, window_size and overlap give the image size, window size and shared
    pixels. Centres start half a window from the edge. center_on_field=True adds
    an offset to centre the whole window layout in the available image area;
    it does not change from window corners to centres.
    """
    # Count how many window positions fit along this axis.
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
    """Return x and y window-centre positions for the result grid.

    dim_sizes, window_sizes and overlap are (y, x) pairs in pixels.
    center_on_field=True shifts the window layout to centre it in the image.
    Return the one-dimensional x and y coordinate arrays; the fitted field has
    one row per y position and one column per x position.
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
    """Subtract the mean and divide by standard deviation over time or per frame.

    imgs ends in (time, y, x); a leading window dimension is also allowed.
    mode='time' uses all supplied frames at each pixel. mode='xy' uses each
    frame's spatial mean and standard deviation. Use population standard
    deviation, dividing variance by the sample count rather than count minus one.
    copy=True makes a separate float64 array and protects supplied images.
    copy=False changes an existing writable floating-point array directly.
    Return the same shape and apply the existing cleanup for undefined values,
    including constant-brightness pixels.
    """
    # Use copy=False only for a private floating-point array that is safe to change.
    # The default public call protects the images supplied by the user.
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
    # Calculate the mean and standard deviation before changing image values.
    # Use population standard deviation (ddof=0) over the whole clip, not per segment.
    std = imgs.std(axis=axes, keepdims=True)
    mean = imgs.mean(axis=axes, keepdims=True)
    np.subtract(imgs, mean, out=imgs)
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(imgs, std, out=imgs)
    return np.nan_to_num(imgs, copy=False)


def rectangular_window_origins(win_x, win_y, image_shape):
    """Return each rectangle's starting pixel, or None for irregular selections.

    Check the layout once when setting up the windows, rather than comparing the
    whole index grid again for every batch.
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
    """Copy one selected batch into a separate (window, time, y, x) float64 array.

    Read rectangles directly rather than collecting and joining frame copies.
    The source can be a cropped image view or a memory-mapped file array.
    Do not change it or extract all windows in the video.
    """
    ny, nx = window_shape
    result = np.empty((len(origins), len(imgs), ny, nx), dtype=np.float64)
    for i, (y0, x0) in enumerate(origins):
        result[i] = imgs[:, y0:y0 + ny, x0:x0 + nx]
    return result


@nb.njit(cache=True, nogil=True, error_model='numpy')
def _gather_normalize_time_uint8(imgs, origin, output):
    """Copy one uint8 rectangle and calculate its time means in the same loop.

    Calculate mean first, then variance in a separate pass, in the original time
    order. With nonnegative uint8 pixels, a window with zero mean is entirely
    zero and stays zero after normalisation. Do not apply that shortcut to signed
    or floating-point images; use their general calculation instead.
    """
    nt, ny, nx = output.shape
    data = output.reshape(nt, ny*nx)
    for start in range(0, ny*nx, 128):
        size = min(128, ny*nx-start)
        mean, variance = np.zeros(size), np.zeros(size)
        ys, xs = np.empty(size, np.intp), np.empty(size, np.intp)
        for j in range(size):
            ys[j] = origin[0] + (start+j)//nx
            xs[j] = origin[1] + (start+j)%nx
        for t in range(nt):
            for j in range(size):
                value = float(imgs[t, ys[j], xs[j]])
                data[t, start+j] = value
                mean[j] += value
        mean /= nt
        for t in range(nt):
            for j in range(size):
                delta = data[t, start+j] - mean[j]
                variance[j] += delta*delta
        for j in range(size):
            variance[j] = np.sqrt(variance[j]/nt)
        for t in range(nt):
            for j in range(size):
                value = (data[t, start+j] - mean[j])/variance[j]
                if np.isnan(value):
                    value = 0.
                elif value == np.inf:
                    value = np.finfo(np.float64).max
                elif value == -np.inf:
                    value = -np.finfo(np.float64).max
                data[t, start+j] = value


@nb.njit(parallel=True, cache=True, nogil=True, error_model='numpy')
def _normalize_time_owned(windows, nonzero):
    """Calculate time variance in small pixel groups without a video-sized copy.

    Calculate the mean first and squared deviations second, following the same
    frame order. Process neighbouring pixels together for efficient memory access.
    Different groups can run in parallel even when the batch has few windows.
    Keep the usual floating-point rules, not the less stable
    mean(x*x) - mean(x)**2 variance shortcut.
    """
    n, nt, ny, nx = windows.shape
    # Process small groups of pixels so the standard-deviation calculation does not
    # need a second video-sized array. Keep neighbouring pixels together in memory.
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
        # Turn the accumulated variance into standard deviation, reusing the same array.
        for j in range(size):
            variance[j] = np.sqrt(variance[j]/nt)
        for t in range(nt):
            for j in range(size):
                value = (data[t, start+j] - mean[j])/variance[j]
                # Handle undefined results as nan_to_num does, including
                # constant-brightness pixels.
                # The selected NumPy error rule allows division by zero before that cleanup.
                if np.isnan(value):
                    value = 0.
                elif value == np.inf:
                    value = np.finfo(np.float64).max
                elif value == -np.inf:
                    value = -np.finfo(np.float64).max
                data[t, start+j] = value


def normalize_owned_windows(windows, mode, nonzero):
    """Normalise batches using a separate array that is safe to change directly.

    The internal caller supplies 64-bit floating-point values in
    (window, time, y, x) order, with adjacent x values next to each other in memory.
    Do not pass the shared source images to this routine.
    """
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

