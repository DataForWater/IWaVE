import numpy as np
from typing import Tuple, Literal


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
    mode: Literal["xy", "time"] = "time"
):
    """Adjust image values by subtracting a mean and dividing by standard deviation.

    imgs can contain one stack (time, y, x), or multiple window stacks.
    mode chooses normalisation over time or over each image's spatial dimensions.
    Return normalised data of the same shape, cleaning undefined numerical values.
    """
    # Calculate the mean and standard deviation for normalisation.
    if mode == "xy":
        imgs_std = np.expand_dims(
            imgs.reshape(imgs.shape[0], imgs.shape[1], -1).std(axis=-1),
            axis=(-1, -2)
        )
        imgs_mean = np.expand_dims(
            imgs.reshape(imgs.shape[0], imgs.shape[1], -1).mean(axis=-1),
            axis=(-1, -2)
        )
    elif mode == "time":
        imgs_std = np.expand_dims(
            imgs.std(axis=-3),
            axis=-3
        )
        imgs_mean = np.expand_dims(
            imgs.mean(axis=-3),
            axis=-3
        )
    else:
        raise ValueError(f'mode must be "xy" or "time", but is "{mode}"')
    # This makes a full-size temporary array; a block-by-block version could use less
    # memory.
    with np.errstate(divide='ignore', invalid='ignore'):
        imgs = (imgs - imgs_mean) / imgs_std
    imgs = np.nan_to_num(imgs)
    return imgs

