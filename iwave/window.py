import numpy as np
from typing import Tuple, Literal


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
    swap_time_dim=False
) -> np.ndarray:

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
    mode: Literal["xy", "time"] = "time"
):
    """
    normalizes images assuming the last two dimensions contain the x/y image
    intensities

    Parameters
    ----------
    imgs : np.ndarray (n x Y x X) or (n x m x Y x X)
        input images, organized in at least one stack
    Returns
    -------

    imgs_norm : np.ndarray (n x Y x X) or (n x m x Y x X)
        output normalized images, organized in at least one stack, 
        similar to imgs
    """
    # compute means and stds
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
    imgs_norm = (imgs - imgs_mean) / imgs_std
    imgs_norm = np.nan_to_num(imgs_norm, nan=0.0)
    return imgs_norm

