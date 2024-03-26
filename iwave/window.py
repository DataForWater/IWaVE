import numpy as np
from typing import Tuple

# def sliding_window_array(
#     image: np.ndarray,
#     window_size: Tuple[int, int] = (64, 64),
#     overlap: Tuple[int, int] = (32, 32),
# ) -> np.ndarray:
#
#     x, y = get_rect_coordinates(
#         image.shape,
#         window_size,
#         overlap,
#         center_on_field=False
#     )
#     x = (x - window_size[1] // 2).astype(int)
#     y = (y - window_size[0] // 2).astype(int)
#     x, y = np.reshape(x, (-1, 1, 1)), np.reshape(y, (-1, 1, 1))
#
#     win_x, win_y = np.meshgrid(np.arange(0, window_size[1]),
#                                np.arange(0, window_size[0]))
#     win_x = win_x[np.newaxis, :, :] + x
#     win_y = win_y[np.newaxis, :, :] + y
#     windows = image[win_y, win_x]
#
#     return windows
#


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
    dim_size,
    window_size,
    overlap,
    center_on_field=False,
):

    coords = np.arange(dim_size) * (window_size - overlap) + (window_size) / 2.0
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

# def get_rect_coordinates(
#     image_size: Tuple[int, int],
#     window_size: Union[int, Tuple[int, int]],
#     overlap: Union[int, Tuple[int, int]],
#     center_on_field: bool = False,
# ):
#     if isinstance(window_size, int):
#         window_size = (window_size, window_size)
#     if isinstance(overlap, int):
#         overlap = (overlap, overlap)
#
#     # @alexlib why the center_on_field is False?
#     # todo: test True as well
#     _, y = get_coordinates(image_size, window_size[0], overlap[0],
#                            center_on_field=center_on_field)
#     x, _ = get_coordinates(image_size, window_size[1], overlap[1],
#                            center_on_field=center_on_field)
#
#     X, Y = np.meshgrid(x[0, :], y[:, 0])
#
#     return (X, Y)