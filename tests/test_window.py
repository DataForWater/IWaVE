""" tests for window manipulations """
import numpy as np

from iwave import window


def test_get_axis_shape(imgs):
    # get the last dimension (x-axis), assert if it fits
    dim_size = imgs.shape[-1]

    x_shape = window.get_axis_shape(
        dim_size=dim_size,
        window_size=64,
        overlap=32,
    )
    assert(x_shape == 11)


def test_get_array_shape(imgs):
    # get last two dimensions, assert numbers in returned dims
    dim_sizes = imgs.shape[-2:]
    xy_shape = window.get_array_shape(
        dim_sizes=dim_sizes,
        window_sizes=(64, 64),
        overlaps=(32, 32)
    )
    assert(xy_shape == (11, 11))


def test_get_axis_coords(imgs):
    dim_size = imgs.shape[-1]
    coords = window.get_axis_coords(
        dim_size,
        64,
        32,
    )
    assert(np.allclose(np.array(coords[0:4]), np.array([32., 64., 96., 128.])))




