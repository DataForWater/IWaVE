"""Check window positions, image extraction, and brightness normalisation."""
import numpy as np
import pytest

from iwave import window, LazyWindowArray


def test_get_axis_shape(imgs):
    # Check how many windows fit along the image x direction.
    dim_size = imgs.shape[-1]

    x_shape = window.get_axis_shape(
        dim_size=dim_size,
        window_size=64,
        overlap=32,
    )
    assert(x_shape == 10)


def test_get_array_shape(imgs):
    # Check the number of window rows and columns.
    dim_sizes = imgs.shape[-2:]
    xy_shape = window.get_array_shape(
        dim_sizes=dim_sizes,
        window_sizes=(64, 64),
        overlaps=(32, 32)
    )
    assert(xy_shape == (23, 10))


def test_get_axis_coords(imgs):
    dim_size = imgs.shape[-1]
    coords = window.get_axis_coords(
        dim_size,
        64,
        32,
    )
    assert(len(coords)==10)
    assert(np.allclose(np.array(coords[0:4]), np.array([32., 64., 96., 128.])))


def test_get_rect_coordinates(imgs):
    x, y = window.get_rect_coordinates(
        dim_sizes=imgs.shape[-2:],
        window_sizes=(64, 64),
        overlap=(32, 32),
    )
    # Check the first group of window positions.
    assert len(x) == 10
    assert len(y) == 23
    assert np.allclose(x[0:2], np.array([32., 64.]))
    assert np.allclose(y[0:2], np.array([32., 64.]))


def test_sliding_window_array(imgs):
    win_x, win_y = window.sliding_window_idx(imgs[0])
    img_wins = window.sliding_window_array(
        imgs[0],
        win_x,
        win_y
    )
    assert img_wins.shape == (23*10, 64, 64)


@pytest.mark.parametrize(
    ("swap_time_dim", "test_dims"),
    [
        (False, (4, 23*10, 64, 64)),
        (True, (23*10, 4, 64, 64))
    ]
)
def test_multi_sliding_window_array(imgs, swap_time_dim, test_dims):
    # Find the x/y pixel positions belonging to each window.
    win_x, win_y = window.sliding_window_idx(imgs[0])
    # Extract those positions from every image.
    window_stack = window.multi_sliding_window_array(
        imgs,
        win_x,
        win_y,
        swap_time_dim=swap_time_dim
    )
    assert(window_stack.shape == test_dims)


def test_normalize(img_windows):
    img_norm = window.normalize(img_windows, mode="xy")
    # Check that normalisation leaves the array dimensions unchanged.
    assert img_windows.shape == img_norm.shape
    # Check that the selected mean is zero and standard deviation is one.
    assert np.isclose(img_norm[0][0].std(), 1.)
    assert np.isclose(img_norm[0][0].mean(), 0.)
    # Also check normalisation over time.
    img_norm = window.normalize(img_windows, mode="time")
    # Check the mean and standard deviation of one selected frame.
    assert np.isclose(img_norm[1, :, 1, 1].std(), 1.)
    assert np.isclose(img_norm[1, :, 1, 1].mean(), 0.)


def test_lazy_windows_match_original_extraction():
    """Check that faster window preparation preserves values and does not change source images."""
    imgs = np.random.default_rng(3).integers(0, 255, (24, 12, 16), dtype=np.uint8)
    original = imgs.copy()
    imgs.setflags(write=False)
    win_x, win_y = window.sliding_window_idx(imgs[0], (6, 8), (0, 0))
    for mode in ('time', 'xy', None):
        lazy = LazyWindowArray(imgs, window.multi_sliding_window_array, win_x, win_y, mode)
        expected = window.multi_sliding_window_array(
            imgs, win_x[:2], win_y[:2], swap_time_dim=True,
        ).astype(float)
        if mode is not None:
            expected = window.normalize(expected, mode)
        actual = lazy[:2]
        assert np.allclose(actual, expected)
        assert actual.flags.c_contiguous
        assert not np.shares_memory(actual, imgs)
    assert np.array_equal(imgs, original)
