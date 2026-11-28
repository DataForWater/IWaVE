"""Shared inputs and helper functions for the tests."""
import numpy as np
import pytest

from iwave import sample_data, io


@pytest.fixture
def fn_video():
    return sample_data.get_sheaf_dataset()


@pytest.fixture
def imgs(fn_video):
    """Read four example frames and return an array in (frame, y, x) order."""
    return io.get_video(fn_video)



@pytest.fixture
def img_windows():
    fn_windows = sample_data.get_sheaf_windows()
    with open(fn_windows, "rb") as f:
        windows = np.load(f)
    return windows


@pytest.fixture
def img_windows_norm(img_windows):
    img_windows = img_windows - img_windows.mean(axis=0)
    img_windows = img_windows / img_windows.std(axis=0)
    img_windows[np.isinf(img_windows)] = 0
    img_windows[np.isnan(img_windows)] = 0
    return img_windows

