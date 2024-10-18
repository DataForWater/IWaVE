""" test fixtures """
from random import sample

from PIL import Image
import glob
import numpy as np
import os
import pytest

from iwave import sample_data


@pytest.fixture
def fn_video():
    return sample_data.get_sheaf_dataset()


@pytest.fixture
def imgs(fn_video):
    """ 4 selected frames from sample dataset, read with reader helper function.
    Result is [4 x n x m ] np.ndarray """
    return sample_data.get_frames(fn_video)
    # return np.stack([np.array(Image.open(fn)) for fn in fns_img])
    # concatenate


# @pytest.fixture
# def img_windows(fn_windows):
#     with open(fn_windows, "rb") as f:
#         windows = np.load(f)
#     return windows

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

