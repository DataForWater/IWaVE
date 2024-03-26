""" test fixtures """

from PIL import Image
import glob
import numpy as np
import os
from pytest import fixture

@fixture
def path_img():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "sheaf",
    )


@fixture
def fns_img(path_img):
    fns = glob.glob(
        os.path.join(
            path_img,
            "*.jpg"
        )
    )
    fns.sort()
    return fns

@fixture
def fn_windows(path_img):
    return os.path.join(
        path_img,
        "window_sample.asc"
    )

@fixture
def imgs(fns_img):
    """ 4 selected frames from sample dataset, read with reader helper function.
    Result is [4 x n x m ] np.ndarray """
    return np.stack([np.array(Image.open(fn)) for fn in fns_img])
    # concatenate

@fixture
def img_windows(fn_windows):
    """
    100 timestamps for 4 (64x64) windows
    :return: 
    """

