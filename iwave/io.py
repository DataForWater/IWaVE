"""File I/O functions for IWaVE."""

import glob
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
from typing import Optional


def get_video(fn: str, start_frame: int = 0, end_frame: int = 4):
    """Read video frames from file

    Parameters
    ----------
    fn : str
        Path to the video file.
    start_frame : int, optional
        The starting frame number from which to begin extraction (default is 0).
    end_frame : int, optional
        The frame number at which to stop extraction (default is 4).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("This function needs cv2. Install iwave with pip install iwave[extra]")
    cap = cv2.VideoCapture(fn)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # retrieve images from start to end frame
    return np.stack(
        [cv2.cvtColor(cap.read()[-1], cv2.COLOR_BGR2GRAY) for _ in tqdm(range(end_frame - start_frame))]
    )


def get_imgs(path: str, wildcard: Optional[str] = None):
    # list all files in folder
    if wildcard:
        path = os.path.join(path, wildcard)

    fns = glob.glob(path)
    fns.sort()
    # read in memory as black-and-white
    return np.stack(
        [np.array(Image.open(fn).convert('L')) for fn in tqdm(fns)])






