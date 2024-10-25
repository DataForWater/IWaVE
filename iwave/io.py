"""File I/O functions for IWaVE."""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
from typing import Optional, Literal


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

    Returns
    -------
    np.ndarray
        3D array with images stacked over first dimension

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
    """Read frames from images.

    Parameters
    ----------
    path : str
        Directory path where image files are stored.
    wildcard : Optional[str], default: None
        Optional wildcard pattern to match specific files within the directory.

    Returns
    -------
    np.ndarray
        3D array with images stacked over first dimension
    """
    # list all files in folder
    if wildcard:
        path = os.path.join(path, wildcard)

    fns = glob.glob(path)
    fns.sort()
    # read in memory as black-and-white
    return np.stack(
        [np.array(Image.open(fn).convert('L')) for fn in tqdm(fns)])


def plot_spectrum(
    spectrum: np.ndarray,
    kt: np.ndarray,
    ky: np.ndarray,
    kx: np.ndarray,
    dim: Literal["x", "y", "time"],
    slice: int,
    ax=None,
    **kwargs
):
    """Plot 2D slice of spectrum of a window.

    Parameters
    ----------
    spectrum : np.ndarray
        The spectrum data array.
    kt : np.ndarray
        Array of wave numbers or time indices corresponding to the first (time) dimension.
    ky : np.ndarray
        Array of wave numbers corresponding to the second (x) dimension.
    kx : np.ndarray
        Array of wave numbers corresponding to the third (y) dimension.
    dim : Literal["x", "y", "time"]
        Dimension along which to slice the spectrum for plotting ("x", "y", or "time").
    slice : int
        Index along the specified dimension to slice the spectrum.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new Axes object is created.
    kwargs : dict
        Additional arguments passed to the plotting function `matplotlib.pyplot.pcolormesh`.

    Returns
    -------
    matplotlib.collections.QuadMesh
        matplotlib handle to mesh plot
    """
    if ax is None:
        ax = plt.axes()
    match dim:
        case "x":
            xlabel = "kt"
            ylabel = "ky"
            xvals = kt
            yvals = ky
            slice_vals = kx
            imvals = np.log(spectrum[:, :, slice])
        case "y":
            xlabel = "kt"
            ylabel = "kx"
            xvals = kt
            yvals = kx
            slice_vals = ky
            imvals = np.log(spectrum[:, slice, :])
        case "time":
            xlabel = "ky"
            ylabel = "kx"
            xvals = ky
            yvals = kx
            slice_vals = kt
            imvals = np.log(spectrum[slice, :])
        case _:
            raise ValueError(f"Invalid dimension: {dim}")
    p = ax.pcolormesh(yvals, xvals, imvals, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Spectrum {dim} wn: {round(slice_vals[slice], 3)}")
    return p