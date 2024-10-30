"""File I/O functions for IWaVE."""

import glob

import matplotlib.axes
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
    slice: Optional[int] = None,
    log: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
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
    slice : int, optional
        Index along the specified dimension to slice the spectrum. If not provided the middle index is used.
    log : bool, optional
        If True (default), spectrum is plotted on log scale.
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
            xlabel = "ky"
            ylabel = "kt"
            xvals = kt
            yvals = ky
            slice_vals = kx
            if slice is None:
                # define middle of dimension
                slice = spectrum.shape[-1] // 2
            imvals = spectrum[:, :, slice]
        case "y":
            xlabel = "kx"
            ylabel = "kt"
            xvals = kt
            yvals = kx
            slice_vals = ky
            if slice is None:
                # define middle of dimension
                slice = spectrum.shape[-2] // 2
            imvals = spectrum[:, slice, :]
        case "time":
            xlabel = "ky"
            ylabel = "kx"
            xvals = ky
            yvals = kx
            slice_vals = kt
            if slice is None:
                # define middle of dimension
                slice = spectrum.shape[-3] // 2
            imvals = spectrum[slice, :]
        case _:
            raise ValueError(f"Invalid dimension: {dim}")
    if log:
        imvals = np.log(imvals)
    p = ax.pcolormesh(yvals, xvals, imvals, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Spectrum {dim} wn: {round(slice_vals[slice], 3)}")
    return p


def plot_spectrum_fitted(
    spectrum: np.ndarray,
    kt_waves_theory: np.ndarray, 
    kt_advected_theory: np.ndarray, 
    kt: np.ndarray,
    ky: np.ndarray,
    kx: np.ndarray,
    dim: Literal["x", "y", "time"],
    slice: Optional[int] = None,
    log: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs
):
    """Plot 2D slice of spectrum of a window.

    Parameters
    ----------
    spectrum : np.ndarray
        The spectrum data array.
    kt_waves_theory : np.ndarray
        Theoretical frequency of gravity waves according to optimised velocity vector.
    kt_advected_theory : np.ndarray
        Theoretical frequency of turbulence-driven waves according to optimised velocity vector.
    kt : np.ndarray
        Array of wave numbers or time indices corresponding to the first (time) dimension.
    ky : np.ndarray
        Array of wave numbers corresponding to the second (x) dimension.
    kx : np.ndarray
        Array of wave numbers corresponding to the third (y) dimension.
    dim : Literal["x", "y", "time"]
        Dimension along which to slice the spectrum for plotting ("x", "y", or "time").
    slice : int, optional
        Index along the specified dimension to slice the spectrum. If not provided the middle index is used.
    log : bool, optional
        If True (default), spectrum is plotted on log scale.
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
            xlabel = "ky"
            ylabel = "kt"
            xvals = kt
            yvals = ky
            slice_vals = kx
            if slice is None:
                # define middle of dimension
                slice = spectrum.shape[-1] // 2
            imvals = spectrum[:, :, slice]
            linevals_waves = np.squeeze(kt_waves_theory[0,:,slice])
            linevals_advected = np.squeeze(kt_advected_theory[0,:,slice])
        case "y":
            xlabel = "kx"
            ylabel = "kt"
            xvals = kt
            yvals = kx
            slice_vals = ky
            if slice is None:
                # define middle of dimension
                slice = spectrum.shape[-2] // 2
            imvals = spectrum[:, slice, :]
            print(f"kt_waves dim = {kt_waves_theory.shape}")
            print(f"kt_adv dim = {kt_advected_theory.shape}")
            linevals_waves = np.squeeze(kt_waves_theory[0,slice,:])
            linevals_advected = np.squeeze(kt_advected_theory[0,slice,:])
        case "time":
            xlabel = "ky"
            ylabel = "kx"
            xvals = ky
            yvals = kx
            slice_vals = kt
            if slice is None:
                # define middle of dimension
                slice = spectrum.shape[-3] // 2
            imvals = spectrum[slice, :]
        case _:
            raise ValueError(f"Invalid dimension: {dim}")
    if log:
        imvals = np.log(imvals)
    p = ax.pcolormesh(yvals, xvals, imvals, **kwargs)
    ax.plot(yvals,linevals_waves, color='red')
    ax.plot(yvals,linevals_advected, color='red')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Spectrum {dim} wn: {round(slice_vals[slice], 3)}")
    ax.set_ylim([0, np.max(xvals)])
    return p