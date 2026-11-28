"""Read image and video files, and plot measured and predicted spectra."""

import glob
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
from typing import Optional, Literal

try:
    import cv2
    ENABLE_CV2 = True
except:
    ENABLE_CV2 = False


def get_video(fn: str, start_frame: int = 0, end_frame: int = 4, stride: int = 1):
    """Read selected video frames into a greyscale image array.

    fn is the video path. start_frame is included; end_frame is excluded.
    stride=1 keeps every frame; stride=2 keeps every other frame, and so on.
    Return images in (frame, y, x) order. This function does not return or adjust
    a frame rate; the caller must use the rate of the retained frames.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("This function needs cv2. Install iwave with pip install iwave[extra]")
    if os.path.isfile(fn) is False:
        raise FileNotFoundError(
            f"The provided file path {fn} does not exist. Please check the path and try again."
        )
    cap = cv2.VideoCapture(fn)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read the selected video frames.
    return np.stack(
        [cv2.cvtColor(cap.read()[-1][::stride, ::stride], cv2.COLOR_BGR2GRAY) for _ in tqdm(range(end_frame - start_frame))]
    )
    

def get_imgs(path: str, wildcard: Optional[str] = None):
    """Read greyscale image files from a directory.

    path is the directory and wildcard optionally restricts the filenames.
    Return the sorted image sequence in (frame, y, x) order.
    """
    # Find the image files to read.
    if wildcard:
        path = os.path.join(path, wildcard)

    fns = glob.glob(path)
    fns.sort()
    # Load each image in greyscale.
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
    """Plot one two-dimensional slice through a window's power spectrum.

    spectrum is in (frequency, ky, kx) order. kt is angular frequency in rad/s;
    ky and kx are spatial wavenumbers in rad/m. dim selects the dimension held
    fixed: 'time', 'y' or 'x'. slice selects its index, or the middle if omitted.
    log=True uses a logarithmic colour scale. ax selects a Matplotlib plot area;
    when omitted, create one. Forward kwargs to pcolormesh for colour and style.
    Return the Matplotlib QuadMesh object, which can be used for a colour bar.
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
                # Use the middle slice when no slice number was supplied.
                slice = spectrum.shape[-1] // 2
            imvals = spectrum[:, :, slice]
        case "y":
            xlabel = "kx"
            ylabel = "kt"
            xvals = kt
            yvals = kx
            slice_vals = ky
            if slice is None:
                # Use the middle slice when no slice number was supplied.
                slice = spectrum.shape[-2] // 2
            imvals = spectrum[:, slice, :]
        case "time":
            xlabel = "ky"
            ylabel = "kx"
            xvals = ky
            yvals = kx
            slice_vals = kt
            if slice is None:
                # Use the middle slice when no slice number was supplied.
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
    """Plot a spectrum slice together with the supplied predicted frequency lines.

    spectrum is in (frequency, ky, kx) order. kt_waves_theory and
    kt_advected_theory contain predictions for waves and flow-carried patterns.
    kt is angular frequency in rad/s; ky and kx are wavenumbers in rad/m.
    dim and slice choose the fixed dimension and its index, as in plot_spectrum.
    The middle slice is used if slice is omitted. log selects a logarithmic colour
    scale; ax chooses an existing plot area. Forward kwargs to pcolormesh.
    Return the Matplotlib QuadMesh object. This function does not fit parameters;
    the caller supplies the lines to draw.
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
                # Use the middle slice when no slice number was supplied.
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
                # Use the middle slice when no slice number was supplied.
                slice = spectrum.shape[-2] // 2
            imvals = spectrum[:, slice, :]
            linevals_waves = np.squeeze(kt_waves_theory[0,slice,:])
            linevals_advected = np.squeeze(kt_advected_theory[0,slice,:])
        case "time":
            xlabel = "ky"
            ylabel = "kx"
            xvals = ky
            yvals = kx
            slice_vals = kt
            if slice is None:
                # Use the middle slice when no slice number was supplied.
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