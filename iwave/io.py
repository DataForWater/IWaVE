"""Read images and video, plot results, and organise optional uncertainty checks.

Spatial plots use image pixels unless metres are requested. Conversion to
metres uses the supplied image resolution, not surveyed river coordinates.
This module also stores result objects and calculates how a fitting score
changes near an estimate. spectral.py contains the noise assumptions.
"""

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

# Plots of velocity and depth across the image.
# Plot existing results along one row or column of windows; do not fit again.
# A column follows y; a row follows x. Show vx, vy, optional depth, then quality.
from numbers import Integral


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a Boolean.')
    return bool(value)


def _shared_legend(figure, axes):
    """Place one shared legend beside the plots, not over the measurements.

    Collect repeated labels only once. Measure the legend so long labels or
    several overlaid results receive enough space without shrinking the plots
    unnecessarily.
    """
    handles, labels, seen = [], [], set()
    for ax in axes:
        # Remove old legends from individual panels if the axes have been reused.
        old = ax.get_legend()
        if old is not None:
            old.remove()
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    # Replace the legend created by this function, rather than adding another one.
    # This keeps repeated calls from drawing several legends in the same place.
    old = getattr(figure, '_iwave_transect_legend', None)
    if old is not None:
        old.remove()
    figure._iwave_transect_legend = None
    if not handles:
        figure.tight_layout()
        return
    legend = figure.legend(handles, labels, loc='upper right',
                           bbox_to_anchor=(1, 1), fontsize='small', frameon=False)
    figure._iwave_transect_legend = legend
    figure.canvas.draw()
    box = legend.get_window_extent()
    legend_width, legend_height = box.width / figure.dpi, box.height / figure.dpi
    width, height = figure.get_size_inches()
    # Leave at least 4.8 inches for the plots, plus a gap before the legend.
    # Make the figure taller if the legend needs more room.
    width = max(width, 4.8 + legend_width + .4)
    height = max(height, legend_height + .6)
    figure.set_size_inches(width, height, forward=True)
    panel_right = 1 - (legend_width + .3) / width
    figure.tight_layout(rect=(0, 0, panel_right, 1))


def plot_transect(iw, *, row=None, column=None, include_depth=None,
                   show_uncertainty=True, show_modes=True, distance_units='pixels',
                   axes=None, label=None, color=None):
    """Plot existing velocity, optional depth, and quality along a row or column.

    Parameters
    ----------
    row, column : nonnegative integer
        Supply exactly one window index, counted from zero, not a pixel position.
        For 13 rows and one column, column=0 selects all 13 windows; row=0 selects
        just the first window.
    include_depth : None or bool
        None includes depth only if it was fitted. False hides it. True requests
        it, but raises an error when depth was supplied rather than measured.
    show_uncertainty : bool
        Draw accepted lower/upper bounds for each parameter. A depth check can
        fail while velocity bounds remain usable. Missing bounds do not mean
        zero uncertainty.
    show_modes : bool
        Draw other solutions already found by fitting. Their own bounds have
        not been calculated, so do not attach the selected fit's bars to them.
    distance_units : 'pixels', 'm' or 'index'
        Choose image pixels (default), metres from image resolution, or window
        number. Image coordinates keep the analysed image's origin; they are
        not automatically converted to surveyed river positions.
    axes : sequence of Matplotlib axes, optional
        Reuse returned axes to compare another method on the same plots.
        Both calls must have the same number of panels: vx, vy, optional depth,
        and quality. A shared legend is placed to the right.
    label, color : optional
        Choose the name and colour of the results being added.

    Returns
    -------
    figure, axes
        The figure and a flat array of plot areas, in vx, vy, optional depth,
        quality order. Call plt.show() after adding all results.

    Reading the symbols
    -------------------
    Filled circles are successful fits with accepted bounds, or with bars hidden.
    Open circles are successful fits without accepted bounds. Crosses are failed
    fits. A red ring means another solution was competitive; no ring does not
    prove uniqueness. Other solutions use red diamonds when competitive, grey
    plus signs when clearly separated on the assumed noise scale, and grey
    triangles when the comparison could not be made.

    The bars describe noise effects near the chosen fit under the assumed model.
    They are not a complete, independently calibrated error budget. This plot
    uses existing results only; it does not run more fits or uncertainty checks.
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    show_uncertainty = _boolean(show_uncertainty, 'show_uncertainty')
    show_modes = _boolean(show_modes, 'show_modes')
    if (row is None) == (column is None):
        raise ValueError('Supply exactly one of row or column (window index).')
    if getattr(iw, 'vx', None) is None or getattr(iw, 'vy', None) is None:
        raise ValueError('Run velocimetry before plotting a transect.')
    vx, vy = np.asarray(iw.vx), np.asarray(iw.vy)
    if vx.ndim != 2 or vy.shape != vx.shape:
        raise ValueError('vx and vy must be matching two-dimensional fields.')
    shape = vx.shape
    along_column = column is not None
    number = column if along_column else row
    limit = shape[1] if along_column else shape[0]
    if isinstance(number, (bool, np.bool_)) or not isinstance(number, Integral):
        raise ValueError('row/column must be a nonnegative integer window index.')
    if not 0 <= number < limit:
        raise ValueError(f'Window index {number} is outside [0, {limit}).')
    selector = (slice(None), int(number)) if along_column else (int(number), slice(None))
    n = shape[0] if along_column else shape[1]
    coordinate_name = 'y' if along_column else 'x'
    if distance_units not in ('m', 'pixels', 'index'):
        raise ValueError("distance_units must be 'm', 'pixels' or 'index'.")
    if distance_units == 'index':
        position = np.arange(n)
        xlabel = f'{"Row" if along_column else "Column"} window index'
    else:
        position = np.asarray(getattr(iw, coordinate_name), dtype=float)
        if position.shape != (n,) or not np.isfinite(position).all():
            raise ValueError('Window-centre coordinates do not match the fitted field.')
        if distance_units == 'm':
            resolution = float(iw.resolution)
            if not np.isfinite(resolution) or resolution <= 0:
                raise ValueError('A positive resolution is required for metre coordinates.')
            position = position * resolution
        xlabel = f'{coordinate_name} [{distance_units}] (analysed-image coordinates)'

    depth_active = bool(getattr(iw, 'depth_estimated', False))
    if include_depth is None:
        include_depth = depth_active
    else:
        include_depth = _boolean(include_depth, 'include_depth')
        if include_depth and not depth_active:
            raise ValueError('Depth was supplied/fixed, not reconstructed; no depth panel is available.')
    names = ['vx', 'vy'] + (['depth'] if include_depth else []) + ['quality']
    values = {'vx': vx, 'vy': vy, 'quality': np.asarray(iw.quality)}
    if values['quality'].shape != shape:
        raise ValueError('Quality does not match the fitted field.')
    if include_depth:
        values['depth'] = np.asarray(iw.d)
        if values['depth'].shape != shape:
            raise ValueError('Depth does not match the velocity field.')
    status = getattr(getattr(iw, 'status', None), 'windows', None)
    if status is None or np.shape(status) != shape:
        raise ValueError('Per-window fit status is missing; run velocimetry first.')
    successful = np.asarray(status, dtype=bool)[selector]
    spectral = getattr(iw, 'spectral_uncertainty', None)
    modes = getattr(iw, 'mode_ambiguity', None)

    if axes is None:
        figure, axes = plt.subplots(len(names), 1, sharex=True,
                                    figsize=(8, 3 * len(names)), squeeze=False)
        axes = axes.ravel()
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) != len(names) or not all(isinstance(ax, Axes) for ax in axes):
            raise ValueError(f'Supply exactly {len(names)} Matplotlib axes for this fit.')
        figure = axes[0].figure
        if not all(ax.figure is figure for ax in axes):
            raise ValueError('All supplied axes must belong to the same figure.')
    if color is None:
        color = axes[0]._get_lines.get_next_color()
    prefix = f'{label}: ' if label else ''

    for ax, name in zip(axes, names):
        # Show each legend label only once, even when it appears at many windows.
        seen = set(ax.get_legend_handles_labels()[1])
        def legend(text):
            text = prefix + text
            if text in seen:
                return None
            seen.add(text)
            return text

        y = np.asarray(values[name][selector], dtype=float)
        finite = np.isfinite(y)
        ok = successful & finite
        # Quality describes agreement with the fitted spectrum; it is not velocity or depth.
        # Plot it along the same line of windows, without uncertainty bars.
        # Do not assign quality scores to other solutions unless they were calculated.
        if name == 'quality':
            ax.plot(position, np.where(ok, y, np.nan), '-o', color=color,
                    label=legend('selected fit'), ms=4)
            failed = ~successful & finite
            if failed.any():
                ax.plot(position[failed], y[failed], 'x', color=color,
                        label=legend('failed fit'), ms=7)
            ax.set_ylabel('Quality')
            ax.grid(alpha=.25)
            continue
        endpoints = np.full((n, 2), np.nan)
        if show_uncertainty and spectral is not None:
            field_bounds = np.asarray(getattr(spectral, name + '_bounds'))
            if field_bounds.shape != (*shape, 2):
                raise ValueError(f'{name}_bounds does not match the field.')
            endpoints = field_bounds[selector]
        lower, upper = y - endpoints[:, 0], endpoints[:, 1] - y
        has_bounds = (ok & np.isfinite(endpoints).all(axis=1)
                      & (lower >= 0) & (upper >= 0))
        # Break the line at a failed fit so it is not presented as a reliable estimate.
        ax.plot(position, np.where(ok, y, np.nan), '-', color=color, alpha=.55)
        filled = ok if not show_uncertainty else has_bounds
        if filled.any():
            ax.plot(position[filled], y[filled], 'o', color=color,
                    label=legend('selected fit'), ms=4)
        missing = ok & ~filled
        if missing.any():
            ax.plot(position[missing], y[missing], 'o', color=color,
                    markerfacecolor='none', label=legend('bounds unavailable/rejected'), ms=5)
        failed = ~successful & finite
        if failed.any():
            ax.plot(position[failed], y[failed], 'x', color=color,
                    label=legend('failed fit'), ms=7)
        if has_bounds.any():
            ax.errorbar(position[has_bounds], y[has_bounds],
                        yerr=np.vstack([lower[has_bounds], upper[has_bounds]]),
                        fmt='none', ecolor=color, capsize=3,
                        label=legend('accepted local bounds'))
        if show_modes and modes is not None:
            flags = modes.ambiguous_solution[selector]
            competitive = np.array([v is True for v in flags]) & finite
            if competitive.any():
                ax.plot(position[competitive], y[competitive], 'o', ms=10,
                        markerfacecolor='none', markeredgecolor='red',
                        label=legend('competitive alternative exists'))
            for j, alternatives in enumerate(modes.additional_estimates[selector]):
                for alternative in alternatives:
                    value = getattr(alternative, name)
                    if not np.isfinite(value):
                        continue
                    flag = alternative.ambiguous_solution
                    if flag is True:
                        marker, edge, text = 'D', 'red', 'competitive rival (no bounds)'
                    elif flag is False:
                        marker, edge, text = '+', '0.5', 'separated rival (no bounds)'
                    else:
                        marker, edge, text = '^', '0.5', 'unassessed rival (no bounds)'
                    ax.plot(position[j], value, marker=marker, linestyle='none',
                            color=edge, markerfacecolor='none', ms=6,
                            label=legend(text))
        ax.set_ylabel({'vx': 'Velocity x [m/s]', 'vy': 'Velocity y [m/s]',
                       'depth': 'Reconstructed depth [m]'}[name])
        ax.grid(alpha=.25)
    axes[-1].set_xlabel(xlabel)
    axes[0].set_title(f'{"Column" if along_column else "Row"} {number} transect')
    _shared_legend(figure, axes)
    return figure, axes


def img_windows_norm(img_windows):
    """Adjust each pixel using its mean and standard deviation over time.

    Replace NaN (not-a-number) and infinite results with zero. For example,
    a constant pixel gives 0/0 before this cleanup.
    This utility was moved from the earlier plots.py example.
    """
    import numpy as np
    with np.errstate(divide='ignore', invalid='ignore'):
        values = (img_windows - img_windows.mean(axis=0)) / img_windows.std(axis=0)
    return np.where(np.isfinite(values), values, 0.)


def plot_velocimetry(iw, ax=None, *, validate_vectors=True,
                     distance_units='pixels', color='C0',
                     ambiguous_color='tab:red', **kwargs):
    """Plot the fitted velocities as arrows at window centres.

    Positions use pixels by default; distance_units='m' converts them using image
    resolution. Velocity components remain in m/s in either case.
    validate_vectors=True omits failed fits; False also shows finite failed fits.
    Always omit undefined velocities. Red marks a known competing solution;
    unknown competition uses the normal colour, not a warning colour.

    Return a Matplotlib Quiver object. Standard arrow options such as scale,
    width and angles are passed through. Base and ambiguity colours must each
    be a single colour. This function displays existing results without refitting.
    """
    from matplotlib.colors import to_rgba
    validate_vectors = _boolean(validate_vectors, 'validate_vectors')
    if distance_units not in ('pixels', 'm'):
        raise ValueError("distance_units must be 'pixels' or 'm'.")
    if getattr(iw, 'vx', None) is None or getattr(iw, 'vy', None) is None:
        raise ValueError('Run velocimetry before plotting vectors.')
    vx, vy = np.asarray(iw.vx), np.asarray(iw.vy)
    x, y = np.asarray(iw.x, float), np.asarray(iw.y, float)
    if vx.ndim != 2 or vy.shape != vx.shape or vx.shape != (len(y), len(x)):
        raise ValueError('Velocity fields and window coordinates do not match.')
    if distance_units == 'm':
        resolution = float(iw.resolution)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('A positive resolution is required for metre coordinates.')
        x, y = x * resolution, y * resolution
    X, Y = np.meshgrid(x, y)
    keep = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(X) & np.isfinite(Y)
    if validate_vectors:
        status = getattr(getattr(iw, 'status', None), 'windows', None)
        if status is None or np.shape(status) != vx.shape:
            raise ValueError('Per-window fit status is missing; run velocimetry first.')
        keep &= np.asarray(status, bool)
    colours = np.tile(to_rgba(color), (vx.size, 1)).reshape((*vx.shape, 4))
    modes = getattr(iw, 'mode_ambiguity', None)
    if modes is not None:
        if modes.ambiguous_solution.shape != vx.shape:
            raise ValueError('Ambiguity flags do not match the velocity field.')
        known_rival = np.array([flag is True for flag in modes.ambiguous_solution.ravel()]).reshape(vx.shape)
        colours[known_rival] = to_rgba(ambiguous_color)
    if ax is None:
        _, ax = plt.subplots()
    if not keep.any():
        # Do not ask Matplotlib to choose arrow sizes when there are no arrows.
        kwargs.setdefault('scale', 1.)
    arrows = ax.quiver(X[keep], Y[keep], vx[keep], vy[keep],
                       color=colours[keep], **kwargs)
    ax.set_xlabel(f'x [{distance_units}] (analysed-image coordinates)')
    ax.set_ylabel(f'y [{distance_units}] (analysed-image coordinates)')
    return arrows


# ================================================================
# Result objects: parameter bounds, success flags, and other solutions found.
# Use [row, column] to select a window; bounds also have [lower, upper].
# NaN means no bound is available. More information is kept in .details.
# ================================================================
from dataclasses import dataclass, field


@dataclass(eq=False)
class AlternativeEstimate:
    """Store another solution reached during the existing search.

    Its parameter values are known, but its own uncertainty bounds have not
    been calculated. Those bounds and their validity flags therefore stay None.
    ambiguous_solution is True if its score is competitive with the chosen fit,
    False if separated on the approximate noise scale, and None if no comparison
    was possible. These flags are not probabilities.
    """
    vx: float
    vy: float
    depth: float
    ambiguous_solution: Optional[bool] = None
    vx_bounds: Optional[np.ndarray] = None
    vy_bounds: Optional[np.ndarray] = None
    depth_bounds: Optional[np.ndarray] = None
    valid: Optional[bool] = None
    details: dict = field(default_factory=dict, repr=False)


class SpectralUncertainty:
    """Store accepted bounds and flags for the selected fit at each window.

    vx_bounds, vy_bounds and depth_bounds contain [lower, upper] endpoints in
    m/s, m/s and metres. Their shape is (rows, columns, 2). Depth bounds can be
    uneven around the estimate; no depth bounds are calculated for supplied depth.
    valid is True only if every fitted parameter passed its bound checks.
    Velocity bounds can still be usable if the depth check failed.
    ambiguous_solution copies the separate competition flag: True, False or None.
    None means unknown, not False. details holds the full dictionaries, including
    error estimates, how those errors vary together (covariance), rejected bounds,
    noise assumptions and reasons for rejection.

    These bounds describe the chosen fit under the assumed spectral noise.
    They do not cover all physical errors or a jump to another solution.
    """
    vx_bounds: np.ndarray
    vy_bounds: np.ndarray
    depth_bounds: np.ndarray
    valid: np.ndarray
    ambiguous_solution: np.ndarray
    details: np.ndarray

    def __init__(self, shape):
        self.vx_bounds = np.full((*shape, 2), np.nan)
        self.vy_bounds = np.full((*shape, 2), np.nan)
        self.depth_bounds = np.full((*shape, 2), np.nan)
        self.valid = np.zeros(shape, dtype=bool)
        self.ambiguous_solution = np.full(shape, None, dtype=object)
        self.details = np.full(shape, None, dtype=object)

    def __repr__(self):
        return (f'SpectralUncertainty(shape={self.valid.shape}, '
                f'fully_valid={np.count_nonzero(self.valid)}/{self.valid.size}; '
                'use vx_bounds, vy_bounds, depth_bounds and details)')

    def _set_window(self, index, record, ambiguous_solution):
        """Store one window's bounds and checks while Iwave collects the full result field."""
        self.details[index] = record
        self.valid[index] = record['valid']
        self.ambiguous_solution[index] = ambiguous_solution
        for name in ('vx', 'vy', 'depth'):
            endpoints = record['accepted_intervals'][name]
            if endpoints is not None:
                getattr(self, name + '_bounds')[index] = endpoints


class ModeAmbiguity:
    """Store other solutions found and whether they compete with the chosen fit.

    valid says whether the comparison was complete and usable. It does not say
    whether the velocity fit or its uncertainty bounds were valid.
    ambiguous_solution is True when a competitive alternative was found, False
    when none was found among the eligible solutions examined, and None when the
    comparison was unavailable. False does not prove no other solution exists.

    additional_estimates contains a separate list of AlternativeEstimate objects
    for each [row, column]. The chosen solution is not repeated there; it remains
    in iw.vx, iw.vy and iw.d. details stores the comparisons, exclusions and limits.
    An incomplete comparison can have ambiguous_solution=True but valid=False:
    a competitor was found, but not every comparison could be checked.
    """
    valid: np.ndarray
    ambiguous_solution: np.ndarray
    additional_estimates: np.ndarray
    details: np.ndarray

    def __init__(self, shape):
        self.valid = np.zeros(shape, dtype=bool)
        self.ambiguous_solution = np.full(shape, None, dtype=object)
        self.additional_estimates = np.empty(shape, dtype=object)
        # For example, explain why a single starting guess cannot reveal other solutions.
        self.message = None
        self.details = np.full(shape, None, dtype=object)
        for index in np.ndindex(shape):
            self.additional_estimates[index] = []

    def __repr__(self):
        flags = self.ambiguous_solution.ravel()
        return (f'ModeAmbiguity(shape={self.valid.shape}, '
                f'competitive={sum(v is True for v in flags)}, '
                f'unassessed={sum(v is None for v in flags)}; '
                'use additional_estimates and details)')

    def _set_window(self, index, record):
        if self.message is None and record.get('message'):
            self.message = record['message']
        self.details[index] = record
        self.valid[index] = record['valid']
        flag = record['ambiguous']
        self.ambiguous_solution[index] = None if flag is None else bool(flag)
        comparisons = {item['mode_index']: item for item in record['comparisons']}
        for mode in record['modes']:
            if mode['selected']:
                continue
            comparison = comparisons.get(mode['mode_index'], {})
            flag = comparison.get('competitive')
            self.additional_estimates[index].append(AlternativeEstimate(
                vx=float(mode['velocity'][1]), vy=float(mode['velocity'][0]),
                depth=float(mode['depth']),
                ambiguous_solution=None if flag is None else bool(flag),
                details={'mode': mode, 'comparison': comparison},
            ))


# ================================================================
# Estimate how the fitting score changes near the chosen result.
# spectral.py decides which uncertainty bounds to accept and how to model noise.
# ================================================================
from copy import deepcopy
from itertools import combinations
from numbers import Real
from time import perf_counter

CURVATURE_DEFAULTS=dict(fd_step=.05,stability_rtol=.25,curvature_rtol=1e-7,
              curvature_atol=1e-10,parameter_scales=None,kernel_truncate=5.)


def resolve_curvature_options(options=None):
    if options is not None and not isinstance(options,dict):raise ValueError('Curvature options must be a dictionary.')
    if set(options or {})-set(CURVATURE_DEFAULTS):raise ValueError('Unknown curvature options.')
    out=deepcopy(CURVATURE_DEFAULTS);out.update(deepcopy(options or {}))
    for key in set(CURVATURE_DEFAULTS)-{'parameter_scales'}:
        v=out[key]
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) or not np.isfinite(v) or v<=0:
            raise ValueError('Invalid curvature option '+key)
        out[key]=float(v)
    if out['fd_step']>.5 or out['curvature_rtol']>=1 or out['kernel_truncate']<5:
        raise ValueError('Require fd_step <= .5, curvature_rtol < 1 and kernel_truncate >= 5.')
    if out['parameter_scales'] is not None:
        a=np.asarray(out['parameter_scales'])
        if a.shape not in ((2,),(3,)) or not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a) or not np.isfinite(a).all() or np.any(a<=0):
            raise ValueError('parameter_scales must contain two or three positive finite numbers.')
        out['parameter_scales']=a.astype(float).tolist()
    return out


def _record(parameters,scales,options):
    p=len(parameters)
    return dict(kind='objective_curvature',valid=False,flags=[],
        parameter_names=['vy','vx']+(['log_depth'] if p==3 else []),
        parameter_units=['m/s','m/s']+(['log(d/1m)'] if p==3 else []),
        parameters=np.asarray(parameters).tolist(),parameter_scales=np.asarray(scales).tolist(),
        depth_estimated=p==3,depth_status='estimated' if p==3 else 'fixed_not_estimated',
        hessian=None,scaled_hessian=None,inverse_curvature=None,eigenvalues=None,
        rank=None,condition_number=None,gradient=None,curvature_step_error=None,
        difference_steps=None,evaluations=0,options=deepcopy(options))


def _curvature(log_objective,parameters,scales,feasible,options):
    """Estimate the score's bending near the fit by small parameter changes.

    Use changes on both sides of each parameter and for parameter pairs, giving
    the matrix of second derivatives (Hessian). Scale the parameters to make
    steps comparable. Never test a point outside the allowed fitting limits.
    """
    theta=np.asarray(parameters,float);scales=np.asarray(scales,float);p=len(theta)
    out=_record(theta,scales,options);flags=out['flags'];cache={}
    def sample(q):
        key=tuple(np.asarray(q,float))
        if key not in cache:cache[key]=float(log_objective(theta+scales*np.asarray(q)))
        return cache[key]
    def finish():
        out['evaluations']=len(cache);out['flags']=list(dict.fromkeys(flags));out['valid']=not out['flags']
        return out
    if not feasible(theta):flags.append('outside_search_constraints');return finish()
    f0=sample(np.zeros(p))
    if not np.isfinite(f0):flags.append('invalid_overlap');return finish()
    out['centre_log_objective']=f0;h=np.full(p,options['fd_step'])
    def stencil(step):
        points=[]
        for i in range(p):
            for sign in (-1,1):
                q=np.zeros(p);q[i]=sign*step[i];points.append(q)
        for i,j in combinations(range(p),2):
            for si in (-1,1):
                for sj in (-1,1):
                    q=np.zeros(p);q[i]=si*step[i];q[j]=sj*step[j];points.append(q)
        return points
    # Reduce the parameter changes until all required test points are inside the limits.
    # Do not replace this symmetric calculation with one that uses only one side.
    for _ in range(7):
        if all(feasible(theta+scales*q) for q in stencil(h)):break
        h*=.5
    else:flags.append('boundary_stencil_unavailable');return finish()
    out['difference_steps']=(h*scales).tolist()
    def differentiate(step):
        H=np.zeros((p,p));g=np.zeros(p)
        for i in range(p):
            q=np.zeros(p);q[i]=step[i];plus,minus=sample(q)-f0,sample(-q)-f0
            g[i]=(plus-minus)/(2*step[i]);H[i,i]=(plus+minus)/step[i]**2
        for i,j in combinations(range(p),2):
            values=[]
            for si,sj in ((1,1),(1,-1),(-1,1),(-1,-1)):
                q=np.zeros(p);q[i]=si*step[i];q[j]=sj*step[j];values.append(sample(q)-f0)
            H[i,j]=H[j,i]=(values[0]-values[1]-values[2]+values[3])/(4*step[i]*step[j])
        return H,g
    # Repeat with two change sizes to check stability; use the smaller changes to form H.
    # Remember repeated test points only until this calculation is finished.
    coarse,_=differentiate(h);H,g=differentiate(h*.5)
    if not np.isfinite(H).all() or not np.isfinite(coarse).all() or not np.isfinite(g).all():flags.append('invalid_stencil');return finish()
    out['hessian']=(H/scales[:,None]/scales[None,:]).tolist();out['scaled_hessian']=H.tolist();out['gradient']=(g/scales).tolist()
    values,V=np.linalg.eigh(H);cutoff=max(options['curvature_atol'],options['curvature_rtol']*float(np.max(np.abs(values))))
    # Check whether the score constrains each combination of scaled parameters.
    # A flat or downward-bending direction means an unresolved bound, not zero error.
    positive=values>cutoff;rank=int(np.count_nonzero(positive))
    out.update(eigenvalues=values.tolist(),eigenvectors_scaled=V.tolist(),rank=rank,eigenvalue_cutoff=cutoff)
    if np.any(values < -cutoff):flags.append('negative_curvature');return finish()
    if rank<p:flags.append('weak_curvature')
    if not rank:return finish()
    vp,lam=V[:,positive],values[positive];W=vp/np.sqrt(lam)
    error=float(np.linalg.norm(W.T@(coarse-H)@W,2));out['curvature_step_error']=error
    if error>options['stability_rtol']:flags.append('unstable_curvature')
    inv=(vp/lam)@vp.T
    out['condition_number']=float(lam.max()/lam.min()) if rank==p else None
    out['inverse_curvature']=(inv*scales[:,None]*scales[None,:]).tolist() if rank==p else None
    return finish()


def _constraints(bounds,search_constraints,depth_estimated):
    p=3 if depth_estimated else 2;box=np.tile([-np.inf,np.inf],(p,1))
    if bounds is not None:
        b=np.asarray(bounds,float)
        if b.shape!=(3,2) or not np.isfinite(b).all() or np.any(b[:,0]>b[:,1]) or b[2,0]<=0:
            raise ValueError('bounds must be finite ordered (vy,vx,depth) intervals with positive depth.')
        box[:2]=b[:2]
        if depth_estimated:box[2]=np.log(b[2])
    disk=None
    if search_constraints is not None:
        if not isinstance(search_constraints,dict):raise ValueError('search_constraints must be a dictionary.')
        if search_constraints.get('velocity_bounds') is not None:
            b=np.asarray(search_constraints['velocity_bounds'],float)
            if b.shape!=(2,2) or not np.isfinite(b).all() or np.any(b[:,0]>b[:,1]):raise ValueError('Invalid search velocity bounds.')
            box[:2,0]=np.maximum(box[:2,0],b[:,0]);box[:2,1]=np.minimum(box[:2,1],b[:,1])
        if depth_estimated and search_constraints.get('log_depth_bounds') is not None:
            b=np.asarray(search_constraints['log_depth_bounds'],float)
            if b.shape!=(2,) or not np.isfinite(b).all() or b[0]>b[1]:raise ValueError('Invalid log-depth bounds.')
            box[2]=[max(box[2,0],b[0]),min(box[2,1],b[1])]
        if search_constraints.get('velocity_disk') is not None:
            d=search_constraints['velocity_disk'];centre=np.asarray(d['centre_vy_vx'],float);radius=float(d['radius'])
            if centre.shape!=(2,) or not np.isfinite(centre).all() or not np.isfinite(radius) or radius<=0:raise ValueError('Invalid velocity disk.')
            disk=(centre,radius)
    def feasible(theta):
        return bool(np.isfinite(theta).all() and np.all(theta>=box[:,0]) and np.all(theta<=box[:,1]) and (disk is None or np.linalg.norm(theta[:2]-disk[0])<=disk[1]))
    return box,disk,feasible


def _auto_scales(evaluator,velocity,depth,free,width):
    from iwave import dispersion
    speed_scale=width/max(float(np.sqrt(np.mean(evaluator.k**2))),1.);scales=[speed_scale,speed_scale]
    if free:
        eta=1e-3;centres=[];a=evaluator.y*velocity[0]+evaluator.x*velocity[1]
        for sign in (-1,1):
            d=depth*np.exp(sign*eta);beta=dispersion.beta_calc(evaluator.k,d,evaluator.alpha);intrinsic=dispersion.omega_i_calc(evaluator.k,d)
            centres.append((1-beta)*a+np.hypot(beta*a,intrinsic))
        sensitivity=float(np.sqrt(np.mean(((centres[1]-centres[0])/(2*eta))**2)))
        scales.append(min(1.,width/max(sensitivity,width)) if np.isfinite(sensitivity) else 1.)
    return np.asarray(scales)


def estimate_objective_curvature(measured_spectrum,velocity,depth,depth_estimated,
        window_dims,res,fps,vel_indx,gravity_waves_switch=True,turbulence_switch=True,
        gauss_width=1.,bounds=None,search_constraints=None,options=None):
    """Calculate how sharply log-overlap changes around the selected fit.

    Use small parameter changes rather than an analytic derivative. Return
    curvature information, not uncertainty from an arbitrary allowed score loss.
    A separate noise model is needed to turn this information into bounds.
    """
    # Import here so io.py and spectral.py can call each other without an import loop.
    from iwave import spectral
    from iwave.spectral import SpectralOverlapObjective
    started=perf_counter();opts=resolve_curvature_options(options)
    if not isinstance(depth_estimated,(bool,np.bool_)):raise ValueError('depth_estimated must be Boolean.')
    if not np.isfinite(gauss_width) or gauss_width<=0:raise ValueError('gauss_width must be positive.')
    if not gravity_waves_switch and not turbulence_switch:raise ValueError('At least one model branch is required.')
    velocity=np.asarray(velocity,float)
    if velocity.shape!=(2,):raise ValueError('velocity must contain (vy,vx).')
    p=3 if depth_estimated else 2
    if not np.isfinite(velocity).all() or not np.isfinite(depth) or depth<=0:
        out=_record([*velocity,*([np.nan] if depth_estimated else [])],np.ones(p),opts)
        out['flags']=['invalid_fit'];out['elapsed_seconds']=perf_counter()-started;return out
    theta=np.array([*velocity,*([np.log(depth)] if depth_estimated else [])]);box,disk,feasible=_constraints(bounds,search_constraints,depth_estimated)
    axes=spectral.wave_numbers(window_dims,res,fps)
    evaluator=SpectralOverlapObjective(measured_spectrum,axes,depth,vel_indx,gravity_waves_switch,turbulence_switch,
        dict(gauss_width=gauss_width,kernel_truncate=opts['kernel_truncate']))
    scales=opts['parameter_scales']
    if scales is None:scales=_auto_scales(evaluator,velocity,depth,depth_estimated,gauss_width)
    elif len(scales)<p:raise ValueError('Joint depth curvature requires three parameter_scales.')
    else:scales=np.asarray(scales[:p])
    def log_objective(parameters):
        score=-evaluator(parameters)
        return -np.log(score) if np.isfinite(score) and score>0 else np.inf
    out=_curvature(log_objective,theta,scales,feasible,opts)
    out.update(evaluations=evaluator.calls,depth=float(depth),gauss_width=float(gauss_width),search_bounds=box.tolist(),
        velocity_disk=None if disk is None else dict(centre_vy_vx=disk[0].tolist(),radius=disk[1]),elapsed_seconds=perf_counter()-started)
    return out
