"""Compare strategies using native IWaVE results and plot_transect.

Requires the iwave-consolidated package update. No uncertainty wrapper or
conversion is used: bounds and flags are available directly on iw.

Edit VIDEO_PATH, then run in your usual IWaVE environment. Window COLUMN zero
contains all thirteen windows in this crop. Because depth=0 below requests
reconstruction, plot_transect automatically adds a depth panel. A supplied,
fixed depth produces velocity and quality panels, without a depth panel.

All fits finish BEFORE any figures are created. This avoids mixing Tk GUI
objects with multiprocessing cleanup. Small native result snapshots are drawn
afterwards; the video itself is NOT copied. Example after running interactively:
    iw.spectral_uncertainty.vx_bounds       # latest strategy
    results['HNM'].spectral_uncertainty.vx_bounds
    results['HNM'].mode_ambiguity.additional_estimates[iy, ix]
"""
from copy import deepcopy
import multiprocessing
from time import perf_counter
from types import SimpleNamespace

# A Windows spawned worker imports this script again. Workers must not use a
# GUI backend; only the original main process should own interactive windows.
if multiprocessing.current_process().name != 'MainProcess':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

VIDEO_PATH = r"..\examples\videos\Fersina_20230630.avi"
CALCULATE_DE_UNCERTAINTY = True
SHOW_OVERVIEW = True
TRANSECT_COLUMN = 0  # Window column, not pixel column. Use row=... for a horizontal line.
DISTANCE_UNITS = 'pixels'  # Set to 'm' for calibrated spatial positions.


def plot_overview(iw, window_Lx, window_Ly):
    """Show where the windows are and preview one window's spectrum."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 7))
    xi, yi = np.meshgrid(iw.x, iw.y)
    axes[0].imshow(iw.imgs[0], cmap='Greys_r')
    axes[0].add_patch(patches.Rectangle(
        (0, 0), window_Lx, window_Ly, linewidth=1,
        edgecolor='r', facecolor='none', label='first window'))
    axes[0].plot(xi.ravel(), yi.ravel(), 'o', label='window centres')
    axes[0].legend(); axes[0].set_title('First frame overview')
    axes[1].imshow(iw.windows[0][0], cmap='Greys_r')
    axes[1].set_title('First frame: first window')
    for ax, dim in zip(axes[2:], ('x', 'y')):
        im = iw.plot_spectrum(window_idx=0, dim=dim, ax=ax)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_saved_transects(iw, results):
    """Draw saved native results, then leave iw at its latest fitted strategy.

    Temporarily select each small snapshot for plotting. The image/video arrays
    stay in place: no video copy, refit or uncertainty conversion is performed.
    The finally block restores the latest state even if drawing raises an error.
    """
    fields = (
        ('vx', 'vx'), ('vy', 'vy'), ('d', 'depth'), ('quality', 'quality'),
        ('depth_estimated', 'depth_estimated'), ('status', 'fit_status'),
        ('spectral_uncertainty', 'spectral_uncertainty'),
        ('mode_ambiguity', 'mode_ambiguity'),
    )
    latest = {target: getattr(iw, target) for target, _ in fields}
    axes = None
    try:
        for i, (label, saved) in enumerate(results.items()):
            for target, source in fields:
                setattr(iw, target, getattr(saved, source))
            _, axes = iw.plot_transect(
                column=TRANSECT_COLUMN, axes=axes, label=label, color=f'C{i}',
                distance_units=DISTANCE_UNITS)
    finally:
        for target, value in latest.items():
            setattr(iw, target, value)
    return axes


def main():
    from iwave import Iwave

    window_Lx, window_Ly = 100, 50
    iw = Iwave(resolution=.01, window_size=(window_Ly, window_Lx),
               overlap=(0, 0), time_size=250, time_overlap=125, smax=10)
    iw.read_video(VIDEO_PATH, start_frame=0, end_frame=500)
    iw.imgs = iw.imgs[:, 15:665, 0:100]
    print(iw)
    print('Images:', iw.imgs.shape, 'windows:', iw.windows.shape)

    # Keep spectra lazy: IWaVE collects uncertainty context in the same pass.
    # Do not create an overview or any other figure before/between the fits.
    # Closed Tk figures can otherwise be garbage-collected by a pool thread.
    strategies = {
        'DE': dict(method='DE',
                   calculate_spectral_uncertainty=CALCULATE_DE_UNCERTAINTY,
                   calculate_mode_ambiguity=False),
        'HNM': dict(method='HNM', hnm_options={'relative_radius': .5},
                    calculate_spectral_uncertainty=True, calculate_mode_ambiguity=True),
        'HNM (1 candidate)': dict(
            method='HNM', hnm_options={'n_candidates': 1, 'relative_radius': .5},
            calculate_spectral_uncertainty=True, calculate_mode_ambiguity=True),
    }
    results = {}
    for label, options in strategies.items():
        start = perf_counter()
        iw.velocimetry(alpha=.85, depth=0, **options)
        elapsed = perf_counter() - start

        # These are copies of the NEW NATIVE outputs, not wrappers/conversions.
        # Without a copy the next strategy would replace the latest results.
        results[label] = SimpleNamespace(
            vx=iw.vx.copy(), vy=iw.vy.copy(), depth=iw.d.copy(),
            depth_estimated=iw.depth_estimated, quality=iw.quality.copy(),
            fit_valid=iw.status.windows.copy(), fit_status=deepcopy(iw.status),
            elapsed=elapsed,
            spectral_uncertainty=deepcopy(iw.spectral_uncertainty),
            mode_ambiguity=deepcopy(iw.mode_ambiguity),
        )
        print(f'{label}: {elapsed:.3f} s, including enabled diagnostics')
        print(iw.spectral_uncertainty)
        print(iw.mode_ambiguity)

    # All worker pools have now shut down. From this point onwards the main
    # thread only draws; no later strategy starts another pool behind a GUI.
    if SHOW_OVERVIEW:
        plot_overview(iw, window_Lx, window_Ly)
    plot_saved_transects(iw, results)
    # Quality is the last transect panel. Show overview and overlays together.
    plt.show()
    return iw, results


if __name__ == '__main__':
    multiprocessing.freeze_support()
    iw, results = main()
