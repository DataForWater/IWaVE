# IWaVE
Image Wave Velocimetry Estimation

> **HNM source snapshot.** This independent branch adds Hough/Nelder–Mead fitting,
> native diagnostic results and transects. It retains the original preprocessing;
> the separate FFT branch is not included. See [ALGORITHMS.md](ALGORITHMS.md) for the
> processing sequence and a guide to the commented source.

This library performs simultaneous analysis of 2D velocimetry and depth in lakes, coastlines, estuaries and
rivers with highly complex bathymetric and wave dynamics through space-time Fourier transform methods using a physics-based
approach. Unlike existing velocimetry approaches such as Particle Image Velocimetry or
Space-Time Image Velocimetry, the uniqueness of this approach lies in the following:

* velocities that are advective of nature, can be distinguished from other wave forms such as wind waves.
  This makes the approach particularly useful in estuaries or river stretches affected strongly by wind,
  or in shallow streams in the presence of standing waves.
* The velocity is estimated based on the physical behavior of the water surface, taking into account the
  speed of propagation of waves and ripples relative to the main flow. This makes the approach more robust
  than traditional methods when there are no visible tracers.
* If the depth is not known, it can be estimated along with the optimization of x and y-directional velocity.
  Depth estimations are most reliable in fast and shallow flows, where wave dynamics are significantly
  affected by the finite depth.

## Gallery

![lake_garda_iwave](https://github.com/user-attachments/assets/edbb863c-ce1c-4448-afec-c10ab479c2cd)

*Image license: Creative Commons Attribution 4.0 International
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). © 2026 Giulio Dolcetti.*

This figure gives an example of surface current velocity distribution reconstruction with IWaVE. The
video was recorded by a drone on Lake Garda (Italy) during a maintenance opening of the Adige-Garda tunnel.
The higher-velocity higher-turbidity plume is caused by the release of water from the Adige River
through the 10 km-long tunnel. The background surface velocity distribution outside of the plume is
wind-generated.

## Purpose

The code is meant to offer an Application Programming Interface (API) for use within more high level applications that
utilize the method in conjunction with more high level functionalities such as Graphical User Interfaces, dashboards,
or automation routines.

The background and validation of methods are outlined and demonstrated in:

Dolcetti, G., Hortobágyi, B., Perks, M., Tait, S. J. & Dervilis, N. (2022).
Using non-contact measurement of water surface dynamics to estimate water discharge.
Water Resources Research, 58(9), e2022WR032829.
https://doi.org/10.1029/2022WR032829

Examples and prospects of this approach are described in a recent HydroLink article, see
https://www.iahr.org/library/infor?pid=40120

The code has been based on the original kOmega Matlab code developed by Giulio Dolcetti
(University of Trento, Italy) and released on https://doi.org/10.5281/zenodo.7998891

The API of the code can:

* ingest a set of frames or frames taken from a video
* Slice these into "interrogation windows" for which x- and y-directional velocities must be estimated
* Analyze advective velocities per interrogation window using the spectral analysis.
* Additionally estimate the water depth if unknown.
* Estimate spectral uncertainty and compare alternative solutions found during the optimisation.
* Plot velocity vectors and transects of velocity, depth and fitting quality.

> [!NOTE]
> The sensitivity of water surface dynamics to variations in water depth is minimal. Depth estimations have specific requirements (see below) and can be prone to significant errors. Do not rely on depth estimates for any activity that could pose a risk.

## Installation

To install IWaVE, set up a python (virtual) environment and follow the instructions
below:

For a direct installation of the latest release, please activate your environment if
needed, and type

```commandline
pip install iwave
```

If you want to run the examples, you will need some extra dependencies. You can install these with

```commandline
pip install "iwave[extra]"
```

This sets you up with ability to retrieve a sample video, read video frames, and make plots.

The commands above install the **published release**, not this source snapshot. For these files,
use a separate virtual environment and install `requirements.txt` (or `requirements-test.txt` for tests).
Run from the directory containing `iwave/`, or replace the corresponding files in a complete project
checkout and use its editable installation. These source-only snapshots do not include upstream build
metadata; `pip install -e .` below therefore refers to the complete checkout, not the bare snapshot.
Verify the active source with `python -c "import iwave; print(iwave.__file__)"`, and restart an already
imported notebook/kernel when changing branches. The dependency list now explicitly includes the
SciPy, NumExpr, Pillow and tqdm imports that were missing from the previous requirements file.
A headless installation may substitute `opencv-python-headless` for `opencv-python`; do not install both.

## Examples

The main functionality is disclosed via an API class `Iwave`.

### Creating an `IWaVE` instance

To create an IWaVE instance, you typically start with some settings for deriving analysis windows:

```python
from iwave import Iwave

# Create the Iwave object and choose the analysis settings.
iw = Iwave(
    resolution=0.01,  # Image resolution, in metres per pixel.
    window_size=(64, 64),  # Window height and width in pixels.
    overlap=(32, 32),  # Shared pixels between neighbouring windows, in (y, x) order.
    time_size=250,  # Frames used in each spectrum time segment.
    time_overlap=125,  # Frames shared by consecutive time segments.
        # time_overlap=time_size/2 gives 50% overlap.
)

# Display the current analysis settings.
print(iw)
```
Initializing a IWaVE instance is done by only setting some parameters for the analysis. At this stage we have not
loaded any video in memory yet. The inputs have the following meaning:
* `window_size`: the size of so-called "interrogation windows" as a tuple (y, x), i.e. slices of pixels from the original
  frames that the images are subdivided in. Advective velocities are estimated per interrogation window by fitting a
  spectral model over space and time within an interrogation window.
* `overlap`: overlaps between the interrogation windows. `(32, 32)` here means that an overlap of 50% in both
  directions is applied.
* `time_size`: spectra are calculated over several subsets of frames and then averaged before fitting the model.
  This reduces noise. You can define how large slices are. If you for instance read 300 frames, and use a
  `time_size` of 100 with no time overlap, 3 subsets of 100 frames are derived and their spectra are averaged.
* `time_overlap`: also for the time, overlap can be used, in the same manner as for spatial overlap using `overlap`.

> [!NOTE]
> Some important remarks on uncertainties: IWaVE employs a spectral approach to compare the observed water surface
> dynamics with the theoretical expectations for given flow conditions. The key parameters determining the uncertainty
> of measurements are the spectral resolution, the number of averages, and the sensitivity of surface dynamics to
> velocity and water depth (see Dolcetti et al., 2022).
> * The spectral resolution improves by increasing the window size and/or the time size. The physical dimensions of
>   the interrogation windows (`window_size` multiplied by `resolution`) should be similar to the water depth or larger.
>   The duration corresponding to `time_size` should be larger than 5 seconds in most applications, ideally around
>   10 seconds. `time_size` itself is specified in frames.
> * The spatial and temporal resolution of the videos (e.g., the pixel size and frame rate) are less critical than
>   the spectral resolution for the accuracy of the estimates. Reasonable results can usually be obtained also with
>   a pixel size of ~5 cm and a frame rate of ~10 fps. Consider downsampling the data if memory or computational time
>   are an issue.
> * More averages can significantly improve the convergence of the method. Ideally, one should aim for at least 3
>   independent slices, regardless of the overlap (e.g., a 30-seconds-long video with slices of 10 seconds).
> * Short waves are more sensitive to flow velocity, while long waves are more sensitive to water depth. Therefore, a
>   better spatial resolution (smaller pixel size) can improve velocity estimates, while a better spectral resolution
>   (larger window size) can improve the depth estimates. However, only the waves with a wavelength similar or larger
>   than the water depth feel the presence of the bed and can be used to estimate the water depth. Typically, these
>   long waves form naturally in flows with Froude number in the range 0.4 to 1.0. These long waves must be visible
>   and their wavelength must be smaller than the window size for the depth estimation to be accurate. Otherwise,
>   depth estimates may fail completely due to lack of sensitivity.

### Reading in a video

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from iwave import Iwave, sample_data

iw = Iwave(
    resolution=0.01,  # Resolution of the analysed images, in metres per pixel.
    window_size=(64, 64),  # Window height and width in pixels.
    overlap=(32, 32),  # Shared pixels between neighbouring windows, in (y, x) order.
    time_size=250,  # Frames per spectrum time segment.
    time_overlap=125,  # Frames shared by consecutive time segments.
        # time_overlap=time_size/2 gives 50% overlap.
)

# Download the built-in example video from Zenodo.
fn_video = sample_data.get_sheaf_dataset()
iw.read_video(fn_video, start_frame=0, end_frame=500)

# Alternatively, load image files and specify their frame rate:
# iw.read_imgs("frames", fps=25, wildcard="*.png")

print(iw)

# Display the loaded image-array dimensions.
print(f"Shape of the available images is {iw.imgs.shape}")

# Display the window-array dimensions.
print(f"Shape of the available windows is {iw.windows.shape}")

# Make two plot areas side by side.
f, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

# On the first image, outline the first window and mark the other window centres.
first_window = patches.Rectangle((0, 0), iw.window_size[1], iw.window_size[0],
                                 linewidth=1, edgecolor='r', facecolor='none', label="first window")
xi, yi = np.meshgrid(iw.x, iw.y)
axs[0].imshow(iw.imgs[0], cmap="Greys_r")
axs[0].add_patch(first_window)
axs[0].plot(xi.flatten(), yi.flatten(), "o", label="centers")
axs[0].legend()
axs[0].set_title("First frame overview")
# In the second plot, show the first frame of the first window.
axs[1].imshow(iw.windows[0][0], cmap="Greys_r")
axs[1].set_title("First frame zoom first window")
plt.show()
```
You can now see that the IWaVE object shows:
* how many frames are available (if the video is shorter than `start_frame` and `end_frame` dictate, you'll get less
  frames)
* how many time slices are expected from the amount of frames (overlap is included in this)
* The dimensions of the windows
* The y- and x-axis expected from the velocimetry analysis.

You can also see that the frames have actually been read into memory. Interrogation windows are generated as needed
and have the following dimensions (in order):
* amount of windows
* amount of frames
* amount of y-pixels per window
* amount of x-pixels per window

Important to note is that the plotted first window on the right is normalized in time by subtracting for each pixel
its temporal mean and dividing by its temporal standard deviation. This is meant to reduce background noise. You may
also normalize in space by passing `norm="xy"` during the initialization of the IWaVE instance. In this case, the mean
and standard deviation are calculated over the pixels within each window, separately for each frame.

Use `iw.read_imgs` as suggested in an inline comment to change reading to a set of frames stored as image files.
You then MUST provide frames-per-second explicitly yourself.

### Estimating x and y-directional velocity

```python
from iwave import Iwave, sample_data, dispersion, io
import matplotlib.pyplot as plt

# Keep the analysis in main(), so new fitting processes can import this file safely.
# Call main() only inside the if __name__ == "__main__": check.
def main():
  iw = Iwave(
      resolution=0.01,
      window_size=(64, 64),
      overlap=(32, 32),
      time_size=250,
      time_overlap=125,
  )
  fn_video = sample_data.get_sheaf_dataset()
  iw.read_video(fn_video, start_frame=0, end_frame=500)

  iw.velocimetry(
    alpha=0.85,  # alpha is depth-averaged velocity divided by surface velocity.
    depth=0.3,  # Supply depth in metres, or set zero to estimate it.
  )

  ax = plt.axes()
  ax.imshow(iw.imgs[0], cmap="Greys_r")

  # Draw velocity arrows.
  iw.plot_velocimetry(ax=ax, color="b", scale=10)  # Choose arrow appearance with Matplotlib quiver options.

  # Show measured power and predicted lines; change window_idx to select another window.
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
  # Draw lines using the fitted depth and the chosen alpha below.
  # The older helper always uses depth=1 m and alpha=0.85 instead.
  j = 4
  waves, advected = dispersion.dispersion(
      iw.ky, iw.kx, (iw.vy.flat[j], iw.vx.flat[j]),
      depth=iw.d.flat[j], vel_indx=0.85,
  )
  measured = iw.spectrum[j]
  p1 = io.plot_spectrum_fitted(measured, waves, advected, iw.kt, iw.ky, iw.kx,
                               dim="x", ax=axs[0])
  axs[0].set_xlim([-100, 100])
  plt.colorbar(p1, ax=axs[0])
  p2 = io.plot_spectrum_fitted(measured, waves, advected, iw.kt, iw.ky, iw.kx,
                               dim="y", ax=axs[1])
  axs[1].set_xlim([-100, 100])
  plt.colorbar(p2, ax=axs[1])
  plt.show()


if __name__ == "__main__":
    main()
```
This estimates velocities in x and y-directions (u, v) per interrogation window and plots it on a background. If the
water depth is not supplied, a water depth of $d=1$ m is assumed. With IWaVE, variations in water depth have a small but
sometimes non-negligible impact on the velocity calculations. Therefore, it is recommended to set a representative
value for depth, even when this is not known accurately.

The results are stored in `iw.vx`, `iw.vy` and `iw.d`, with rows and columns corresponding to the interrogation windows.

### Optimisation algorithm

The flow parameters are calculated by maximising the cross-correlation between the measured spectrum of the image intensity and a synthetic spectrum generated according to linear wave theory. By default, this is done using a differential evolution algorithm (`scipy.optimize.differential_evolution`, `method="DE"`) which maximises the chances to identify a global maximum but may converge slowly, especially when the window size is large and/or there are more parameters to be estimated (e.g., also the water depth).

For DE, `pass_downsampling = [2, 1]` (default) will run the optimisation in two steps. The first step
employs a trimmed
spectrum with reduced dimensions by a factor of 2, and is used to identify an initial estimate of the flow velocity.
During the second step, the search of the optimum is confined
within a region between -0.1 m/s and +0.1 m/s of the initial estimate. The two-steps approach can reduce the computation time
by around 50% for large problems, but could be less robust if the input images are noisy or have a low resolution. In
that case, it is recommended to set `pass_downsampling = [1, 1]` to avoid an excessive loss in resolution.

Alternatively, `method="HNM"` uses a Hough transform to identify initial velocity estimates from the spectral ridges,
followed by a local Nelder–Mead optimisation. The refinement estimates velocity and, if requested, depth, within a
limited velocity region around the primary Hough estimate. This can substantially reduce computation time, although
its success depends on the initial estimates. With a supplied depth, HNM uses one starting estimate by default;
when estimating depth, it uses up to 12.

```python
# To select HNM, use this in place of the velocimetry call in main():
iw.velocimetry(method="HNM", alpha=0.85, depth=0.3)
```

Settings for the two methods can be supplied separately through `de_options` and `hnm_options`. For example,
`de_options={"pass_downsampling": [1, 1]}` selects the full-resolution DE passes described above. For HNM,
`hnm_options={"sampling": "adaptive", "filling_ratio": 0.5}` uses adaptive sampling of the spatial wavenumber pairs
for the initial velocity estimate, with a budget of up to half the spatial wavenumber pairs. These settings do not
change the video resolution.

> [!IMPORTANT]
> Optimization of the velocities occurs with a possibly large amount of spawned parallelized processes.
> The `if __name__ == "__main__":` construction around the script is necessary to optimize
> velocities without rerunning the entire script in each spawned subprocess. Like so:
>
> ```python
> import iwave
>
> def main():
>     # Create Iwave and load your video or image sequence here.
>     # Keep fitting, saving results and plotting inside this function.
>     pass
>
> # Run the analysis function when this file is started as a script.
> if __name__ == "__main__":
>     main()
> ```
>
> When combining the examples in one script, complete all fits before creating figures, including the frame overview.

> [!NOTE]
> By default, optimizations are parallelized using the maximum amount of CPUs available on your system. If you want to
> change the amount of CPUs used for optimization, you can set the environment variable `IWAVE_NUM_THREADS` to the desired
> number of CPUs.
> If you set this environment variable within a python script, make sure it is set before importing IWaVE.
> ```python
> import os
> os.environ["IWAVE_NUM_THREADS"] = "1"
>
> # Set the environment variable before importing iwave.
> import iwave
> ```


### Uncertainties

`iw.quality` is a quality
metric describing the agreement between the measured and fitted spectra, rather than a measure of uncertainty in the
optimised parameters. The quality $q$ is obtained from the ratio of the inverse spectral overlaps calculated with the
measured spectrum and with the (ideal) synthetic spectrum,
$q = 1 - 0.2\log_{10}\frac{c_m}{c_i}$, where $c_m$ is the measured inverse overlap, and $c_i$ the ideal inverse overlap.
A perfect spectral match gives $q=1$; the score is not clipped to a fixed range. `iw.cost` is the
optimisation objective value. Acceptable quality may vary depending on window size, frame rate, and velocity and depth magnitude.
Values of $q < 0.7$ are often indicative of poor fitting between measured and ideal spectra, which may indicate
erroneous estimates of velocity. The water depth has a relatively small effect on the cost function, therefore high
values of $q$ are not sufficient indicators of accurate depth estimation, although low values of $q$ are usually
indicative of large uncertainties in both velocity and depth estimations.

`iw.status.windows` and `iw.status.windows_message` contain the optimisation outcome and exit message for each window.
`iw.status.overall` is true only if all windows report success; `iw.status.overall_message` gives a summary.

Spectral uncertainty can be calculated by setting `calculate_spectral_uncertainty=True`. This uses an approximate
overlap-corrected effective number of temporal averages and the numerical local sensitivity of the fitted spectrum
to estimate provisional bounds on velocity and reconstructed depth. These are noise-scaled local sensitivity bounds,
conditional on the selected solution and fixed fitted mask, not calibrated total confidence intervals. They exclude
calibration and physical-model errors and do not account for other local minima of the cost function.

A separate option, `calculate_mode_ambiguity=True`, compares distinct converged, feasible, interior endpoints from the
existing Nelder–Mead starts. The starts use the same measured data and are **not independent noise realisations**.
Solutions that cannot be distinguished on the estimated shared-data spectral-noise scale are marked as competitive
rivals; alternatives resolved as worse on that scale are separated rivals. No additional searches are performed.
The calculation of mode ambiguity requires more than one starting estimate, so `n_candidates` must be larger than one:

```python
# Use this velocimetry call in main(); each optional check has its own switch.
iw.velocimetry(
    method="HNM", alpha=0.85, depth=0.3,
    hnm_options={"n_candidates": 6},
    calculate_spectral_uncertainty=True,
    calculate_mode_ambiguity=True,
)

# Read results for the first row and column of windows:
print(iw.spectral_uncertainty.vx_bounds[0, 0])  # [lower, upper] in m/s
print(iw.spectral_uncertainty.vy_bounds[0, 0])
print(iw.mode_ambiguity.ambiguous_solution[0, 0])
```

For reconstructed depth, the corresponding bounds are available at `iw.spectral_uncertainty.depth_bounds` in meters.
All bound arrays have shape `(rows, columns, 2)`; inspect `.valid` and the per-window `.details[row, column]` for
acceptance flags, assumptions and failure reasons. `iw.mode_ambiguity.additional_estimates[row, column]` contains
native rival records with `.vx`, `.vy`, `.depth` and `.ambiguous_solution`; no rival bounds are invented.
Missing or rejected bounds are `NaN`. The ambiguity flag is `True` if a competing alternative was found, `False` if
none was found among the assessed solutions, and `None` if ambiguity could not be assessed. DE and single-start HNM
do not provide an ambiguity assessment. The comparison does not establish that all possible solutions have been found.
Both diagnostics are disabled by default and do not change the fitted parameters. Passing a tuning dictionary alone
does not enable either calculation. For DE, use an unpenalised fit (`de_options={"penalty_weight": 0}`) when requesting
spectral bounds: penalised fits are flagged invalid, not treated as calibrated estimates.


### Estimating water depth as well as x and y-directional velocity

The depth estimation is enabled by setting `depth = 0` in `iw.velocimetry`. In the `main()` function above, replace the
velocimetry call with the following and add the plots after fitting:

```python
iw.velocimetry(
  alpha=0.85,  # alpha is depth-averaged velocity divided by surface velocity.
  depth=0,  # A depth of zero asks the method to estimate it.
  dmin=0.01, dmax=3.0,  # Lower and upper allowed depths, in metres.
)

f, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
# Plot x velocity along y, for each x position.
for i in range(iw.vx.shape[1]):
    axs[0].plot(iw.y, iw.vx[:, i], "o", label=f'x={iw.x[i]}')
axs[0].set_title("u vs y")
axs[0].set_xlabel("y [pixels]")
axs[0].set_ylabel("u (m/s)")
# Plot y velocity along y, for each x position.
for i in range(iw.vy.shape[1]):
    axs[1].plot(iw.y, iw.vy[:, i], "o", label=f'x={iw.x[i]}')
axs[1].set_title("v vs y")
axs[1].set_xlabel("y [pixels]")
axs[1].set_ylabel("v (m/s)")
# Plot depth along y, for each x position.
for i in range(iw.d.shape[1]):
    axs[2].plot(iw.y, iw.d[:, i], "o", label=f'x={iw.x[i]}')
axs[2].set_title("depth vs y")
axs[2].set_xlabel("y [pixels]")
axs[2].set_ylabel("depth (m)")
plt.show()

```
This estimates velocities in x and y-directions (u, v) and water depth (d) per interrogation window and plots the results.
The same depth estimation is available with `method="HNM"`.

### Plotting transects

`iw.plot_transect` plots velocity components, fitting quality and, when estimated, depth along one row or column of
windows. Available spectral uncertainty bounds and alternative solutions are included automatically:

```python
# Add this after fitting, before plt.show() in main():
fig, axs = iw.plot_transect(column=0)
# Use row=0 instead to plot the first row of windows.
```

Row and column indices refer to interrogation windows, starting from zero. Positions are in pixels by default;
`distance_units="m"` converts them to meters. Velocity components remain in m/s.

`iw.plot_velocimetry` also uses pixel positions by default and accepts `distance_units="m"`. Failed fits are omitted
unless `validate_vectors=False`; nonfinite vectors are always omitted. Known ambiguous vectors use
`ambiguous_color="tab:red"`; unknown ambiguity keeps the ordinary `color`. Velocity components remain in m/s,
including when positions are pixels. Matplotlib's `scale` controls displayed arrow length, not the physical units.

Transects return `(figure, axes)` in `vx, vy[, depth], quality` order and never call `show()`.
Depth appears automatically only when reconstructed; `include_depth=False` hides it, while `True` rejects fixed-depth
fits. Use `show_uncertainty=False` or `show_modes=False` to hide those layers without recalculation. Reuse `axes` with
`label`/`color` to overlay strategies with matching panel counts; the shared legend is outside the panels.
Transects also accept `distance_units="index"`. Window-centre positions refer to the analysed/cropped image,
not surveyed river coordinates. Open markers indicate unavailable/rejected bounds; red rings indicate a competitive
alternative. Rivals are separate points, not extensions of the selected error bar. Unknown ambiguity has no warning
marker and remains `None`, not `False`.

### API and migration reference

Additional constructor controls are `fps`, `imgs`, `norm`, `spectrum_threshold`, `smax` and `window_chunk_size`.
`imgs` is a `(frames, y, x)` array and needs a supplied frame rate. `norm="time"` is the default; `"xy"` normalises
per frame and `None` disables image normalisation. `spectrum_threshold=1.0` multiplies the frequency-mean normalised
power at each spatial wavenumber; increasing it removes more weak power when that mean is finite. It is not an
absolute intensity threshold. `smax=4.0` m/s sets the velocity search limits; preprocessing retains frequencies up
to `3*smax*sqrt(kx**2+ky**2)`, which is a different operation from fitting bounds. `window_chunk_size=50` limits
windows per batch, not total input-video memory. Smaller batches use less peak memory. Spatial window sizes are
rounded up to even numbers; `time_size` remains in frames. Use positive integer `time_size` no larger than the
loaded clip and integer `0 <= time_overlap < time_size`; only complete temporal segments are averaged.

`read_video(..., stride=k)` subsamples frames but does not automatically divide the stored frame rate: pass
`fps=original_fps/k` when using a stride larger than one. Assign cropped/replacement data through `iw.imgs = ...`
so window coordinates and lazy handles are rebuilt; do not assume in-place mutation invalidates fitted results.

`velocimetry()` still defaults to `method="DE"`, `alpha=0.85`, `depth=1.0`, `dmin=0.01`, `dmax=3.0`, both physical
branches enabled and `progress=True`. `progress=False` suppresses the parent fitting progress bar. Bounds are in m/s
and metres. `gravity_waves_switch=False` disables waves; `turbulence_switch=False` disables the advective branch.
At least one branch is needed for HNM; reconstructing depth additionally requires gravity waves and refinement.

| Settings namespace | Use |
|---|---|
| `de_options` | `gauss_width=1.0`, `penalty_weight=1.0`, `pass_downsampling=[2, 1]`, and nested `scipy_options` |
| `hnm_options` | Flat detector, sampling and local-refinement settings below; there is no HNM velocity penalty |
| `spectral_uncertainty_options` | Tuning for explicitly enabled local spectral sensitivity |
| `mode_ambiguity_options` | Tuning for explicitly enabled existing-endpoint comparisons |

New-style nested DE settings merge with the standard defaults. Legacy direct SciPy kwargs retain their previous
replace-defaults behaviour when nonempty. Do not mix that style or top-level `gauss_width`, `penalty_weight` or
`pass_downsampling` with `de_options`, or pass DE settings to HNM.

**HNM options** (defaults unless stated otherwise):

| Key | Default | Meaning |
|---|---:|---|
| `sampling` | `"fixed"` | Fixed spatial plan or `"adaptive"` scheduling |
| `filling_ratio` | `1.0` | Maximum processed spatial columns: `ceil(filling_ratio*Ny*Nx)`, capped by eligibility |
| `velocity_step` | `0.01` | Requested Hough grid spacing [m/s]; actual spacing follows the bounds |
| `vote_sigma` | `0.015` | Gaussian vote-field smoothing scale [m/s] |
| `k_min_factor` | `1.0` | Radial cutoff factor multiplying twice the smaller spatial Fourier spacing |
| `peak_height`, `peak_prominence` | `0.25`, `0.30` | Peak thresholds after common intensity scaling |
| `scale_percentile` | `99.5` | Percentile for that common intensity scale |
| `min_ridges` | `12` | Minimum extracted ridge points for a Hough estimate |
| `max_accumulator_mb` | `None` | Optional estimated accumulator-workspace cap; not a total-RAM limit |
| `refine` | `True` | Nelder–Mead refinement; `False` supports fixed-depth Hough-only fits |
| `relative_radius`, `min_radius` | `0.5`, `None` | Hard velocity disk: `max(relative_radius*abs(v_Hough), min_radius)`; `None` uses twice the larger actual grid spacing |
| `n_candidates` | `1` fixed depth; `12` free depth | Maximum Hough-derived starts; fewer can be available inside the shared disk |
| `candidate_separation` | `0.1` | Minimum separation between starting velocities [m/s] |
| `maxfev` | `160` fixed depth; `640` free depth | NM evaluation budget **per start**, excluding subsequent diagnostics |
| `xatol`, `fatol` | `1e-4`, `1e-6` | NM coordinate/objective tolerances; depth coordinates use log(depth) |
| `initial_step_factor` | `2.0` | Initial velocity simplex step is this factor times `velocity_step` |
| `depth_initial` | `None` | Hough reference/NM initial depth; `None` uses `sqrt(dmin*dmax)` |
| `log_depth_step` | `0.1` | Initial depth-simplex edge in natural-log depth |
| `gauss_width`, `kernel_truncate` | `1.0`, `5.0` | Forward-model frequency scale [rad/s] and narrow-band tail cutoff in multiples of that scale |
| `batch_size`, `min_columns` | `64`, `64` | Adaptive processing batch and minimum processed columns |
| `stable_batches`, `convergence_tol` | `2`, `0.01` | Adaptive successive-estimate stability count and velocity tolerance [m/s] |
| `exploration_every`, `min_direction_ratio` | `4`, `0.05` | Adaptive geometric exploration cadence and directional-information ratio |

Former `local_options`/`sampling_options` nesting is not accepted: move their entries directly into `hnm_options`.
Use `initial_step_factor`, not the former `initial_step` key.

The trust disk remains centred on the primary Hough estimate for **all** starts. It is not a velocity penalty and
is not automatically expanded. Boundary-limited or unconverged selected fits are unsuccessful even if their
spectral quality is high. Depth-dependent defaults apply independently per key; explicit user values win.
`filling_ratio` affects Hough evidence only, not the FFT grid or the full-spectrum local refinement.
HNM requires `0.5 < alpha <= 1`, positive resolution/frame rate/depth limits, at least three retained frequency bins
and `0 < filling_ratio <= 1`. The reference depth must lie within the positive reconstruction limits.

**Diagnostic tuning:** both dictionaries accept `confidence=0.95`, `variance_estimator="plugin"` (or
`"gamma_unbiased"`) and `n_effective=None`. The default effective count is the rectangular-Welch overlap
approximation; an explicit `n_effective` is the user's modelling choice, not an automatically estimated calibration.
`spectral_uncertainty_options` additionally accepts `fd_step=0.05`, `stability_rtol=0.25`, `curvature_rtol=1e-7`,
`curvature_atol=1e-10`, `stationarity_tol=0.1`, `parameter_scales=None`, `check_region=True`, `quadratic_rtol=0.5`
and `kernel_truncate=5.0` (at least 5). Parameter scales follow `(vy, vx[, log_depth])` order.
`mode_ambiguity_options["merge_tolerances"]` defaults to `None`, meaning ten `xatol` units per coordinate in the
same order. These are numerical endpoint-merging tolerances, not uncertainties. Neither confidence setting is
selection-adjusted global coverage. Keep the forward-model kernel convention consistent when changing expert options.

For precomputed/array spectra, either diagnostic requires one entry in `spectral_contexts` per flattened window:
`{"normalized_spectrum": full, "n_averages": n, "time_size": T, "time_overlap": O}`; `n_effective` may replace
the averaging metadata. `full` has `(frequency, ky, kx)` shape and is complete, spatially normalised power **before**
thresholding/DC/support censoring. The processed spectrum alone is insufficient. Normal lazy spectra supply this
context automatically. Component bounds are accepted independently: a rejected depth interval does not by itself
remove accepted velocity bounds. Use `.details` to understand missing/invalid components.

Native `SpectralUncertainty`, `ModeAmbiguity` and `AlternativeEstimate` classes, spatial plotting functions,
`img_windows_norm` and the curvature helpers live in `iwave.io`. Normal use needs only `Iwave` and its native result
attributes; direct imports from earlier split helper modules should use the current locations. The old standalone
`plots.py` demonstration is not part of this HNM package.

`iw.method_options` records resolved solver settings and `iw.sampling_metadata[row, column]` the HNM audit record.
`bool(iw.status)` is equivalent to `iw.status.overall`; `status_windows`, `message_windows` and `message` are
read-only aliases of the structured status. These are not uncertainty flags. The old `calculate_uncertainty`,
`uncertainty_options`, `calculate_hough_uncertainty` and `hough_uncertainty_options` interface is removed;
use the two explicit diagnostic switches above instead.

There is **no preprocessing-overlap option** in this clean version. Remove `preprocessing_overlap` and private
lookahead assignments from scripts based on the discarded experiment. No FFT-engine, mixed-precision or FFT
spatial-subsampling switch is added here. `IWAVE_NUM_THREADS` controls fitting-process requests, not a global cap
on Numba/NumExpr/native thread teams; set environment controls before importing the package.

The legacy `iw.plot_spectrum_fitted()` convenience method still assumes depth 1 m and alpha 0.85 for its overlay;
use the explicit `dispersion.dispersion`/`io.plot_spectrum_fitted` example above to display another fitted depth or
alpha. This existing limitation was documented, not silently changed in the cleanup.

This standalone HNM snapshot does not expose `iw.warmup()` or the gathered-FFT workspace. Those belong to the
FFT/integrated environment; its preprocessing and per-batch process-pool lifecycle are intentionally unchanged.

## For developers

To install IWaVE from the source code as developer (i.e. you wish to provide
contributions to the code), you can checkout the code base with git using an ssh key
authentication. For instructions how to set this up, please refer to
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

To check out the code and install it, please follow the code below:

```commandline
git clone git@github.com:DataForWater/IWaVE.git
cd IWaVE
pip install -e .
```
This will install the code base in editable mode. Any code
changes will then immediately be reflected in the installation.

In case you wish to install the code base as developer, and have all dependencies
for testing installed as well, you can replace the last line by:

```commandline
pip install -e ".[test]"
```
You can now run the tests by running:

```commandline
pytest ./tests
```

### Credits:

* Giulio Dolcetti
* Hessel Winsemius
* Salvador Peña-Haro
