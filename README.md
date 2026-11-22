# IWaVE
Image Wave Velocimetry Estimation

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

## Examples

The main functionality is disclosed via an API class `Iwave`.

### Creating an `IWaVE` instance

To create an IWaVE instance, you typically start with some settings for deriving analysis windows:

```python
from iwave import Iwave

# Initialize IWaVE object
iw = Iwave(
    resolution=0.01,  # image resolution in meters per pixel
    window_size=(64, 64),  # size of interrogation windows over which velocities are estimated
    overlap=(32, 32),  # overlap in space (y, x) used to select windows from images or frames
    time_size=250,  # amount of frames in time used for one spectral analysis
    time_overlap=125,  # amount of overlap in frames, used to establish time slices. Selecting half of
        # time_size implies that you use a 50% overlap in time between frame sets.
)

# print some information about the IWaVE instance
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
    resolution=0.01,  # resolution of videos you will analyze in meters per pixel.
    window_size=(64, 64),  # size of interrogation windows over which velocities are estimated
    overlap=(32, 32),  # overlap in space (y, x) used to select windows from images or frames
    time_size=250,  # amount of frames used for one spectral analysis
    time_overlap=125,  # amount of overlap in frames, used to establish time slices. Selecting half of
        # time_size implies that you use a 50% overlap in time between frame sets.
)

# retrieve a sample video from zenodo. This is built-in sample functionality...
fn_video = sample_data.get_sheaf_dataset()
iw.read_video(fn_video, start_frame=0, end_frame=500)

# NOTE: you can also read images from a directory, with the frames per second set:
# iw.read_imgs("frames", fps=25, wildcard="*.png")

print(iw)

# show the shape of the read images
print(f"Shape of the available images is {iw.imgs.shape}")

# show the shape of the manipulated windows
print(f"Shape of the available windows is {iw.windows.shape}")

# create a new figure with two subplots in one row
f, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

# plot the first image with a patch at the first window and centers of rest in the first axes instance
first_window = patches.Rectangle((0, 0), iw.window_size[1], iw.window_size[0],
                                 linewidth=1, edgecolor='r', facecolor='none', label="first window")
xi, yi = np.meshgrid(iw.x, iw.y)
axs[0].imshow(iw.imgs[0], cmap="Greys_r")
axs[0].add_patch(first_window)
axs[0].plot(xi.flatten(), yi.flatten(), "o", label="centers")
axs[0].legend()
axs[0].set_title("First frame overview")
# plot the first window of the first image in the second axes instance
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
from iwave import Iwave, sample_data
import matplotlib.pyplot as plt

# the velocity optimization process is parallelized, for this, you MUST wrap
# your script in a function and protect it with an `if __name__ == "__main__":` check
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
    alpha=0.85,  # alpha represents the depth-averaged velocity over surface velocity [-]
    depth=0.3,  # depth in [m] has to be known or estimated. If depth = 0, then the depth is estimated.
  )

  ax = plt.axes()
  ax.imshow(iw.imgs[0], cmap="Greys_r")

  # add velocity vectors
  iw.plot_velocimetry(ax=ax, color="b", scale=10)  # accepts matplotlib.pyplot.quiver keyword arguments

  # plot the measured spectra and fitted dispersion relation (modify window_idx to visualize different windows)
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
  p1 = iw.plot_spectrum_fitted(window_idx=4, dim="x", ax=axs[0])
  axs[0].set_xlim([-100, 100])
  plt.colorbar(p1, ax=axs[0])
  p2 = iw.plot_spectrum_fitted(window_idx=4, dim="y", ax=axs[1])
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
# Replace the iw.velocimetry call in main() to select HNM:
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
>     # ... here your core functionalities, making an IWaVE instance, reading video/frames
>     # calling `iw.velocimetry`, storing results, plotting and so on...
>     pass
>
> # at the end, call your function with main functionalities
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
> # make sure you import iwave AFTER setting the environment variable
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

Spectral uncertainty can be calculated by setting `calculate_spectral_uncertainty=True`. This uses the number of
temporal averages, accounting for overlap, and the local sensitivity of the fitted spectrum to estimate bounds on
velocity and reconstructed depth. These bounds describe the optimiser uncertainty near the selected solution; they do not include calibration errors or
errors in the physical model; they do not account for the presence of potentlially multiple local minima of the cost function.

A separate option, `calculate_mode_ambiguity=True`, compares distinct solutions obtained from multiple independent instances of the Nelder-Mead refinement step. Solutions that cannot be distinguished on the estimated spectral-noise scale are marked as competitive rival. Alternative solutions with a lower score are marked as separate rival.
The calculation of mode ambiguity requires more than one starting estimate, so `n_candidates` must be larger than one:

```python
# Replace the iw.velocimetry call in main(); both diagnostics are optional.
iw.velocimetry(
    method="HNM", alpha=0.85, depth=0.3,
    hnm_options={"n_candidates": 6},
    calculate_spectral_uncertainty=True,
    calculate_mode_ambiguity=True,
)

# Results for the window in row 0, column 0:
print(iw.spectral_uncertainty.vx_bounds[0, 0])  # [lower, upper] in m/s
print(iw.spectral_uncertainty.vy_bounds[0, 0])
print(iw.mode_ambiguity.ambiguous_solution[0, 0])
```

For reconstructed depth, the corresponding bounds are available at `iw.spectral_uncertainty.depth_bounds` in meters.
Missing or rejected bounds are `NaN`. The ambiguity flag is `True` if a competing alternative was found, `False` if
none was found among the assessed solutions, and `None` if ambiguity could not be assessed. DE and single-start HNM
do not provide an ambiguity assessment. The comparison does not establish that all possible solutions have been found.
Both diagnostics are disabled by default and do not change the fitted parameters.


### Estimating water depth as well as x and y-directional velocity

The depth estimation is enabled by setting `depth = 0` in `iw.velocimetry`. In the `main()` function above, replace the
velocimetry call with the following and add the plots after fitting:

```python
iw.velocimetry(
  alpha=0.85,  # alpha represents the depth-averaged velocity over surface velocity [-]
  depth=0,  # If depth = 0, then the depth is estimated.
  dmin=0.01, dmax=3.0,  # lower and upper depth limits [m]
)

f, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
# Plot u against y for all x values
for i in range(iw.vx.shape[1]):
    axs[0].plot(iw.y, iw.vx[:, i], "o", label=f'x={iw.x[i]}')
axs[0].set_title("u vs y")
axs[0].set_xlabel("y [pixels]")
axs[0].set_ylabel("u (m/s)")
# Plot v against y for all x values
for i in range(iw.vy.shape[1]):
    axs[1].plot(iw.y, iw.vy[:, i], "o", label=f'x={iw.x[i]}')
axs[1].set_title("v vs y")
axs[1].set_xlabel("y [pixels]")
axs[1].set_ylabel("v (m/s)")
# Plot d against y for all x values
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
# Add after fitting, before plt.show() in main():
fig, axs = iw.plot_transect(column=0)
# Use row=0 instead to select the first row of windows.
```

Row and column indices refer to interrogation windows, starting from zero. Positions are in pixels by default;
`distance_units="m"` converts them to meters. Velocity components remain in m/s.

`iw.plot_velocimetry` also uses pixel positions by default and accepts `distance_units="m"`. Failed fits are omitted
unless `validate_vectors=False`; known ambiguous vectors are red by default when the ambiguity diagnostic is enabled.

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
