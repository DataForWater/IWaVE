# IWaVE
Image Wave Velocimetry Estimation

This library performs simultaneous analysis of 2D velocimetry and stream depth 
through 2D Fourier transform methods, with a physics-based approach. 
Unlike existing velocimetry approaches such as Particle Image Velocimetry or
Space-Time Image Velocimetry, the uniqueness of this approach lies in the following:
* velocities that are advective of nature, can be distinguished from other wave forms such as wind waves. 
  This makes the approach particularly useful in estuaries or river stretches affected strongly by wind,
  or in shallow streams in the presence of standing waves.
* The velocity is estimated based on the physical behavior of the water surface, taking into account the
  speed of propagation of waves and ripples relative to the main flow. This makes the approach more robust
  than traditional methods when there are no visible tracers.
* If the depth is not known, it can be estimated along with the optimization of x and y-directional velocity.
  Depth estimations are reliable only in fast and shallow flows, where wave dynamics are significantly
  affected by the finite depth.

The code is meant to offer an Application Programming Interface for use within more high level applications that 
utilize the method in conjunction with more high level functionalities such as Graphical User Interfaces, dashboards,
or automization routines.

The background and validation of methods are outlined and demonstrated in:

Dolcetti, G., Hortobágyi, B., Perks, M., Tait, S. J. & Dervilis, N. (2022). 
Using non-contact measurement of water surface dynamics to estimate water discharge. 
Water Resources Research, 58(9), e2022WR032829. 
https://doi.org/10.1029/2022WR032829

The code has been based on the original kOmega code developed by Giulio Dolcetti
(University of Trento, Italy) and released on https://doi.org/10.5281/zenodo.7998891

The API of the code can:
* ingest a set of frames or frames taken from a video
* Slice these into "interrogation window" for which x- and y-directional velocities must be estimated
* Analyze advective velocities per interrogation window using the spectral analysis.

> [!NOTE]
> The methods behind IWaVE can in principle also resolve the depth of the stream at each window area of interest. 
> This method however, is not yet stable enough for us to release. We will work on this in the coming period. 

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
pip install iwave[extra]
```

This sets you up with ability to retrieve a sample video, read video frames, and make plots.

## Examples

The main functionality is disclosed via an API class IWaVE.

### Creating an `IWaVE` instance

To create an IWaVE instance, you typically start with some settings for deriving analysis windows and 

```python
from iwave import Iwave

# Initialize IWaVE object
iw = Iwave(
    resolution=0.01,
    window_size=(128, 128),  # size of interrogation windows over which velocities are estimated
    overlap=(64, 64),  # overlap in space (y, x) used to select windows from images or frames
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
* `overlap`: overlaps between the interrogation window. `(64, 64)` here means that an overlap of 50% in both
  directions is applied.
* `time_size`: a spectral model is fitted over several subsets of frames and then averaged. This reduces noise. You 
  can define how large slices are. If you for instance read 300 frames, and use a slice_size of 100, 3 subsets of 100 
  frames are derived, and the spectral model is fitted for all three and then averaged.
* `time_overlap`: also for the time, overlap can be used, in the same manner as for spatial overlap using `overlap`. 

### Reading in a video

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from iwave import Iwave, sample_data

iw = Iwave(
    resolution=0.01,  # resolution of videos you will analyze in meters. 
    window_size=(128, 128),  # size of interrogation windows over which velocities are estimated
    overlap=(64, 64),  # overlap in space (y, x) used to select windows from images or frames
    time_size=250,  # amount of frames used for one spectral analysis
    time_overlap=125,  # amount of overlap in frames, used to establish time slices. Selecting half of 
        # time_size implies that you use a 50% overlap in time between frame sets.
)

# retrieve a sample video from zenodo. This is built-in sample functionality...
fn_video = sample_data.get_sheaf_dataset()
iw.read_video(fn_video, start_frame=0, end_frame=500)

# NOTE: you can also read a list of ordered images with the frames per second set, using iw.read_imgs([...], fps=...)

print(iw)

# show the shape of the read images
print(f"Shape of the available images is {iw.imgs.shape}")

# show the shape of the manipulated windows
print(f"Shape of the available images is {iw.windows.shape}")

# Get the spectra of all windows and filter spectra with a spectral threshold of 1.0
iw.get_spectra(threshold=1.0)

# create a new figure with two subplots in one row
f, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

# plot the first image with a patch at the first window and centers of rest in the first axes instance
first_window = patches.Rectangle((0, 0), 128, 128, linewidth=1, edgecolor='r', facecolor='none', label="first window")
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
* how many frames are available (if the video is shorter than `start_frame` and `end_frame` dictate you'll get less     
  frames)
* how many time slices are expected from the amount of frames (overlap is included in this)
* The dimensions of the windows
* The y- and x-axis expected from the velocimetry analysis.

You can also see that the frames have actually been read into memory and windowed into a shape that has the following 
dimensions (in order):
* amount of windows
* amount of frames
* amount of y-pixels per window
* amount of x-pixels per window

Use `iw.read_imgs` as suggested in an inline comment to change reading to a set of frames stored as image files.
You then MUST provide frames-per-second explicitly yourself.

### Estimating x and y-directional velocity

```python
from iwave import Iwave, sample_data
import matplotlib.pyplot as plt
from matplotlib import patches

iw = Iwave(
    # repeat from example above...
)

iw.velocimetry(
  alpha=0.85,  # alpha represents the depth-averaged velocity over surface velocity [-]
  depth=0.3  # depth in [m] has to be known or estimated
)

ax = plt.axes()
ax.imshow(iw.imgs[0], cmap="Greys_r")

# add velocity vectors
iw.plot_velocimetry(ax=ax, color="b", scale=10)  # you can add kwargs that belong to matplotlib.pyploy.quiver

# plot the measured spectra and fitted dispersion relation (modify window_idx to visualize different windows)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
p1 = iw.plot_spectrum_fitted(window_idx=4, dim="x", ax=axs[0])
axs[0].set_xlim([-100, 100])
plt.colorbar(p1, ax=axs[0])
p2 = iw.plot_spectrum_fitted(window_idx=4, dim="y", ax=axs[1])
axs[1].set_xlim([-100, 100])
plt.colorbar(p2, ax=axs[1])
plt.show()
```
This estimates velocities in x and y-directions (u, v) per interrogation window and plots it on a background.
## For developers

To install IWaVE from the source code as developer (i.e. you wish to provide 
contributions to the code), you must checkout the code base with git using an ssh key
authentication. for instructions how to set this up, please refer to 
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

To check out the code and install it, please follow the code below:

```commandline
git clone git@github.com:DataForWater/IWaVE.git
cd IWaVE
pip install -e .
```
This will install the code base using symbolic links instead of copies. Any code 
changes will then immediately be reflected in the installation.

In case you wish to install the code base as developer, and have all dependencies 
for testing installed as well, you can replace the last line by: 

```commandline
pip install -e .[test]
```
You can now run the tests by running:

```commandline
pytest ./tests
```
