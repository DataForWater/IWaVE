# Following the HNM calculation

This guide explains the route from images to results. Function names are included so you can find the corresponding code. The README remains the reference for parameter names, defaults and examples.

## The main ideas

A **window** is one small rectangle of the image, followed through time. The program estimates one velocity for that rectangle. A **batch** is a group of windows processed together to limit memory use.

A **spectrum** describes how the images vary in space and time. Its **power** measures the strength of each variation. The Fourier transform, usually called **FFT**, calculates this representation. Frequencies use rad/s; spatial wavenumbers use rad/m.

HNM combines two methods. **Hough** counts which velocities are supported by frequency peaks, giving starting guesses. **Nelder-Mead** improves those guesses by trying nearby velocities and, when requested, depths against the full measured spectrum.

## Follow one call to `iw.velocimetry()`

1. **Locate the image windows — `iwave.py`, `window.py`, `data_models.py`.** Loading images records window positions and creates objects that prepare the data only when requested. It does not immediately create every window and spectrum. The original video frames still remain in memory.
2. **Adjust image brightness — `LazyWindowArray`.** With `norm='time'`, subtract each pixel's mean and divide by its standard deviation over the entire loaded clip. Only afterwards divide the clip into shorter time segments. Normalising each segment separately would give different spectra. `norm='xy'` instead adjusts each frame across its pixels.
3. **Calculate measured power — `spectral.py`.** Transform each complete time segment and average its power. Do not average complex transform values or separately fitted velocities. Then divide each frequency slice by its spatial mean power, remove weak power using the existing threshold, remove zero frequency, and apply the velocity-support filter. This spectrum normalisation is separate from image-brightness normalisation.
4. **Prepare the initial velocity guesses — `optimise_hnm.py`.** Choose spatial-wavenumber columns, find peaks along frequency and add their support to a velocity grid. Each column contains all available frequencies. `filling_ratio` limits the columns examined in this first step only. It does not reduce the spectrum used in later fitting. Fixed selection uses a repeatable plan; adaptive selection considers power and the directions already examined.
5. **Improve each guess — `_refine_estimate()`, `SpectralOverlapObjective`.** Compare the measured spectrum with predicted waves and flow-carried patterns. Try up to `n_candidates` starting guesses. Each has its own evaluation budget, but all searches stay inside one circle around the primary Hough estimate and within the overall limits. Failure does not move or enlarge that circle. HNM adds no penalty for higher speed inside the allowed region.
6. **If depth is unknown — `_depth_setup()`, `_joint_simplex()`.** Hough uses one reference depth. Nelder-Mead can then vary both velocity and `log(depth)`; using the logarithm keeps physical depth positive. Convert it back to metres when returning results. The best finite, allowed endpoint is selected, then checked for success. A worse result is not substituted merely because it passed its stopping checks.
7. **Collect the results — `optimise.py`, `iwave.py`.** Separate Python processes fit individual windows. This HNM-only version starts and closes those processes for each batch. Fits may finish out of order; their window numbers put the results back into the correct image positions. Stored fields use `vx`, `vy` and `d`; internal fitting vectors put y before x.

## Optional checks, without changing the fit

The spectral check keeps the complete normalised power **before filtering**, together with information about time averaging. Discarded values still matter when modelling noise. Overlapping time segments share frames, so the noise calculation adjusts their effective averaging count rather than treating them as independent.

Small parameter changes show how the fitting score bends near the chosen result. Together with the assumed spectral noise, this predicts nearby lower and upper bounds. The calculation also accounts for values sharing the same normalisation mean. Bounds can be rejected when the fit is at a limit, the score is too flat, or the approximation is unreliable. A rejected bound means unavailable information, not zero error. Velocity bounds can remain usable when a depth bound fails.

The alternative-solution check compares different successful results already reached by HNM. Their scores come from the same measurements, so their noise is related. No extra search is run. `False` means no competitor among the usable results examined; `None` means the comparison was unavailable. Neither establishes that the chosen result is globally unique.

These are checks under an assumed noise/model description, not a complete calibrated error budget.

## Displaying results

`io.py` stores bounds and alternative estimates and draws the spatial plots. A transect is a row or column of existing windows; plotting it does not fit again. The quality panel shows spectral agreement, not an error bar. Positions are image pixels unless conversion by image resolution is requested. Velocity remains in m/s.

The older `plot_spectrum_fitted()` convenience method still draws its lines with depth 1 m and alpha 0.85. Use the README's explicit plotting example to draw lines for the actual fitted depth and chosen alpha.
