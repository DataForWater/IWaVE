# Following the FFT calculation

This guide explains how the FFT-only package prepares spectra and sends them to the original DE fitting method. It does not add HNM, the newer uncertainty result objects, or transect plotting. The README remains the parameter reference.

## What the arrays contain

Loaded images use `(frame, y, x)` order. One **window** is a rectangle followed through the loaded video. A requested group uses `(window, frame, y, x)` order.

The **Fourier transform**, usually called **FFT**, describes variation by frequency and spatial wavenumber instead of image position. **Power** measures the strength of that variation. Result arrays use `(window, frequency, ky, kx)`. Frequencies are rad/s and wavenumbers are rad/m. Both signs of the spatial wavenumbers are kept. As in the original calculation, time retains `ceil(time_size/2)` frequency samples.

## Follow the preparation steps

1. **Record where windows are — `iwave.py`, `data_models.py`, `window.py`.** Loading images records the pixel positions without making every window and spectrum. Selecting data later prepares only the requested windows. The loaded source video still occupies memory.
2. **Choose direct rectangle extraction when possible.** Ordinary rectangular windows can be copied directly from the source images. For irregular pixel selections, custom extraction functions, and other unsupported cases, keep the general extraction method. Never change the original image data while preparing a window.
3. **Adjust image brightness over the whole clip.** With `norm='time'`, subtract each pixel's time mean and divide by its standard deviation. Use all loaded frames before making shorter FFT segments. Compute mean first and squared deviations second; do not replace this with a less stable shortcut.
4. **Reuse temporary input arrays — `_FFTWorkspace`.** For ordinary time-normalised uint8 images, each active calculation needs one full-clip window at a time, not an input array holding the whole batch. Reuse that space for later windows. Transform arrays are reused too. Returned power does **not** share this temporary memory, so a later request cannot overwrite an earlier result.
5. **Calculate the transforms — `_window_power()`.** The original transform uses a spatial FFT and inverse-time FFT. This version performs time first, using real-valued input, while preserving the required time sign and power scale `1/T**2`. Each FFT uses one thread; Numba already processes separate windows in parallel. The spatial grid remains complete and calculations retain double precision.
6. **Average segment powers.** Use complete, possibly overlapping time segments. Add their powers in order and divide by the segment count. An all-zero segment still counts in that divisor. Do not average complex transform values or fit each segment separately.
7. **Prepare power for fitting.** Divide each frequency slice by its spatial mean. This is not the earlier image-brightness adjustment. Apply the existing relative threshold, remove zero frequency, apply the velocity-support filter, and clean undefined numerical values in the original order. Changing that order can change the answer for zero-power slices.

## Fit one batch at a time — `optimise.py`

A batch is a group of windows processed together. Prepare one batch, fit it, collect the results, and release its data before preparing the next. There is no background preparation of later batches.

Start the separate Python fitting processes once per `velocimetry()` call and reuse them across batches. Keep the original DE search settings and returned values. Individual fits can finish in any order; their saved window numbers restore the image order before the fields are arranged into rows and columns.

Internal returned values are ordered `(vy, vx, depth)`. Public attributes are named `vx`, `vy`, and `d`, in m/s and metres. Spatial positions are window centres in image pixels, not surveyed river coordinates.

## Memory and first-use setup

Temporary-array memory depends on clip length, window size, and the number of calculations running together. Current output power also grows with batch size. The source video, fitting processes and FFT library can require additional memory, so batch size is not a hard limit on total RAM.

A shorter request reuses existing temporary arrays; it does not necessarily shrink them. A change in transform shape replaces them rather than keeping a separate allocation for every old shape. Directly requesting all `iw.windows[:]` still creates all those image windows; the memory-saving spectrum path does not make every possible request small.

`iw.warmup()` uses tiny test arrays to prepare some Numba calculations before a timed run. It neither prepares the real video nor starts fitting processes. Its time is reported separately, not eliminated. The usual `if __name__ == '__main__':` entry remains necessary for scripts that start fitting processes.

Set `IWAVE_NUM_THREADS` before importing IWaVE to request the fitting-process count. It does not also limit threads used inside every numerical library.

## Existing interface limitations

This package keeps the older DE-only options. Its overall `status` means the run completed; it is not an array of individual optimiser success flags. Arrow positions are pixels and ordinary Matplotlib quiver options control display.

The old fitted-spectrum convenience method draws lines for depth 1 m and alpha 0.85. Use the README's explicit example for the fitted depth. This comment revision does not silently change plotting or any calculation, default, precision or solver budget.
