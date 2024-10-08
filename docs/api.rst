.. currentmodule:: iwave

.. _api:

=============
API reference
=============

spectral functions
==================

Several functions are available to derive the spectrum from a set of images or a theoretical spectrum.

.. autosummary::
    :toctree: _generated

    spectral.wave_numbers
    spectral.spectral_imgs
    dispersion.intensity

Array-related functions
=======================

Several functions to transform, reshape or otherwise manipulate arrays are available.
These generally are meant to help preparing a set of frames for the velocimetry analysis.

.. autosummary::
    :toctree: _generated

    window.sliding_window_idx
    window.sliding_window_array
    window.multi_sliding_window_array
    window.normalize
    window.get_axis_coords
    window.get_array_shape

