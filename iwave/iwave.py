# -*- coding: utf-8 -*-


import os
import sys
import datetime
import numpy as np


class Iwave(object):
    def __init__(self, keeper_data):
        self.keeper_data = keeper_data

    def read_frames(self):
        pass

    def data_norm(self):
        # data normalisation
        normalised_data = window.normalize(imgs, "time")

    def data_split(self):
        """
        this function should read a set of data with 
        dimensions [t, y, x] and extract nw windows, returning an np.ndarray
        with dims [w, t, y, x]
        this is probably done already in window.py, but I am not sure which
        functions do what.
        inputs can be a set of (y, x) indices or window centre coordinates (yc, xc)
        and (Ly, Lx) dimensions eventually these could be automatically 
        assigned along a transect based on starting point, end point, 
        and number of points
        """
        windowed_data = window_data(normalised_data, window_coordinates) 
        
    def data_segmentation(self):
        # data segmentation is currently included in the spectrum calculation using spectra.sliding_window_spectrum
        # data segmentation and calculation of the average spectrum
        measured_spectrum = spectral.sliding_window_spectrum(windowed_data, segment_duration, overlap, engine) #this function 
               already does everything that is needed here

    def wavenumber(self):
        # calculation of the wavenumber arrays
        kt, ky, kx = wave_number_dims((segment_duration, windowed_data.shape[2], windowed_data.shape[3]), resolution, fps)

    def spectrum_preprocessing(self):
        preprocessed_spectrum = optimise.spectrum_preprocessing(measured_spectrum, kt, ky, kx, velocity_threshold, 
            spectrum_threshold) 

        if depthisknown:
            optimised_parameters = optimise.optimise_velocity(measured_spectrum, bounds, depth, velocity_indx, img_size, resolution, fps)
        else:
            optimised_parameters = optimise.optimise_velocity_depth(measured_spectrum, bounds, velocity_indx, img_size, resolution, fps)

    def plot(self:)
        figures = plot_results(optimised_parameter, kt, ky, kx)
        """
        for iw = selected_windows:
            kt_gw, kt_turb = dispersion(ky, kx, optimised_velocity[iw], depth[iw], vel_indx)
            plot(kx, kt, measured_spectrum[iw, ky==0, :, :]) # plot x-t spectrum cross-section
            hold on; plot(kx, kt_gw[ky==0, :]) # plot gravity waves theoretical relation based on optimised parameters
            hold on; plot(kx, kt_turb[ky==0, :]) # plot turbulent waves theoretical relation based on optimised parameters

            plot(kx, kt, measured_spectrum[iw, :, kx==0, :]) # plot y-t spectrum cross-section
            hold on; plot(kx, kt_gw[:, kx==0]) # plot gravity waves theoretical relation based on optimised parameters
            hold on; plot(kx, kt_turb[:, kx==0]) # plot turbulent waves theoretical relation based on optimised parameters
        """


    #output_data = export(optimised_parameters)


if __name__ == '__main__':
    ############################################################################
    vid_file = '/home/sp/workspace/dk_site/config/site_config.dkc'
    ############################################################################
    
    iwave = Iwave(vid_file)
    