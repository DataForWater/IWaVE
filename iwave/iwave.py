# -*- coding: utf-8 -*-


import os
import sys
import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import window

class Iwave(object):
    def __init__(self, frames_path):
        self.frames_path = frames_path

    def readFrames(self, normalize=True):
        images = os.listdir(self.frames_path)

        for nn in range(0, len(images)):
            if nn == 0:
                pass
            elif nn == 1:
                img0 = np.asarray(Image.open(os.path.join(self.frames_path,
                                                    images[0])))
                img1 = np.asarray(Image.open(os.path.join(self.frames_path,
                                                    images[1])))
                self.imgs_array = np.vstack((img0[None], img1[None]))

                #plt.imshow(img0)
                #plt.show()

            else:
                img = np.asarray(Image.open(os.path.join(self.frames_path,
                                                    images[nn])))
                self.imgs_array = np.vstack((self.imgs_array, img[None]))
            print ("Image: ", nn)

        if normalize:
            self.imgNormalization()
            
        print("Done")

    def imgNormalization(self):
        print("Normalizing...")
        self.normalized_data = window.normalize(self.imgs_array, "time")

    def subwindows(self, window_coordinates):
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
        win_x, win_y = sliding_window_idx
        w = windows.multi_sliding_window_array(self.normalized_data,
                                               win_x, win_y)

        #self.windowed_data = window_data(self.normalized_data, window_coordinates) 
        
    def data_segmentation(self):
        """
        data segmentation is currently included in the spectrum calculation
        using spectra.sliding_window_spectrum
        data segmentation and calculation of the average spectrum
        """
        
        measured_spectrum = spectral.sliding_window_spectrum(windowed_data,
                                            segment_duration, overlap, engine)

    def wavenumber(self):
        # calculation of the wavenumber arrays
        kt, ky, kx = wave_number_dims((segment_duration, windowed_data.shape[2],
                                       windowed_data.shape[3]), resolution, fps)

    def spectrum_preprocessing(self):
        preprocessed_spectrum = optimise.spectrum_preprocessing(measured_spectrum,
                            kt, ky, kx, velocity_threshold, spectrum_threshold) 

        if depthisknown:
            optimised_parameters = optimise.optimise_velocity(measured_spectrum, bounds, depth, velocity_indx, img_size, resolution, fps)
        else:
            optimised_parameters = optimise.optimise_velocity_depth(measured_spectrum, bounds, velocity_indx, img_size, resolution, fps)

    def plot(self):
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
    frames_path = '/home/sp/pCloudDrive/Docs/d4w/iwave/transformed'
    ############################################################################
    
    iwave = Iwave(frames_path)
    iwave.readFrames(normalize=True)
    iwave.subwindows([64, 64])
    