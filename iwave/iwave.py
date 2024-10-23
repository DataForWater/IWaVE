# -*- coding: utf-8 -*-


import os
import cv2
import sys
import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import window
import spectral

class Iwave(object):
    def __init__(self):
        #self.video_path = video_path
        pass

    def read_frames(self, frames_path):
        """Reads saved images 

        Args:
            normalize (bool, optional): _description_. Defaults to True.
        """
        images = os.listdir(frames_path)

        for nn in range(0, len(images)):
            if nn == 0:
                pass
            elif nn == 1:
                img0 = np.asarray(Image.open(os.path.join(frames_path,
                                                    images[0])))
                img1 = np.asarray(Image.open(os.path.join(frames_path,
                                                    images[1])))
                self.imgs_array = np.vstack((img0[None], img1[None]))

                #plt.imshow(img0)
                #plt.show()

            else:
                img = np.asarray(Image.open(os.path.join(frames_path,
                                                    images[nn])))
                self.imgs_array = np.vstack((self.imgs_array, img[None]))
            print ("Image: ", nn)
          
        print("Done")

    def frames_from_video(self, vid, start_frame=0, end_frame=4):
        """Extract frames from video

        Args:
            vid (str): path to video
            start_frame (int, optional): first frame. Defaults to 0.
            end_frame (int, optional): last frame. Defaults to 4.

        Returns:
            _type_: _description_
        """
        cap = cv2.VideoCapture(vid)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # retrieve images from start to end frame
        imgs = np.stack(
            [cv2.cvtColor(cap.read()[-1],
             cv2.COLOR_BGR2GRAY) for _ in range(end_frame - start_frame)]
        )
        return imgs
    
    def save_frames(self, vid, dst):
        pass

    def save_windows(self):
        pass

    def img_normalization(self, imgs_array):
        """normalizes images assuming the last two dimensions contain the 
        x/y image intensities
        """
        return window.normalize(imgs_array, "time")

    def subwindows(self, window_coordinates):
        """
        this function should read a set of data with 
        dimensions [t, y, x] and extract nw windows, returning an np.ndarray
        with dims [w, t, y, x]

        inputs can be a set of (y, x) indices or window centre coordinates (yc, xc)
        and (Ly, Lx) dimensions eventually these could be automatically 
        assigned along a transect based on starting point, end point, 
        and number of points
        """
        win_x, win_y = sliding_window_idx
        w = windows.multi_sliding_window_array(frames, win_x, win_y)

        #self.windowed_data = window_data(self.normalized_data, window_coordinates) 
        
    def data_segmentation(self, windows, segment_duration, overlap, engine):
        """
        data segmentation is currently included in the spectrum calculation
        using spectra.sliding_window_spectrum
        data segmentation and calculation of the average spectrum
        """
        
        return spectral.sliding_window_spectrum(windowds, segment_duration,
                                                overlap, engine)

    def subwindow_spectra(self, imgs: np.ndarray, win_t: int, overlap: int, engine):

        return spectral.sliding_window_spectrum(frames, win_t, overlap, engine)

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
    video_path = '/home/sp/pCloudDrive/Docs/d4w/iwave/vid/Fersina_20230630.avi'
    ############################################################################
    
    # Initialize
    iwave = Iwave()

    # Use video
    frames = iwave.frames_from_video(video_path, start_frame=0, end_frame=4)

    # or use frames
    #frames = iwave.read_frames(frames_path)

    # Normalize frames
    frames = iwave.img_normalization(frames)

    # 3D Spectrum of windows
    iwave.subwindow_spectra(frames, 32, 16, "numpy")

    print('ok')
    