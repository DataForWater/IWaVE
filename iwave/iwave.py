# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List
from iwave import window, spectral, optimise, io

repr_template = """
Resolution [m]: {}
Window size (y, x): {}
Overlap (y, x): {}
Size of time slices: {}
Overlap in time slices: {}
Number of images: {}
Frames per second: {}
""".format


class Iwave(object):
    def __init__(
        self,
        resolution: float,
        window_size: Tuple[int, int] = (128, 128),  # size of interrogation windows over which velocities are estimated
        overlap: Tuple[int, int] = (0, 0),  # overlap in space (y, x) used to select windows from images or frames
        time_size: int = 128,  # amount of frames in time used for one spectral analysis
        time_overlap: int = 0,  # amount of overlap in frames, used to establish time slices. Selecting half of
        imgs: Optional[np.ndarray] = None
    ):
        self.resolution = 0.02
        self.window_size = window_size
        self.overlap = overlap
        self.time_size = time_size
        self.time_overlap = time_overlap
        if imgs:
            self.imgs = imgs
        else:
            self.imgs = None
        self.fps = None

    def __repr__(self):
        if self.imgs is not None:
            no_imgs = len(self.imgs)
        else:
            no_imgs = None

        return repr_template(
            self.resolution,
            self.window_size,
            self.overlap,
            self.time_size,
            self.time_overlap,
            no_imgs,
            self.fps
        )

    @property
    def imgs(self):
        return self._imgs

    @imgs.setter
    def imgs(self, images):
        if images is not None:
            if images.ndim != 3:
                raise ValueError(f"Provided image array must have 3 dimensions. Provided dimensions are {images.ndim}: {images.shape}")
        self._imgs = images
        if images is not None:
            # TODO: check if image set is large enough for the given dimension of subwindowing and time windowing
            # subwindow images and get axes. This always necessary, so in-scope methods only.
            self._get_subwindow(images)
            self._get_x_y_axes(images)

    @property
    def windows(self):
        """Return images represented in subwindows."""
        return self._windows

    @windows.setter
    def windows(self, win):
        self._windows = win

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, _x):
        self._x = _x

    @property
    def y(self):
        return self._x

    @y.setter
    def y(self, _y):
        self._y = _y

    def _get_subwindow(self, images: np.ndarray):
        """Create windows following provided parameters."""
        # get the x and y coordinates per window
        win_x, win_y = window.sliding_window_idx(
            images[0],
            window_size=self.window_size,
            overlap=self.overlap,
        )
        # apply the coordinates on all images
        self.windows = window.multi_sliding_window_array(
            images,
            win_x,
            win_y,
            swap_time_dim=True
        )

    def _get_x_y_axes(self, images: np.ndarray):
        x, y = window.get_rect_coordinates(
            dim_sizes=images.shape[-2:],
            window_sizes=self.window_size,
            overlap=self.overlap,
        )
        self.x = x
        self.y = y


    def read_imgs(self, path: str, fps: float, wildcard: str = None):
        """Read frames stored as images on disk from path and wildcard.

        Parameters
        ----------
        path : str
            The directory path where the image frames are located.
        fps : float
           frames per second (must be explicitly set if it cannot be read from the video)
        wildcard : str, optional
            The pattern to match filenames. Defaults to None, meaning all files in the directory will be read.
        """
        self.imgs = io.get_imgs(path=path, wildcard=wildcard)

    def read_video(self, file: str, start_frame: int = 0, end_frame: int = 4):
        """
        Parameters
        ----------
        file : str
            Path to the video file.
        start_frame : int, optional
            The starting frame number from which to begin reading the video.
        end_frame : int, optional
            The ending frame number until which to read the video.

        Returns
        -------
        numpy.ndarray
            An array of grayscale images from the video between the specified frames.
        """
        self.imgs = io.get_video(fn=file, start_frame=start_frame, end_frame=end_frame)
        # get the frame rate from the video

        cap = cv2.VideoCapture(file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()
        del cap

    def save_frames(self, dst: str):
        raise NotImplementedError

    def save_windows(self):
        raise NotImplementedError

    def img_normalization(self, imgs_array):
        """normalizes images assuming the last two dimensions contain the 
        x/y image intensities
        """
        return window.normalize(imgs_array, "time")

    def velocimetry(
        self,
        alpha=0.85,
        depth=1.,
    ):
        raise NotImplementedError

    # def data_segmentation(self, windows, segment_duration, overlap, engine):
    #     """
    #     data segmentation is currently included in the spectrum calculation
    #     using spectra.sliding_window_spectrum
    #     data segmentation and calculation of the average spectrum
    #     """
    #
    #     return spectral.sliding_window_spectrum(windowds, segment_duration,
    #                                             overlap, engine)
    #
    # def subwindow_spectra(self, imgs: np.ndarray, win_t: int, overlap: int, engine):
    #
    #     return spectral.sliding_window_spectrum(frames, win_t, overlap, engine)
    #
    # def get_spectra(self, imgs, engine):
    #
    #     return spectral.spectral_imgs(imgs, engine)
    #
    #
    # def wavenumber(self):
    #     # calculation of the wavenumber arrays
    #     kt, ky, kx = wave_number_dims((segment_duration, windowed_data.shape[2],
    #                                    windowed_data.shape[3]), resolution, fps)
    #
    # def spectrum_preprocessing(self):
    #     preprocessed_spectrum = optimise.spectrum_preprocessing(measured_spectrum,
    #                         kt, ky, kx, velocity_threshold, spectrum_threshold)
    #
    #     if depthisknown:
    #         optimised_parameters = optimise.optimise_velocity(measured_spectrum, bounds, depth, velocity_indx, img_size, resolution, fps)
    #     else:
    #         optimised_parameters = optimise.optimise_velocity_depth(measured_spectrum, bounds, velocity_indx, img_size, resolution, fps)
    #
    # def plot(self):
    #     pass
    #
    #
    # def export_opt_param(self):
    #     pass


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

    # Subwidnows
    subwins = iwave.subwindows(frames, [64, 64], [32, 32])

    # 3D Spectrum 
    iwave.get_spectra(subwins, engine="numpy")
    
    print('ok')
    