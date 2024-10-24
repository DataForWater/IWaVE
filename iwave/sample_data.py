"""Retrieval of sample dataset."""
import os.path
import numpy as np
from iwave import window, io


def get_sheaf_dataset():
    """Retrieve and cache sample dataset of Sheaf river."""
    try:
        import pooch
    except:
        raise ImportError("This function needs pooch. Install iwave with pip install iwave[extra]")
    # Define the DOI link
    filename = "Fersina_20230630.avi"
    base_url = "doi:10.5281/zenodo.13935860"
    url = base_url + "/" + filename
    print(f"Retrieving or providing cached version of dataset from {url}")
    # Create a Pooch registry to manage downloads
    registry = pooch.create(
        # Define the cache directory
        path=pooch.os_cache("iwave"),
        # Define the base URL for fetching data
        base_url=base_url,
        # Define the registry with the file we're expecting to download
        registry={
            filename: None
        },
    )
    # Fetch the dataset
    file_path = registry.fetch(filename, progressbar=True)
    print(f"Sheaf dataset is available in {file_path}")
    return file_path


def get_sheaf_windows(start_frame=0, end_frame=200, window_size=(64, 64), overlap=(32, 32), max_windows=4):
    src_path = get_sheaf_dataset()
    dst_path = os.path.join(
        os.path.split(src_path)[0], "windows_{:04d}_{:04d}.bin".format(start_frame, end_frame)
    )
    if not os.path.exists(dst_path):
        imgs = io.get_video(src_path, start_frame=start_frame, end_frame=end_frame)
    # get the x and y coordinates per window
        win_x, win_y = window.sliding_window_idx(imgs[0], window_size=window_size, overlap=overlap)
        # apply the coordinates on all images
        window_stack = window.multi_sliding_window_array(
            imgs,
            win_x,
            win_y,
            swap_time_dim=True
        )
        # only select the ones we are interested in
        window_stack = window_stack[0:max_windows]

        with open(dst_path, "wb") as f:
            np.save(f, window_stack)
    return dst_path