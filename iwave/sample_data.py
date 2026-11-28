"""Download and prepare the example image data."""
import os.path
import numpy as np
from iwave import window, io


def get_sheaf_dataset():
    """Download the Sheaf river example if it has not already been saved locally."""
    try:
        import pooch
    except:
        raise ImportError("This function needs pooch. Install iwave with pip install iwave[extra]")
    # Identify the sample data by its DOI.
    filename = "Fersina_20230630.avi"
    base_url = "doi:10.5281/zenodo.13935859"
    url = base_url + "/" + filename
    print(f"Retrieving or providing cached version of dataset from {url}")
    # Use Pooch to download the file and remember where it was saved.
    registry = pooch.create(
        # Choose the local folder for downloaded sample files.
        path=pooch.os_cache("iwave"),
        # Set the download address.
        base_url=base_url,
        # List the expected file and its checksum.
        registry={
            filename: None
        },
    )
    # Download the file if it is not already saved locally.
    file_path = registry.fetch(filename, progressbar=True)
    print(f"Sheaf dataset is available in {file_path}")
    return file_path


def get_sheaf_windows(start_frame=0, end_frame=500, window_size=(64, 64), overlap=(32, 32), max_windows=4):
    src_path = get_sheaf_dataset()
    dst_path = os.path.join(
        os.path.split(src_path)[0], "windows_{:04d}_{:04d}.bin".format(start_frame, end_frame)
    )
    if not os.path.exists(dst_path):
        imgs = io.get_video(src_path, start_frame=start_frame, end_frame=end_frame)
    # Find the x/y pixel positions belonging to each window.
        win_x, win_y = window.sliding_window_idx(imgs[0], window_size=window_size, overlap=overlap)
        # Extract those positions from every image.
        window_stack = window.multi_sliding_window_array(
            imgs,
            win_x,
            win_y,
            swap_time_dim=True
        )
        # Keep only the requested windows.
        window_stack = window_stack[0:max_windows]

        with open(dst_path, "wb") as f:
            np.save(f, window_stack)
    return dst_path