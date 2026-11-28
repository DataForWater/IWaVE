"""Shared inputs and helper functions for the tests."""
import numpy as np
import pytest

from iwave import sample_data, io


@pytest.fixture
def fn_video():
    return sample_data.get_sheaf_dataset()


@pytest.fixture
def imgs(fn_video):
    """Read four example frames and return an array in (frame, y, x) order."""
    return io.get_video(fn_video)



@pytest.fixture
def img_windows():
    fn_windows = sample_data.get_sheaf_windows()
    with open(fn_windows, "rb") as f:
        windows = np.load(f)
    return windows


@pytest.fixture
def img_windows_norm(img_windows):
    img_windows = img_windows - img_windows.mean(axis=0)
    img_windows = img_windows / img_windows.std(axis=0)
    img_windows[np.isinf(img_windows)] = 0
    img_windows[np.isnan(img_windows)] = 0
    return img_windows


# Small repeatable test arrays; no video download is needed.
@pytest.fixture
def flow():
    from types import SimpleNamespace
    from iwave import Iwave, dispersion, spectral
    dims, res, fps, depth = (128, 12, 14), .25, 20., .15
    axes = spectral.wave_numbers(dims, res, fps)
    raw = dispersion.intensity((.13, .47), depth, .85, dims, res, fps, 1.) + .03
    measured = spectral.spectrum_preprocessing(raw[None], *axes, 3., spectrum_threshold=.2)[0]
    context = dict(normalized_spectrum=raw/raw.mean(axis=(-2, -1), keepdims=True),
                   n_averages=3, time_size=128, time_overlap=64)
    def make_iwave():
        iw = Iwave(resolution=res, window_size=dims[1:], time_size=dims[0],
                   time_overlap=64, fps=fps, smax=1.)
        iw.spectrum = measured[None]; iw.x = np.array([0.]); iw.y = np.array([0.])
        return iw
    return SimpleNamespace(dims=dims, res=res, fps=fps, depth=depth, axes=axes,
        raw=raw, measured=measured, context=context, make_iwave=make_iwave,
        hnm_options=dict(velocity_step=.025, peak_height=.03, peak_prominence=.03, min_ridges=3))


@pytest.fixture
def inline_pool(monkeypatch):
    """Run the real fitting function directly, so this test checks results without starting processes."""
    from concurrent.futures import Future
    from iwave import optimise
    class InlinePool:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, *args, **kwargs):
            future = Future()
            try: future.set_result(function(*args, **kwargs))
            except Exception as error: future.set_exception(error)
            return future
    monkeypatch.setattr(optimise, 'ProcessPoolExecutor', InlinePool)


@pytest.fixture
def plot_field(flow):
    from iwave import io
    from iwave.iwave import VelocimetryStatus
    iw = flow.make_iwave(); shape = (3, 1)
    iw.x = np.array([50.]); iw.y = np.array([25., 75., 125.]); iw.resolution = .01
    iw.vx = np.full(shape, .6); iw.vy = np.full(shape, -.1); iw.d = np.full(shape, .3)
    iw.quality = np.full(shape, .9); iw.depth_estimated = True
    iw.status = VelocimetryStatus(np.ones(shape, bool), np.full(shape, 'ok', object))
    iw.spectral_uncertainty = io.SpectralUncertainty(shape); iw.mode_ambiguity = io.ModeAmbiguity(shape)
    for name, value in [('vx', iw.vx), ('vy', iw.vy), ('depth', iw.d)]:
        getattr(iw.spectral_uncertainty, name + '_bounds')[:] = np.stack([value-.01, value+.02], axis=-1)
    iw.spectral_uncertainty.valid[:] = True
    return iw

