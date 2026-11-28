import numpy as np
import pytest
import time
from iwave import spectral, dispersion, optimise, window, Iwave


def test_preprocessing():
    """Check the dimensions after processing the spectrum."""
    img = np.random.rand(3, 32, 32, 32)
    kt, ky, kx = spectral.wave_numbers(img.shape, res=0.02, fps=25)
    kt_gw, kt_turb = dispersion.dispersion(ky, kx, velocity=[1, 0], depth=1, vel_indx=1)
    synthetic_spectrum = dispersion.theoretical_spectrum(kt_gw, kt_turb, kt, gauss_width=1, gravity_waves_switch=True, 
                                                         turbulence_switch=True)
    measured_spectrum = spectral.sliding_window_spectrum(img, img.shape[1], 0)
    preprocessed_spectrum = spectral.spectrum_preprocessing(measured_spectrum, kt, ky, kx, velocity_threshold=5)
    # Check that processing keeps the measured and model spectrum dimensions compatible.
    assert preprocessed_spectrum[0].shape == synthetic_spectrum.shape

    
def test_nsp(img_size=(256, 64, 64), res=0.02, fps=25):
    """Compare the inverse-overlap score with a known result."""
    kt, ky, kx = spectral.wave_numbers(img_size, res, fps)
    kt_gw, kt_turb = dispersion.dispersion(ky, kx, velocity=[1, 0], depth=1, vel_indx=1)
    synthetic_spectrum = dispersion.theoretical_spectrum(kt_gw, kt_turb, kt, gauss_width=1, gravity_waves_switch=True, 
                                                         turbulence_switch=True)
    cost = optimise.nsp_inv(synthetic_spectrum,synthetic_spectrum)
    expected_cost = np.sum(synthetic_spectrum)**2 / np.sum(synthetic_spectrum**2)
    # Compare a model spectrum with itself to check the overlap score.
    assert np.allclose(cost, expected_cost)

    
def test_cost_function_velocity_depth(img_size=(256, 64, 64), res=0.02, fps=25):
    """Check how the fitting score changes when the trial depth changes."""
    kt, ky, kx = spectral.wave_numbers(img_size, res, fps)
    depth_1 = 0.30
    depth_2 = 0.29
    depth_3 = 0.31
    velocity_y = 1
    velocity_x = 0
    params_1 = [velocity_y, velocity_x, np.log(depth_1)]
    params_2 = [velocity_y, velocity_x, np.log(depth_2)]
    params_3 = [velocity_y, velocity_x, np.log(depth_3)]
    vel_indx = 1
    kt_gw_1, kt_turb_1 = dispersion.dispersion(ky, kx, [velocity_y, velocity_x], depth_1, vel_indx)
    synthetic_spectrum_1 = dispersion.theoretical_spectrum(
        kt_gw_1,
        kt_turb_1,
        kt,
        gauss_width=1,
        gravity_waves_switch=True,
        turbulence_switch=True
    )
    cost_11 = optimise.cost_function_velocity_depth(
        params_1, synthetic_spectrum_1, vel_indx,
        img_size, res, fps, gauss_width=1, penalty_weight = 1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    cost_12 = optimise.cost_function_velocity_depth(
        params_2, synthetic_spectrum_1, vel_indx,
        img_size, res, fps, gauss_width=1, penalty_weight = 1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    cost_13 = optimise.cost_function_velocity_depth(
        params_3, synthetic_spectrum_1, vel_indx,
        img_size, res, fps, gauss_width=1, penalty_weight = 1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    # Check that changing depth away from the known value makes agreement worse.
    assert cost_12 > cost_11
    assert cost_13 > cost_11

def test_optimise_velocity_depth(img_size=(128, 64, 64), res=0.02, fps=12):
    """Fit a generated single-window spectrum with known velocity and depth."""
    kt, ky, kx = spectral.wave_numbers(img_size, res, fps)
    velocity = [1, 0]
    depth = 0.2
    velocity_indx = 1
    kt_gw, kt_turb = dispersion.dispersion(
        ky,
        kx,
        velocity,
        depth,
        velocity_indx
    )
    synthetic_spectrum = dispersion.theoretical_spectrum(
        kt_gw,
        kt_turb,
        kt,
        gauss_width=1,
        gravity_waves_switch=True,
        turbulence_switch=True
    )
    # Set the allowed velocity and depth ranges.

    # Repeat the same spectrum to represent several windows.
    synthetic_spectrum = np.tile(synthetic_spectrum, (2,1,1,1))
    vel_y_min = 0
    vel_y_max = 2
    vel_x_min = -0.5
    vel_x_max = 0.5
    depth_min = 0.01
    depth_max = 1
    bounds = [[(vel_y_min, vel_y_max), (vel_x_min, vel_x_max), (depth_min, depth_max)]] * 2
    t1 = time.time()
    output, _, _ = optimise.optimise_velocity(
        synthetic_spectrum,
        bounds,
        velocity_indx,
        img_size,
        res,
        fps,
        gauss_width=1,
        penalty_weight=0,
        gravity_waves_switch=True,
        turbulence_switch=True,
        pass_downsampling=[1],
        popsize=20,
        workers=1,
        maxiter=1000,
    )
    vel_y_optimal = output[:, 0]
    vel_x_optimal = output[:, 1]  
    depth_optimal = output[:, 2] 
    t2 = time.time()
    print(f"Took {t2 - t1} seconds")
    assert vel_x_max >= vel_x_min
    assert vel_y_max >= vel_y_min
    assert depth_max >= depth_min
    assert np.all(np.abs(vel_y_optimal - velocity[0]) < 0.01)
    assert np.all(np.abs(vel_x_optimal - velocity[1]) < 0.01)
    assert np.all(np.abs(depth_optimal - depth) < 0.05)
    
    
def test_iwave(img_size=(128, 64, 64), res=0.02, fps=12):
    """Compare one-pass and reduced-then-full fitting on a generated spectrum."""
    np.random.seed(0)
    kt, ky, kx = spectral.wave_numbers(img_size, res, fps)
    velocity = [1, 0]
    depth = 0.2
    velocity_indx = 1
    depth_min = 0.01
    depth_max = 1
    
    iw = Iwave(
        resolution=res,
        window_size=(img_size[1], img_size[2]),
        overlap=(0, 0),
        time_size=128,
        time_overlap=0,
        fps=fps,
    )
    
    kt_gw, kt_turb = dispersion.dispersion(
        ky,
        kx,
        velocity,
        depth,
        velocity_indx
    )
    synthetic_spectrum = dispersion.theoretical_spectrum(
        kt_gw,
        kt_turb,
        kt,
        gauss_width=1,
        gravity_waves_switch=True,
        turbulence_switch=True
    )
    # Repeat the same spectrum to represent several windows.
    iw.spectrum = np.tile(synthetic_spectrum, (2,1,1,1))

    iw.x = np.array([0., 1.])
    iw.y = np.array([0.])
    
    # Fit once using the full spectrum.
    iw.velocimetry(
        alpha=0.85,  # alpha is depth-averaged velocity divided by surface velocity.
        depth=depth,
        popsize= 20,
        pass_downsampling=[1],
        dmin=depth_min,
        dmax=depth_max,
        gravity_waves_switch=True,
        turbulence_switch=True,
    )
    
    vy_1step = iw.vy
    vx_1step = iw.vx 
    d_1step = iw.d
       
    # Fit a reduced spectrum first, then the full spectrum.
    iw.velocimetry(
        alpha=0.85,  # alpha is depth-averaged velocity divided by surface velocity.
        depth=depth,
        popsize= 20,
        dmin=depth_min,
        dmax=depth_max,
        gravity_waves_switch=True,
        turbulence_switch=True,
    )

    vy_2steps = iw.vy
    vx_2steps = iw.vx 
    d_2steps = iw.d
    print("vy_1step:", vy_1step, "expected:", velocity[0])
    print("vx_1step:", vx_1step, "expected:", velocity[1])
    print("d_1step:", d_1step, "expected:", depth)
    print("vy_2steps:", vy_2steps, "expected:", velocity[0])
    print("vx_2steps:", vx_2steps, "expected:", velocity[1])
    print("d_2steps:", d_2steps, "expected:", depth)

    assert np.all(np.abs(vy_1step - velocity[0]) < 0.01)
    assert np.all(np.abs(vx_1step - velocity[1]) < 0.01)
    assert np.all(np.abs(d_1step - depth) < 0.05)

    assert np.all(np.abs(vy_2steps - velocity[0]) < 0.01)
    assert np.all(np.abs(vx_2steps - velocity[1]) < 0.01)
    assert np.all(np.abs(d_2steps - depth) < 0.05)


def test_window_batches_reuse_one_worker_pool(monkeypatch):
    """Check that the DE pipeline handles one batch at a time and reuses its fitting processes."""
    from concurrent.futures import ThreadPoolExecutor

    imgs = np.random.default_rng(6).integers(0, 255, (64, 8, 24), dtype=np.uint8)
    iw = Iwave(.1, window_size=(8, 8), time_size=32, time_overlap=16,
               fps=20., imgs=imgs, smax=1., window_chunk_size=2)
    batches, pools = [], []
    extract = spectral._FFTWorkspace._power_rectangular

    def tracked_extract(workspace, images, origins, shape, win_t, overlap):
        batches.append(len(origins))
        return extract(workspace, images, origins, shape, win_t, overlap)

    class CountPool(ThreadPoolExecutor):
        def __init__(self, **kwargs):
            pools.append(kwargs['max_workers'])
            # Run the actual DE fitting function in this process for the small test,
            # without starting extra Python processes or redirecting all program output.
            super().__init__(max_workers=kwargs['max_workers'])

    monkeypatch.setattr(spectral._FFTWorkspace, '_power_rectangular', tracked_extract)
    monkeypatch.setattr(optimise, 'ProcessPoolExecutor', CountPool)
    iw.velocimetry(depth=.3, pass_downsampling=[1], popsize=2,
                   maxiter=1, seed=0, polish=False)
    assert batches == [2, 1]
    assert len(pools) == 1
    assert pools[0] <= 2
    assert iw.vx.shape == (1, 3)
