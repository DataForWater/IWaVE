import numpy as np
import time
import pytest
from iwave import spectral, dispersion, optimise


def test_preprocessing():
    """Check shape of preprocessed spectrum against expected shape."""
    img = np.random.rand(3, 32, 32, 32)
    kt, ky, kx = spectral.wave_numbers(img.shape, res=0.02, fps=25)
    kt_gw, kt_turb = dispersion.dispersion(ky, kx, velocity=[1, 0], depth=1, vel_indx=1)
    synthetic_spectrum = dispersion.theoretical_spectrum(kt_gw, kt_turb, kt, gauss_width=1, gravity_waves_switch=True, 
                                                         turbulence_switch=True)
    measured_spectrum = spectral.sliding_window_spectrum(img, img.shape[1], 0, "numpy")
    preprocessed_spectrum = optimise.spectrum_preprocessing(measured_spectrum, kt, ky, kx, velocity_threshold=5)
    #test if the size of the preprocessed spectrum matches the one of the theoretical spectrum
    assert preprocessed_spectrum[0].shape == synthetic_spectrum.shape

    
def test_nsp(img_size=(256, 64, 64), res=0.02, fps=25):
    """Compute nsp cost function, compare against known value."""
    kt, ky, kx = spectral.wave_numbers(img_size, res, fps)
    kt_gw, kt_turb = dispersion.dispersion(ky, kx, velocity=[1, 0], depth=1, vel_indx=1)
    synthetic_spectrum = dispersion.theoretical_spectrum(kt_gw, kt_turb, kt, gauss_width=1, gravity_waves_switch=True, 
                                                         turbulence_switch=True)
    cost = optimise.nsp_inv(synthetic_spectrum,synthetic_spectrum)
    expected_cost = np.sum(synthetic_spectrum) / np.sum(synthetic_spectrum**2)
    #test if the auto-correlation matches the theoretical expectation based on a synthetic spectrum
    assert np.allclose(cost, expected_cost)


def test_cost_function_velocity(img_size=(256, 64, 64), res=0.02, fps=25):
    """Check cost function (without depth) against expected gradients."""
    kt, ky, kx = spectral.wave_numbers(img_size, res, fps)
    velocity_1 = [1, 0]
    velocity_2 = [1.01, 0]
    velocity_3 = [0.99, 0]
    depth = 1
    vel_indx=1
    kt_gw_1, kt_turb_1 = dispersion.dispersion(ky, kx, velocity_1, depth, vel_indx)
    synthetic_spectrum_1 = dispersion.theoretical_spectrum(
        kt_gw_1,kt_turb_1, kt, gauss_width=1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    cost_11 = optimise.cost_function_velocity(
        velocity_1, synthetic_spectrum_1, depth, vel_indx,
        img_size, res, fps, gauss_width=1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    cost_12 = optimise.cost_function_velocity(
        velocity_2, synthetic_spectrum_1, depth, vel_indx,
        img_size, res, fps, gauss_width=1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    cost_13 = optimise.cost_function_velocity(
        velocity_3, synthetic_spectrum_1, depth, vel_indx,
        img_size, res, fps, gauss_width=1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    #test if the cost function increases when the velocity deviates from optimal
    assert cost_12 > cost_11
    assert cost_13 > cost_11
    

def test_optimise_velocity(img_size=(64, 32, 32), res=0.02, fps=25):
    """Check hypothetical case optimization without depth (only u, v) for one single window."""
    kt, ky, kx = spectral.wave_numbers(img_size, res, fps)
    velocity = [1, 0]
    depth = 0.3
    velocity_indx = 1
    vel_y_min = 0
    vel_y_max = 2
    vel_x_min = -0.5
    vel_x_max = 0.5
    vel_threshold = 5
    kt_gw, kt_turb = dispersion.dispersion(ky, kx, velocity, depth, velocity_indx)
    synthetic_spectrum = dispersion.theoretical_spectrum(
        kt_gw,
        kt_turb,
        kt,
        gauss_width=1,
        gravity_waves_switch=True,
        turbulence_switch=True
    )
    bounds = [(vel_y_min, vel_y_max), (vel_x_min, vel_x_max)]
    t1 = time.time()

    synthetic_spectrum = np.tile(synthetic_spectrum, (2,1,1,1)) # simulate multiple windows
    optimal = optimise.optimise_velocity(
        synthetic_spectrum,
        bounds,
        depth,
        velocity_indx,
        img_size,
        res,
        fps,
        gauss_width=1,
        gravity_waves_switch=True,
        turbulence_switch=True,
        popsize=10,
        maxiter=10000,
        workers=1
    )
    t2 = time.time()
    print(f"Took {t2 - t1} seconds")
    vel_y_optimal = optimal[:,0]
    vel_x_optimal = optimal[:,1]
    assert vel_x_max >= vel_x_min
    assert vel_y_max >= vel_y_min
    assert np.all(np.abs(vel_y_optimal - velocity[0]) < 0.01)
    assert np.all(np.abs(vel_x_optimal - velocity[1]) < 0.01)
    print(f"Original velocity was {velocity}, optimized {optimal}")


def test_cost_function_velocity_depth(img_size=(256, 64, 64), res=0.02, fps=25):
    """Check cost function (with depth) against expected gradients."""
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
        img_size, res, fps, gauss_width=1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    cost_12 = optimise.cost_function_velocity_depth(
        params_2, synthetic_spectrum_1, vel_indx,
        img_size, res, fps, gauss_width=1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    cost_13 = optimise.cost_function_velocity_depth(
        params_3, synthetic_spectrum_1, vel_indx,
        img_size, res, fps, gauss_width=1,
        gravity_waves_switch=True, turbulence_switch=True
    )
    #test if the cost function increases when the depth deviates from optimal
    assert cost_12 > cost_11
    assert cost_13 > cost_11


@pytest.mark.skip(reason="Optimization with depth is not yet stable")
def test_optimise_velocity_depth(img_size=(128, 64, 64), res=0.02, fps=12):
    """Check hypothetical case optimization with depth for one single window."""
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
    # synthetic_spectrum = optimise.spectrum_preprocessing(synthetic_spectrum, kt, ky, kx, velocity_threshold=10, spectrum_threshold=1)
    # define ranges for optimization

    synthetic_spectrum = np.tile(synthetic_spectrum, (2,1,1,1)) # simulate multiple windows
    vel_y_min = 0
    vel_y_max = 2
    vel_x_min = -0.5
    vel_x_max = 0.5
    depth_min = 0.01
    depth_max = 1
    vel_threshold = 5
    bounds = [(vel_y_min, vel_y_max), (vel_x_min, vel_x_max), (depth_min, depth_max)]
    t1 = time.time()
    optimal = optimise.optimise_velocity_depth(
        synthetic_spectrum,
        bounds,
        velocity_indx,
        img_size,
        res,
        fps,
        gauss_width=1,
        gravity_waves_switch=True,
        turbulence_switch=True,
        popsize=10,
        workers=1,
        maxiter=1000,
        updating="deferred"
    )
    vel_y_optimal = optimal[:,0]
    vel_x_optimal = optimal[:,1]
    depth_optimal = optimal[:,2]
    print(f"Original velocity/depth was {velocity, depth}, optimized {optimal}")
    t2 = time.time()
    print(f"Took {t2 - t1} seconds")
    assert vel_x_max >= vel_x_min
    assert vel_y_max >= vel_y_min
    assert depth_max >= depth_min
    assert np.all(np.abs(vel_y_optimal - velocity[0]) < 0.01)
    assert np.all(np.abs(vel_x_optimal - velocity[1]) < 0.01)
    assert np.all(np.abs(depth_optimal - depth) < 0.05)