"""Check noise and alternative-solution calculations using small generated spectra."""
import numpy as np
import pytest
from iwave import LazySpectrumArray, dispersion, io, optimise, optimise_hnm as hnm, spectral


def test_lazy_context_preserves_power_in_one_averaging_pass(flow, monkeypatch):
    """Keep full normalised power for noise checks without repeating the FFT."""
    windows = np.random.default_rng(3).normal(size=(1, 256, 12, 14))
    kt, ky, kx = flow.axes
    lazy = LazySpectrumArray(windows, 128, 64, kt, kx, ky, smax=1., threshold=.2)
    original = spectral.sliding_window_spectrum; calls = []
    def tracked(*args, **kwargs):
        raw = original(*args, **kwargs); calls.append(raw.copy()); return raw
    monkeypatch.setattr(spectral, 'sliding_window_spectrum', tracked)
    measured, context = lazy.get_with_spectral_context(0)
    assert len(calls) == 1 and context['n_averages'] == 3
    expected = calls[0][0]/calls[0][0].mean(axis=(-2,-1), keepdims=True)
    np.testing.assert_allclose(context['normalized_spectrum'], expected)
    np.testing.assert_array_equal(measured, lazy[0])
    assert np.isclose(spectral.effective_averages(3, 128, 64), 2.25)
    assert np.any((context['normalized_spectrum'] > 0) & (measured == 0))


def test_optional_native_outputs_and_strict_boolean_gates(flow, inline_pool):
    """Check that each optional calculation needs its own switch and does not alter the fit."""
    iw = flow.make_iwave(); args = dict(method='HNM', depth=flow.depth,
        hnm_options=flow.hnm_options, progress=False)
    iw.velocimetry(**args, spectral_uncertainty_options={'confidence': .9})
    point = np.array([iw.vy, iw.vx, iw.d]); assert iw.spectral_uncertainty is None
    with pytest.warns(UserWarning, match='Ambiguity cannot be assessed'):
        iw.velocimetry(**args, calculate_spectral_uncertainty=True,
                       calculate_mode_ambiguity=True, spectral_contexts=[flow.context])
    np.testing.assert_array_equal(point, [iw.vy, iw.vx, iw.d])
    assert isinstance(iw.spectral_uncertainty, io.SpectralUncertainty)
    assert iw.spectral_uncertainty.valid.item() and np.isfinite(iw.spectral_uncertainty.vx_bounds).all()
    # Depth was supplied, not estimated.
    assert np.isnan(iw.spectral_uncertainty.depth_bounds).all()
    assert iw.mode_ambiguity.ambiguous_solution.item() is None
    assert 'n_candidates = 1' in iw.mode_ambiguity.message
    with pytest.warns(UserWarning):
        iw.velocimetry(**args, calculate_mode_ambiguity=True, spectral_contexts=[flow.context])
    assert iw.spectral_uncertainty is None and iw.mode_ambiguity is not None
    iw.velocimetry(method='HNM', depth=0, dmin=.08, dmax=.4, hnm_options=flow.hnm_options,
        progress=False, calculate_spectral_uncertainty=True, spectral_contexts=[flow.context])
    lo, hi = iw.spectral_uncertainty.depth_bounds[0,0]
    assert 0 < lo < iw.d.item() < hi and iw.mode_ambiguity is None
    assert not np.isclose(hi-iw.d.item(), iw.d.item()-lo, rtol=1e-3)
    with pytest.raises(ValueError): iw.velocimetry(**args, calculate_spectral_uncertainty=1)
    with pytest.raises(ValueError): iw.velocimetry(**args, calculate_mode_ambiguity='True')
    with pytest.raises(TypeError): iw.velocimetry(**args, calculate_uncertainty=True)


def test_covariance_scales_with_averaging_and_curvature_respects_bounds(flow, inline_pool):
    """Check that more averages reduce predicted noise, and reject score checks at search limits."""
    iw = flow.make_iwave()
    iw.velocimetry(method='HNM', depth=flow.depth, hnm_options=flow.hnm_options, progress=False)
    records = [spectral.estimate_spectral_uncertainty(flow.measured, [iw.vy.item(), iw.vx.item()],
        flow.depth, False, flow.dims, flow.res, flow.fps, .85, flow.context,
        bounds=[(-1,1),(-1,1),(.15,.15)], options={'n_effective': count}) for count in [1.,4.]]
    assert all(record['valid'] for record in records)
    np.testing.assert_allclose(records[0]['covariance'], 4*np.array(records[1]['covariance']), rtol=1e-9)
    feasible = lambda p: bool(np.all(np.abs(p) <= .1))
    hessian = np.array([[3., .5], [.5, 2.]])
    exact = io._curvature(lambda p: .5*p@hessian@p, np.zeros(2), np.ones(2),
                          feasible, io.resolve_curvature_options())
    assert exact['valid']
    np.testing.assert_allclose(exact['hessian'], hessian, rtol=1e-9)
    visited = []
    def quadratic(p): visited.append(p.copy()); return float(p@p)
    failed = io._curvature(quadratic, [.1,0.], np.ones(2), feasible, io.resolve_curvature_options())
    assert not failed['valid'] and 'boundary_stencil_unavailable' in failed['flags']
    assert all(feasible(point) for point in visited)


def test_observed_mode_ambiguity_reuses_existing_endpoints():
    """Check that two patterns can give competing fits, and reduced noise can distinguish them."""
    dims, res, fps, depth = (128,16,16), .25, 20., .15
    axes = spectral.wave_numbers(dims, res, fps)
    raw = dispersion.intensity((.05,.9), depth, .85, dims, res, fps, 1.)
    raw += dispersion.intensity((-.2,.75), depth, .85, dims, res, fps, 1.) + .02
    measured = spectral.spectrum_preprocessing(raw[None], *axes, 4.5, spectrum_threshold=.1)[0]
    context = dict(normalized_spectrum=raw/raw.mean(axis=(-2,-1), keepdims=True),
                   n_averages=3, time_size=128, time_overlap=64)
    bounds = [(-1.5,1.5),(-1.5,1.5),(.15,.15)]
    options = dict(n_candidates=12, maxfev=300, velocity_step=.025,
                   peak_height=.03, peak_prominence=.03, min_ridges=3)
    args = dict(measured_spectrum=measured, bnds=bounds, vel_indx=.85, window_dims=dims,
        res=res, fps=fps, gravity_waves_switch=True, turbulence_switch=True,
        method='HNM', hnm_options=options, _return_metadata=True)
    base = optimise.optimize_single_spectrum_velocity_unpack(args)
    diagnosed = optimise.optimize_single_spectrum_velocity_unpack(dict(args,
        calculate_mode_ambiguity=True, _spectral_context=context))
    np.testing.assert_array_equal(base[:6], diagnosed[:6])
    assert base[7]['local_runs'] == diagnosed[7]['local_runs']
    mode = diagnosed[7]['mode_ambiguity']
    assert mode['ambiguous'] and mode['n_distinct_modes'] >= 2 and mode['extra_searches'] == 0
    resolved = hnm.estimate_mode_ambiguity(measured, diagnosed[:2], depth, bounds, dims,
        res, fps, .85, context, diagnosed[7], hnm.resolve_hnm_options(options, bnds=bounds),
        options={'n_effective': 225.}, optimizer_success=True)
    assert not resolved['ambiguous'] and resolved['extra_searches'] == 0
