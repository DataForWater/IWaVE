"""Main solver additions, using synthetic spectra like test_optimise.py."""
import numpy as np
import pytest
from iwave import dispersion, optimise, optimise_hnm as hnm, spectral


def test_de_default_and_separate_backend_options(flow, inline_pool):
    """DE stays the default; its old and new option styles give the same fit."""
    old, new = flow.make_iwave(), flow.make_iwave()
    # Old nonempty kwargs replace defaults; the new dictionary merges them.
    # Supply the same complete settings to compare like with like.
    options = dict(seed=7, maxiter=150, popsize=8, tol=1e-8, atol=1e-12,
                   polish=True, strategy='best1bin', init='sobol', workers=1)
    old.velocimetry(depth=flow.depth, progress=False, penalty_weight=0,
                    pass_downsampling=[1], **options)
    new.velocimetry(method='DE', depth=flow.depth, progress=False,
        de_options=dict(penalty_weight=0, pass_downsampling=[1], scipy_options=options))
    np.testing.assert_array_equal([old.vy, old.vx, old.d], [new.vy, new.vx, new.d])
    assert np.allclose([new.vy.item(), new.vx.item()], [.13, .47], atol=.01)
    assert bool(new.status) and new.spectral_uncertainty is None and new.mode_ambiguity is None
    with pytest.raises(ValueError): optimise.resolve_method_options('HNM', de_options={})
    with pytest.raises(ValueError): optimise.normalise_method('Hough')


def test_compiled_ridge_points_keep_plateaus_and_column_order():
    """Plateau interpolation and point order match the original peak loop."""
    columns = np.array([[0., 0.], [1., 2.], [1., 0.], [0., 2.],
                        [3., 0.], [0., 0.], [0., 0.]])
    points = hnm._ridge_points(columns, np.array([[0, 0], [0, 1]]),
                               np.arange(7.), np.array([-1.]), np.array([2., 3.]), .5, .5)
    np.testing.assert_array_equal(points, [[2., -1., 1.5, 1.], [2., -1., 4., 1.],
                                           [3., -1., 1., 1.], [3., -1., 3., 1.]])


def test_compiled_votes_match_original_rasterisation():
    """Keep both physical branches, axial cases and adjacent-bin deposits."""
    vy, vx = np.linspace(-.8, .8, 17), np.linspace(-1.2, 1.2, 25)
    dy, dx = vy[1]-vy[0], vx[1]-vx[0]
    points = np.array([[1.5, 0., .25, .7], [0., 2., .35, 1.],
                       [-2., 1., 1.8, .4], [1., -2., 2.4, .2]])
    k = np.hypot(points[:, 0], points[:, 1])
    beta, intrinsic = dispersion.beta_calc(k, .15, .85), dispersion.omega_i_calc(k, .15)
    expected = np.zeros((len(vy), len(vx)))
    for (x, y, w, weight), b, oi in zip(points, beta, intrinsic):
        branches = [w, ((1-b)*w-np.sqrt(b*b*w*w+(1-2*b)*oi*oi))/(1-2*b)]
        for a in branches:
            swap = abs(y*dy) < abs(x*dx)
            pos = ((a-y*vy)/x-vx[0])/dx if swap else ((a-x*vx)/y-vy[0])/dy
            low = np.floor(pos).astype(np.int64); frac = pos-low
            for shift, weights in ((0, 1-frac), (1, frac)):
                dest = low+shift; limit = len(vx) if swap else len(vy)
                valid = (dest >= 0) & (dest < limit); other = np.flatnonzero(valid)
                index = (other, dest[valid]) if swap else (dest[valid], other)
                np.add.at(expected, index, weight*weights[valid])
    actual = np.zeros_like(expected)
    hnm._accumulate_votes(actual, points, beta, intrinsic, vy, vx, dy, dx, True, True)
    np.testing.assert_array_equal(actual, expected)


def test_hough_fixed_adaptive_sampling_and_window_scaled_cutoff(flow):
    """Both sampling plans recover the ridge without exceeding their k-pair budget."""
    bounds = [(-1, 1), (-1, 1), (flow.depth, flow.depth)]
    ky, kx = flow.axes[1:]
    pairs, cutoff = hnm.eligible_pairs(ky, kx)
    expected = 2*min(abs(ky[1]-ky[0]), abs(kx[1]-kx[0]))
    assert np.isclose(cutoff, expected)
    assert np.isclose(hnm.eligible_pairs(ky/2, kx/2)[1], cutoff/2)
    for sampling in ['fixed', 'adaptive']:
        result = hnm.optimize_single_spectrum_velocity_hnm(flow.measured, bounds, .85,
            flow.dims, flow.res, flow.fps, return_metadata=True,
            hnm_options={**flow.hnm_options, 'sampling': sampling, 'filling_ratio': .8, 'refine': False})
        assert np.allclose(result[:2], [.13, .47], atol=.05)
        assert result[5] and len(result[7]['selected_pairs']) <= hnm.budget(.8, len(ky), len(kx), len(pairs))


def test_hnm_fixed_and_free_depth_obey_trust_region(flow, inline_pool):
    """Recover velocity/depth while keeping all existing starts inside the disk."""
    for depth, candidates, evaluations in [(flow.depth, 1, 160), (0, 12, 640)]:
        iw = flow.make_iwave()
        iw.velocimetry(method='HNM', depth=depth, dmin=.08, dmax=.4,
                       hnm_options=flow.hnm_options, progress=False)
        assert bool(iw.status)
        assert np.allclose([iw.vy.item(), iw.vx.item(), iw.d.item()], [.13, .47, .15], atol=.01)
        assert iw.depth_estimated == (depth == 0)
        record = iw.sampling_metadata[0, 0]; trust = record['trust_region']
        assert 1 <= len(record['local_runs']) <= candidates
        assert hnm.resolve_hnm_options(bnds=[(-1,1),(-1,1),(.08,.4) if depth == 0 else (.15,.15)])['maxfev'] == evaluations
        for run in record['local_runs']:
            assert run['nfev'] <= evaluations and run['feasible']
            assert np.linalg.norm(np.array(run['velocity'])-trust['centre_vy_vx']) <= trust['radius'] + 1e-12
        assert np.isclose(trust['radius'], max(.5*np.linalg.norm(trust['centre_vy_vx']), trust['min_radius']))
    empty = hnm.optimize_single_spectrum_velocity_hnm(np.zeros_like(flow.measured),
        [(-1,1),(-1,1),(.15,.15)], .85, flow.dims, flow.res, flow.fps, hnm_options=flow.hnm_options)
    assert not empty[5]  # No fabricated successful vector when no ridge exists.


def test_fast_overlap_matches_original_dense_model(flow):
    """The fast evaluator changes runtime, not the public theoretical spectrum."""
    before = flow.measured.copy()
    for waves, turbulence in [(True,True), (True,False), (False,True)]:
        objective = spectral.SpectralOverlapObjective(flow.measured, flow.axes, flow.depth,
            .85, waves, turbulence, dict(gauss_width=1., kernel_truncate=5.))
        velocity = (.2, .4)
        model = dispersion.intensity(velocity, flow.depth, .85, flow.dims, flow.res,
                                     flow.fps, 1., waves, turbulence)
        expected = -np.sum(flow.measured*model)/model.sum()
        assert np.isclose(objective(velocity), expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(flow.measured, before)


def test_fused_overlap_statistics_match_model_weights(flow):
    """Preserve dense/narrow support, log-depth and final fit statistics."""
    q = np.array([.17, .43, np.log(.19)])
    for width in (1., 8.):
        objective = spectral.SpectralOverlapObjective(flow.measured, flow.axes,
            flow.depth, .85, True, True, dict(gauss_width=width, kernel_truncate=5.))
        model = dispersion.intensity(q[:2], np.exp(q[2]), .85, flow.dims,
                                     flow.res, flow.fps, width, True, True)
        # The diagnostic weights use the original support convention, including
        # sparse-band truncation and counting overlapping branches only once.
        model *= objective.normalized_weights(q) > 0
        mass, square = model.sum(), np.sum(model*model)
        overlap = np.sum(model*flow.measured)
        expected = (-overlap/mass, mass, square, overlap)
        np.testing.assert_allclose(objective.statistics(q), expected, rtol=1e-11, atol=1e-11)
        assert objective.calls == 0
        assert np.isclose(objective(q), expected[0], rtol=1e-11, atol=1e-11)
        assert objective.calls == 1
