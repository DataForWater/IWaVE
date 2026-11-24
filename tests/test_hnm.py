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
