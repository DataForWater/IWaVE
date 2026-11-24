"""Main result/plot additions; figures use Agg so tests need no GUI."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pytest
from iwave import io


def test_native_results_and_curvature_live_in_io():
    """Moved objects remain native, with partial bounds and honest unknown flags."""
    for obj in [io.AlternativeEstimate, io.SpectralUncertainty, io.ModeAmbiguity,
                io.estimate_objective_curvature]:
        assert obj.__module__ == 'iwave.io'
    for name in ['results', 'uncertainty', 'plots', 'plotting', 'spectral_objective',
                 'spectral_uncertainty', 'mode_ambiguity']:
        assert importlib.util.find_spec('iwave.'+name) is None
    s = io.SpectralUncertainty((1,2))
    s._set_window((0,0), dict(valid=False, accepted_intervals=dict(vx=[.4,.6], vy=[-.1,.1], depth=None)), None)
    assert not s.valid[0,0] and np.isfinite(s.vx_bounds[0,0]).all()
    assert np.isnan(s.depth_bounds[0,0]).all() and s.ambiguous_solution[0,0] is None
    m = io.ModeAmbiguity((1,2)); rival = io.AlternativeEstimate(.7,-.1,.3,True)
    m.additional_estimates[0,0].append(rival)
    assert m.additional_estimates[0,1] == []  # Lists are not shared between windows.
    assert rival.vx_bounds is None and rival.valid is None


def test_transect_quality_depth_positions_and_unknown_markers(plot_field):
    """Quality is last; depth is automatic; pixels and metres locate the same fits."""
    iw = plot_field; iw.status.windows[1,0] = False
    iw.mode_ambiguity.ambiguous_solution[0,0] = True
    iw.mode_ambiguity.additional_estimates[0,0] = [
        io.AlternativeEstimate(.7, -.2, .4, True), io.AlternativeEstimate(.8, -.3, .5, False)]
    fig, axes = iw.plot_transect(column=0)
    assert len(axes) == 4 and axes[-1].get_ylabel() == 'Quality'
    np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), [25,75,125])
    assert any(line.get_marker() == 'x' for line in axes[0].lines)
    assert not any(line.get_marker() == 's' for ax in axes for line in ax.lines)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert 'ambiguity unassessed' not in labels
    assert 'competitive rival (no bounds)' in labels and 'separated rival (no bounds)' in labels
    assert not any(line.get_marker() == 'D' for line in axes[-1].lines)  # No rival quality invented.
    fig.canvas.draw(); legend = fig.legends[0].get_window_extent()
    assert all(legend.x0 >= ax.get_window_extent().x1 for ax in axes)
    metric, ma = iw.plot_transect(column=0, distance_units='m')
    np.testing.assert_allclose(ma[0].lines[0].get_xdata(), [.25,.75,1.25])
    iw.depth_estimated = False
    fixed, fa = iw.plot_transect(column=0)
    assert len(fa) == 3 and fa[-1].get_ylabel() == 'Quality'
    plt.close(fig); plt.close(metric); plt.close(fixed)


def test_vector_validation_units_and_known_ambiguity(plot_field):
    """Filter failed fits, colour only known rivals, and convert positions not m/s."""
    iw = plot_field; iw.status.windows[1,0] = False
    iw.mode_ambiguity.ambiguous_solution[0,0] = True
    p = iw.plot_velocimetry(); m = iw.plot_velocimetry(distance_units='m')
    assert len(p.U) == 2
    np.testing.assert_allclose(m.get_offsets(), p.get_offsets()*.01)
    np.testing.assert_array_equal(p.U, m.U)
    np.testing.assert_allclose(p.get_facecolors()[0], to_rgba('tab:red'))
    np.testing.assert_allclose(p.get_facecolors()[1], to_rgba('C0'))
    all_vectors = iw.plot_velocimetry(validate_vectors=False)
    assert len(all_vectors.U) == 3
    plt.close('all')


