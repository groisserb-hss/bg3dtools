"""Taubin lambda|mu smoothing."""

import numpy as np
import pytest

from bg3dtools.mesh.generate import build_plane
from bg3dtools.mesh.laplace import taubin_smoothing
from bg3dtools.mesh.utils import adj_from_edges


def icosphere(subdiv=3):
    """A closed genus-0 sphere of radius 1, via trimesh."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def plain_laplacian(v, f, n):
    """Unweighted Laplacian smoothing — the shrinking comparison baseline."""
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = adj_from_edges(e, len(v))
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)[:, None]
    v = np.asarray(v, float).copy()
    for _ in range(n):
        v = (A @ v) / deg
    return v


def test_zero_iterations_returns_a_copy():
    f, v = build_plane(7, 7, return_vertices=True)
    v = np.asarray(v, float)
    out = taubin_smoothing(v, f, 0)
    assert np.array_equal(out, v)
    out[0, 0] = 99.0
    assert v[0, 0] != 99.0


def test_high_frequency_spike_is_attenuated():
    f, v = build_plane(11, 11, return_vertices=True)
    v = np.asarray(v, float).copy()
    apex = 5 * 11 + 5
    v[apex, 2] = 1.0
    out = taubin_smoothing(v, f, 20)
    assert abs(out[apex, 2]) < 0.15, "a one-vertex spike should be largely removed"


def test_scale_is_preserved_where_plain_laplacian_collapses():
    """The property that makes Taubin the right filter for measured smoothing.

    Plain Laplacian smoothing is mean-curvature flow: on a closed surface it
    contracts without bound. Any statistic that compares a smoothed surface to
    its input would then report contraction rather than feature removal.
    """
    v, f = icosphere()
    n = 60
    taubin_r = np.linalg.norm(taubin_smoothing(v, f, n), axis=1).mean()
    plain_r = np.linalg.norm(plain_laplacian(v, f, n), axis=1).mean()

    # Measured at 60 iterations: Taubin 0.997, plain Laplacian 0.506.
    assert taubin_r > 0.95, f"Taubin lost scale: mean radius {taubin_r:.3f}"
    assert plain_r < 0.75, (
        f"the baseline did not collapse (plain {plain_r:.3f} vs taubin {taubin_r:.3f}); "
        "this test is not comparing what it claims to")


def test_pinned_vertices_do_not_move():
    f, v = build_plane(9, 9, return_vertices=True)
    v = np.asarray(v, float).copy()
    v[:, 2] = np.random.default_rng(0).normal(0, 0.1, len(v))
    pinned = np.zeros(len(v), bool)
    pinned[:9] = True                       # one edge of the grid

    out = taubin_smoothing(v, f, 30, pinned=pinned)
    assert np.array_equal(out[pinned], v[pinned])
    assert not np.allclose(out[~pinned], v[~pinned])


def test_all_pinned_is_a_no_op():
    f, v = build_plane(7, 7, return_vertices=True)
    v = np.asarray(v, float)
    out = taubin_smoothing(v, f, 10, pinned=np.ones(len(v), bool))
    assert np.array_equal(out, v)


def test_more_iterations_smooth_monotonically():
    v, f = icosphere()
    rng = np.random.default_rng(1)
    noisy = v + rng.normal(0, 0.02, v.shape)

    prev = np.inf
    for n in (5, 10, 20, 40):
        resid = np.abs(np.linalg.norm(taubin_smoothing(noisy, f, n), axis=1) - 1.0).mean()
        assert resid < prev, f"{n} iterations did not improve on the previous count"
        prev = resid


@pytest.mark.parametrize("lam,mu", [(0.9, 0.5), (0.9, -0.5), (-0.9, -0.905), (0.9, -0.9)])
def test_invalid_lambda_mu_raises(lam, mu):
    """Requires lam > 0 > mu and |mu| > lam, or it is not a low-pass filter."""
    f, v = build_plane(5, 5, return_vertices=True)
    with pytest.raises(ValueError, match="lam"):
        taubin_smoothing(np.asarray(v, float), f, 5, lam=lam, mu=mu)


def test_empty_face_array_returns_input():
    v = np.zeros((4, 3))
    out = taubin_smoothing(v, np.zeros((0, 3), np.int64), 10)
    assert np.array_equal(out, v)
