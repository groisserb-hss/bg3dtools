"""nonrigid_ICP and its parts: the direct vertex solve against LSQR, the stratified candidate sampler, the
inner-loop window and cap, the iteration record, and the default behaviour every existing caller relies on.

The meshes are subdivided icosahedra on the unit sphere (nonrigid_ICP asserts faces > 1.8 * vertices: the
plain icosahedron fails that, a once-subdivided one passes). The scan is a noisy ellipsoid, never an exact
fit, so the convergence ratio never meets 0/0."""
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from bg3dtools import igl_compat
from bg3dtools.mesh import (NRICPResult, discrete_match, equivalent_membrane, fit_vertices, nonrigid_ICP, ordered_edges,
                            per_face_normals, stratified_candidate_count, stratified_surface_sample,
                            surface_match)
from bg3dtools.mesh.generate import generate_icosahedron


def _icosphere(k):
    v, f = generate_icosahedron()
    v, f = igl_compat.upsample(v, f.astype(np.int64), k)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v, f


def _area(v, f):
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


@pytest.fixture(scope='module')
def mesh():
    return _icosphere(2)              # 162 vertices, 320 faces


@pytest.fixture(scope='module')
def scan(mesh):
    """2000 points on a noisy ellipsoid around the sphere, with the ellipsoid's normals."""
    v, f = mesh
    rng = np.random.default_rng(0)
    scale = np.array([1.0, 0.8, 1.2])
    pmap, fidx, _ = stratified_surface_sample(v, f, 2000, rng)
    pts = (pmap @ v) * scale + 0.01 * rng.standard_normal((2000, 3))
    normals = per_face_normals(v * scale, f)[fidx]
    return pts, normals


def _system(mesh, scan, rad=0.3, landmarks=True):
    """The correspondences and weights of one NRICP iteration, as nonrigid_ICP builds them."""
    v, f = mesh
    pts, _ = scan
    edges = ordered_edges(f)
    d2, fidx, bc = surface_match(pts, v, f)
    K = 3 * len(f)
    scan_w = np.sqrt(1.0 / (rad ** 2 + d2)) * rad / K
    edge_w = np.full(len(edges), 0.2 / len(edges))
    kw = {}
    if landmarks:
        idx = np.array([0, 17, 40, 80, 150])
        kw = dict(landmark_regressor=sparse.csr_matrix((np.ones(5), (np.arange(5), idx)), (5, len(v))),
                  landmark_targets=v[idx] + 0.01, landmark_w=np.full(5, 0.1))
    return dict(initV=v, faces=f, modelV=v, fidx=fidx, bc=bc, scanpts=pts, edges=edges,
                scan_w=scan_w, edge_w=edge_w, **kw)


# ---------------------------------------------------------------------------
# fit_vertices: the solver
# ---------------------------------------------------------------------------

def test_fit_vertices_direct_matches_lsqr(mesh, scan):
    sys_ = _system(mesh, scan)
    lsqr = fit_vertices(**sys_, solver='lsqr')
    direct = fit_vertices(**sys_, solver='direct')
    np.testing.assert_allclose(direct, lsqr, atol=1e-4)      # metres on a unit sphere


def test_fit_vertices_direct_holds_undetermined_vertices_at_init(mesh, scan):
    """Two disjoint spheres, the scan matched to the first only: the second has edge rows alone, which a
    translation leaves unchanged, so the normal matrix is singular there. The ridge keeps that sphere at
    its init (LSQR's warm start does the same), and the first sphere is the same solve as before."""
    v, f = mesh
    sys_ = _system(mesh, scan, landmarks=False)
    shift = np.array([5.0, 0.0, 0.0])
    v2 = np.vstack([v, v + shift])
    f2 = np.vstack([f, f + len(v)])
    per_edge = sys_['edge_w'][0]                      # the same weight per edge as the single sphere
    sys_.update(initV=v2, faces=f2, modelV=v2, edges=ordered_edges(f2))
    sys_['edge_w'] = np.full(len(sys_['edges']), per_edge)
    direct = fit_vertices(**sys_, solver='direct')
    lsqr = fit_vertices(**sys_, solver='lsqr')
    assert np.all(np.isfinite(direct))
    np.testing.assert_allclose(direct[len(v):], v + shift, atol=1e-6)      # untouched: nothing pulls it
    np.testing.assert_allclose(direct[:len(v)], lsqr[:len(v)], atol=1e-4)  # the determined sphere agrees
    one = fit_vertices(**_system(mesh, scan, landmarks=False), solver='direct')
    np.testing.assert_allclose(direct[:len(v)], one, atol=1e-6)            # and is the single-sphere solve


def test_fit_vertices_rejects_unknown_solver(mesh, scan, monkeypatch):
    dumped = []
    monkeypatch.setattr(np, 'savez', lambda *a, **k: dumped.append(a))
    with pytest.raises(ValueError):
        fit_vertices(**_system(mesh, scan), solver='cholmod')
    assert dumped == []                                       # refused before anything was built


# ---------------------------------------------------------------------------
# the stratified sampler
# ---------------------------------------------------------------------------

def test_stratified_counts_follow_area():
    v, f = _icosphere(2)
    v = v * np.array([1.0, 1.0, 3.0])                        # unequal face areas
    n = 200_000
    _, fidx, _ = stratified_surface_sample(v, f, n, np.random.default_rng(1))
    counts = np.bincount(fidx, minlength=len(f))
    assert counts.sum() == n
    area = _area(v, f)
    np.testing.assert_allclose(counts / n, area / area.sum(), atol=1e-3)
    weights = np.ones(len(f))
    weights[: len(f) // 2] = 0.0
    _, fidx_w, _ = stratified_surface_sample(v, f, n, np.random.default_rng(1), face_weights=weights)
    assert np.bincount(fidx_w, minlength=len(f))[: len(f) // 2].sum() == 0


def test_stratified_points_lie_on_their_faces(mesh):
    v, f = mesh
    n = 5000
    pmap, fidx, bc = stratified_surface_sample(v, f, n, np.random.default_rng(2))
    assert pmap.shape == (n, len(v)) and fidx.shape == (n,) and bc.shape == (n, 3)
    assert bc.min() >= 0.0
    np.testing.assert_allclose(bc.sum(axis=1), 1.0, atol=1e-12)
    pts = pmap @ v
    assert igl_compat.point_mesh_squared_distance(pts, v, f)[0].max() < 1e-20
    # each point lies on the face it is assigned to
    tri = v[f[fidx]]
    np.testing.assert_allclose(np.einsum('nk,nkd->nd', bc, tri), pts, atol=1e-12)


def test_stratified_is_a_fresh_seeded_draw(mesh):
    v, f = mesh
    a = stratified_surface_sample(v, f, 1000, np.random.default_rng(5))
    b = stratified_surface_sample(v, f, 1000, np.random.default_rng(5))
    c = stratified_surface_sample(v, f, 1000, np.random.default_rng(6))
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert not (np.array_equal(a[1], c[1]) and np.array_equal(a[2], c[2]))
    rng = np.random.default_rng(7)
    d = stratified_surface_sample(v, f, 1000, rng)
    e = stratified_surface_sample(v, f, 1000, rng)
    assert not np.array_equal(d[2], e[2])                     # two calls on one generator differ


def test_stratified_rejects_a_zero_area_mesh():
    v = np.zeros((4, 3))
    f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    with pytest.raises(ValueError):
        stratified_surface_sample(v, f, 10, np.random.default_rng(0))


def test_stratified_candidate_count(mesh):
    v, f = mesh
    area = _area(v, f).sum()
    assert stratified_candidate_count(v, f, res=10.0) == round(area * 100)
    assert stratified_candidate_count(v, f, res=10.0, sample_scale=1.3) == round(1.3 * area * 100)
    assert stratified_candidate_count(v, f, res=1e-6) == 1


@pytest.mark.parametrize('sampler', ['voxel', 'stratified'])
def test_discrete_match_both_samplers_return_valid_matches(mesh, scan, sampler):
    v, f = mesh
    pts, normals = scan
    d2, fidx, bc = discrete_match(pts, normals, v, f, per_face_normals(v, f), radius=0.3, sampler=sampler,
                                  rng=np.random.default_rng(0))
    assert np.all(np.isfinite(d2)) and d2.shape == (len(pts),)
    assert fidx.min() >= 0 and fidx.max() < len(f)
    assert bc.min() >= -1e-12 and np.allclose(bc.sum(axis=1), 1.0)


def test_discrete_match_rejects_unknown_sampler(mesh, scan):
    v, f = mesh
    pts, normals = scan
    with pytest.raises(ValueError):
        discrete_match(pts, normals, v, f, per_face_normals(v, f), sampler='grid')


# ---------------------------------------------------------------------------
# nonrigid_ICP: defaults, window, cap, stats
# ---------------------------------------------------------------------------

def _run(mesh, scan, **kw):
    v, f = mesh
    pts, normals = scan
    kw.setdefault('rad', 0.3)
    kw.setdefault('discrete', True)
    return nonrigid_ICP(pts, f, v, pt_normals=normals, **kw)


def test_default_call_returns_the_tuple_and_the_defaults_are_pinned(mesh, scan):
    verts, err = _run(mesh, scan)
    assert verts.shape == mesh[0].shape and np.isfinite(err)
    params = inspect.signature(nonrigid_ICP).parameters
    assert params['window'].default == 3 and params['max_iter'].default is None
    assert params['solver'].default == 'direct' and params['sampler'].default == 'stratified'
    assert params['sample_scale'].default == 1.0 and params['rng'].default is None
    assert params['return_stats'].default is False
    assert params['converge_thresh'].default == 0.01 and params['model_weight'].default == 0.2


def test_window_sets_the_iteration_floor(mesh, scan):
    assert _run(mesh, scan, converge_thresh=np.inf, return_stats=True).n_iters == 6
    assert _run(mesh, scan, converge_thresh=np.inf, window=2, return_stats=True).n_iters == 4


def test_cap_stops_at_the_cap(mesh, scan):
    assert _run(mesh, scan, converge_thresh=0.0, max_iter=3, return_stats=True).n_iters == 3
    assert _run(mesh, scan, converge_thresh=0.0, max_iter=1, return_stats=True).n_iters == 1


def test_stats_are_consistent(mesh, scan):
    r = _run(mesh, scan, return_stats=True)
    assert isinstance(r, NRICPResult)
    n = r.n_iters
    assert n >= 6 and len(r.errors) == len(r.iter_s) == len(r.match_s) == len(r.solve_s) == n
    assert r.err == r.errors[-1]
    assert r.total_s >= r.iter_s.sum() - 1e-9
    assert np.all(r.iter_s + 1e-9 >= r.match_s + r.solve_s)
    assert r.verts.shape == mesh[0].shape


def test_direct_equals_lsqr_with_a_shared_seed(mesh, scan):
    kw = dict(max_iter=1, sampler='stratified', return_stats=True)
    a = _run(mesh, scan, rng=7, solver='lsqr', **kw)
    b = _run(mesh, scan, rng=7, solver='direct', **kw)
    np.testing.assert_allclose(a.verts, b.verts, atol=1e-4)   # identical correspondences, exact vs LSQR
    assert a.errors[0] == b.errors[0]


def test_stratified_run_is_reproducible_from_a_seed(mesh, scan):
    a = _run(mesh, scan, rng=11, sampler='stratified', max_iter=2)[0]
    b = _run(mesh, scan, rng=11, sampler='stratified', max_iter=2)[0]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('bad', [dict(solver='cholmod'), dict(sampler='grid'), dict(window=0), dict(max_iter=0)])
def test_rejects_bad_knobs(mesh, scan, bad):
    with pytest.raises(ValueError):
        _run(mesh, scan, **bad)


# ---------------------------------------------------------------------------
# The membrane in the solve (2026-10-20): no post-solve filter, the result the last solve
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).parent / 'data' / 'nricp_smoothing_golden.npz'     # the code before the change


def test_no_membrane_reproduces_the_old_smooth_false(mesh, scan):
    """Without a membrane the loop is the old one without its filter (smooth=False), to the bit."""
    with np.load(GOLDEN) as z:
        old = z['nosmooth']
    np.testing.assert_allclose(_run(mesh, scan, rng=7)[0], old, atol=1e-10)


def test_the_membrane_default_is_pinned():
    params = inspect.signature(nonrigid_ICP).parameters
    assert params['membrane'].default is None
    assert 'smooth' not in params and 'smoothing' not in params


def _dense_membrane_solve(sys_, reg_op, reg_target):
    """The same energy built row by row, densely, per coordinate, and solved by lstsq: an independent reference."""
    v, f, fidx, bc = sys_['modelV'], sys_['faces'], sys_['fidx'], sys_['bc']
    pts, sw, edges, ew = sys_['scanpts'], sys_['scan_w'], sys_['edges'], sys_['edge_w']
    nV = len(v)
    B = np.zeros((len(pts), nV))
    for s_ in range(len(pts)):
        np.add.at(B[s_], f[fidx[s_]], bc[s_])
    E = np.zeros((len(edges), nV))
    E[np.arange(len(edges)), edges[:, 0]], E[np.arange(len(edges)), edges[:, 1]] = ew, -ew
    root = np.linalg.cholesky(reg_op.toarray() + 1e-12 * np.eye(nV)).T     # reg_op = root' root
    A = np.vstack([sw[:, None] * B, E, root])
    out = np.empty_like(v)
    for c in range(3):
        b = np.concatenate([sw * pts[:, c], ew * (v[edges[:, 0], c] - v[edges[:, 1], c]), root @ reg_target[:, c]])
        out[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
    return out


def test_the_membrane_solve_is_the_least_squares_minimiser(mesh, scan):
    from bg3dtools.mesh.laplace import cotangent_weights
    sys_ = _system(mesh, scan, landmarks=False)
    v, f = mesh
    reg_op = (-1e-5 * sparse.csr_matrix(-cotangent_weights(v, f))).tocsr()
    target = 1.01 * v
    got = fit_vertices(**sys_, reg_op=reg_op, reg_target=target)
    np.testing.assert_allclose(got, _dense_membrane_solve(sys_, reg_op, target), atol=1e-6)


def _bump_scan(v, f, n_points, width, amplitude=0.005, seed=0):
    """Points on the sphere of *v* with a Gaussian radial bump of *width* (m) on top, and radial normals."""
    rng = np.random.default_rng(seed)
    pmap, _, _ = stratified_surface_sample(v, f, n_points, rng)
    pts = pmap @ v
    radius = np.linalg.norm(v, axis=1).mean()
    pts = radius * pts / np.linalg.norm(pts, axis=1, keepdims=True)
    top = np.array([0.0, 0.0, radius])
    unit = pts / radius
    pts = pts + amplitude * np.exp(-0.5 * (np.linalg.norm(pts - top, axis=1) / width) ** 2)[:, None] * unit
    return pts, unit


def _bump_fitted(v, f, n_points, width, membrane, amplitude=0.005):
    """How much of the bump's height the fit takes up at its peak (the top vertex)."""
    pts, normals = _bump_scan(v, f, n_points, width, amplitude)
    out, _ = nonrigid_ICP(pts, f, v, pt_normals=normals, membrane=membrane, rad=0.05, model_weight=0.01,
                          max_iter=8, converge_thresh=0.0, rng=1)
    top = int(np.argmax(v[:, 2]))
    return (np.linalg.norm(out[top]) - np.linalg.norm(v[top])) / amplitude


@pytest.fixture(scope='module')
def fine_sphere():
    v, f = _icosphere(4)
    return 0.15 * v, f                                # a 15 cm sphere: ~8 mm edges


def test_a_membrane_keeps_broad_offsets_and_damps_fine_ones(fine_sphere):
    v, f = fine_sphere
    assert _bump_fitted(v, f, 20000, 0.06, 0.01) > 0.9
    fine = _bump_fitted(v, f, 20000, 0.008, 0.01)
    assert fine < 0.5
    assert _bump_fitted(v, f, 20000, 0.008, None) > fine + 0.3


def test_the_membrane_does_not_depend_on_the_scan_density(fine_sphere):
    """The membrane's weight follows the data's weight per area, so a sparser scan is held the same."""
    v, f = fine_sphere
    assert abs(_bump_fitted(v, f, 5000, 0.025, 0.01) - _bump_fitted(v, f, 20000, 0.025, 0.01)) < 0.05


def test_a_membrane_resists_a_slide(mesh):
    """Matches held while the scan points turn along the sphere drag the vertices along it -- a rotation of the
    displacement field, which costs the data the same wherever the surface is not pinned. The membrane
    charges its gradient and holds it, the more the wider (weights as nonrigid_ICP sets them: width^2
    times the data's nominal weight per area)."""
    from bg3dtools.mesh.laplace import cotangent_weights
    v, f = mesh
    rng = np.random.default_rng(3)
    pmap, fidx, bc = stratified_surface_sample(v, f, 3000, rng)
    on = pmap @ v
    on /= np.linalg.norm(on, axis=1, keepdims=True)
    a = 0.1
    turn = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    edges = ordered_edges(f)
    K = 3 * len(f)
    L = sparse.csr_matrix(-cotangent_weights(v, f))
    rho = len(on) / (K * K * igl_compat.massmatrix(v, f).diagonal().sum())
    common = dict(initV=v, faces=f, modelV=v, fidx=fidx, bc=bc, scanpts=on @ turn.T, edges=edges,
                  scan_w=np.full(len(on), 1.0 / K), edge_w=np.full(len(edges), 0.01 / len(edges)))
    drift = [np.linalg.norm(fit_vertices(**common) - v, axis=1).mean()]
    for width in (0.3, 1.0):                          # on a unit sphere
        op = (-(width ** 2) * rho * L).tocsr()
        drift.append(np.linalg.norm(fit_vertices(**common, reg_op=op, reg_target=v) - v, axis=1).mean())
    assert drift[0] > drift[1] > drift[2]
    assert drift[2] < 0.5 * drift[0]


@pytest.mark.parametrize('bad', [dict(membrane=0.0), dict(membrane=-0.01), dict(membrane=0.01, solver='lsqr')])
def test_the_membrane_rejects_bad_settings(mesh, scan, bad):
    with pytest.raises(ValueError):
        _run(mesh, scan, **bad)


def test_the_equivalent_membrane_is_as_stiff_as_the_old_filter(fine_sphere):
    """The Laplace filter the membrane replaced, (I - t L)^-1, and the membrane filter of the equivalent width,
    (M - w^2 L)^-1 M, take up a smooth bump alike (on a near-uniform mesh, where the mean vertex area is every
    vertex's)."""
    from bg3dtools.mesh.laplace import cotangent_weights
    v, f = fine_sphere
    L = sparse.csr_matrix(-cotangent_weights(v, f))
    M = sparse.diags(igl_compat.massmatrix(v, f).diagonal())
    centre = int(np.argmax(v[:, 2]))
    bump = np.exp(-0.5 * (np.linalg.norm(v - v[centre], axis=1) / 0.02) ** 2)
    for t in (1.0, 5.0):
        width = equivalent_membrane(v, f, t)
        old = sparse.linalg.spsolve((sparse.eye(len(v)) - t * L).tocsc(), bump)[centre]
        new = sparse.linalg.spsolve((M - width ** 2 * L).tocsc(), M @ bump)[centre]
        assert abs(old - new) < 0.05


def test_the_equivalent_membrane_is_a_width_in_the_mesh_units(fine_sphere):
    v, f = fine_sphere
    assert np.isclose(equivalent_membrane(1000.0 * v, f, 5.0), 1000.0 * equivalent_membrane(v, f, 5.0))
    assert np.isclose(equivalent_membrane(v, f, 4.0), 2.0 * equivalent_membrane(v, f, 1.0))
