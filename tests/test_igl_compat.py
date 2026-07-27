"""Contract tests for :mod:`bg3dtools.igl_compat`.

These pin the igl 2.5.1 contract that every wrapper presents — return arity and
order, dtypes, contiguity, and the leading-dimension normalizations — so a
libigl upgrade cannot silently change bg3dtools' behavior again.

**Must pass on both bindings.** Deliberately import-light (numpy + scipy +
``bg3dtools.igl_compat`` only, no torch/trimesh/open3d) so it can run in a bare
scratch env::

    conda create -y -n igl261 -c conda-forge python=3.12 igl=2.6.1 numpy scipy pytest
    ~/opt/anaconda3/envs/igl261/bin/python -m pytest tests/test_igl_compat.py -v

Verified green on igl 2.5.1 (canonical) and igl 2.6.1.
"""

import inspect

import numpy as np
import pytest

from bg3dtools import igl_compat as ic


# ---------------------------------------------------------------------------
# fixtures — tiny synthetic meshes, no file or heavy-dep dependencies
# ---------------------------------------------------------------------------

def _quad():
    """Unit square as 2 triangles: open, one boundary loop of 4, 2 ears."""
    v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return v, f


def _tetra():
    """Closed tetrahedron: no boundary, edge- and vertex-manifold."""
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return v, f


def _icosahedron():
    """Closed genus-0 sphere-like mesh: 12 verts, 20 faces, unit radius."""
    p = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0],
                  [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
                  [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], dtype=np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
                 dtype=np.int64)
    return v / np.linalg.norm(v, axis=1, keepdims=True), f


def _two_islands():
    """Two disjoint triangles — the facet_components repro from the incident."""
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                  [5., 0., 0.], [6., 0., 0.], [5., 1., 0.]])
    f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return v, f


def _nonmanifold_edge():
    """Three triangles sharing one edge — not edge-manifold."""
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                  [0., -1., 0.], [0., 0., 1.]])
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    return v, f


def _bowtie():
    """Two triangles meeting at a single vertex — vertex 0 is non-manifold."""
    v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                  [-1., 0., 0.], [-1., -1., 0.]])
    f = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    return v, f


def assert_index(a, shape=None):
    """Every index/label array the layer returns is C-contiguous int64."""
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.int64, "expected int64 indices, got %s" % a.dtype
    assert a.flags["C_CONTIGUOUS"]
    if shape is not None:
        assert a.shape == shape, "expected %s, got %s" % (shape, a.shape)


def assert_float(a, shape=None):
    """Every coordinate/scalar array the layer returns is C-contiguous float64."""
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.float64, "expected float64, got %s" % a.dtype
    assert a.flags["C_CONTIGUOUS"]
    if shape is not None:
        assert a.shape == shape, "expected %s, got %s" % (shape, a.shape)


# ---------------------------------------------------------------------------
# module-level invariants
# ---------------------------------------------------------------------------

WRAPPERS = [
    n for n in ic.__all__
    if n not in ("AVAILABLE", "PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA")
]


def test_available_is_a_subset_of_all():
    assert ic.AVAILABLE <= set(ic.__all__)


def test_no_version_string_is_ever_parsed():
    """igl.__version__ is absent from the 2.6.1 conda build; the layer must not read it."""
    src = open(ic.__file__).read()
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "__version__" not in code.split('"""')[-1], \
        "igl_compat must feature-detect, never parse a version string"


@pytest.mark.parametrize("name", WRAPPERS)
def test_unavailable_wrappers_raise_actionable_errors(name):
    """Requirement: never silently pass junk through for an unsupported API shape."""
    if name in ic.AVAILABLE:
        pytest.skip("%s is available on this binding" % name)
    fn = getattr(ic, name)
    n_required = sum(
        p.default is inspect.Parameter.empty
        for p in inspect.signature(fn).parameters.values()
    )
    with pytest.raises(RuntimeError) as exc:
        fn(*([None] * n_required))
    msg = str(exc.value)
    assert name in msg
    assert "igl=2.5.1" in msg, "the error must say how to fix it"


def test_no_module_shadows_the_names_it_imports_from_igl_compat():
    """A module that imports ``foo`` from igl_compat must not also ``def foo``.

    Several bg3dtools modules wrap an igl function under the *same* name
    (``mesh.laplace.gaussian_curvature``, ``mesh.utils.per_vertex_normals``, …).
    Importing the compat function unaliased there binds a name the module's own
    ``def`` then overwrites, so the wrapper ends up calling itself — an infinite
    recursion that no type checker catches. Import such names as ``igl_<name>``.
    """
    import ast
    import pathlib

    import bg3dtools

    offenders = []
    for path in sorted(pathlib.Path(bg3dtools.__path__[0]).rglob("*.py")):
        src = path.read_text()
        if "igl_compat" not in src:
            continue
        tree = ast.parse(src)
        imported = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("igl_compat")
            for alias in node.names
        }
        defined = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for name in sorted(imported & defined):
            offenders.append("%s defines def %s() but also imports it from igl_compat"
                             % (path.name, name))
    assert not offenders, "\n".join(offenders)


def test_unrecognized_return_arity_raises_rather_than_guessing(monkeypatch):
    """A future binding returning an unseen arity must fail loudly, not be duck-typed."""
    _, f = _two_islands()
    monkeypatch.setattr(ic, "_facet_components", lambda ff: (1, 2, 3))
    with pytest.raises(RuntimeError, match="not supported by bg3dtools.igl_compat"):
        ic.facet_components(f)


# ---------------------------------------------------------------------------
# facet_components — the function that caused the production incident
# ---------------------------------------------------------------------------

def test_facet_components_returns_a_bare_array_not_a_tuple():
    """igl 2.6 returns (n_components, ids); the contract is the bare ids array.

    Downstream code did ``np.bincount(facet_components(F))``, which raised
    ``ValueError: setting an array element with a sequence`` on the 2-tuple.
    """
    _, f = _two_islands()
    labels = ic.facet_components(f)
    assert not isinstance(labels, tuple)
    assert_index(labels, (2,))
    assert len(np.unique(labels)) == 2                  # two islands
    assert np.array_equal(np.bincount(labels), [1, 1])  # the exact crashing call


def test_facet_components_single_component():
    _, f = _quad()
    labels = ic.facet_components(f)
    assert_index(labels, (2,))
    assert len(np.unique(labels)) == 1


# ---------------------------------------------------------------------------
# exact_geodesic — the silent-wrong-answer case
# ---------------------------------------------------------------------------

def test_exact_geodesic_known_distances():
    """On a flat mesh geodesic distance is Euclidean, so the answer is known.

    igl 2.6 reordered the positional arguments to (V, F, VS, FS, VT, FT), so a
    4-positional call binds target *vertices* to the source *faces* slot and
    returns an EMPTY array instead of raising. This test fails loudly if the
    wrapper ever regresses to a positional call.
    """
    v, f = _quad()
    d = ic.exact_geodesic(v, f, np.array([0]), np.array([1, 2, 3]))
    assert_float(d, (3,))
    assert d == pytest.approx([1.0, np.sqrt(2.0), 1.0], abs=1e-9)


def test_exact_geodesic_source_to_itself_is_zero():
    v, f = _icosahedron()
    d = ic.exact_geodesic(v, f, np.array([0]), np.arange(len(v)))
    assert_float(d, (len(v),))
    assert d[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(d[1:] > 0)


@pytest.mark.parametrize("face_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_exact_geodesic_immune_to_index_dtype_mismatch(face_dtype, index_dtype):
    """libigl rejects a call whose array integer arguments disagree in dtype::

        ValueError: Invalid type (int64, Row Major) for argument 'vs'.
        Expected it to match argument 'f' which is of type (int32, Row Major).

    NumPy's default int is int64 on Linux/macOS but int32 on Windows, so this only
    ever bit on some hosts. The layer pins faces *and* index sets to int64, making
    the mismatch structurally impossible — whatever the caller passes in.
    """
    v, f = _quad()
    d = ic.exact_geodesic(
        v, f.astype(face_dtype),
        np.array([0], dtype=index_dtype), np.array([1, 2, 3], dtype=index_dtype),
    )
    assert d == pytest.approx([1.0, np.sqrt(2.0), 1.0], abs=1e-9)


def test_heat_geodesic_one_shot_contract():
    """igl 2.6 replaced the one-shot call with a precompute/solve pair around a
    HeatGeodesicsData object; the wrapper drives it so the one-shot survives."""
    v, f = _icosahedron()
    d = ic.heat_geodesic(v, f, 0.1, np.array([0]))
    assert_float(d, (len(v),))
    assert np.all(np.isfinite(d))
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert d.argmax() == 3          # the antipode of vertex 0 on this icosahedron
    # the heat method approximates exact geodesics; agree to within 20%
    exact = ic.exact_geodesic(v, f, np.array([0]), np.arange(len(v)))
    assert np.allclose(d, exact, rtol=0.2, atol=0.05)


# ---------------------------------------------------------------------------
# arity / order changes
# ---------------------------------------------------------------------------

def test_is_edge_manifold_returns_a_plain_bool():
    """igl 2.6 returns a 5-tuple (is_manifold, BF, E, EMAP, BE)."""
    _, quad_f = _quad()
    _, nm_f = _nonmanifold_edge()
    for f, expected in ((quad_f, True), (_tetra()[1], True), (nm_f, False)):
        out = ic.is_edge_manifold(f)
        assert isinstance(out, bool), "got %r" % type(out)
        assert out is expected


def test_lscm_returns_success_flag_first():
    """igl 2.6 returns (uv, Q) — the flag is gone and uv moved to position 0."""
    v, f = _quad()
    success, uv = ic.lscm(v, f, np.array([0, 1]), np.array([[0., 0.], [1., 0.]]))
    assert isinstance(success, bool) and success
    assert_float(uv, (4, 2))
    assert np.all(np.isfinite(uv))


def test_decimate_returns_five_elements_with_success_first():
    """igl 2.6 drops the success flag (4 elements) and returns Fortran-ordered U/G."""
    v, f = _icosahedron()
    out = ic.decimate(v, f, 10)
    assert len(out) == 5
    success, u, g, j, i = out
    assert isinstance(success, bool) and success
    assert_float(u, (u.shape[0], 3))       # contiguity restored from Fortran order
    assert_index(g, (10, 3))
    assert_index(j, (10,))
    assert_index(i, (u.shape[0],))
    assert g.max() < len(u)


def test_write_triangle_mesh_returns_bool_both_ways(tmp_path):
    """igl 2.6 returns None on success and raises on failure, where 2.5.1 returns a bool."""
    v, f = _quad()
    good = tmp_path / "out.obj"
    assert ic.write_triangle_mesh(good, v, f) is True
    assert good.stat().st_size > 0
    assert ic.write_triangle_mesh(tmp_path / "no" / "such" / "dir" / "o.obj", v, f) is False


def test_all_boundary_loop_rename_and_contract():
    """igl 2.6 renamed this to boundary_loop_all; the return type is unchanged."""
    _, open_f = _quad()
    loops = ic.all_boundary_loop(open_f)
    assert isinstance(loops, list) and len(loops) == 1
    assert sorted(loops[0]) == [0, 1, 2, 3]
    # a closed mesh yields an empty list, so `not all_boundary_loop(f)` tests closedness
    assert ic.all_boundary_loop(_tetra()[1]) == []


def test_read_obj_rename_and_contract(tmp_path):
    """igl 2.6 renamed read_obj to readOBJ; order, arity and dtypes are unchanged."""
    p = tmp_path / "quad.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                 "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                 "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    out = ic.read_obj(p)
    assert len(out) == 6
    verts, tc, n, faces, ftc, fn = out
    assert_float(verts, (4, 3))
    assert_float(tc, (4, 2))
    assert_index(faces, (2, 3))
    assert_index(ftc, (2, 3))
    assert np.array_equal(faces, [[0, 1, 2], [0, 2, 3]])
    assert n.shape == (0, 0) and fn.shape == (0, 0)     # no normals in this OBJ


def test_read_obj_single_face_stays_two_dimensional(tmp_path):
    """2.5.1 squeezes a one-row block to 1-D; 2.6.1 does not. The contract is 2-D."""
    p = tmp_path / "tri.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nf 1/1 2/2 3/3\n")
    verts, tc, _, faces, ftc, _ = ic.read_obj(p)
    assert_float(verts, (3, 3))
    assert_index(faces, (1, 3))
    assert_index(ftc, (1, 3))


# ---------------------------------------------------------------------------
# leading-dimension normalization (2.5.1 squeezes, 2.6.1 does not)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_points", [1, 2, 5])
def test_point_mesh_squared_distance_keeps_the_query_dimension(n_points):
    """2.5.1 squeezes sqrD/I to 0-d and C to (3,) when nP == 1."""
    v, f = _quad()
    p = np.tile([0.25, 0.25, 2.0], (n_points, 1))
    sqr_d, idx, closest = ic.point_mesh_squared_distance(p, v, f)
    assert_float(sqr_d, (n_points,))
    assert_index(idx, (n_points,))
    assert_float(closest, (n_points, 3))
    assert sqr_d == pytest.approx([4.0] * n_points)                 # straight down z
    assert closest == pytest.approx(np.tile([0.25, 0.25, 0.0], (n_points, 1)))


@pytest.mark.parametrize("n_points", [1, 3])
def test_winding_number_keeps_the_query_dimension(n_points):
    v, f = _tetra()
    inside = np.tile([0.1, 0.1, 0.1], (n_points, 1))
    w = ic.winding_number(v, f, inside)
    assert_float(w, (n_points,))
    assert np.all(np.abs(np.abs(w) - 1.0) < 1e-6)                   # inside a closed mesh
    assert np.all(np.abs(ic.winding_number(v, f, inside + 10.0)) < 1e-6)   # outside


def test_remove_unreferenced_single_surviving_face_stays_two_dimensional():
    """2.5.1 squeezes NF to (ss,) when only one face survives."""
    v, f = _two_islands()
    nv, nf, i, j = ic.remove_unreferenced(v, f[:1])
    assert_float(nv, (3, 3))
    assert_index(nf, (1, 3))                                        # not (3,)
    assert_index(i, (6,))
    assert_index(j, (3,))
    assert np.array_equal(nv, v[j])                                 # J maps new -> old


def test_remove_unreferenced_multiple_faces():
    v, f = _two_islands()
    nv, nf, i, j = ic.remove_unreferenced(v, f)
    assert_float(nv, (6, 3))
    assert_index(nf, (2, 3))
    assert np.array_equal(nv[nf], v[f])                             # geometry preserved


# ---------------------------------------------------------------------------
# dtype pinning — subsumes the class-2 (int32-return) bugs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("face_dtype", [np.int32, np.int64])
def test_index_returns_are_int64_whatever_the_input_dtype(face_dtype):
    """The 2.5.1 binding propagates the input face dtype to its outputs (int32 in →
    int32 out, on every platform), and int32 faces make predicates like
    is_edge_manifold report spurious non-manifoldness. The layer always pins int64.
    """
    v, f_int64 = _icosahedron()
    f = np.ascontiguousarray(f_int64, dtype=face_dtype)

    assert_index(ic.edges(f))
    assert_index(ic.bfs_orient(f)[0])
    assert_index(ic.bfs_orient(f)[1])
    assert_index(ic.upsample(v, f)[1])
    assert_index(ic.decimate(v, f, 10)[2])
    assert_index(ic.remove_unreferenced(v, f)[1])
    assert_index(ic.facet_components(f))
    assert_index(ic.triangle_triangle_adjacency(f)[0])
    assert_index(ic.vertex_triangle_adjacency(f, len(v))[0])
    assert_index(ic.vertex_triangle_adjacency(f, len(v))[1])
    assert_index(ic.boundary_loop(f))
    assert_index(ic.ears(f)[0])
    assert ic.is_edge_manifold(f) is True     # would be False on int32 without the pin


def test_fortran_ordered_input_is_accepted():
    """Callers hand us whatever numpy gave them; the layer must not care."""
    v, f = _icosahedron()
    vf = np.asfortranarray(v)
    ff = np.asfortranarray(f)
    assert_float(ic.doublearea(vf, ff), (20,))
    assert_index(ic.edges(ff))
    assert ic.is_edge_manifold(ff) is True


# ---------------------------------------------------------------------------
# pass-through wrappers: arity, order, shape, dtype
# ---------------------------------------------------------------------------

def test_edges_and_adjacency():
    v, f = _quad()
    e = ic.edges(f)
    assert_index(e, (5, 2))                                # 4 boundary + 1 diagonal
    a = ic.adjacency_matrix(f)
    assert a.shape == (4, 4)
    assert a.nnz == 10                                     # 5 edges, both directions


def test_vertex_triangle_adjacency_offsets():
    v, f = _quad()
    vf, ni = ic.vertex_triangle_adjacency(f, 4)
    assert_index(vf, (6,))                                 # 2 faces x 3 corners
    assert_index(ni, (5,))                                 # nV + 1 offsets
    assert ni[0] == 0 and ni[-1] == 6
    assert sorted(vf[ni[0]:ni[1]]) == [0, 1]               # vertex 0 is in both faces


def test_triangle_triangle_adjacency_marks_boundaries_with_minus_one():
    v, f = _quad()
    tt, tti = ic.triangle_triangle_adjacency(f)
    assert_index(tt, (2, 3))
    assert_index(tti, (2, 3))
    assert (tt == -1).sum() == 4                           # 4 boundary edges
    assert (tt == 1).sum() == 1 and (tt == 0).sum() == 1   # the shared diagonal


def test_boundary_loop_open_and_closed():
    _, f = _quad()
    loop = ic.boundary_loop(f)
    assert_index(loop, (4,))
    assert sorted(loop) == [0, 1, 2, 3]
    assert_index(ic.boundary_loop(_tetra()[1]), (0,))      # closed -> empty


def test_ears_present_and_absent():
    _, quad_f = _quad()
    ears, ear_opp = ic.ears(quad_f)
    assert_index(ears, (2,))                               # both triangles are ears
    assert_index(ear_opp, (2,))
    _, ico_f = _icosahedron()
    assert_index(ic.ears(ico_f)[0], (0,))                  # closed mesh has none


def test_bfs_orient_fixes_a_flipped_face():
    v, f = _tetra()
    flipped = f.copy()
    flipped[1] = flipped[1][::-1]
    ff, c = ic.bfs_orient(flipped)
    assert_index(ff, (4, 3))
    assert_index(c, (4,))
    assert ic.is_edge_manifold(ff)
    assert np.array_equal(np.sort(ff, axis=1), np.sort(f, axis=1))   # same triangles


def test_doublearea_and_barycenter():
    v, f = _quad()
    da = ic.doublearea(v, f)
    assert_float(da, (2,))
    assert da.sum() == pytest.approx(2.0)                  # unit square, twice the area
    bc = ic.barycenter(v, f)
    assert_float(bc, (2, 3))
    assert bc.mean(axis=0) == pytest.approx([0.5, 0.5, 0.0])


def test_internal_angles_sum_to_pi_per_triangle():
    v, f = _quad()
    ang = ic.internal_angles(v, f)
    assert_float(ang, (2, 3))
    assert ang.sum(axis=1) == pytest.approx([np.pi, np.pi])


def test_gaussian_curvature_of_a_closed_sphere_obeys_gauss_bonnet():
    v, f = _icosahedron()
    k = ic.gaussian_curvature(v, f)
    assert_float(k, (12,))
    assert k.sum() == pytest.approx(4 * np.pi)             # genus 0


def test_per_vertex_and_per_face_normals():
    v, f = _quad()
    vn = ic.per_vertex_normals(v, f, ic.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
    assert_float(vn, (4, 3))
    assert np.allclose(np.abs(vn), [[0, 0, 1]] * 4)        # planar mesh
    fn = ic.per_face_normals(v, f, np.array([0., 1., 0.]))
    assert_float(fn, (2, 3))
    assert np.allclose(np.abs(fn), [[0, 0, 1]] * 2)


def test_per_vertex_normals_defaults_to_area_weighting():
    """The weighting argument became an enum in 2.6; the default must work on both."""
    v, f = _quad()
    assert np.allclose(ic.per_vertex_normals(v, f),
                       ic.per_vertex_normals(v, f,
                                             ic.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA))


def test_average_onto_faces_argument_order():
    """Argument order is (faces, per-vertex values) — easy to transpose by mistake."""
    _, f = _quad()
    s = np.array([0., 0., 1., 1.])
    out = ic.average_onto_faces(f, s)
    assert_float(out, (2,))
    assert out == pytest.approx([1 / 3, 2 / 3])


def test_cotmatrix_and_intrinsic_delaunay_cotmatrix():
    import scipy.sparse as sp
    v, f = _icosahedron()
    lap = ic.cotmatrix(v, f)
    assert sp.issparse(lap) and lap.shape == (12, 12)
    ldd, lengths, faces = ic.intrinsic_delaunay_cotmatrix(v, f)
    assert sp.issparse(ldd) and ldd.shape == (12, 12)
    assert_float(lengths, (20, 3))
    assert_index(faces, (20, 3))


def test_connected_components_takes_an_adjacency_matrix():
    """2.5.1 rejects a face array here; pass adjacency_matrix(f) to stay portable."""
    _, f = _two_islands()
    n, c, k = ic.connected_components(ic.adjacency_matrix(f))
    assert n == 2
    assert_index(c, (6,))
    assert_index(k, (2,))
    assert k.sum() == 6


def test_upsample_quadruples_faces():
    v, f = _icosahedron()
    nv, nf = ic.upsample(v, f)
    assert_float(nv, (42, 3))                              # 12 verts + 30 edge midpoints
    assert_index(nf, (80, 3))                              # 20 * 4
    assert ic.is_edge_manifold(nf)


def test_remove_duplicate_vertices_merges_a_seam():
    v, f = _quad()
    # duplicate vertex 2 and rewire the second triangle through the copy
    v2 = np.vstack([v, v[2]])
    f2 = np.array([[0, 1, 2], [0, 4, 3]], dtype=np.int64)
    sv, svi, svj, sf = ic.remove_duplicate_vertices(v2, f2, 1e-7)
    assert_float(sv, (4, 3))                               # the copy is merged away
    assert_index(svi, (4,))
    assert_index(svj, (5,))
    assert_index(sf, (2, 3))
    assert np.array_equal(sv[sf], v2[f2])                  # geometry preserved


def test_random_points_on_mesh_samples_the_surface():
    v, f = _icosahedron()
    b, fi, p = ic.random_points_on_mesh(50, v, f)
    assert_float(b, (50, 3))
    assert_index(fi, (50,))
    assert_float(p, (50, 3))
    assert np.all(fi >= 0) and np.all(fi < len(f))
    assert b.sum(axis=1) == pytest.approx(np.ones(50))     # barycentric
    # points reconstructed from (face, barycentric) match the returned positions
    assert np.allclose(np.einsum('ij,ijk->ik', b, v[f[fi]]), p)


def test_cylinder_generates_a_valid_open_mesh():
    v, f = ic.cylinder(8, 4)
    assert_float(v, (32, 3))
    assert_index(f, (48, 3))
    assert ic.is_edge_manifold(f)
    assert len(ic.all_boundary_loop(f)) == 2               # top and bottom rims


# ---------------------------------------------------------------------------
# functions that exist in only one binding
# ---------------------------------------------------------------------------

@pytest.mark.skipif("extract_manifold_patches" not in ic.AVAILABLE,
                    reason="removed in igl 2.6")
def test_extract_manifold_patches():
    _, f = _two_islands()
    n, labels = ic.extract_manifold_patches(f)
    assert isinstance(n, int) and n == 2
    assert_index(labels, (2,))


@pytest.mark.skipif("resolve_duplicated_faces" not in ic.AVAILABLE,
                    reason="removed in igl 2.6")
def test_resolve_duplicated_faces():
    _, f = _icosahedron()
    dup = np.vstack([f, f[:2]])
    f2, j = ic.resolve_duplicated_faces(dup)
    assert_index(f2, (18, 3))          # both copies of each duplicate pair cancel
    assert_index(j, (18,))
    assert np.array_equal(f2, dup[j])


@pytest.mark.skipif("collapse_small_triangles" not in ic.AVAILABLE,
                    reason="removed in igl 2.6")
def test_collapse_small_triangles_leaves_a_healthy_mesh_alone():
    v, f = _icosahedron()
    out = ic.collapse_small_triangles(v, f, 1e-12)
    assert_index(out, (20, 3))
    assert np.array_equal(out, f)


@pytest.mark.skipif("point_simplex_squared_distance" not in ic.AVAILABLE,
                    reason="removed in igl 2.6")
def test_point_simplex_squared_distance_contract():
    """Only the *shape* contract is asserted — the 2.5.1 binding's values are wrong.

    See :func:`test_point_simplex_squared_distance_misreads_column_major_verts`.
    """
    v, f = _quad()
    sqr_d, closest, bc = ic.point_simplex_squared_distance(
        np.array([0.25, 0.25, 2.0]), v, f, 0)
    assert isinstance(sqr_d, float)
    assert_float(closest, (3,))
    assert_float(bc, (3,))
    assert bc.sum() == pytest.approx(1.0)


@pytest.mark.skipif("point_simplex_squared_distance" not in ic.AVAILABLE,
                    reason="removed in igl 2.6")
def test_point_simplex_squared_distance_misreads_column_major_verts():
    """KNOWN BUG in the igl 2.5.1 binding, pinned here so any fix is deliberate.

    ``igl.point_simplex_squared_distance`` reads the vertex matrix in **Fortran
    (column-major)** order, so a normal C-contiguous ``V`` yields a "closest
    point" that is not even on the mesh. Passing ``np.asfortranarray(v)`` gives
    the right answer, which is how the misread was diagnosed.

    ``igl_compat`` deliberately does NOT apply that workaround: correcting it
    would change numerical results for ``mesh.highD.HighDimMesh``, whose nearest
    -point search is built on this call, and that is a behavior change rather
    than a compatibility fix. Tracked as a follow-up; this test documents the
    hazard and fails loudly if the underlying binding is ever fixed.
    """
    import igl  # deliberately raw: this pins the *binding's* behavior, not the layer's
    v, f = _quad()
    on_face_0 = np.array([0.9, 0.1, 0.0])           # exactly on triangle (0, 1, 2)

    correct = igl.point_simplex_squared_distance(on_face_0, np.asfortranarray(v), f, 0)
    assert correct[0] == pytest.approx(0.0), "F-order V is the one that works"

    wrong = ic.point_simplex_squared_distance(on_face_0, v, f, 0)
    assert wrong[0] > 0.5, "C-order V still misreads; remove this test once fixed"
    # the reported closest point is off the z=0 plane the whole mesh lies in
    assert abs(wrong[1][2]) > 1e-6


@pytest.mark.skipif("is_vertex_manifold" not in ic.AVAILABLE,
                    reason="added in igl 2.6, absent from 2.5.1")
def test_is_vertex_manifold():
    _, f = _bowtie()
    mask = ic.is_vertex_manifold(f)
    assert mask.dtype == np.bool_
    assert mask.shape == (5,)                              # length is max(f) + 1
    assert not mask[0]                                     # the pinch vertex
    assert mask[1:].all()
    assert ic.is_vertex_manifold(_tetra()[1]).all()
