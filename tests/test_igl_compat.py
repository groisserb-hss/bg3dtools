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

#: The callable wrappers in ``__all__`` — excludes AVAILABLE and the re-exported
#: igl constants (PER_VERTEX_NORMALS_WEIGHTING_TYPE_*, MASSMATRIX_TYPE_*), which
#: are values rather than functions.
WRAPPERS = [n for n in ic.__all__ if callable(getattr(ic, n))]


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

    repo_root = pathlib.Path(bg3dtools.__path__[0]).parent
    sources = []
    for pkg in ("bg3dtools", "spectral_match"):
        pkg_dir = repo_root / pkg
        if pkg_dir.is_dir():                      # spectral_match may not be installed
            sources.extend(sorted(pkg_dir.rglob("*.py")))
    assert sources, "found no package sources to scan"

    offenders = []
    for path in sources:
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


def test_barycentric_coordinates_tri_rename_and_contract():
    """igl 2.6 renamed this to barycentric_coordinates (tri/tet split by arity)."""
    a = np.tile([0., 0., 0.], (3, 1))
    b = np.tile([1., 0., 0.], (3, 1))
    c = np.tile([0., 1., 0.], (3, 1))
    q = np.array([[0., 0., 0.],          # corner a
                  [1 / 3, 1 / 3, 0.],    # centroid
                  [0.5, 0.5, 0.]])       # midpoint of edge bc
    bc = ic.barycentric_coordinates_tri(q, a, b, c)
    assert_float(bc, (3, 3))
    assert bc == pytest.approx(np.array([[1, 0, 0],
                                         [1 / 3, 1 / 3, 1 / 3],
                                         [0, 0.5, 0.5]]))
    # reconstructing from the coordinates returns the query points
    tri = np.stack([a, b, c], axis=1)
    assert np.einsum('ij,ijk->ik', bc, tri) == pytest.approx(q)


def test_barycentric_coordinates_tri_single_point_stays_two_dimensional():
    """2.5.1 squeezes the result to (3,) for one point; the contract is (nP, 3)."""
    bc = ic.barycentric_coordinates_tri(
        np.array([[0.25, 0.25, 0.]]), np.array([[0., 0., 0.]]),
        np.array([[1., 0., 0.]]), np.array([[0., 1., 0.]]))
    assert_float(bc, (1, 3))
    assert bc[0] == pytest.approx([0.5, 0.25, 0.25])


def test_cotmatrix_and_intrinsic_delaunay_cotmatrix():
    import scipy.sparse as sp
    v, f = _icosahedron()
    lap = ic.cotmatrix(v, f)
    assert sp.issparse(lap) and lap.shape == (12, 12)
    ldd, lengths, faces = ic.intrinsic_delaunay_cotmatrix(v, f)
    assert sp.issparse(ldd) and ldd.shape == (12, 12)
    assert_float(lengths, (20, 3))
    assert_index(faces, (20, 3))


def _irregular_mesh():
    """Irregular enough that Voronoi and barycentric vertex areas differ."""
    v = np.array([[0., 0, 0], [3., 0, 0.2], [0., 1, 0.5], [-0.2, -2., 0.1],
                  [1., 1., 2.0], [2., -1., 0.3]])
    f = np.array([[0, 1, 2], [0, 2, 4], [0, 4, 1], [0, 3, 1], [0, 2, 3], [1, 5, 3]],
                 dtype=np.int64)
    return v, f


def test_massmatrix_lumped_diagonal_sums_to_surface_area():
    import scipy.sparse as sp
    v, f = _irregular_mesh()
    area = ic.doublearea(v, f).sum() / 2
    for mtype in (ic.MASSMATRIX_TYPE_VORONOI, ic.MASSMATRIX_TYPE_BARYCENTRIC):
        m = ic.massmatrix(v, f, mtype)
        assert sp.issparse(m) and m.shape == (6, 6)
        diag = np.asarray(m.diagonal()).ravel()
        assert diag.sum() == pytest.approx(area)     # a lumped mass matrix partitions area
        assert np.all(diag > 0)


def test_massmatrix_defaults_to_voronoi():
    """Both bindings default to MASSMATRIX_TYPE_DEFAULT, which is Voronoi for
    triangles; the wrapper pins Voronoi rather than relying on that."""
    v, f = _irregular_mesh()
    np.testing.assert_allclose(
        ic.massmatrix(v, f).diagonal(),
        ic.massmatrix(v, f, ic.MASSMATRIX_TYPE_VORONOI).diagonal())


def test_massmatrix_types_are_distinct():
    """Voronoi and barycentric agree in total but not per vertex, so the type matters."""
    v, f = _irregular_mesh()
    vor = np.asarray(ic.massmatrix(v, f, ic.MASSMATRIX_TYPE_VORONOI).diagonal()).ravel()
    bar = np.asarray(ic.massmatrix(v, f, ic.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()).ravel()
    full = ic.massmatrix(v, f, ic.MASSMATRIX_TYPE_FULL)
    assert not np.allclose(vor, bar)
    assert vor.sum() == pytest.approx(bar.sum())
    assert full.nnz > len(v)                                   # unlumped: off-diagonals
    assert np.asarray(full.diagonal()).sum() == pytest.approx(vor.sum() / 2)


def test_massmatrix_accepts_int32_faces():
    v, f = _irregular_mesh()
    np.testing.assert_allclose(
        ic.massmatrix(v, np.ascontiguousarray(f, np.int32)).diagonal(),
        ic.massmatrix(v, f).diagonal())


def test_cotmatrix_and_massmatrix_are_not_all_zeros():
    """Pins the correction of a long-standing repo comment.

    Several comments claimed igl.cotmatrix/igl.massmatrix "return all zeros" on
    2.5.1. They do not, on either binding, for any input shape tried here — so the
    hand-rolled mesh.laplace implementations are kept for their different
    normalisation, not because igl's are broken.
    """
    for v, f in (_quad(), _icosahedron(), _irregular_mesh(), _nonmanifold_edge()):
        for faces in (f, np.ascontiguousarray(f, np.int32)):
            lap = np.asarray(ic.cotmatrix(v, faces).todense())
            mass = np.asarray(ic.massmatrix(v, faces).todense())
            assert np.any(lap != 0), "cotmatrix returned all zeros"
            assert np.any(mass != 0), "massmatrix returned all zeros"
            # a valid Laplacian annihilates constants
            assert np.max(np.abs(lap.sum(axis=1))) < 1e-9


def test_read_triangle_mesh_contract(tmp_path):
    p = tmp_path / "quad.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n")
    v, f = ic.read_triangle_mesh(p)
    assert_float(v, (4, 3))
    assert_index(f, (2, 3))
    assert np.array_equal(f, [[0, 1, 2], [0, 2, 3]])
    assert v[2] == pytest.approx([1.0, 1.0, 0.0])


def test_read_triangle_mesh_single_face_stays_two_dimensional(tmp_path):
    """2.5.1 squeezes F to (3,) for a one-triangle file; the contract is (nF, 3)."""
    p = tmp_path / "tri.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n")
    v, f = ic.read_triangle_mesh(p)
    assert_float(v, (3, 3))
    assert_index(f, (1, 3))


def test_read_triangle_mesh_roundtrips_write_triangle_mesh(tmp_path):
    v, f = _icosahedron()
    for ext in ("obj", "off", "ply"):
        p = tmp_path / ("m." + ext)
        assert ic.write_triangle_mesh(p, v, f) is True
        v2, f2 = ic.read_triangle_mesh(p)
        assert_float(v2, (12, 3))
        assert_index(f2, (20, 3))
        assert v2 == pytest.approx(v, abs=1e-6), ext
        assert np.array_equal(f2, f), ext


def test_read_triangle_mesh_accepts_str_and_pathlike(tmp_path):
    p = tmp_path / "tri.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n")
    from_path = ic.read_triangle_mesh(p)
    from_str = ic.read_triangle_mesh(str(p))
    assert np.array_equal(from_path[0], from_str[0])
    assert np.array_equal(from_path[1], from_str[1])


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
    v, f = _quad()
    sqr_d, closest, bc = ic.point_simplex_squared_distance(
        np.array([0.25, 0.25, 2.0]), v, f, 0)
    assert isinstance(sqr_d, float)
    assert_float(closest, (3,))
    assert_float(bc, (3,))
    assert bc.sum() == pytest.approx(1.0)
    # the query sits directly above face 0, so the closest point is straight down
    assert sqr_d == pytest.approx(4.0)
    assert closest == pytest.approx([0.25, 0.25, 0.0])


def _ref_closest_point_on_tri(p, a, b, c):
    """Independent closest-point-on-triangle reference.

    Ericson, *Real-Time Collision Detection* §5.1.5. Used to check libigl rather
    than assume it: the 2.5.1 binding misreads the vertex matrix's storage order,
    which is why :func:`bg3dtools.igl_compat.point_simplex_squared_distance`
    passes ``asfortranarray(v)``.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 <= d1 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cq = p - c
    d5, d6 = ab @ cq, ac @ cq
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 <= d2 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


@pytest.mark.skipif("point_simplex_squared_distance" not in ic.AVAILABLE,
                    reason="removed in igl 2.6")
def test_point_simplex_squared_distance_matches_an_independent_reference():
    """The wrapper corrects the 2.5.1 binding's column-major misread of ``v``.

    Without ``asfortranarray(v)`` the binding returns closest points that are not
    on the mesh at all. Every point/face pair below is checked against
    :func:`_ref_closest_point_on_tri`, spanning the face interior, the edges and
    the vertices — i.e. every branch of the closest-point algorithm.
    """
    rng = np.random.RandomState(0)
    for v, f in (_quad(), _icosahedron(), _two_islands()):
        lo, hi = v.min(0) - 0.6, v.max(0) + 0.6
        queries = np.vstack([v,                            # on vertices
                             v[f].mean(1),                 # face interiors
                             v[f][:, :2].mean(1),          # over edges
                             rng.rand(15, 3) * (hi - lo) + lo])
        for q in queries:
            for i in range(len(f)):
                sqr_d, closest, bc = ic.point_simplex_squared_distance(q, v, f, i)
                want = _ref_closest_point_on_tri(q, *v[f[i]])
                assert closest == pytest.approx(want, abs=1e-9)
                assert sqr_d == pytest.approx(np.sum((q - want) ** 2), abs=1e-9)
                # the closest point must lie on the simplex it was asked about
                assert bc @ v[f[i]] == pytest.approx(closest, abs=1e-9)
                assert bc.sum() == pytest.approx(1.0)
                assert np.all(bc >= -1e-12)


@pytest.mark.skipif("point_simplex_squared_distance" not in ic.AVAILABLE,
                    reason="removed in igl 2.6")
def test_point_simplex_min_over_faces_matches_point_mesh_squared_distance():
    """The two distance queries must agree — they did not before the storage fix."""
    rng = np.random.RandomState(1)
    v, f = _icosahedron()
    pts = rng.rand(25, 3) * 4 - 2
    per_face_min = np.array([
        min(ic.point_simplex_squared_distance(q, v, f, i)[0] for i in range(len(f)))
        for q in pts
    ])
    assert per_face_min == pytest.approx(ic.point_mesh_squared_distance(pts, v, f)[0])


@pytest.mark.skipif("point_simplex_squared_distance" not in ic.AVAILABLE,
                    reason="removed in igl 2.6")
def test_point_simplex_squared_distance_accepts_a_row_vector_point():
    """A (1, dim) query is the same single point as a (dim,) one.

    The raw binding silently returns a *different* wrong answer for the row-vector
    form, so the wrapper flattens.
    """
    v, f = _quad()
    p = np.array([0.25, 0.25, 2.0])
    assert (ic.point_simplex_squared_distance(p[None, :], v, f, 0)[0]
            == pytest.approx(ic.point_simplex_squared_distance(p, v, f, 0)[0]))


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
