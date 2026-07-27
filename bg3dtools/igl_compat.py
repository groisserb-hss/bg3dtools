"""libigl version-compatibility layer — the ONLY place in bg3dtools that imports ``igl``.

Why this module exists
----------------------
igl 2.6 is a **full rewrite of the python bindings** (pybind11 → nanobind) with
breaking changes to names, return arity, return order and dtypes. A machine that
drifted from the canonical **igl 2.5.1** to 2.6.1 broke a downstream consumer with
``ValueError: setting an array element with a sequence`` because
``igl.facet_components`` had silently started returning a ``(n, ids)`` tuple
instead of a bare ``ids`` array.

Every function here presents **one stable contract — the igl 2.5.1 convention** —
regardless of which binding is installed. Call sites in bg3dtools import from here
and never touch ``igl`` directly.

The contract
------------
Per wrapper the docstring states the 2.5.1 return arity/order, plus a ``2.6``
line recording what the 2.6.1 binding does and whether it was empirically
verified. On top of the per-function contract, these rules hold everywhere:

* index/label arrays are returned as **C-contiguous int64**,
* coordinate/scalar arrays as **C-contiguous float64**,
* arrays whose leading dimension is the element count keep that dimension even
  when it is 1 (the 2.5.1 binding squeezes those; 2.6.1 does not — normalizing
  is the only way to have a single contract).

Those three rules make the wrapper contract *stricter* than raw 2.5.1, never
different in shape or order. In particular the int64 pinning subsumes the class
of dtype bugs described in ``mesh/utils.py:as_igl_faces``: the 2.5.1 binding
propagates the input face dtype to its outputs (int32 in → int32 out, on every
platform), and int32 faces make predicates like ``is_edge_manifold`` report
spurious non-manifoldness.

No version strings
------------------
``igl.__version__`` **does not exist** in the 2.6.1 conda build, so this module
never parses a version. Symbols are feature-detected once at import; return
shapes are normalized with arity-explicit handling of the verified 2.6 behavior.
Anything else raises a clear :class:`RuntimeError` rather than passing junk
through.

Availability
------------
A handful of functions exist in only one of the two bindings. Use
:data:`AVAILABLE` (a frozenset of wrapper names whose underlying symbol
resolved) instead of ``hasattr(igl, ...)``::

    from bg3dtools.igl_compat import AVAILABLE, is_vertex_manifold
    if "is_vertex_manifold" in AVAILABLE:
        mask = is_vertex_manifold(faces)

Calling an unavailable wrapper raises ``RuntimeError`` with remediation advice.

Testing
-------
``tests/test_igl_compat.py`` pins every contract below and is deliberately
import-light (numpy + scipy only) so it can run under a bare scratch env::

    conda create -y -n igl261 -c conda-forge python=3.12 igl=2.6.1 numpy scipy pytest
    ~/opt/anaconda3/envs/igl261/bin/python -m pytest tests/test_igl_compat.py -v
"""

from typing import Any, List, Optional, Tuple

import igl
import numpy as np

__all__ = [
    "AVAILABLE",
    "MASSMATRIX_TYPE_BARYCENTRIC",
    "MASSMATRIX_TYPE_FULL",
    "MASSMATRIX_TYPE_VORONOI",
    "PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA",
    "adjacency_matrix",
    "all_boundary_loop",
    "average_onto_faces",
    "barycenter",
    "barycentric_coordinates_tri",
    "bfs_orient",
    "boundary_loop",
    "collapse_small_triangles",
    "connected_components",
    "cotmatrix",
    "cylinder",
    "decimate",
    "doublearea",
    "ears",
    "edges",
    "exact_geodesic",
    "extract_manifold_patches",
    "facet_components",
    "gaussian_curvature",
    "heat_geodesic",
    "internal_angles",
    "intrinsic_delaunay_cotmatrix",
    "is_edge_manifold",
    "is_vertex_manifold",
    "lscm",
    "massmatrix",
    "per_face_normals",
    "per_vertex_normals",
    "point_mesh_squared_distance",
    "point_simplex_squared_distance",
    "random_points_on_mesh",
    "read_obj",
    "read_triangle_mesh",
    "remove_duplicate_vertices",
    "remove_unreferenced",
    "resolve_duplicated_faces",
    "triangle_triangle_adjacency",
    "upsample",
    "vertex_triangle_adjacency",
    "winding_number",
    "write_triangle_mesh",
]


# ---------------------------------------------------------------------------
# symbol resolution (feature detection, once, at import)
# ---------------------------------------------------------------------------

_resolved = {}


def _sym(wrapper_name: str, *igl_names: str):
    """Resolve the first of *igl_names* present on ``igl``; record availability."""
    for n in igl_names:
        fn = getattr(igl, n, None)
        if fn is not None:
            _resolved[wrapper_name] = n
            return fn
    return None


def _missing(wrapper_name: str, *igl_names: str):
    raise RuntimeError(
        "igl API for %r not supported by bg3dtools.igl_compat: the installed "
        "libigl python binding provides none of %s — install igl 2.5.1 "
        "(conda install -c conda-forge igl=2.5.1)"
        % (wrapper_name, ", ".join(repr(n) for n in igl_names))
    )


def _unexpected(wrapper_name: str, got: Any):
    """Raised when a symbol exists but hands back an arity we have never seen."""
    shape = (
        "tuple of %d" % len(got)
        if isinstance(got, tuple)
        else type(got).__name__
    )
    raise RuntimeError(
        "igl API for %r not supported by bg3dtools.igl_compat: %r returned %s, "
        "which matches neither the igl 2.5.1 nor the 2.6.1 contract — install "
        "igl 2.5.1 (conda install -c conda-forge igl=2.5.1)"
        % (wrapper_name, _resolved.get(wrapper_name, wrapper_name), shape)
    )


_adjacency_matrix = _sym("adjacency_matrix", "adjacency_matrix")
_all_boundary_loop = _sym("all_boundary_loop", "all_boundary_loop", "boundary_loop_all")
_average_onto_faces = _sym("average_onto_faces", "average_onto_faces")
_barycenter = _sym("barycenter", "barycenter")
_barycentric_coordinates_tri = _sym(
    "barycentric_coordinates_tri", "barycentric_coordinates_tri", "barycentric_coordinates"
)
_bfs_orient = _sym("bfs_orient", "bfs_orient")
_boundary_loop = _sym("boundary_loop", "boundary_loop")
_collapse_small_triangles = _sym("collapse_small_triangles", "collapse_small_triangles")
_connected_components = _sym("connected_components", "connected_components")
_cotmatrix = _sym("cotmatrix", "cotmatrix")
_cylinder = _sym("cylinder", "cylinder")
_decimate = _sym("decimate", "decimate")
_doublearea = _sym("doublearea", "doublearea")
_ears = _sym("ears", "ears")
_edges = _sym("edges", "edges")
_exact_geodesic = _sym("exact_geodesic", "exact_geodesic")
_extract_manifold_patches = _sym("extract_manifold_patches", "extract_manifold_patches")
_facet_components = _sym("facet_components", "facet_components")
_gaussian_curvature = _sym("gaussian_curvature", "gaussian_curvature")
_heat_geodesic = _sym("heat_geodesic", "heat_geodesic")
_heat_precompute = getattr(igl, "heat_geodesics_precompute", None)
_heat_solve = getattr(igl, "heat_geodesics_solve", None)
_HeatData = getattr(igl, "HeatGeodesicsData", None)
_internal_angles = _sym("internal_angles", "internal_angles")
_intrinsic_delaunay_cotmatrix = _sym(
    "intrinsic_delaunay_cotmatrix", "intrinsic_delaunay_cotmatrix"
)
_is_edge_manifold = _sym("is_edge_manifold", "is_edge_manifold")
_is_vertex_manifold = _sym("is_vertex_manifold", "is_vertex_manifold")
_lscm = _sym("lscm", "lscm")
_massmatrix = _sym("massmatrix", "massmatrix")
_per_face_normals = _sym("per_face_normals", "per_face_normals")
_per_vertex_normals = _sym("per_vertex_normals", "per_vertex_normals")
_point_mesh_squared_distance = _sym(
    "point_mesh_squared_distance", "point_mesh_squared_distance"
)
_point_simplex_squared_distance = _sym(
    "point_simplex_squared_distance", "point_simplex_squared_distance"
)
_random_points_on_mesh = _sym("random_points_on_mesh", "random_points_on_mesh")
_read_obj = _sym("read_obj", "read_obj", "readOBJ")
_read_triangle_mesh = _sym("read_triangle_mesh", "read_triangle_mesh")
_remove_duplicate_vertices = _sym("remove_duplicate_vertices", "remove_duplicate_vertices")
_remove_unreferenced = _sym("remove_unreferenced", "remove_unreferenced")
_resolve_duplicated_faces = _sym("resolve_duplicated_faces", "resolve_duplicated_faces")
_triangle_triangle_adjacency = _sym(
    "triangle_triangle_adjacency", "triangle_triangle_adjacency"
)
_upsample = _sym("upsample", "upsample")
_vertex_triangle_adjacency = _sym("vertex_triangle_adjacency", "vertex_triangle_adjacency")
_winding_number = _sym("winding_number", "winding_number")
_write_triangle_mesh = _sym("write_triangle_mesh", "write_triangle_mesh")

# ``heat_geodesic`` is a single call in 2.5.1 and a precompute/solve pair in 2.6.1;
# register the composite so it shows up in AVAILABLE either way.
if _heat_geodesic is None and None not in (_heat_precompute, _heat_solve, _HeatData):
    _resolved["heat_geodesic"] = "heat_geodesics_precompute+heat_geodesics_solve"

#: Names in ``__all__`` whose underlying igl symbol resolved on this install.
#: Prefer ``"name" in AVAILABLE`` over ``hasattr(igl, "name")`` at call sites.
AVAILABLE = frozenset(_resolved)

#: Weighting enum for :func:`per_vertex_normals`. An ``int`` in 2.5.1 and a
#: ``PerVertexNormalsWeightingType`` enum member in 2.6.1 — both accepted by
#: their own binding, so this is re-exported verbatim rather than normalized.
PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA = getattr(
    igl, "PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA", 1
)

#: Mass-matrix types for :func:`massmatrix`. Plain ``int`` in 2.5.1 and
#: ``MassMatrixType`` enum members in 2.6.1 — both bindings accept either, with the
#: same integer mapping (barycentric 0, Voronoi 1, full 2), so these are re-exported
#: verbatim rather than normalized. Pass these rather than a literal.
MASSMATRIX_TYPE_BARYCENTRIC = getattr(igl, "MASSMATRIX_TYPE_BARYCENTRIC", 0)
MASSMATRIX_TYPE_VORONOI = getattr(igl, "MASSMATRIX_TYPE_VORONOI", 1)
MASSMATRIX_TYPE_FULL = getattr(igl, "MASSMATRIX_TYPE_FULL", 2)


# ---------------------------------------------------------------------------
# dtype / shape normalizers
# ---------------------------------------------------------------------------

def _idx(a) -> np.ndarray:
    """C-contiguous int64 view of an index/label array."""
    return np.ascontiguousarray(a, dtype=np.int64)


def _flt(a) -> np.ndarray:
    """C-contiguous float64 view of a coordinate/scalar array."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _idx2d(a, ncols: int) -> np.ndarray:
    return _idx(a).reshape(-1, ncols)


def _flt2d(a, ncols: int) -> np.ndarray:
    return _flt(a).reshape(-1, ncols)


def _idx1d(a, n: Optional[int] = None) -> np.ndarray:
    out = _idx(a)
    return out.reshape(-1) if n is None else out.reshape(n)


def _flt1d(a, n: Optional[int] = None) -> np.ndarray:
    out = _flt(a)
    return out.reshape(-1) if n is None else out.reshape(n)


def _rows(a) -> int:
    """Row count of a face/point array, treating a bare row as one row."""
    a = np.asarray(a)
    return 1 if a.ndim == 1 else a.shape[0]


def _cols(a, default: int = 3) -> int:
    a = np.asarray(a)
    return a.shape[-1] if a.ndim >= 1 and a.shape[-1] else default


# ---------------------------------------------------------------------------
# connectivity / topology
# ---------------------------------------------------------------------------

def edges(f: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a mesh.

    Contract: ``(nE, 2)`` int64, C-contiguous.

    2.6: same name and shape; the 2.5.1 binding propagates the input face dtype
    (int32 faces → int32 edges) while 2.6.1 always returns int64 — normalized here.
    Verified on both.
    """
    if _edges is None:
        _missing("edges", "edges")
    return _idx2d(_edges(_idx(f)), 2)


def adjacency_matrix(f: np.ndarray):
    """Vertex-vertex adjacency matrix.

    Contract: ``(nV, nV)`` scipy sparse matrix of int64 counts.

    2.6: identical (name, dtype, sparse layout). Verified on both.
    """
    if _adjacency_matrix is None:
        _missing("adjacency_matrix", "adjacency_matrix")
    return _adjacency_matrix(_idx(f))


def vertex_triangle_adjacency(f: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Flattened vertex→incident-triangle lists.

    Contract: ``(VF, NI)``; ``VF`` is ``(3*nF,)`` int64 and ``NI`` is ``(n+1,)``
    int64 such that vertex ``v``'s faces are ``VF[NI[v]:NI[v+1]]``.

    2.6: same name/order. The 2.5.1 binding returns **int32** here even on
    macOS/Linux; 2.6.1 returns int64 — normalized to int64. Verified on both.
    """
    if _vertex_triangle_adjacency is None:
        _missing("vertex_triangle_adjacency", "vertex_triangle_adjacency")
    vf, ni = _vertex_triangle_adjacency(_idx(f), int(n))
    return _idx1d(vf), _idx1d(ni)


def triangle_triangle_adjacency(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Face→face adjacency across each of the three edges.

    Contract: ``(TT, TTi)``, both ``(nF, 3)`` int64. Boundary edges are ``-1``.

    2.6: identical name, order and dtype. Verified on both.
    """
    if _triangle_triangle_adjacency is None:
        _missing("triangle_triangle_adjacency", "triangle_triangle_adjacency")
    tt, tti = _triangle_triangle_adjacency(_idx(f))
    return _idx2d(tt, 3), _idx2d(tti, 3)


def facet_components(f: np.ndarray) -> np.ndarray:
    """Connected-component id per face (edge-edge connectivity).

    Contract: bare ``(nF,)`` int64 array of component ids — **not** a tuple.

    2.6: ``igl.facet_components`` returns ``(n_components, ids)``. This is the
    incident that motivated the module: downstream ``np.bincount(result)`` on the
    2-tuple raised ``ValueError: setting an array element with a sequence``. The
    count is recoverable as ``ids.max() + 1``; use :func:`extract_manifold_patches`
    if you want it returned. Verified on both.
    """
    if _facet_components is None:
        _missing("facet_components", "facet_components")
    out = _facet_components(_idx(f))
    if isinstance(out, tuple):
        if len(out) != 2:  # 2.6.1: (n_components, ids)
            _unexpected("facet_components", out)
        out = out[1]
    return _idx1d(out, _rows(f))


def extract_manifold_patches(f: np.ndarray) -> Tuple[int, np.ndarray]:
    """Split faces into patches that are manifold-connected.

    Contract: ``(n_patches: int, labels: (nF,) int64)``.

    2.6: **REMOVED** — raises ``RuntimeError``. Guard with
    ``"extract_manifold_patches" in AVAILABLE``. There is no drop-in 2.6
    replacement: :func:`facet_components` is the nearest, but it joins across
    non-manifold edges that ``extract_manifold_patches`` splits at, so a caller
    falling back to it gets *coarser* patches. Verified 2.5.1-only.
    """
    if _extract_manifold_patches is None:
        _missing("extract_manifold_patches", "extract_manifold_patches")
    out = _extract_manifold_patches(_idx(f))
    if not isinstance(out, tuple) or len(out) != 2:
        _unexpected("extract_manifold_patches", out)
    n, labels = out
    return int(n), _idx1d(labels, _rows(f))


def connected_components(a) -> Tuple[int, np.ndarray, np.ndarray]:
    """Connected components of a sparse **adjacency matrix** (not faces).

    Contract: ``(n_components: int, C: (nV,) int64, K: (n_components,) int64)``
    where ``C`` labels each vertex and ``K`` counts members per component.

    2.6: same name, order and arity, and additionally tolerates a face array as
    input (2.5.1 raises ``TypeError`` for that). Pass an adjacency matrix — build
    one with :func:`adjacency_matrix` — to stay portable. Verified on both.
    """
    if _connected_components is None:
        _missing("connected_components", "connected_components")
    out = _connected_components(a)
    if not isinstance(out, tuple) or len(out) != 3:
        _unexpected("connected_components", out)
    n, c, k = out
    return int(n), _idx1d(c), _idx1d(k)


def is_edge_manifold(f: np.ndarray) -> bool:
    """True when every edge is incident to one or two consistently oriented faces.

    Contract: a plain ``bool``.

    2.6: returns a 5-tuple ``(is_manifold, BF, E, EMAP, BE)``; the flag is
    element 0. Verified on both, manifold and non-manifold.
    """
    if _is_edge_manifold is None:
        _missing("is_edge_manifold", "is_edge_manifold")
    out = _is_edge_manifold(_idx(f))
    if isinstance(out, tuple):
        if len(out) != 5:  # 2.6.1: (bool, BF, E, EMAP, BE)
            _unexpected("is_edge_manifold", out)
        out = out[0]
    return bool(out)


def is_vertex_manifold(f: np.ndarray) -> np.ndarray:
    """Per-vertex manifoldness mask (a vertex is non-manifold when its incident
    faces form more than one fan).

    Contract: ``(max(f) + 1,)`` bool array. Note the length follows the highest
    referenced index, *not* any nominal vertex count — pad if you need ``nV``.

    2.6: **NEW** — absent from 2.5.1, so this is the reverse of the usual
    situation. Guard with ``"is_vertex_manifold" in AVAILABLE``. Verified 2.6-only.
    """
    if _is_vertex_manifold is None:
        _missing("is_vertex_manifold", "is_vertex_manifold")
    return np.ascontiguousarray(_is_vertex_manifold(_idx(f)), dtype=bool)


def boundary_loop(f: np.ndarray) -> np.ndarray:
    """Longest ordered boundary loop.

    Contract: ``(n,)`` int64 vertex indices; empty for a closed mesh.

    2.6: identical name, shape and dtype. Verified on both, open and closed.
    """
    if _boundary_loop is None:
        _missing("boundary_loop", "boundary_loop")
    return _idx1d(_boundary_loop(_idx(f)))


def all_boundary_loop(f: np.ndarray) -> List[List[int]]:
    """Every ordered boundary loop.

    Contract: a ``list`` of ``list``s of python ``int``; empty list for a closed
    mesh (so ``not all_boundary_loop(f)`` tests closedness).

    2.6: **RENAMED** to ``igl.boundary_loop_all``; return type unchanged.
    Verified on both, open and closed.
    """
    if _all_boundary_loop is None:
        _missing("all_boundary_loop", "all_boundary_loop", "boundary_loop_all")
    return _all_boundary_loop(_idx(f))


def ears(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Faces with two boundary edges, and which of their edges is interior.

    Contract: ``(ears, ear_opp)``, both ``(n_ears,)`` int64; both empty when the
    mesh has no ears.

    2.6: identical name, order, shape and dtype. Verified on both.
    """
    if _ears is None:
        _missing("ears", "ears")
    e, eo = _ears(_idx(f))
    return _idx1d(e), _idx1d(eo)


def bfs_orient(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orient each connected patch consistently by breadth-first traversal.

    Contract: ``(FF, C)`` — reoriented ``(nF, 3)`` int64 faces and ``(nF,)`` int64
    patch ids.

    2.6: identical name and order. 2.5.1 propagates the input face dtype
    (int32 in → int32 out); pinned to int64 here. Verified on both.
    """
    if _bfs_orient is None:
        _missing("bfs_orient", "bfs_orient")
    ff, c = _bfs_orient(_idx(f))
    return _idx2d(ff, _cols(f)), _idx1d(c)


# ---------------------------------------------------------------------------
# cleanup / remeshing
# ---------------------------------------------------------------------------

def remove_unreferenced(
    v: np.ndarray, f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop vertices no face references, reindexing faces.

    Contract: ``(NV, NF, I, J)`` — kept vertices ``(nV', dim)`` float64, remapped
    faces ``(nF, ss)`` int64, ``I`` ``(nV,)`` old→new and ``J`` ``(nV',)`` new→old,
    both int64.

    2.6: same name, order and arity. 2.5.1 squeezes ``NF`` to ``(ss,)`` when only
    one face survives (and propagates int32 faces); both normalized here.
    Verified on both.
    """
    if _remove_unreferenced is None:
        _missing("remove_unreferenced", "remove_unreferenced")
    out = _remove_unreferenced(_flt(v), _idx(f))
    if not isinstance(out, tuple) or len(out) != 4:
        _unexpected("remove_unreferenced", out)
    nv, nf, i, j = out
    return _flt2d(nv, _cols(v)), _idx2d(nf, _cols(f)), _idx1d(i), _idx1d(j)


def remove_duplicate_vertices(
    v: np.ndarray, f: np.ndarray, epsilon: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge vertices closer together than *epsilon*.

    Contract: ``(SV, SVI, SVJ, SF)`` — merged vertices ``(nV', dim)`` float64,
    ``SVI`` ``(nV',)`` new→old, ``SVJ`` ``(nV,)`` old→new, remapped faces
    ``(nF, ss)`` int64.

    2.6: identical name, order, arity and dtypes. Verified on both.
    """
    if _remove_duplicate_vertices is None:
        _missing("remove_duplicate_vertices", "remove_duplicate_vertices")
    out = _remove_duplicate_vertices(_flt(v), _idx(f), float(epsilon))
    if not isinstance(out, tuple) or len(out) != 4:
        _unexpected("remove_duplicate_vertices", out)
    sv, svi, svj, sf = out
    return _flt2d(sv, _cols(v)), _idx1d(svi), _idx1d(svj), _idx2d(sf, _cols(f))


def resolve_duplicated_faces(f1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cancel duplicated faces against their oppositely oriented copies.

    Contract: ``(F2, J)`` — surviving ``(nF', 3)`` int64 faces and ``(nF',)`` int64
    indices into the input.

    2.6: **REMOVED** — raises ``RuntimeError``. 2.5.1 squeezes ``F2``/``J`` when a
    single face survives; normalized here. Verified 2.5.1-only.
    """
    if _resolve_duplicated_faces is None:
        _missing("resolve_duplicated_faces", "resolve_duplicated_faces")
    out = _resolve_duplicated_faces(_idx(f1))
    if not isinstance(out, tuple) or len(out) != 2:
        _unexpected("resolve_duplicated_faces", out)
    f2, j = out
    return _idx2d(f2, _cols(f1)), _idx1d(j)


def collapse_small_triangles(v: np.ndarray, f: np.ndarray, eps: float) -> np.ndarray:
    """Collapse triangles whose relative area is below *eps*.

    Contract: ``(nF, 3)`` int64 faces (degenerate ones become repeated indices;
    callers usually follow up with :func:`remove_unreferenced`).

    2.6: **REMOVED** — raises ``RuntimeError``. Verified 2.5.1-only.
    """
    if _collapse_small_triangles is None:
        _missing("collapse_small_triangles", "collapse_small_triangles")
    return _idx2d(_collapse_small_triangles(_flt(v), _idx(f), float(eps)), _cols(f))


def upsample(
    v: np.ndarray, f: np.ndarray, number_of_subdivs: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Loop-style topological subdivision (vertices are not smoothed).

    Contract: ``(NV, NF)`` — ``(nV', dim)`` float64 and ``(nF', 3)`` int64.

    2.6: identical name and order. 2.5.1 propagates int32 faces; pinned to int64.
    Verified on both.
    """
    if _upsample is None:
        _missing("upsample", "upsample")
    out = _upsample(_flt(v), _idx(f), int(number_of_subdivs))
    if not isinstance(out, tuple) or len(out) != 2:
        _unexpected("upsample", out)
    nv, nf = out
    return _flt2d(nv, _cols(v)), _idx2d(nf, _cols(f))


def decimate(
    v: np.ndarray, f: np.ndarray, max_m: int
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse edges until at most *max_m* faces remain.

    Contract: ``(success, U, G, J, I)`` — ``success`` bool, ``U`` ``(nV', dim)``
    float64, ``G`` ``(nF', 3)`` int64, ``J`` ``(nF',)`` int64 birth faces,
    ``I`` ``(nV',)`` int64 birth vertices.

    2.6: drops the leading success flag (returns 4 elements) and hands back
    **Fortran-ordered** ``U``/``G``. The flag is synthesized as ``True`` — the 2.6
    binding raises on failure rather than reporting it — and contiguity is
    restored. Verified on both.
    """
    if _decimate is None:
        _missing("decimate", "decimate")
    out = _decimate(_flt(v), _idx(f), int(max_m))
    if not isinstance(out, tuple):
        _unexpected("decimate", out)
    if len(out) == 5:  # 2.5.1: (success, U, G, J, I)
        success, u, g, j, i = out
        success = bool(success)
    elif len(out) == 4:  # 2.6.1: (U, G, J, I) — no success flag
        u, g, j, i = out
        success = True
    else:
        _unexpected("decimate", out)
    return success, _flt2d(u, _cols(v)), _idx2d(g, _cols(f)), _idx1d(j), _idx1d(i)


# ---------------------------------------------------------------------------
# differential geometry / measures
# ---------------------------------------------------------------------------

def doublearea(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Twice the area of each triangle.

    Contract: ``(nF,)`` float64.

    2.6: identical. Verified on both.
    """
    if _doublearea is None:
        _missing("doublearea", "doublearea")
    return _flt1d(_doublearea(_flt(v), _idx(f)), _rows(f))


def barycenter(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Centroid of each face.

    Contract: ``(nF, dim)`` float64.

    2.6: identical. Verified on both.
    """
    if _barycenter is None:
        _missing("barycenter", "barycenter")
    return _flt2d(_barycenter(_flt(v), _idx(f)), _cols(v))


def internal_angles(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Interior angle at each corner of each triangle, in radians.

    Contract: ``(nF, 3)`` float64.

    2.6: identical. Verified on both.
    """
    if _internal_angles is None:
        _missing("internal_angles", "internal_angles")
    return _flt2d(_internal_angles(_flt(v), _idx(f)), 3)


def gaussian_curvature(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Integrated (angle-defect) Gaussian curvature per vertex.

    Contract: ``(nV,)`` float64.

    2.6: identical. Verified on both.
    """
    if _gaussian_curvature is None:
        _missing("gaussian_curvature", "gaussian_curvature")
    return _flt1d(_gaussian_curvature(_flt(v), _idx(f)), _rows(v))


def per_vertex_normals(v: np.ndarray, f: np.ndarray, weighting=None) -> np.ndarray:
    """Area/angle-weighted vertex normals.

    Contract: ``(nV, 3)`` float64. Rows for unreferenced vertices are ``NaN``
    (both bindings) — callers normally zero those out.

    2.6: same name and shape, but *weighting* changed from an ``int`` to a
    ``PerVertexNormalsWeightingType`` enum. Pass
    :data:`PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA` (re-exported from whichever
    binding is installed) rather than a literal. Verified on both.
    """
    if _per_vertex_normals is None:
        _missing("per_vertex_normals", "per_vertex_normals")
    if weighting is None:
        weighting = PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
    return _flt2d(_per_vertex_normals(_flt(v), _idx(f), weighting), 3)


def per_face_normals(v: np.ndarray, f: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Unit face normals, with *z* substituted for degenerate faces.

    Contract: ``(nF, 3)`` float64.

    2.6: identical. Verified on both.
    """
    if _per_face_normals is None:
        _missing("per_face_normals", "per_face_normals")
    return _flt2d(_per_face_normals(_flt(v), _idx(f), _flt(z)), 3)


def cotmatrix(v: np.ndarray, f: np.ndarray):
    """Cotangent (discrete Laplace-Beltrami) matrix.

    Contract: ``(nV, nV)`` float64 scipy sparse matrix.

    2.6: identical name, return type and values. Verified on both.

    Notes
    -----
    bg3dtools computes its Laplacians with the hand-rolled
    ``mesh.laplace.cotangent_weights`` instead of this. A long-standing comment
    said that was because igl's version "returns all zeros" on 2.5.1; that is
    **not reproducible** — see :func:`massmatrix` for the measurements. The two
    agree to 1.3e-15 but with **opposite sign** (igl's is negative
    semi-definite), so they are not interchangeable without a sign flip.
    """
    if _cotmatrix is None:
        _missing("cotmatrix", "cotmatrix")
    return _cotmatrix(_flt(v), _idx(f))


def massmatrix(v: np.ndarray, f: np.ndarray, type=None):
    """Mass matrix of a mesh (vertex areas on the diagonal, for lumped types).

    Contract: ``(nV, nV)`` float64 scipy sparse matrix. *type* selects the
    quadrature: pass one of :data:`MASSMATRIX_TYPE_VORONOI` (default),
    :data:`MASSMATRIX_TYPE_BARYCENTRIC` or :data:`MASSMATRIX_TYPE_FULL`. The
    first two are lumped (diagonal, and the diagonal sums to the surface area);
    ``FULL`` is the unlumped Galerkin matrix, whose diagonal sums to half that.
    Voronoi and barycentric differ per vertex on irregular meshes, so the choice
    matters even though their totals agree.

    The argument is named *type* to match libigl's own keyword; it shadows the
    builtin only inside this signature.

    2.6: same name, return type and values. The type constants became
    ``MassMatrixType`` enum members, but the integer mapping is unchanged and both
    bindings accept either form. Both bindings' own default is
    ``MASSMATRIX_TYPE_DEFAULT``, which resolves to Voronoi for triangle meshes;
    this wrapper passes Voronoi explicitly rather than relying on that.
    Verified on both.

    Notes
    -----
    Contrary to a long-standing comment in this repo, neither this nor
    :func:`cotmatrix` returns all zeros on igl 2.5.1. Measured on 2.5.1 and 2.6.1
    across planar, closed, non-manifold, zero-area and int32/float32 inputs: the
    Voronoi diagonal sums to exactly the surface area, and ``cotmatrix`` matches
    ``mesh.laplace.cotangent_weights`` up to sign. The hand-rolled
    ``mesh.laplace`` implementations are kept because they are *not* drop-in
    equivalents — ``fem_mass_matrix``'s diagonal sums to half this one's and
    ``lumped_vertex_areas`` differs by ~0.1% (a different lumping scheme).
    """
    if _massmatrix is None:
        _missing("massmatrix", "massmatrix")
    if type is None:
        type = MASSMATRIX_TYPE_VORONOI
    return _massmatrix(_flt(v), _idx(f), type)


def intrinsic_delaunay_cotmatrix(v: np.ndarray, f: np.ndarray):
    """Cotangent matrix of the intrinsic Delaunay triangulation.

    Contract: ``(L, l, F)`` — ``(nV, nV)`` float64 sparse, ``(nF, 3)`` float64
    edge lengths, ``(nF, 3)`` int64 flipped faces.

    2.6: identical name, order and arity. Verified on both.
    """
    if _intrinsic_delaunay_cotmatrix is None:
        _missing("intrinsic_delaunay_cotmatrix", "intrinsic_delaunay_cotmatrix")
    out = _intrinsic_delaunay_cotmatrix(_flt(v), _idx(f))
    if not isinstance(out, tuple) or len(out) != 3:
        _unexpected("intrinsic_delaunay_cotmatrix", out)
    ldd, l, ff = out
    return ldd, _flt2d(l, 3), _idx2d(ff, 3)


def average_onto_faces(f: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Average a per-vertex scalar field onto faces.

    Contract: ``(nF,)`` float64. Argument order is ``(faces, vertex_values)``.

    2.6: identical name, argument order and return shape. Verified on both.
    """
    if _average_onto_faces is None:
        _missing("average_onto_faces", "average_onto_faces")
    return _flt1d(_average_onto_faces(_idx(f), _flt(s)), _rows(f))


def winding_number(v: np.ndarray, f: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Generalized winding number of a mesh at query points *o*.

    Contract: ``(nO,)`` float64 (``≈1`` inside a closed mesh, ``≈0`` outside).

    2.6: same name and semantics. 2.5.1 squeezes the result to 0-d for a single
    query point; normalized to ``(nO,)``. Verified on both.
    """
    if _winding_number is None:
        _missing("winding_number", "winding_number")
    o = _flt(o)
    if o.ndim == 1:
        o = o[None, :]
    return _flt1d(_winding_number(_flt(v), _idx(f), o), o.shape[0])


# ---------------------------------------------------------------------------
# distance queries
# ---------------------------------------------------------------------------

def point_mesh_squared_distance(
    p: np.ndarray, v: np.ndarray, ele: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest point on a mesh for each query point.

    Contract: ``(sqrD, I, C)`` — ``(nP,)`` float64 squared distances, ``(nP,)``
    int64 element indices, ``(nP, dim)`` float64 closest points.

    2.6: same name, order and arity. 2.5.1 squeezes ``sqrD``/``I`` to 0-d and
    ``C`` to ``(dim,)`` when ``nP == 1``; all normalized to keep the leading
    query dimension. Verified on both for ``nP == 1`` and ``nP > 1``.
    """
    if _point_mesh_squared_distance is None:
        _missing("point_mesh_squared_distance", "point_mesh_squared_distance")
    p = _flt(p)
    if p.ndim == 1:
        p = p[None, :]
    out = _point_mesh_squared_distance(p, _flt(v), _idx(ele))
    if not isinstance(out, tuple) or len(out) != 3:
        _unexpected("point_mesh_squared_distance", out)
    sqr_d, i, c = out
    n = p.shape[0]
    return _flt1d(sqr_d, n), _idx1d(i, n), _flt2d(c, p.shape[1])


def barycentric_coordinates_tri(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Barycentric coordinates of points *p* within triangles ``(a, b, c)``.

    One triangle per query point: all four arguments are ``(nP, dim)``. Points
    outside their triangle get coordinates outside ``[0, 1]``; degenerate
    triangles yield ``NaN`` (both bindings), which callers usually replace with
    ``1/3``.

    Contract: ``(nP, 3)`` float64, C-contiguous.

    2.6: **RENAMED** to ``igl.barycentric_coordinates`` (the tri/tet distinction
    moved into the argument count). Values agree exactly and both are insensitive
    to array storage order. 2.5.1 squeezes the result to ``(3,)`` when
    ``nP == 1``; normalized here. Verified on both.
    """
    if _barycentric_coordinates_tri is None:
        _missing("barycentric_coordinates_tri",
                 "barycentric_coordinates_tri", "barycentric_coordinates")
    p, a, b, c = (_flt(x) for x in (p, a, b, c))
    if p.ndim == 1:
        p, a, b, c = p[None, :], a[None, :], b[None, :], c[None, :]
    return _flt2d(_barycentric_coordinates_tri(p, a, b, c), 3)


def point_simplex_squared_distance(
    p: np.ndarray, v: np.ndarray, ele: np.ndarray, i: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Squared distance from the single point *p* to simplex *i* of a mesh.

    Contract: ``(sqrD: float, c: (dim,) float64, bc: (ss,) float64)`` — the
    distance, the closest point on that simplex, and its barycentric coordinates.
    *p* is ONE ``dim``-long point, not a batch; use
    :func:`point_mesh_squared_distance` for batches.

    2.6: **REMOVED** — raises ``RuntimeError``. Note *i* is a scalar index and so
    is exempt from the array-dtype matching libigl enforces elsewhere.
    Verified 2.5.1-only.

    Notes
    -----
    **The igl 2.5.1 binding misreads *v*: it expects column-major storage.** Fed a
    normal C-contiguous vertex array it returns a "closest point" that is not on
    the mesh at all — e.g. for a planar mesh in ``z == 0`` it returns points with
    ``z != 0``. This wrapper therefore passes ``np.asfortranarray(v)``, which is
    correct: checked against an independent point-triangle reference (Ericson,
    *Real-Time Collision Detection* §5.1.5) over 1387 point/face pairs across three
    meshes, max error 8.9e-16, where the C-order call was right in only 24 of them.
    ``ele`` and ``p`` are order-insensitive.

    This is the one place where the layer **changes** 2.5.1's numerical answers
    rather than preserving them — the previous answers were simply wrong. It fixes
    ``mesh.highD.HighDimMesh``, whose nearest-point search is built on this call.
    """
    if _point_simplex_squared_distance is None:
        _missing("point_simplex_squared_distance", "point_simplex_squared_distance")
    # reshape(-1): a (1, dim) point silently yields a different wrong answer
    out = _point_simplex_squared_distance(
        _flt(p).reshape(-1), np.asfortranarray(v, dtype=np.float64), _idx(ele), int(i)
    )
    if not isinstance(out, tuple) or len(out) != 3:
        _unexpected("point_simplex_squared_distance", out)
    sqr_d, c, bc = out
    return float(sqr_d), _flt1d(c), _flt1d(bc)


_EG_CANDIDATES = (("vs", "vt"), ("VS", "VT"))
_EG_KWARGS = None  # resolved on first exact_geodesic() call, then cached


def exact_geodesic(
    v: np.ndarray, f: np.ndarray, vs: np.ndarray, vt: np.ndarray
) -> np.ndarray:
    """Exact geodesic distances from source vertices *vs* to target vertices *vt*.

    Contract: ``(len(vt),)`` float64.

    2.6: **SILENT WRONG ANSWER if called positionally.** The signature changed
    from ``(v, f, vs, vt, fs=None, ft=None)`` to ``(V, F, VS, FS, VT, FT)``, so a
    four-positional call binds the target *vertices* to the source *faces* slot
    and returns an empty array instead of raising. The keyword names also changed
    case (``vs``/``vt`` → ``VS``/``VT``). This wrapper therefore always calls by
    keyword and resolves which spelling the installed binding accepts on first use
    — a wrong spelling raises ``TypeError``, so the dispatch can never silently
    pick the broken calling convention. Verified on both against known distances.
    """
    if _exact_geodesic is None:
        _missing("exact_geodesic", "exact_geodesic")
    v, f = _flt(v), _idx(f)
    # libigl requires every *array* integer argument of a call to share one
    # integer dtype; _idx() pins faces and index sets alike to int64.
    vs, vt = _idx1d(vs), _idx1d(vt)

    global _EG_KWARGS
    for names in _EG_CANDIDATES if _EG_KWARGS is None else (_EG_KWARGS,):
        src, tgt = names
        try:
            d = _exact_geodesic(v, f, **{src: vs, tgt: vt})
        except TypeError:
            continue
        _EG_KWARGS = names
        return _flt1d(d, vt.size)
    raise RuntimeError(
        "igl API for 'exact_geodesic' not supported by bg3dtools.igl_compat: the "
        "installed binding accepts neither the igl 2.5.1 (vs=/vt=) nor the 2.6.1 "
        "(VS=/VT=) keyword spelling, and calling positionally is unsafe because "
        "2.6 reorders the arguments — install igl 2.5.1 "
        "(conda install -c conda-forge igl=2.5.1)"
    )


def heat_geodesic(
    v: np.ndarray, f: np.ndarray, t: float, gamma: np.ndarray
) -> np.ndarray:
    """Heat-method approximate geodesic distances from source vertices *gamma*.

    Contract: ``(nV,)`` float64.

    2.6: the one-shot ``igl.heat_geodesic`` is **REMOVED**, replaced by a
    precompute/solve pair around a ``HeatGeodesicsData`` object. This wrapper
    drives that pair, so the one-shot contract keeps working; it re-factorizes on
    every call, exactly as 2.5.1's one-shot does. Verified on both — the two
    paths agree to float64 round-off on an icosahedron.
    """
    v, f = _flt(v), _idx(f)
    gamma = _idx1d(gamma)
    if _heat_geodesic is not None:
        return _flt1d(_heat_geodesic(v, f, float(t), gamma), _rows(v))
    if None in (_heat_precompute, _heat_solve, _HeatData):
        _missing("heat_geodesic", "heat_geodesic", "heat_geodesics_precompute")
    data = _HeatData()
    _heat_precompute(v, f, float(t), data)
    return _flt1d(_heat_solve(data, gamma), _rows(v))


def random_points_on_mesh(
    n: int, v: np.ndarray, f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample *n* points uniformly by area.

    Contract: ``(B, FI, P)`` — ``(n, 3)`` float64 barycentric coordinates,
    ``(n,)`` int64 face indices, ``(n, dim)`` float64 positions.

    2.6: identical name, order, arity and dtypes. Verified on both.

    Warning
    -------
    ``FI`` has been observed to contain out-of-range face indices; callers should
    mask with ``FI < len(f)`` (see ``mesh.utils.surface_sample``). Not reproduced
    on either binding here, but the guard is cheap and is kept.
    """
    if _random_points_on_mesh is None:
        _missing("random_points_on_mesh", "random_points_on_mesh")
    out = _random_points_on_mesh(int(n), _flt(v), _idx(f))
    if not isinstance(out, tuple) or len(out) != 3:
        _unexpected("random_points_on_mesh", out)
    b, fi, p = out
    return _flt2d(b, 3), _idx1d(fi), _flt2d(p, _cols(v))


# ---------------------------------------------------------------------------
# parametrization / generation / IO
# ---------------------------------------------------------------------------

def lscm(
    v: np.ndarray, f: np.ndarray, b: np.ndarray, bc: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """Least-squares conformal map with boundary vertices *b* pinned to *bc*.

    Contract: ``(success: bool, uv: (nV, 2) float64)``.

    2.6: returns ``(uv, Q)`` — the success flag is gone and a sparse quadratic
    form takes its place, i.e. both the arity *and* the position of ``uv``
    changed. The flag is synthesized from ``np.isfinite(uv).all()``, since a
    failed solve surfaces as non-finite coordinates. Verified on both.
    """
    if _lscm is None:
        _missing("lscm", "lscm")
    out = _lscm(_flt(v), _idx(f), _idx1d(b), _flt2d(bc, 2))
    if not isinstance(out, tuple) or len(out) != 2:
        _unexpected("lscm", out)
    first, second = out
    if isinstance(first, (bool, np.bool_)):  # 2.5.1: (success, uv)
        return bool(first), _flt2d(second, 2)
    uv = _flt2d(first, 2)  # 2.6.1: (uv, Q)
    return bool(np.isfinite(uv).all()), uv


def cylinder(axis_devisions: int, height_devisions: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unit open cylinder mesh (libigl's spelling of "divisions" is preserved).

    Contract: ``(V, F)`` — ``(nV, 3)`` float64 and ``(nF, 3)`` int64.

    2.6: identical name, order and dtypes. Verified on both.
    """
    if _cylinder is None:
        _missing("cylinder", "cylinder")
    out = _cylinder(int(axis_devisions), int(height_devisions))
    if not isinstance(out, tuple) or len(out) != 2:
        _unexpected("cylinder", out)
    v, f = out
    return _flt2d(v, 3), _idx2d(f, 3)


def read_obj(obj_file) -> Tuple[np.ndarray, ...]:
    """Read a Wavefront OBJ, including texture coordinates and normals.

    Contract: ``(V, TC, N, F, FTC, FN)`` — coordinates float64, indices int64,
    all C-contiguous. Absent blocks come back as ``(0, 0)``-shaped arrays.

    2.6: **RENAMED** to ``igl.readOBJ``; return order, arity and dtypes are
    unchanged, and it parses both 2019-era and 2026-era 3dMD scanner OBJs
    identically. 2.5.1 squeezes any block holding a single row to 1-D;
    ``np.atleast_2d`` restores it without assuming a corner count (OBJ faces are
    not necessarily triangles). Verified on both.
    """
    if _read_obj is None:
        _missing("read_obj", "read_obj", "readOBJ")
    out = _read_obj(str(obj_file))
    if not isinstance(out, tuple) or len(out) != 6:
        _unexpected("read_obj", out)
    v, tc, n, f, ftc, fn = out
    return (
        np.atleast_2d(_flt(v)),
        np.atleast_2d(_flt(tc)),
        np.atleast_2d(_flt(n)),
        np.atleast_2d(_idx(f)),
        np.atleast_2d(_idx(ftc)),
        np.atleast_2d(_idx(fn)),
    )


def read_triangle_mesh(filename) -> Tuple[np.ndarray, np.ndarray]:
    """Read a triangle mesh, picking the format from the extension (obj/off/ply/stl/…).

    Contract: ``(V, F)`` — ``(nV, 3)`` float64 and ``(nF, 3)`` int64, both
    C-contiguous, **always 2-D**. Use this rather than
    :func:`bg3dtools.mesh.mesh_io.read_triangle_mesh` when you want libigl's own
    parser; the latter is a trimesh-based reader that also handles scenes and
    point clouds.

    2.6: same name, arity, order and dtypes. Two differences, both normalized
    here: 2.5.1 squeezes ``F`` to ``(3,)`` when the file holds a single triangle
    (2.6.1 returns ``(1, 3)``), and 2.5.1 takes an extra ``dtypef`` argument that
    2.6.1 dropped — this wrapper never passes it, so both read float64.
    Verified on both.
    """
    if _read_triangle_mesh is None:
        _missing("read_triangle_mesh", "read_triangle_mesh")
    out = _read_triangle_mesh(str(filename))
    if not isinstance(out, tuple) or len(out) != 2:
        _unexpected("read_triangle_mesh", out)
    v, f = out
    return _flt2d(v, 3), _idx2d(f, 3)


def write_triangle_mesh(filename, v: np.ndarray, f: np.ndarray) -> bool:
    """Write a mesh, picking the format from the extension (obj/off/stl/ply/…).

    Contract: ``True`` on success, ``False`` on failure.

    2.6: returns ``None`` and *raises* on an unwritable path instead of returning
    ``False``. Normalized back to the bool contract, so a failed write is a
    falsy return here on both bindings rather than an exception on one — check
    the result if you care. Verified on both, including the failure path.
    """
    if _write_triangle_mesh is None:
        _missing("write_triangle_mesh", "write_triangle_mesh")
    try:
        out = _write_triangle_mesh(str(filename), _flt(v), _idx(f))
    except (OSError, RuntimeError):  # 2.6.1 raises where 2.5.1 returns False
        return False
    return True if out is None else bool(out)
