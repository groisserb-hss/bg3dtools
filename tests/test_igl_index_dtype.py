"""Guard against libigl integer-dtype mismatches.

libigl's pybind bindings require every *array* integer argument of a call (the
faces ``f`` plus index arrays such as ``exact_geodesic``'s source/target sets
``vs``/``vt``) to share ONE integer dtype, or they reject the call::

    ValueError: Invalid type (int64, Row Major) for argument 'vs'.
    Expected it to match argument 'f' which is of type (int32, Row Major).

NumPy's default integer is int64 on Linux/macOS but int32 on Windows, and mesh
I/O historically mixed the two, so this only ever bit on some hosts. These tests
pin the remaining defenses: (1) ``match_index_dtype`` harmonizes index arrays to
f's dtype at the call site; (2) the mesh readers canonicalize faces to int64.

The primary defense now lives in :mod:`bg3dtools.igl_compat`, which pins every
array index argument *and* return to int64 — see ``tests/test_igl_compat.py`` for
the tests that libigl calls are structurally immune to the mismatch. This module
keeps the tests for the two repo-level helpers and the call sites that depend on
them.

(Scalar index arguments — e.g. ``point_simplex_squared_distance``'s face index —
are NOT subject to this matching, verified separately, so they need no fix.)
"""

import numpy as np

from bg3dtools.igl_compat import all_boundary_loop, is_edge_manifold
from bg3dtools.mesh.utils import match_index_dtype


def _mesh():
    """A tiny non-degenerate triangle mesh (int64 faces)."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]],
                 dtype=np.float64)
    f = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=np.int64)
    return v, f


# ---------------------------------------------------------------------------
# match_index_dtype
# ---------------------------------------------------------------------------

def test_match_index_dtype_matches_faces():
    f32 = np.array([[0, 1, 2]], dtype=np.int32)
    f64 = f32.astype(np.int64)
    assert match_index_dtype(f32, np.arange(3, dtype=np.int64)).dtype == np.int32
    assert match_index_dtype(f64, np.arange(3, dtype=np.int32)).dtype == np.int64


def test_match_index_dtype_multiple_and_contiguous():
    f = np.array([[0, 1, 2]], dtype=np.int32)
    a, b = match_index_dtype(f, np.arange(3, dtype=np.int64), np.arange(2, dtype=np.int64))
    assert a.dtype == np.int32 and b.dtype == np.int32
    assert a.flags['C_CONTIGUOUS'] and b.flags['C_CONTIGUOUS']


def test_match_index_dtype_float_faces_fall_back_to_int64():
    # an empty / float face placeholder must still yield a usable integer dtype
    out = match_index_dtype(np.zeros((0, 3), dtype=np.float64), np.arange(2))
    assert out.dtype == np.int64


# ---------------------------------------------------------------------------
# I/O boundary: readers canonicalize faces to int64
# ---------------------------------------------------------------------------

def test_mesh_readers_return_int64_faces(tmp_path):
    from bg3dtools.mesh.mesh_io import (read_triangle_mesh, read_colored_plyfile,
                                        write_colored_plyfile)
    v, f = _mesh()
    p = str(tmp_path / "mesh.ply")
    write_colored_plyfile(p, v, f)                # PLY stores int32 indices on disk
    assert read_colored_plyfile(p)[1].dtype == np.int64
    assert read_triangle_mesh(p)[1].dtype == np.int64


# ---------------------------------------------------------------------------
# make_manifold: int64-pinned collapse + graceful post-collapse fallback
#
# On the Windows libigl wheel, collapse_small_triangles / remove_unreferenced
# hand back int32 faces, and the manifold checks then report spurious
# non-manifoldness — which aborted the whole preprocessing run. The fix pins
# int64 across the collapse and, if the (cosmetic) collapse still can't be kept
# manifold + closed, keeps the already-good pre-collapse mesh instead of raising.
# ---------------------------------------------------------------------------

def _closed_tetra():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)
    return v, f


def test_make_manifold_survives_broken_collapse(monkeypatch, caplog):
    """A post-collapse mesh that fails the manifold check must not abort the run.

    Forces the cosmetic-collapse branch (a < area_thresh) and makes that collapse
    return a non-manifold mesh — the Windows-wheel failure mode. make_manifold
    must keep the (already manifold + closed) pre-collapse mesh and warn, not raise.
    """
    from bg3dtools.mesh import clean
    v, f = _closed_tetra()
    monkeypatch.setattr(clean, "doublearea", lambda vv, ff: np.array([1e-30]))
    monkeypatch.setattr(
        clean, "collapse_small_triangles",
        lambda vv, ff, t: np.vstack([np.asarray(ff), np.asarray(ff)[:1]]),  # dup a face
    )
    with caplog.at_level("WARNING"):
        v2, f2 = clean.make_manifold(v.copy(), f.copy())        # must not raise
    assert f2.dtype == np.int64                                 # int64 pinned
    assert not all_boundary_loop(f2)                             # closed
    assert is_edge_manifold(f2)                                 # edge-manifold
    assert np.array_equal(np.sort(f2, 1), np.sort(f, 1))        # reverted to input
    assert "retaining the pre-collapse mesh" in caplog.text


# ---------------------------------------------------------------------------
# as_igl_faces + class-2 (int32-return) guards
#
# The libigl bindings propagate the input face dtype out of remove_unreferenced /
# decimate / upsample / bfs_orient, and the manifold predicates
# (is_edge_manifold / all_boundary_loop) then misbehave on int32. igl_compat now
# pins int64 on both sides of every igl call; as_igl_faces keeps the *repo-wide*
# face dtype canonical for the non-igl consumers. These guard both.
# ---------------------------------------------------------------------------

def test_as_igl_faces_forces_contiguous_int64():
    from bg3dtools.mesh.utils import as_igl_faces
    for arr in (np.array([[0, 1, 2]], np.int32),                        # int32 (Windows default)
                np.asfortranarray(np.array([[0, 1, 2], [1, 2, 3]], np.int64)),  # non-contiguous
                np.zeros((0, 3), np.float64)):                          # float placeholder
        out = as_igl_faces(arr)
        assert out.dtype == np.int64 and out.flags['C_CONTIGUOUS']


def test_submesh_pins_int64_even_if_remove_unreferenced_returns_int32(monkeypatch):
    """submesh must return int64 faces even on a wheel whose remove_unreferenced hands back
    int32 (the Windows failure mode) — this is what protects make_manifold's main-loop
    manifold checks and every other submesh caller (error.py, vertebrae.py, ...)."""
    from bg3dtools.mesh import utils
    v, f = _mesh()
    real = utils.remove_unreferenced
    def int32_return(V, F):
        nv, nf, i, j = real(V, F)
        return nv, np.ascontiguousarray(nf, np.int32), i, j            # emulate the Windows wheel
    monkeypatch.setattr(utils, "remove_unreferenced", int32_return)
    _, nf = utils.submesh(v, f, np.array([True, True, False]), return_indices=False)
    assert nf.dtype == np.int64


def test_edge_neighbors_canonicalizes_faces_before_manifold_predicate(monkeypatch):
    """edge_neighbors must pin int64 before is_edge_manifold so a binding that (wrongly) reports
    int32 faces as non-manifold does not fire the assert. Emulates that behaviour: without
    the int64 chokepoint, passing int32 faces would raise AssertionError."""
    from bg3dtools.mesh import modify
    v, f = _mesh()
    real_iem = modify.is_edge_manifold
    monkeypatch.setattr(modify, "is_edge_manifold",
                        lambda F: real_iem(F) if F.dtype == np.int64 else False)
    edges, neigh = modify.edge_neighbors(f.astype(np.int32))           # must NOT raise
    assert len(edges) == len(neigh)
