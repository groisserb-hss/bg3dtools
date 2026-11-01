"""bg3dtools.mesh.faces_absent_from: the faces a cleaning took out of a mesh, recovered by centroid matching."""
import numpy as np

from bg3dtools.mesh import faces_absent_from, submesh

V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], float)
F = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])


def test_recovers_the_dropped_faces_through_submesh():
    keep = np.array([True, True, False, True])
    kept_v, kept_f = submesh(V, F, keep)[:2]              # re-sorted faces, re-indexed vertices
    assert faces_absent_from(V, F, kept_v, kept_f, tol=1e-6).tolist() == [False, False, True, False]


def test_identical_and_empty_meshes():
    assert not faces_absent_from(V, F, V, F, tol=1e-6).any()
    assert faces_absent_from(V, F, V[:0], F[:0], tol=1e-6).all()
    assert faces_absent_from(V, F[:0], V, F, tol=1e-6).shape == (0,)


def test_storage_jitter_below_tol_still_matches():
    jittered = V.astype(np.float32).astype(np.float64) + 1e-7     # a float32 PLY round trip plus noise
    assert not faces_absent_from(V, F, jittered, F, tol=1e-5).any()
    assert faces_absent_from(V, F, V + 0.5, F, tol=1e-5).all()    # a shifted copy is a different mesh
