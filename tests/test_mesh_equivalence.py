"""
Equivalence tests: verify that library-based replacements produce the same
results as the custom implementations they replace.

Each test runs *both* the old code (copied inline) and the proposed replacement,
then asserts numerical equivalence within tolerance.

Note: igl.cotmatrix and igl.massmatrix are broken in igl 2.5.1 (return all
zeros), so cotangent_weights and fem_mass_matrix are kept as custom code.
"""

import numpy as np
import pytest
import igl


# ---------------------------------------------------------------------------
# Helpers — build test meshes
# ---------------------------------------------------------------------------

def _icosahedron():
    from bg3dtools.mesh.generate import generate_icosahedron
    return generate_icosahedron()


def _subdivided_icosahedron():
    """Return a denser mesh by loop-subdividing the icosahedron once."""
    v, f = _icosahedron()
    v, f = igl.upsample(v, f.astype(np.int32), 1)
    return v.astype(np.float64), f


def _unit_tetrahedron():
    """Tet with known volume = 1/6 (signed)."""
    v = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    f = np.array([
        [0, 2, 1],  # base (outward normal = -z)
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ], dtype=np.int32)
    return v, f


# ===========================================================================
# 1. points_to_barycentric  vs  trimesh
# ===========================================================================

def _old_diagonal_dot(a, b):
    a = np.asanyarray(a)
    return np.dot(a * b, [1.0] * a.shape[1])


def _old_points_to_barycentric(triangles, points, method=None):
    """Copy of the custom implementation from utils.py (Cramer's rule path)."""
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)

    edge_vectors = triangles[:, 1:] - triangles[:, :1]
    w = points - triangles[:, 0].reshape((-1, 3))

    dot00 = _old_diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
    dot01 = _old_diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
    dot02 = _old_diagonal_dot(edge_vectors[:, 0], w)
    dot11 = _old_diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
    dot12 = _old_diagonal_dot(edge_vectors[:, 1], w)

    denominator = np.maximum(0.001, (dot00 * dot11 - dot01 * dot01))
    inverse_denominator = 1.0 / denominator

    barycentric = np.zeros((len(triangles), 3), dtype=np.float64)
    barycentric[:, 2] = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
    barycentric[:, 1] = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
    barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]

    barycentric = np.clip(barycentric, 0, 1)
    barycentric /= np.sum(barycentric, axis=1, keepdims=True)

    return barycentric


def test_points_to_barycentric_basic():
    """Points at triangle vertices and centroid."""
    from bg3dtools.mesh.barycentric import points_to_barycentric

    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)

    # Test vertex points
    for i in range(3):
        p = tri[0, i:i + 1]
        t = np.tile(tri, (1, 1, 1))
        old = _old_points_to_barycentric(t, p)
        new = points_to_barycentric(t, p)
        np.testing.assert_allclose(old, new, atol=1e-10)

    # Test centroid
    centroid = tri[0].mean(axis=0, keepdims=True)
    old = _old_points_to_barycentric(tri, centroid)
    new = points_to_barycentric(tri, centroid)
    np.testing.assert_allclose(old, new, atol=1e-10)


def test_points_to_barycentric_random():
    """Random points inside random triangles."""
    from bg3dtools.mesh.barycentric import points_to_barycentric

    rng = np.random.default_rng(42)
    n = 200
    triangles = rng.standard_normal((n, 3, 3))
    # Generate random barycentric coords and reconstruct points
    bc_true = rng.uniform(0.01, 1, (n, 3))
    bc_true /= bc_true.sum(axis=1, keepdims=True)
    points = np.einsum('nij,ni->nj', triangles, bc_true)

    old = _old_points_to_barycentric(triangles, points)
    new = points_to_barycentric(triangles, points)
    np.testing.assert_allclose(old, new, atol=1e-8)


def test_points_to_barycentric_clipping():
    """Verify clip+renormalize on points outside triangle."""
    from bg3dtools.mesh.barycentric import points_to_barycentric

    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    # Point outside triangle
    p = np.array([[-0.1, -0.1, 0]], dtype=np.float64)
    bc = points_to_barycentric(tri, p)
    # All coords should be >= 0 after clipping
    assert np.all(bc >= 0)
    # Should sum to 1 after renormalization
    np.testing.assert_allclose(bc.sum(axis=1), 1.0, atol=1e-12)


# ===========================================================================
# 2. mesh_volume  old vs new formula
# ===========================================================================

def _old_mesh_volume(V, F):
    """Copy of the custom implementation from utils.py."""
    F = F.astype(np.int64)
    center = igl.barycenter(V, F)
    e1 = V[F[:, 1], :] - V[F[:, 0], :]
    e2 = V[F[:, 2], :] - V[F[:, 0], :]
    FNdA = np.cross(e1, e2) / 2

    return np.mean([center[:, 0].T @ FNdA[:, 0],
                    center[:, 1].T @ FNdA[:, 1],
                    center[:, 2].T @ FNdA[:, 2]])


def test_mesh_volume_tetrahedron():
    from bg3dtools.mesh.utils import mesh_volume

    v, f = _unit_tetrahedron()
    old = _old_mesh_volume(v, f)
    new = mesh_volume(v, f)
    np.testing.assert_allclose(old, new, atol=1e-12,
                               err_msg="mesh_volume must match on tetrahedron")
    # Known value: volume of unit tet = 1/6
    np.testing.assert_allclose(abs(new), 1 / 6, atol=1e-12)


def test_mesh_volume_icosahedron():
    from bg3dtools.mesh.utils import mesh_volume

    v, f = _icosahedron()
    old = _old_mesh_volume(v, f)
    new = mesh_volume(v, f)
    np.testing.assert_allclose(old, new, atol=1e-10,
                               err_msg="mesh_volume must match on icosahedron")


def test_mesh_volume_signed():
    """Flipping faces should negate volume."""
    from bg3dtools.mesh.utils import mesh_volume

    v, f = _icosahedron()
    vol_pos = mesh_volume(v, f)
    vol_neg = mesh_volume(v, f[:, ::-1])
    np.testing.assert_allclose(vol_pos, -vol_neg, atol=1e-12)


# ===========================================================================
# 3. get_genus  old vs trimesh
# ===========================================================================

def _old_get_genus(verts, faces):
    """Copy of old implementation (Euler formula without manifold assertions)."""
    nV = verts.shape[0]
    nE = len(igl.edges(faces))
    nF = faces.shape[0]
    return 1 - (nV - nE + nF) / 2


def test_get_genus_icosahedron():
    """Icosahedron is a closed genus-0 surface."""
    from bg3dtools.mesh.utils import get_genus

    v, f = _icosahedron()
    old = _old_get_genus(v, f)
    new = get_genus(v, f)
    assert old == new == 0, f"Genus mismatch: old={old}, new={new}"


def test_get_genus_subdivided():
    """Subdivided icosahedron is still genus-0."""
    from bg3dtools.mesh.utils import get_genus

    v, f = _subdivided_icosahedron()
    old = _old_get_genus(v, f)
    new = get_genus(v, f)
    assert old == new == 0, f"Genus mismatch: old={old}, new={new}"
