"""
Barycentric coordinate operations for triangle meshes.

This module provides functions for computing barycentric coordinates,
converting them to sparse matrices, and projecting points onto mesh surfaces.
"""

import igl
import numpy as np
from scipy.sparse import coo_matrix

__all__ = [
    "points_to_barycentric",
    "bc2sparse",
    "project_to_bccoord",
    "blend_vert_face",
]


def points_to_barycentric(triangles, points, method=None):
    """
    Find the barycentric coordinates of points relative to triangles.

    Thin wrapper around ``trimesh.triangles.points_to_barycentric`` with
    clip + renormalize for numerical safety.

    Parameters
    ----------
    triangles : (n, 3, 3) float
        Triangle vertices in space.
    points : (n, 3) float
        Point in space associated with a triangle.
    method : str, optional
        Passed to trimesh (``'cross'`` or default Cramer's rule).

    Returns
    -------
    barycentric : (n, 3) float
        Barycentric coordinates of each point, clipped to [0, 1]
        and renormalized to sum to 1.
    """
    import trimesh
    bc = trimesh.triangles.points_to_barycentric(triangles, points, method=method)
    bc = np.clip(bc, 0, 1)
    bc /= np.sum(bc, axis=1, keepdims=True)
    return bc


def bc2sparse(faces: np.ndarray, fidx: np.ndarray, bccoord: np.ndarray, nV=None) -> coo_matrix:
    """
    :param faces: [nF x 3] source mesh simplex (triangulation)
    :param fidx: [nV] indices into source mesh,
    :param bccoord: [nV x 3] barycentric coordinates
    :param nV: scalar value (size of destination points)
    :return bc_map: coo_matrix mapping source vertices to destination points
    """

    if nV is None:
        nV = np.max(faces) + 1
    D = faces.shape[1]

    mask = np.isfinite(fidx) & np.all(np.isfinite(bccoord), axis=1)
    finite_fidx = fidx.copy()
    finite_fidx[~mask] = 0

    nP = fidx.shape[0]
    iidx = np.tile(np.arange(nP)[None, :], [D, 1]).T.flatten()
    try:
        jidx = faces[finite_fidx, :].flatten()
    except Exception as e:
        print('fidx min = %d , fidx max = %d, face size = %d' %(np.min(fidx), np.max(fidx), faces.shape[0]))
        raise e
    bccoord = bccoord.flatten()

    return coo_matrix((bccoord, (iidx, jidx)), [nP, nV])


def project_to_bccoord(verts, faces, points, return_map=False):
    """
    convenience function calling several other tools. Project points onto mesh and
    return the barycentric coordinates of projected points.

    :param verts, faces: triangulated mesh
    :param points: cartesian coordinates

    :return bccoord: barycentric coordinates
    :return fidx: for each point, index of matching simplex
    """
    bad = np.logical_not(np.all(np.isfinite(points), axis=1))
    d2, fidx, projected = igl.point_mesh_squared_distance(points, verts, faces)
    fidx[bad] = 0
    tris = verts[faces[fidx, :], :]
    bc = points_to_barycentric(tris, projected)

    bc[bad] = np.nan
    fidx[bad] = -1

    if return_map:
        return bc, fidx, bc2sparse(faces, fidx, bc)
    else:
        return bc, fidx


def blend_vert_face(v_feat, f_feat, bccoord):
    """
    blend features defined on vertices and faces, weighted according to barycentric coordinates

    :param v_feat: #P x F feature defined on vertices
    :param f_feat: #P x F feature defined on faces
    :param bccoord: #P x D barycentric coordinates
    """
    d = np.sum(bccoord**2, axis=1, keepdims=True)
    p = np.arctan(20 * (d - 0.5)) / np.pi + 0.5  # 20 and -0.5 are a hand-picked by personal intuition
    return p * v_feat + (1 - p) * f_feat
