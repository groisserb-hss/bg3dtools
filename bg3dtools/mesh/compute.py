"""
Mesh computational geometry operations.

This module provides functions for solving PDEs on meshes and
computing geometric transformations.
"""

import logging
from typing import Optional, Tuple
import igl
import numpy as np
from bg3dtools.mesh.utils import per_face_normals
from bg3dtools.mesh.barycentric import project_to_bccoord
from scipy.spatial import KDTree

__all__ = [
    "heat_equation",
    "move_points",
    "face_tforms",
]


def heat_equation(
    verts: np.ndarray,
    faces: np.ndarray,
    heat: np.ndarray,
    dirichlet: Optional[np.ndarray] = None,
    neumann: Optional[np.ndarray] = None,
    step: float = 0.1,
    thresh: float = 0.0001,
    max_iter: int = 5000000
) -> np.ndarray:
    """
    Solve the heat equation on a mesh using explicit time stepping.

    Iteratively diffuses heat values across the mesh surface using
    the intrinsic Delaunay cotangent Laplacian.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle face indices.
    heat : (nV,) ndarray
        Initial heat values at vertices. Modified in-place.
    dirichlet : (nV,) ndarray, optional
        Dirichlet boundary conditions. Finite values are fixed.
    neumann : (nV,) ndarray, optional
        Neumann boundary conditions (flux). Finite values add gradient.
    step : float, optional
        Time step scaling factor. Default is 0.1.
    thresh : float, optional
        Convergence threshold. Default is 0.0001.
    max_iter : int, optional
        Maximum iterations. Default is 5000000.

    Returns
    -------
    heat : (nV,) ndarray
        Converged heat values.
    """
    log = logging.getLogger('heat_equation')
    nV = verts.shape[0]

    A = igl.intrinsic_delaunay_cotmatrix(verts, faces)[0]
    a = np.abs(A.diagonal())
    scale = np.percentile(a, 97)
    A *= (step / scale)

    d_mask = np.zeros(nV, dtype=bool) if dirichlet is None else np.isfinite(dirichlet)
    d_values = None if dirichlet is None else dirichlet[d_mask]

    n_mask = np.zeros(nV, dtype=bool) if neumann is None else np.isfinite(neumann)
    n_values = None if neumann is None else neumann[n_mask]

    delta = np.zeros(nV)
    count, stable = 0, 0
    while stable < 3 and count < max_iter:
        assert np.all(np.isfinite(delta))

        # enforce neumann
        if neumann is not None:
            delta[n_mask] = n_values * step

        heat += delta

        # enforce dirichlet boundary conditions
        if dirichlet is not None:
            heat[d_mask] = d_values

        delta = A @ heat

        # test for convergence
        delta[d_mask | n_mask] = 0
        diff = np.max(np.abs(delta / heat))
        stable = stable + 1 if diff < thresh else 0

        if count % 100000 == 0:
            log.debug('step %d  diff = %.4f (%d/3)' % (count, diff, stable))
        count += 1

    if count == max_iter:
        log.warning('max iterations reached')

    return heat


def move_points(
    verts: np.ndarray,
    verts_: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    pt_normals: Optional[np.ndarray] = None,
    fidx: Optional[np.ndarray] = None,
    w: float = 0.2,
    N: int = 7,
    keep_frac: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Move off-surface points following mesh deformation.

    Uses per-face affine transforms with outlier rejection to propagate
    deformation from mesh vertices to arbitrary points.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Original mesh vertices.
    verts_ : (nV, 3) ndarray
        Deformed mesh vertices.
    faces : (nF, 3) ndarray
        Triangle face indices.
    points : (nP, 3) ndarray
        Points to move.
    pt_normals : (nP, 3) ndarray, optional
        Point normals for directional matching (6D KD-tree search).
    fidx : (nP, N) ndarray, optional
        Pre-computed face indices. Computed if None.
    w : float, optional
        Weight for normal direction in matching. Default is 0.2.
    N : int, optional
        Number of nearest faces to consider. Default is 7.
    keep_frac : float, optional
        Fraction of candidates to keep after outlier rejection. Default is 0.6.

    Returns
    -------
    moved : (nP, 3) ndarray
        Moved point positions.
    fidx : (nP, N) ndarray
        Face indices used for each point.
    """
    log = logging.getLogger('move_points')
    nF = faces.shape[0]
    nP = points.shape[0]

    # Compute face transforms
    T_orig = face_tforms(verts, faces)  # (nF, 4, 4)
    T_defo = face_tforms(verts_, faces)  # (nF, 4, 4)
    T = T_defo @ np.linalg.inv(T_orig)  # (nF, 4, 4)

    if fidx is None:
        # Build KDTree for N-nearest face search
        face_centers = verts[faces].mean(axis=1)  # (nF, 3)
        if pt_normals is None:
            points_plus = points
            face_plus = face_centers
        else:
            face_normals = per_face_normals(verts, faces)
            face_plus = np.hstack((face_centers, w * face_normals))
            points_plus = np.hstack((points, w * pt_normals))

        kdtree = KDTree(face_plus)
        _, fidx = kdtree.query(points_plus, k=N)  # (nP, N)
    else:
        assert fidx.shape == (nP, N)

    # Gather candidate transforms
    T_nearest = T[fidx]  # (nP, N, 4, 4)
    points_h = np.hstack((points, np.ones((nP, 1))))[:, None, :]  # (nP, 1, 4)
    moved_all = np.einsum('pnij,pmj->pni', T_nearest, points_h)[:, :, :3]  # (nP, N, 3)

    # Filter by L2 distance to median or mean
    center = np.median(moved_all, axis=1, keepdims=True)  # (nP, 1, 3)
    dists = np.linalg.norm(moved_all - center, axis=2)  # (nP, N)

    k = max(1, int(np.floor(keep_frac * N)))
    # Get top-k indices (smallest distances)
    topk_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]  # (nP, k)

    # Gather filtered moved points
    rows = np.arange(nP)[:, None]
    moved_topk = moved_all[rows, topk_idx]  # (nP, k, 3)
    moved = moved_topk.mean(axis=1)  # (nP, 3)

    return moved, fidx


def face_tforms(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-face local coordinate frames as transformation matrices.

    Each face's coordinate frame has:
    - X-axis along edge v0->v1
    - Y-axis in the face plane, orthogonal to X
    - Z-axis as face normal
    - Origin at v0

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Mesh vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle face indices.

    Returns
    -------
    T : (nF, 4, 4) ndarray
        Homogeneous transformation matrices for each face.
    """
    nF = faces.shape[0]

    v0 = verts[faces[:, 0], :]
    v1 = verts[faces[:, 1], :]
    v2 = verts[faces[:, 2], :]

    # compute transformation matrices for each face
    e1 = v1 - v0
    e2 = v2 - v0
    # normalize vectors
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = e2 - np.sum(e2 * e1, axis=1, keepdims=True) * e1
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    e3 = np.cross(e1, e2)

    # compute transformation matrices
    T = np.tile(np.eye(4), (nF, 1, 1))
    T[:, 0:3, 0] = e1
    T[:, 0:3, 1] = e2
    T[:, 0:3, 2] = e3
    T[:, 0:3, 3] = v0

    return T
