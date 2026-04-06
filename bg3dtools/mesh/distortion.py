"""
Mesh distortion metrics.

This module provides functions for measuring geometric distortion
between a deformed mesh and its reference configuration.
"""

from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sparse
from bg3dtools.mesh.utils import ordered_edges, per_vertex_normals
from bg3dtools.mesh.laplace import cotangent_weights

__all__ = [
    "edge_distortion",
    "cotangent_smooth_matrix",
    "normal_fold_score",
    "triangle_matrix",
    "face_distortion",
]


def edge_distortion(
    verts: np.ndarray,
    faces: np.ndarray,
    reference_verts: np.ndarray,
    edges: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute edge length distortion between meshes.

    Measures how much each edge has changed length compared to a reference.

    Parameters
    ----------
    verts : (n, 3) ndarray
        Deformed mesh vertex coordinates.
    faces : (m, 3) ndarray
        Triangle face indices.
    reference_verts : (n, 3) ndarray
        Reference mesh vertex coordinates.
    edges : (e, 2) ndarray, optional
        Edge vertex indices. Computed from faces if None.

    Returns
    -------
    distortion : (e,) ndarray
        L2 norm of edge vector difference for each edge.
    """
    if edges is None:
        edges = ordered_edges(faces)

    v0, v1 = edges[:, 0], edges[:, 1]  # indices of vertices on each edge
    reg_edges = verts[v0] - verts[v1]
    ref_edges = reference_verts[v0] - reference_verts[v1]

    edge_diff = np.linalg.norm(reg_edges - ref_edges, axis=-1)

    return edge_diff


def cotangent_smooth_matrix(cot_L: sparse.spmatrix) -> sparse.spmatrix:
    """Build a row-normalized smoothing matrix from a cotangent Laplacian.

    Each row of the result averages a vertex's neighbors using cotangent
    weights.  One application replaces each vertex value with its
    1-ring weighted average; repeated application extends the radius.

    Parameters
    ----------
    cot_L : (nV, nV) sparse matrix
        Cotangent Laplacian from ``cotangent_weights``.

    Returns
    -------
    S : (nV, nV) sparse matrix (CSR)
        Row-stochastic smoothing matrix.
    """
    diag = -cot_L.diagonal()
    eps = 1e-6 * np.mean(diag)
    diag_inv = sparse.diags(1.0 / np.maximum(diag, eps))
    return (sparse.eye(cot_L.shape[0]) + diag_inv @ cot_L).tocsr()


def normal_fold_score(
    verts: np.ndarray,
    faces: np.ndarray,
    reference_verts: np.ndarray,
    n_rings: int = 3,
    cot_L: Optional[sparse.spmatrix] = None,
    ref_normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-vertex fold detection via multi-ring Laplacian of normal similarity.

    Computes cosine similarity between per-vertex normals of the deformed
    and reference meshes, then measures each vertex's deviation from its
    cotangent-weighted multi-ring neighborhood average.  Flags vertices
    at fold boundaries (where normal similarity changes abruptly) while
    ignoring uniform rotations (e.g. an arm rotating 45 degrees).

    Parameters
    ----------
    verts : (nV, 3) array
        Deformed mesh vertex positions.
    faces : (nF, 3) array
        Triangle face indices.
    reference_verts : (nV, 3) array
        Reference mesh vertex positions (used to build ``cot_L`` if not
        provided).
    n_rings : int
        Smoothing radius for the neighborhood average.  Default 3.
    cot_L : sparse matrix, optional
        Precomputed cotangent Laplacian (from ``cotangent_weights``).
        Built from *reference_verts* if None.
    ref_normals : (nV, 3) array, optional
        Precomputed per-vertex normals of the reference mesh.
        Computed from *reference_verts* if None.

    Returns
    -------
    fold_score : (nV,) array
        Non-negative per-vertex score.  Zero for uniformly-rotated
        regions; large at fold boundaries.
    """
    if ref_normals is None:
        ref_normals = per_vertex_normals(reference_verts, faces)
    if cot_L is None:
        # igl.cotmatrix is broken in igl 2.5.1 (returns all zeros)
        cot_L = -cotangent_weights(reference_verts, faces)

    S = cotangent_smooth_matrix(cot_L)
    n_def = per_vertex_normals(verts, faces)
    cos_sim = np.sum(n_def * ref_normals, axis=1)

    smooth_cos = cos_sim.copy()
    for _ in range(n_rings):
        smooth_cos = S @ smooth_cos

    return np.abs(cos_sim - smooth_cos)


def triangle_matrix(
    verts: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute local coordinate frame matrices for each triangle.

    Parameters
    ----------
    verts : (n, 3) ndarray
        Vertex coordinates.
    faces : (m, 3) ndarray
        Triangle face indices.

    Returns
    -------
    T : (m, 3, 3) ndarray
        Per-face matrices with rows [e0, e1, normal] where e0, e1
        are edge vectors and normal is the unit face normal.
    """
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    e3 = np.cross(e0, e1)
    e3 /= np.linalg.norm(e3, axis=1, keepdims=True)

    return np.stack([e0, e1, e3], axis=1)


def face_distortion(
    verts: np.ndarray,
    faces: np.ndarray,
    reference_verts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-face deformation via SVD of local transforms.

    Analyzes how each triangle has been deformed relative to reference
    by computing the SVD of the deformation gradient.

    Parameters
    ----------
    verts : (n, 3) ndarray
        Deformed mesh vertex coordinates.
    faces : (m, 3) ndarray
        Triangle face indices.
    reference_verts : (n, 3) ndarray
        Reference mesh vertex coordinates.

    Returns
    -------
    U : (m, 3, 3) ndarray
        Left singular vectors per face.
    S : (m, 3) ndarray
        Singular values per face (stretch factors).
    Vt : (m, 3, 3) ndarray
        Right singular vectors per face.
    """
    ref_faces = triangle_matrix(reference_verts, faces)
    reg_faces = triangle_matrix(verts, faces)

    face_diff = np.linalg.solve(ref_faces, reg_faces)
    s, v, d = np.linalg.svd(face_diff)

    return s, v, d

