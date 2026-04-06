"""
Mesh metric computations.

This module provides functions for computing geodesic distances on meshes.
"""

from typing import Optional
import igl
import numpy as np

__all__ = [
    "calc_geodesic",
]


def calc_geodesic(
    verts: np.ndarray,
    faces: np.ndarray,
    v_idx: Optional[np.ndarray] = None,
    exact: bool = True,
    t: float = 1.0
) -> np.ndarray:
    """
    Compute geodesic distances from source vertices.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle face indices.
    v_idx : (k,) ndarray, optional
        Source vertex indices. All vertices if None.
    exact : bool, optional
        If True, use exact geodesic. If False, use heat method. Default is True.
    t : float, optional
        Time parameter for heat method. Default is 1.0.

    Returns
    -------
    D : (nV, k) ndarray
        Distance matrix where D[i, j] is the geodesic distance from
        vertex i to source vertex v_idx[j].
    """
    nV = verts.shape[0]

    if v_idx is None:
        v_idx = np.arange(nV)
    nI = len(v_idx)

    D = np.zeros((nV, nI))
    dst = np.arange(nV, dtype=faces.dtype)
    for ii, vv in enumerate(v_idx):
        src = np.array([vv], dtype=faces.dtype)
        if exact:
            D[:, ii] = igl.exact_geodesic(verts, faces, vs=src, vt=dst)
        else:
            D[:, ii] = igl.heat_geodesic(verts, faces, t, src)

    return D
