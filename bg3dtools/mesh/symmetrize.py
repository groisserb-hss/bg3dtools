"""
Mesh symmetrization utilities.

This module provides functions for making meshes symmetric by
averaging with their mirror reflection.
"""

import numpy as np
from bg3dtools.mesh.registration import nonrigid_ICP
from bg3dtools.mesh.utils import surface_sample, per_vertex_normals
from igl import point_mesh_squared_distance

__all__ = [
    "symmetrize",
]


def symmetrize(
    orig_v: np.ndarray,
    faces: np.ndarray,
    axis: int
) -> np.ndarray:
    """
    Symmetrize a mesh by averaging with its mirror reflection.

    Mirrors the mesh along the specified axis, registers the original
    to the mirrored version using non-rigid ICP, then averages the
    positions to create a symmetric mesh.

    Parameters
    ----------
    orig_v : (N, 3) ndarray
        Original vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face indices.
    axis : int
        Axis to mirror along (0=X, 1=Y, 2=Z).

    Returns
    -------
    symm_v : (N, 3) ndarray
        Symmetrized vertex coordinates.
    """
    verts = orig_v.copy()
    verts[:, axis] -= np.mean(verts[:, axis])

    flip = np.array([[-1 if i == axis else 1 for i in range(3)]])
    # Mirror the vertices along the x-axis
    new_v = verts * flip
    new_f = faces[:, ::-1]
    # Sample the surface of the flipped mesh
    N = 4 * len(verts)
    point_map = surface_sample(new_v, faces, N=N)[0]
    sampled = point_map @ new_v
    new_normals = point_map @ per_vertex_normals(new_v, new_f)

    # register original mesh to flipped surface
    d = point_mesh_squared_distance(verts, sampled, new_f)[0]
    r = 2 * np.percentile(np.sqrt(d), 90)
    fitted_v = nonrigid_ICP(sampled, faces, verts, pt_normals=new_normals, rad=r)[0]

    symm_v = (fitted_v + verts) / 2

    return symm_v

