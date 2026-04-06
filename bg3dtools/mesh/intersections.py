"""
Mesh intersection and boolean operations.

This module provides functions for slicing meshes with planes
and performing boolean operations using trimesh.
"""

from typing import Tuple
import numpy as np
from bg3dtools.pointclouds.fitting import project_to_plane

__all__ = [
    "boolean_slice",
]


def boolean_slice(
    verts: np.ndarray,
    faces: np.ndarray,
    plane: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice a mesh with a plane, keeping the negative half-space.

    Parameters
    ----------
    verts : (N, 3) ndarray
        Mesh vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face indices.
    plane : (4,) ndarray
        Plane coefficients [a, b, c, d] where ax + by + cz + d = 0.

    Returns
    -------
    sliced_verts : (N', 3) ndarray
        Vertices of sliced mesh.
    sliced_faces : (M', 3) ndarray
        Faces of sliced mesh.
    """
    import trimesh

    # Create a trimesh object from the input vertices and faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # project vertices onto the plane
    proj = project_to_plane(plane, verts)[0]
    center = np.mean(proj, axis=0)

    # scale cube mesh bounding box
    s = np.max(np.max(verts, axis=0) - np.min(verts, axis=0))
    # align cube with x-y plane
    A = np.eye(4)
    A[:3, 3] = [0, 0, s]
    # align cube z-axis with plane normal
    z_axis = plane[:3]
    z_axis /= np.linalg.norm(z_axis)
    temp = np.array([0, 1, 0]) if np.allclose(z_axis, [0, 0, 1]) else np.array([0, 0, 1])
    x_axis = np.cross(z_axis, temp)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    B = np.eye(4)
    B[:3, :3] = np.array([x_axis, y_axis, z_axis]).T
    B[:3, 3] = center
    tform = B @ A

    cube = trimesh.creation.box(extents=[2*s, 2*s, 2*s], transform=tform)

    # slice the mesh with the cube
    sliced = trimesh.boolean.difference([mesh, cube], engine='manifold')
    return sliced.vertices, sliced.faces
