"""
Open3D mesh utilities.

This module provides helper functions for manipulating Open3D triangle meshes.
"""

import open3d as o3d
import numpy as np

from bg3dtools.mesh.preprocess import MeshCleaner
from bg3dtools.mesh import submesh

__all__ = [
    "o3d_submesh",
]


def o3d_submesh(
    mesh: o3d.geometry.TriangleMesh,
    f_idx: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """
    Extract a submesh from an Open3D TriangleMesh.

    Preserves vertex colors, normals, UVs, and material IDs where available.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Input mesh.
    f_idx : (k,) ndarray
        Indices of faces to keep.

    Returns
    -------
    new_mesh : o3d.geometry.TriangleMesh
        Submesh containing only the specified faces.
    """
    new_mesh = o3d.geometry.TriangleMesh(mesh)

    # faces and vertices
    old_verts = np.asarray(mesh.vertices)
    old_faces = np.asarray(mesh.triangles)
    new_verts, new_faces, f_idx, v_idx = submesh(old_verts, old_faces, f_idx)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
    new_mesh.vertices = o3d.utility.Vector3dVector(new_verts)

    # face normals
    if mesh.has_triangle_normals():
        new_mesh.compute_triangle_normals()

    # vertex normals
    if mesh.has_vertex_normals():
        new_mesh.compute_vertex_normals()

    # vertex colors
    if mesh.has_vertex_colors():
        new_vertex_colors = np.asarray(new_mesh.vertex_colors)[v_idx]
        new_mesh.vertex_colors = o3d.utility.Vector3iVector(new_vertex_colors)

    # face materials
    if mesh.has_triangle_material_ids():
        new_face_materials = np.asarray(new_mesh.triangle_material_ids)[f_idx]
        new_mesh.triangle_material_ids = o3d.utility.IntVector(new_face_materials)

    # triangle_uvs
    if mesh.has_triangle_uvs():
        old_uvs = np.asarray(mesh.triangle_uvs)
        new_U = old_uvs[:, 0].reshape([-1, 3])[f_idx]
        new_V = old_uvs[:, 1].reshape([-1, 3])[f_idx]
        new_triangle_uvs = np.column_stack([new_U.flatten(), new_V.flatten()])
        new_mesh.triangle_uvs = o3d.utility.Vector2dVector(new_triangle_uvs)

    return new_mesh




