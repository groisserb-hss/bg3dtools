"""
Tangent plane projection and 2D rasterization for mesh patches.

This module provides functions for flattening a local mesh patch onto a 2D
plane and rasterizing it onto a regular pixel grid with per-pixel face/barycentric
mapping for back-projection to 3D.
"""

import numpy as np
from typing import Optional, Tuple

__all__ = [
    "tangent_plane_project",
    "rasterize_mesh_2d",
]


def tangent_plane_project(
    verts: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """
    Project 3D vertices onto a 2D tangent plane.

    Builds an oriented orthonormal frame at ``center`` with the given
    surface ``normal`` and an approximate ``up`` direction (cranial).
    The ``up`` vector is orthogonalized against the normal to define the
    y-axis of the 2D coordinate system.

    Parameters
    ----------
    verts : (N, 3) ndarray
        Vertex coordinates to project.
    center : (3,) ndarray
        Origin of the tangent plane (point on the surface).
    normal : (3,) ndarray
        Surface normal at ``center`` (outward-pointing).
    up : (3,) ndarray
        Approximate "up" direction (e.g. toward C7). Will be
        orthogonalized against ``normal``.

    Returns
    -------
    xy : (N, 2) ndarray
        Projected 2D coordinates. ``xy[:, 0]`` is the right-axis,
        ``xy[:, 1]`` is the up-axis.

    Examples
    --------
    >>> xy = tangent_plane_project(patch_verts, center, normal, up_dir)
    >>> xy.shape
    (500, 2)
    """
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    normal = normal / np.linalg.norm(normal)

    # Orthogonalize up against normal
    y_axis = up - np.dot(up, normal) * normal
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-12:
        raise ValueError(
            "up direction is nearly parallel to surface normal; "
            "cannot define tangent plane orientation."
        )
    y_axis = y_axis / y_norm

    # Right-hand frame: x = y × normal
    x_axis = np.cross(y_axis, normal)

    # Project centered vertices
    verts_c = np.asarray(verts, dtype=np.float64) - center
    xy = verts_c @ np.column_stack([x_axis, y_axis])  # (N, 2)
    return xy


def rasterize_mesh_2d(
    verts_2d: np.ndarray,
    faces: np.ndarray,
    grid_size: int,
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize a 2D triangle mesh onto a regular pixel grid.

    For each pixel center, determines which triangle (if any) contains it
    and computes the barycentric coordinates within that triangle. This
    provides a per-pixel mapping back to the 3D mesh surface.

    Parameters
    ----------
    verts_2d : (N, 2) ndarray
        Flattened 2D vertex positions (e.g. from :func:`tangent_plane_project`).
    faces : (F, 3) ndarray
        Triangle vertex indices.
    grid_size : int
        Output resolution (``grid_size × grid_size`` pixels).
    radius : float, optional
        Half-width of the square grid in mesh units. The grid covers
        ``[-radius, radius]`` in both axes. If None, fits to the bounding
        box of ``verts_2d`` with 5% padding.

    Returns
    -------
    face_map : (grid_size, grid_size) ndarray, int32
        Face index per pixel (−1 where no triangle covers the pixel).
    bary_map : (grid_size, grid_size, 3) ndarray, float64
        Barycentric coordinates per pixel (zeros where ``face_map == -1``).

    Examples
    --------
    >>> face_map, bary_map = rasterize_mesh_2d(xy, sub_faces, 256, radius=0.08)
    >>> valid = face_map >= 0
    >>> valid.sum()  # number of covered pixels
    42000
    """
    from matplotlib.tri import Triangulation, TrapezoidMapTriFinder

    verts_2d = np.asarray(verts_2d, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    if radius is None:
        extent = np.max(np.abs(verts_2d)) * 1.05
        radius = max(extent, 1e-12)

    # Build pixel grid centers: [-radius, radius] mapped to [0, grid_size-1]
    half_pixel = radius / grid_size
    lin = np.linspace(-radius + half_pixel, radius - half_pixel, grid_size)
    gx, gy = np.meshgrid(lin, lin[::-1])  # row 0 = top = +y
    px = gx.ravel()
    py = gy.ravel()

    # Use matplotlib's TrapezoidMapTriFinder for O(log N) point location
    tri = Triangulation(verts_2d[:, 0], verts_2d[:, 1], faces)
    finder = TrapezoidMapTriFinder(tri)
    tri_idx = finder(px, py)  # (grid_size^2,) int, -1 if outside

    face_map = tri_idx.reshape(grid_size, grid_size).astype(np.int32)
    bary_map = np.zeros((grid_size, grid_size, 3), dtype=np.float64)

    # Compute barycentric coords for valid pixels
    valid_mask = tri_idx >= 0
    if valid_mask.any():
        valid_tri = tri_idx[valid_mask]
        valid_px = px[valid_mask]
        valid_py = py[valid_mask]

        # Triangle vertex positions for each valid pixel
        v0 = verts_2d[faces[valid_tri, 0]]  # (K, 2)
        v1 = verts_2d[faces[valid_tri, 1]]
        v2 = verts_2d[faces[valid_tri, 2]]

        # Barycentric via inverse area method
        d00 = v1 - v0
        d01 = v2 - v0
        d02 = np.column_stack([valid_px, valid_py]) - v0

        dot00 = np.einsum("ij,ij->i", d00, d00)
        dot01 = np.einsum("ij,ij->i", d00, d01)
        dot02 = np.einsum("ij,ij->i", d00, d02)
        dot11 = np.einsum("ij,ij->i", d01, d01)
        dot12 = np.einsum("ij,ij->i", d01, d02)

        inv_denom = 1.0 / np.maximum(dot00 * dot11 - dot01 * dot01, 1e-30)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        bary_flat = np.zeros((grid_size * grid_size, 3), dtype=np.float64)
        bary_flat[valid_mask, 0] = 1.0 - u - v
        bary_flat[valid_mask, 1] = u
        bary_flat[valid_mask, 2] = v
        bary_map = bary_flat.reshape(grid_size, grid_size, 3)

    return face_map, bary_map
