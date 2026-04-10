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
    "has_flipped_triangles",
    "mds_flatten",
    "lscm_flatten",
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


def _point_in_triangle_scan(
    verts_2d: np.ndarray,
    faces: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
) -> np.ndarray:
    """Brute-force point-in-triangle test for all pixels against all faces.

    Slower than TrapezoidMapTriFinder but works on degenerate triangulations.
    Returns an array of face indices per pixel (-1 if outside all triangles).
    """
    n_pts = len(px)
    tri_idx = np.full(n_pts, -1, dtype=np.int32)

    for fi in range(faces.shape[0]):
        v0 = verts_2d[faces[fi, 0]]
        v1 = verts_2d[faces[fi, 1]]
        v2 = verts_2d[faces[fi, 2]]

        # Skip degenerate triangles
        area2 = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
        if abs(area2) < 1e-20:
            continue

        # Vectorized barycentric test for all unassigned pixels
        mask = tri_idx < 0
        qx, qy = px[mask], py[mask]

        d00 = v1 - v0
        d01 = v2 - v0
        d02x = qx - v0[0]
        d02y = qy - v0[1]

        inv = 1.0 / area2
        u = (d01[1] * d02x - d01[0] * d02y) * inv
        v = (d00[0] * d02y - d00[1] * d02x) * inv

        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8)
        idx_in_mask = np.where(inside)[0]
        if len(idx_in_mask) > 0:
            orig_idx = np.where(mask)[0][idx_in_mask]
            tri_idx[orig_idx] = fi

    return tri_idx


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

    # Use matplotlib's TrapezoidMapTriFinder for O(log N) point location.
    # Falls back to per-triangle scan if TrapezoidMapTriFinder rejects the
    # triangulation (degenerate triangles from MDS flattening).
    try:
        tri = Triangulation(verts_2d[:, 0], verts_2d[:, 1], faces)
        finder = TrapezoidMapTriFinder(tri)
        tri_idx = finder(px, py)  # (grid_size^2,) int, -1 if outside
    except RuntimeError:
        tri_idx = _point_in_triangle_scan(verts_2d, faces, px, py)

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


def has_flipped_triangles(verts_2d: np.ndarray, faces: np.ndarray) -> bool:
    """Check whether a 2D triangulation has any flipped (overlapping) triangles.

    In a valid planar triangulation from a surface projection, all triangles
    should have consistent winding (all positive or all negative signed area).
    Mixed signs indicate the projection has produced overlapping geometry,
    typically from high surface curvature.

    Parameters
    ----------
    verts_2d : (N, 2) ndarray
        2D vertex positions.
    faces : (F, 3) ndarray
        Triangle vertex indices.

    Returns
    -------
    bool
        True if any triangles have opposite winding from the majority.
    """
    v0 = verts_2d[faces[:, 0]]
    v1 = verts_2d[faces[:, 1]]
    v2 = verts_2d[faces[:, 2]]
    # Signed area (2x) via cross product of edge vectors
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
            (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    n_positive = np.sum(cross > 0)
    n_negative = np.sum(cross < 0)
    return n_positive > 0 and n_negative > 0


def mds_flatten(
    verts: np.ndarray,
    faces: np.ndarray,
    center_idx: int,
    up_direction: np.ndarray,
) -> np.ndarray:
    """Flatten a mesh patch to 2D using classical MDS on geodesic distances.

    Computes all-pairs geodesic distances on the submesh, then applies
    classical (Torgerson) MDS to embed the vertices in 2D while preserving
    geodesic distances as well as possible. Unlike tangent-plane projection,
    MDS unrolls curvature rather than projecting through it, so it won't
    produce overlapping triangles on disk-topology patches.

    Parameters
    ----------
    verts : (N, 3) ndarray
        Submesh vertex coordinates.
    faces : (F, 3) ndarray
        Submesh triangle indices.
    center_idx : int
        Index of the center vertex in the submesh (will be placed at origin).
    up_direction : (3,) ndarray
        Cranial direction in 3D, used to orient the 2D embedding so that
        the most "upward" vertex (in 3D) maps to +y in 2D.

    Returns
    -------
    coords_2d : (N, 2) ndarray
        Flattened 2D coordinates, centered on ``center_idx`` with
        consistent orientation.
    """
    from .metrics import calc_geodesic

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    up_direction = np.asarray(up_direction, dtype=np.float64)
    n = verts.shape[0]

    # 1. All-pairs geodesic distance matrix (heat method avoids segfaults
    #    in igl.exact_geodesic when called ~N times on small submeshes)
    D = calc_geodesic(verts, faces, exact=False)  # (N, N)

    # 2. Symmetrize and zero diagonal
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)

    # 3. Classical MDS: B = -0.5 * H @ D^2 @ H
    D_sq = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    # 4. Eigendecompose, take top 2 eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # eigh returns ascending order; take the two largest
    idx = np.argsort(eigenvalues)[::-1][:2]
    vals = np.maximum(eigenvalues[idx], 0.0)  # clamp numerical noise
    coords_2d = eigenvectors[:, idx] * np.sqrt(vals)[None, :]  # (N, 2)

    # 5. Center on center_idx
    coords_2d -= coords_2d[center_idx]

    # 6. Orient: rotate so the most "upward" vertex (in 3D) maps to +y
    rel_3d = verts - verts[center_idx]
    up_proj = rel_3d @ up_direction
    up_vertex = np.argmax(up_proj)

    target_2d = coords_2d[up_vertex]
    angle = np.arctan2(target_2d[0], target_2d[1])  # angle from +y axis
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    coords_2d = coords_2d @ R

    # 7. Fix handedness: flip x if mean signed area is negative
    v0 = coords_2d[faces[:, 0]]
    v1 = coords_2d[faces[:, 1]]
    v2 = coords_2d[faces[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
            (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    if np.mean(cross) < 0:
        coords_2d[:, 0] = -coords_2d[:, 0]

    return coords_2d


def lscm_flatten(
    verts: np.ndarray,
    faces: np.ndarray,
    center_idx: int,
    up_direction: np.ndarray,
) -> np.ndarray:
    """Flatten a mesh patch to 2D using Least-Squares Conformal Mapping.

    LSCM computes a conformal (angle-preserving) parameterization via a sparse
    linear solve, making it orders of magnitude faster than MDS for large
    submeshes while producing comparable quality on disk-topology patches.

    Parameters
    ----------
    verts : (N, 3) ndarray
        Submesh vertex coordinates.
    faces : (F, 3) ndarray
        Submesh triangle indices.
    center_idx : int
        Index of the center vertex in the submesh (will be placed at origin).
    up_direction : (3,) ndarray
        Cranial direction in 3D, used to orient the 2D embedding so that
        the most "upward" vertex (in 3D) maps to +y in 2D.

    Returns
    -------
    coords_2d : (N, 2) ndarray
        Flattened 2D coordinates, centered on ``center_idx`` with
        consistent orientation.

    Raises
    ------
    ValueError
        If LSCM fails (e.g. degenerate mesh or no boundary).
    """
    import igl

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    up_direction = np.asarray(up_direction, dtype=np.float64)

    # 1. Get boundary loop
    boundary = igl.boundary_loop(faces)
    if len(boundary) < 2:
        raise ValueError("Mesh has no boundary; LSCM requires a boundary loop.")

    # 2. Pin two boundary vertices:
    #    - closest to center (anchors the center region)
    #    - farthest in the "up" direction (gives stable orientation)
    boundary_verts = verts[boundary]
    center_pos = verts[center_idx]

    dists_to_center = np.linalg.norm(boundary_verts - center_pos, axis=1)
    pin_near = boundary[np.argmin(dists_to_center)]

    rel_3d = boundary_verts - center_pos
    up_proj = rel_3d @ up_direction
    pin_up = boundary[np.argmax(up_proj)]

    # If both pins are the same vertex, pick the farthest from center instead
    if pin_near == pin_up:
        pin_up = boundary[np.argmax(dists_to_center)]

    b = np.array([pin_near, pin_up], dtype=np.int32)
    bc = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    # 3. Solve LSCM
    success, uv = igl.lscm(verts, faces, b, bc)
    if not success:
        raise ValueError("LSCM solve failed on this submesh.")

    coords_2d = uv.astype(np.float64)

    # 4. Center on center_idx
    coords_2d -= coords_2d[center_idx]

    # 5. Orient: rotate so the most "upward" vertex (in 3D) maps to +y
    rel_3d_all = verts - verts[center_idx]
    up_proj_all = rel_3d_all @ up_direction
    up_vertex = np.argmax(up_proj_all)

    target_2d = coords_2d[up_vertex]
    angle = np.arctan2(target_2d[0], target_2d[1])  # angle from +y axis
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    coords_2d = coords_2d @ R

    # 6. Fix handedness: flip x if mean signed area is negative
    v0 = coords_2d[faces[:, 0]]
    v1 = coords_2d[faces[:, 1]]
    v2 = coords_2d[faces[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
            (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    if np.mean(cross) < 0:
        coords_2d[:, 0] = -coords_2d[:, 0]

    return coords_2d
