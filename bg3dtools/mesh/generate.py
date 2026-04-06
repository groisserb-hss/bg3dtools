"""
Mesh generation utilities.

This module provides functions for generating basic geometric primitives
as triangle meshes, including cylinders, cubes, planes, and icosahedra.
"""

from typing import Optional, Tuple, Union
import igl
import numpy as np

__all__ = [
    "build_cylinder_capped",
    "build_cube",
    "build_plane",
    "generate_icosahedron",
    "build_camera_frustum",
    "pointcloud_to_splatted_mesh",
]


def build_cylinder_capped(
    nR: int,
    nC: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a capped cylinder mesh.

    Creates a cylinder with closed caps at both ends.

    Parameters
    ----------
    nR : int
        Number of rows (height subdivisions).
    nC : int
        Number of columns (circumference subdivisions). Must be >= 3.

    Returns
    -------
    vertices : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle face indices.

    Raises
    ------
    ValueError
        If nR < 0 or nC < 3.
    """
    if nR < 0 or nC < 3:
        raise ValueError('nR, nC must be positive')
    v, f = igl.cylinder(nR, nC)
    nV = len(v)

    # add vertex in the center of top and bottom caps
    end_pts = np.array([[0, 0, 0], [0, 0, 1]])
    v = np.row_stack([v, end_pts])

    bottom_cap = np.column_stack([np.arange(0, nR),
                                  np.arange(1, nR+1) % nR,
                                  nV * np.ones(nR)]).astype(np.int64)
    top_cap = np.column_stack([bottom_cap[:, :2] + nV - nR,
                               (nV+1) * np.ones(nR)]).astype(np.int64)
    f = np.row_stack([f, bottom_cap[:, [0, 2, 1]], top_cap])
    return v, f


def build_cube(
    x: float = 1,
    y: float = 1,
    z: float = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a centered cube mesh.

    Parameters
    ----------
    x : float, optional
        Width (x dimension). Default is 1.
    y : float, optional
        Depth (y dimension). Default is 1.
    z : float, optional
        Height (z dimension). Default is 1.

    Returns
    -------
    vertices : (8, 3) ndarray
        Vertex coordinates, centered at origin.
    faces : (12, 3) ndarray
        Triangle face indices.
    """
    verts = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
                      [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]], dtype=np.float64)
    faces = np.array([[1, 0, 2], [3, 2, 0],
                      [1, 5, 0], [4, 0, 5],
                      [2, 6, 1], [5, 1, 6],
                      [3, 7, 2], [6, 2, 7],
                      [0, 4, 3], [7, 3, 4],
                      [5, 6, 4], [7, 4, 6]])

    verts -= np.mean(verts, axis=0)
    return verts, faces


def build_plane(
    R: int,
    C: int,
    return_vertices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a triangulated planar mesh grid.

    Parameters
    ----------
    R : int
        Number of rows in the grid.
    C : int
        Number of columns in the grid.
    return_vertices : bool, optional
        If True, return both faces and vertices. Default is False.

    Returns
    -------
    faces : (nF, 3) ndarray
        Triangle face indices. Returned always.
    vertices : (R*C, 3) ndarray
        Vertex coordinates on [-0.5, 0.5] x [-0.5, 0.5] plane.
        Only returned if return_vertices is True.
    """
    # Generate vertex indices for the grid
    vertex_indices = np.arange(R * C).reshape(R, C)

    # Generate indices for the first set of triangles (top-left, bottom-left, bottom-right)
    tl = vertex_indices[:-1, :-1].reshape(-1, 1)
    bl = vertex_indices[1:, :-1].reshape(-1, 1)
    br = vertex_indices[1:, 1:].reshape(-1, 1)
    triangles1 = np.hstack([tl, bl, br])

    # Generate indices for the second set of triangles (top-left, bottom-right, top-right)
    tr = vertex_indices[:-1, 1:].reshape(-1, 1)
    triangles2 = np.hstack([tl, br, tr])

    # Combine both sets of triangles
    faces = np.vstack([triangles1, triangles2])

    if not return_vertices:
        return faces

    # Generate vertex positions for the grid
    x = np.linspace(-0.5, 0.5, C)
    y = np.linspace(-0.5, 0.5, R)
    xv, yv = np.meshgrid(x, y)
    vertices = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(R * C)])

    return faces, vertices


def generate_icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a unit icosahedron centered at the origin.

    Creates a regular icosahedron with vertices on the unit sphere.

    Returns
    -------
    vertices : (12, 3) ndarray
        Vertex coordinates on unit sphere.
    faces : (20, 3) ndarray
        Triangle face indices.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Normalize to unit length
    r = np.sqrt(1 + phi ** 2)

    # Define the 12 vertices of the icosahedron
    vertices = np.array([
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1]
    ]) / r  # Normalize to have unit length

    # Define the 20 faces (each face is a triangle defined by three vertex indices)
    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ])

    return vertices, faces




def build_camera_frustum(
    hfov: float = 60.0,
    vfov: float = 45.0,
    scale: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a camera frustum wireframe (rectangular pyramid).

    The camera looks along +Z with X right and Y down (OpenCV convention).
    The apex is at the origin; the four corners of the near plane are at
    depth ``scale`` along +Z.

    Parameters
    ----------
    hfov : float
        Horizontal field of view in degrees.
    vfov : float
        Vertical field of view in degrees.
    scale : float
        Depth of the frustum (distance from apex to near plane).

    Returns
    -------
    vertices : (5, 3) ndarray
        Frustum vertices: [apex, top-left, top-right, bottom-right, bottom-left].
    edges : (8, 2) ndarray
        Edge index pairs (4 edges from apex + 4 edges on rectangle).
    """
    hw = scale * np.tan(np.radians(hfov / 2))
    hh = scale * np.tan(np.radians(vfov / 2))

    vertices = np.array([
        [0.0, 0.0, 0.0],          # apex
        [-hw, -hh, scale],         # top-left  (Y down → -Y is up in image)
        [ hw, -hh, scale],         # top-right
        [ hw,  hh, scale],         # bottom-right
        [-hw,  hh, scale],         # bottom-left
    ], dtype=np.float64)

    edges = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # apex to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # rectangle
    ], dtype=np.int64)

    return vertices, edges


def pointcloud_to_splatted_mesh(
    points: np.ndarray,
    cube_size: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace every point in an N×3 cloud with a little cube and return a single
    triangulated mesh.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) of point coordinates (dtype need not be float32).
    cube_size : float, optional
        Edge length of each cube.  If None, a heuristic based on cloud density
        is used.

    Returns
    -------
    faces : np.ndarray (int32)
        Array of shape (N × 12, 3) listing triangle vertex indices.
    verts : np.ndarray (float32)
        Array of shape (N × 8, 3) with vertex coordinates.

    Notes
    -----
    • Each cube has 8 vertices and 12 triangles (2 per face).
    • The heuristic cube size is
      `0.8 × (volume / N) ** (1/3)` where *volume* is the axis‑aligned bounding
      box volume.  Adjust the `0.8` multiplier to taste.
    """
    points = np.asarray(points, dtype=np.float32)
    n_pts = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("`points` must be an (N, 3) array")

    # --------- choose cube edge length if not given ---------
    if cube_size is None:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        extents = maxs - mins
        # Guard against degenerate volume
        vol = np.prod(extents) if np.all(extents > 0) else extents.max() ** 3
        # Reasonable spacing between points
        spacing = (vol / n_pts) ** (1.0 / 3.0) if vol > 0 else 1.0
        cube_size = 0.8 * spacing  # empirical scale factor

    half = cube_size / 2.0

    # --------- template cube (8 verts, 12 faces) ---------
    template_verts = np.array(
        [
            [-half, -half, -half],  # 0
            [ half, -half, -half],  # 1
            [ half,  half, -half],  # 2
            [-half,  half, -half],  # 3
            [-half, -half,  half],  # 4
            [ half, -half,  half],  # 5
            [ half,  half,  half],  # 6
            [-half,  half,  half],  # 7
        ],
        dtype=np.float32,
    )

    # triangles (int32) in CCW order
    template_faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],       # bottom  (‑z)
            [4, 6, 5], [4, 7, 6],       # top     (+z)
            [0, 5, 1], [0, 4, 5],       # back    (‑y)
            [3, 2, 6], [3, 6, 7],       # front   (+y)
            [0, 3, 7], [0, 7, 4],       # left    (‑x)
            [1, 5, 6], [1, 6, 2],       # right   (+x)
        ],
        dtype=np.int32,
    )

    # --------- fill arrays (vectorized) ---------
    # Broadcast template verts + each point: (N, 8, 3)
    verts = (template_verts[None, :, :] + points[:, None, :]).reshape(-1, 3)
    # Offset template faces per point: (N, 12, 3)
    offsets = (np.arange(n_pts) * 8)[:, None, None]  # (N, 1, 1)
    faces = (template_faces[None, :, :] + offsets).reshape(-1, 3)

    return verts, faces