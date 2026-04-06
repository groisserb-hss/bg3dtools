"""
Trimesh visualization wrappers.

This module provides convenient functions for visualizing point clouds
and triangle meshes using the trimesh library.
"""

from typing import Optional, Tuple, List, Union
import numpy as np
import trimesh
from .colors import default_colors, get_heatmap_color
from bg3dtools.mesh.generate import pointcloud_to_splatted_mesh


def draw_geometries(
    geometries: Union[trimesh.Trimesh, List[trimesh.Trimesh]],
    render: bool = True
) -> trimesh.Scene:
    """
    Display one or more trimesh geometries in a scene.

    Parameters
    ----------
    geometries : trimesh.Trimesh or list of trimesh.Trimesh
        Geometry or list of geometries to display.
    render : bool, optional
        If True, show the visualization window. Default is True.

    Returns
    -------
    scene : trimesh.Scene
        Scene containing all geometries.
    """
    if not isinstance(geometries, list):
        geometries = [geometries]

    scene = trimesh.Scene(geometries)
    if render:
        scene.show()
    return scene


def scatt(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    render: bool = True,
    colormap: str = 'parula',
    caxis: Tuple[Optional[float], Optional[float]] = (None, None)
) -> trimesh.PointCloud:
    """
    Visualize a point cloud.

    Parameters
    ----------
    points : (N, 3) ndarray
        Point coordinates.
    colors : ndarray, optional
        Point colors. Can be:
        - (3,) for uniform color
        - (N,) for scalar values mapped through colormap
        - (N, 3) or (N, 4) for per-point RGB/RGBA
    render : bool, optional
        If True, show the visualization. Default is True.
    colormap : str, optional
        Colormap name for scalar colors. Default is 'parula'.
    caxis : tuple, optional
        Color axis limits (vmin, vmax) for scalar mapping.

    Returns
    -------
    pc : trimesh.PointCloud
        Point cloud object.
    """
    points = np.asarray(points)
    if np.any(np.isnan(points)):
        print('Warning: NaN values found in points, replacing with zeros.')
        points = np.nan_to_num(points)

    if colors is None:
        colors = np.array([1., 0., 0.])
    else:
        colors = np.array(colors).astype(np.float32)

    num_pts = len(points)

    if colors.size == num_pts:
        colors = colors.flatten()
        colors = get_heatmap_color(colors, mapname=colormap, caxis=caxis)
    elif (colors.shape == (num_pts, 3) or
          colors.shape == (num_pts, 4) or
          colors.shape == (3,)):
        pass
    else:
        print('color rejected; improper shape')
        colors = np.array([0.5, 0.5, 0.5])

    pc = trimesh.points.PointCloud(points, colors=colors)
    if render:
        pc.show()
    return pc


def scatts(
    point_list: List[np.ndarray],
    render: bool = True
) -> List[trimesh.PointCloud]:
    """
    Visualize multiple point clouds with automatic coloring.

    Each point cloud is assigned a different color from the default palette.

    Parameters
    ----------
    point_list : list of (N, 3) ndarray
        List of point clouds to visualize.
    render : bool, optional
        If True, show the visualization. Default is True.

    Returns
    -------
    clouds : list of trimesh.PointCloud
        List of point cloud objects.
    """
    clouds = []

    for ii, pts in enumerate(point_list):
        c = default_colors[ii % 7]
        clouds.append(scatt(pts, c, render=False))

    scene = trimesh.Scene(clouds)
    if render:
        scene.show()

    return clouds


def splatt(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    render: bool = True,
    cube_size: Optional[float] = None
) -> trimesh.Trimesh:
    """
    Visualize points as small cubes (splats).

    Converts each point to a small cube mesh for visualization.

    Parameters
    ----------
    points : (N, 3) ndarray
        Point coordinates.
    colors : ndarray, optional
        Colors for the mesh (same format as trisurfsm).
    render : bool, optional
        If True, show the visualization. Default is True.
    cube_size : float, optional
        Size of each cube. Auto-computed if None.

    Returns
    -------
    mesh : trimesh.Trimesh
        Combined mesh of all cube splats.
    """
    verts, faces = pointcloud_to_splatted_mesh(points, cube_size=cube_size)
    mesh = trisurfsm(verts, faces, colors=colors, render=render)
    return mesh


def draw_axes(
    length: float = 1,
    render: bool = True
) -> trimesh.Trimesh:
    """
    Draw coordinate axes at the origin.

    Parameters
    ----------
    length : float, optional
        Length of each axis arrow. Default is 1.
    render : bool, optional
        If True, show the visualization. Default is True.

    Returns
    -------
    axes : trimesh.Trimesh
        Axes geometry.
    """
    axes = trimesh.creation.axis(length / 10)
    if render:
        axes.show()
    return axes


def draw_lines(
    p0: np.ndarray,
    p1: np.ndarray,
    colors: Optional[np.ndarray] = None,
    render: bool = True,
    radius: float = 0.1
) -> List[trimesh.Trimesh]:
    """
    Draw lines as cylinders between point pairs.

    Parameters
    ----------
    p0 : (N, 3) ndarray
        Start points of lines.
    p1 : (N, 3) ndarray
        End points of lines.
    colors : (N, 3) ndarray, optional
        RGB colors for each line. Uses default palette if None.
    render : bool, optional
        If True, show the visualization. Default is True.
    radius : float, optional
        Cylinder radius. Default is 0.1.

    Returns
    -------
    lines : list of trimesh.Trimesh
        Cylinder geometries representing the lines.
    """
    p0 = np.array(p0).reshape([-1, 3])
    p1 = np.array(p1).reshape([-1, 3])

    # Calculate the distance between the start and end points
    line_length = np.linalg.norm(p1 - p0, axis=1, keepdims=True)

    # Calculate the midpoint for the placement of the cylinder
    mid_point = (p0 + p1) / 2

    # Calculate the orientation of the line for the cylinder rotation
    direction = (p1 - p0) / (line_length + 1e-6)

    if colors is None:
        colors = [[0, 255, 0]] * len(p0)
    colors = np.array(colors)
    if colors.size == 3:
        colors = np.tile(colors.reshape([1, 3]), (len(p0), 1))

    # Create a cylinder to represent the line
    lines = []
    for ii in range(len(p0)):
        color = colors[ii]
        orientation = trimesh.geometry.align_vectors(np.array([0, 0, 1]), direction[ii].flatten())
        lines.append(trimesh.creation.cylinder(face_colors=color, radius=radius, height=line_length[ii], sections=None,
                                               transform=trimesh.transformations.concatenate_matrices(
                                                   trimesh.transformations.translation_matrix(mid_point[ii]),
                                                   orientation)))

    if render:
        # Visualize the lines
        scene = trimesh.Scene(lines)
        scene.show()
    return lines


def trisurfsm(
    verts: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
    render: bool = True,
    colormap: str = 'parula',
    caxis: Tuple[Optional[float], Optional[float]] = (None, None)
) -> trimesh.Trimesh:
    """
    Visualize a triangle mesh with optional coloring.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.
    colors : ndarray, optional
        Mesh colors. Can be:
        - (3,) for uniform color
        - (nV,) or (nF,) for scalar values mapped through colormap
        - (nV, 3) or (nV, 4) for per-vertex RGB/RGBA
        - (nF, 3) or (nF, 4) for per-face RGB/RGBA
    render : bool, optional
        If True, show the visualization. Default is True.
    colormap : str, optional
        Colormap name for scalar colors. Default is 'parula'.
    caxis : tuple, optional
        Color axis limits (vmin, vmax) for scalar mapping.

    Returns
    -------
    mesh : trimesh.Trimesh
        Triangle mesh object.
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    if colors is None:
        colors = np.array([0.5, 0.5, 0.5])
    else:
        colors = np.array(colors)

    if (colors.shape == (len(verts), 3)) and (np.max(colors) <= 1) and (np.min(colors) >= 0):
        # treat this as RGB
        colors = (colors * 255).astype(np.uint8)

    elif colors.size == len(verts) or colors.size == len(faces):
        colors = get_heatmap_color(colors, mapname=colormap, caxis=caxis)

    if colors.shape == (len(verts), 3) or colors.shape == (len(verts), 4) or colors.shape == (3,):
        mesh.visual.vertex_colors = colors
    elif colors.shape == (len(faces), 3) or colors.shape == (len(faces), 4):
        mesh.visual.face_colors = colors

    if render:
        mesh.show()
    return mesh



