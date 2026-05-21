"""
Open3D visualization wrappers.

This module provides convenient functions for visualizing point clouds
and triangle meshes using the Open3D library.
"""

import functools
import logging
from typing import Optional, Tuple, List

import numpy as np
import open3d as o3d
from open3d.visualization import draw_geometries

from .colors import default_colors, get_heatmap_color

_log = logging.getLogger(__name__)


def get_cam_params_o3d(
    w: Optional[int] = None,
    h: Optional[int] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    twist: Optional[np.ndarray] = None,
    trans: Optional[np.ndarray] = None
) -> o3d.camera.PinholeCameraParameters:
    """
    Create default Open3D camera parameters.

    Parameters
    ----------
    w : int, optional
        Image width. Default is 1920.
    h : int, optional
        Image height. Default is 1080.
    fx : float, optional
        Focal length x. Default is 1000.
    fy : float, optional
        Focal length y. Default is 1000.
    cx : float, optional
        Principal point x. Default is (w-1)/2.
    cy : float, optional
        Principal point y. Default is (h-1)/2.
    twist : ndarray, optional
        Not currently used.
    trans : ndarray, optional
        Not currently used.

    Returns
    -------
    cam_param : o3d.camera.PinholeCameraParameters
        Camera parameters with identity extrinsic.
    """
    w = w or 1920
    h = h or 1080
    fx = fx or 1000.0
    fy = fy or 1000.0
    cx = cx if cx is not None else (w - 1) / 2
    cy = cy if cy is not None else (h - 1) / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    extrinsic = np.eye(4)
    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.extrinsic = extrinsic
    cam_param.intrinsic = intrinsic

    return cam_param


def scatt(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    render: bool = True,
    colormap: str = 'parula',
    caxis: Tuple[Optional[float], Optional[float]] = (None, None)
) -> o3d.geometry.PointCloud:
    """
    Visualize a point cloud using Open3D.

    Parameters
    ----------
    points : (N, 3) ndarray
        Point coordinates.
    colors : ndarray, optional
        Point colors. Can be:
        - (3,) for uniform color
        - (N,) for scalar values mapped through colormap
        - (N, 3) for per-point RGB (0-1 range)
        Default is [0.1, 0.1, 0.1].
    render : bool, optional
        If True, show the visualization. Default is True.
    colormap : str, optional
        Colormap name for scalar colors. Default is 'parula'.
    caxis : tuple, optional
        Color axis limits (vmin, vmax) for scalar mapping.

    Returns
    -------
    pc : o3d.geometry.PointCloud
        Point cloud object.
    """
    if colors is None:
        colors = [.1, .1, .1]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    colors = np.asarray(colors, dtype=np.float64)

    if colors.size == 3:
        pc.paint_uniform_color(colors)
    elif colors.size == len(points):
        rgb = get_heatmap_color(colors, mapname=colormap, caxis=caxis)
        pc.colors = o3d.utility.Vector3dVector(rgb)
    elif (len(colors) == len(points)) and (colors.shape[1] == 3):
        pc.colors = o3d.utility.Vector3dVector(colors)
    else:
        raise ValueError(f"Invalid number of colors: {colors.size}")

    if render:
        o3d.visualization.draw_geometries([pc])
    return pc


def scatts(
    point_list: List[np.ndarray],
    render: bool = True
) -> List[o3d.geometry.PointCloud]:
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
    show : list of o3d.geometry.PointCloud
        List of point cloud objects.
    """
    show = []

    for ii, pts in enumerate(point_list):
        c = default_colors[ii % 7]
        show.append(scatt(pts, c, render=False))

    if render:
        draw_geometries(show)
    return show


def draw_line(
    p0: np.ndarray,
    p1: np.ndarray,
    colors: Optional[np.ndarray] = None,
    render: bool = True,
    colormap: str = 'parula',
    caxis: Tuple[Optional[float], Optional[float]] = (None, None),
    resolution: float = 0.1
) -> o3d.geometry.LineSet:
    """
    Draw lines between point pairs using a LineSet.

    Parameters
    ----------
    p0 : (N, 3) ndarray
        Start points of lines.
    p1 : (N, 3) ndarray
        End points of lines.
    colors : ndarray, optional
        Line colors. Can be:
        - (3,) for uniform RGB colour
        - (N,) for scalar values mapped through colormap
        - (N, 3) for per-line RGB (0-1 range)
    render : bool, optional
        If True, show the visualization. Default is True.
    colormap : str, optional
        Colormap name for scalar colors. Default is 'parula'.
    caxis : tuple, optional
        Color axis limits for scalar mapping.
    resolution : float, optional
        Unused, kept for backward compatibility.

    Returns
    -------
    ls : o3d.geometry.LineSet
        LineSet geometry.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1, 3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 3)
    N = len(p0)
    assert N == len(p1)

    points = np.concatenate([p0, p1], axis=0)  # (2N, 3)
    lines = np.column_stack([np.arange(N), np.arange(N, 2 * N)])  # (N, 2)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)

    if colors is None:
        ls.paint_uniform_color([0.1, 0.1, 0.1])
    else:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.size == 3:
            ls.paint_uniform_color(colors.ravel())
        elif colors.ndim == 1 and colors.size == N:
            rgb = get_heatmap_color(colors, mapname=colormap, caxis=caxis)
            ls.colors = o3d.utility.Vector3dVector(rgb)
        elif colors.ndim == 2 and colors.shape == (N, 3):
            ls.colors = o3d.utility.Vector3dVector(colors)
        else:
            ls.paint_uniform_color([0.1, 0.1, 0.1])

    if render:
        o3d.visualization.draw_geometries([ls])
    return ls


def _apply_triangle_colors(mesh: o3d.geometry.TriangleMesh,
                           verts: np.ndarray,
                           faces: np.ndarray,
                           tri_rgb: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Apply per-triangle RGB colors.

    If legacy Open3D supports mesh.triangle_colors, use it.
    Otherwise, "explode" the mesh (duplicate vertices per triangle) and assign
    per-vertex colors so the rendered result is per-face colored.
    """
    tri_rgb = np.asarray(tri_rgb, dtype=np.float64)
    if tri_rgb.shape[0] != faces.shape[0] or tri_rgb.shape[1] != 3:
        raise ValueError(f"tri_rgb must be (nF,3); got {tri_rgb.shape}")

    if np.nanmax(tri_rgb) > 1.0:
        tri_rgb = tri_rgb / 255.0
    tri_rgb = np.nan_to_num(tri_rgb, nan=0.0, posinf=1.0, neginf=0.0)

    # Some Open3D versions expose triangle_colors on legacy TriangleMesh
    if hasattr(mesh, "triangle_colors"):
        mesh.triangle_colors = o3d.utility.Vector3dVector(tri_rgb)
        return mesh

    # Fallback: explode mesh so triangle colors can be represented as vertex colors
    nF = faces.shape[0]
    verts_exploded = verts[faces.reshape(-1)].reshape(-1, 3)          # (nF*3, 3)
    faces_exploded = np.arange(nF * 3, dtype=np.int64).reshape(nF, 3) # (nF, 3)
    vcol_exploded = np.repeat(tri_rgb, 3, axis=0)                     # (nF*3, 3)

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts_exploded)
    m.triangles = o3d.utility.Vector3iVector(faces_exploded)
    m.vertex_colors = o3d.utility.Vector3dVector(vcol_exploded)
    m.compute_vertex_normals()
    return m


def trisurfsm(
    verts: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
    render: bool = True,
    colormap: str = 'parula',
    caxis: Tuple[Optional[float], Optional[float]] = (None, None)
) -> o3d.geometry.TriangleMesh:
    """
    Visualize a triangle mesh using Open3D.

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
        - (nV, 3) for per-vertex RGB (0-1 range)
        - (nF, 3) for per-face RGB (0-1 or 0-255 range)
    render : bool, optional
        If True, show the visualization. Default is True.
    colormap : str, optional
        Colormap name for scalar colors. Default is 'parula'.
    caxis : tuple, optional
        Color axis limits (vmin, vmax) for scalar mapping.

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
        Triangle mesh object.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    if colors is None:
        mesh.paint_uniform_color([.5, .5, .5])

    elif len(colors) == 3:
        mesh.paint_uniform_color(colors)

    # per-face scalar (nF,)
    elif colors.size == faces.shape[0] and (colors.ndim == 1 or (colors.ndim == 2 and colors.shape[1] == 1)):
        s = colors.reshape(-1).astype(np.float64)
        tri_rgb = get_heatmap_color(s, mapname=colormap, caxis=caxis).astype(np.float64)
        mesh = _apply_triangle_colors(mesh, verts, faces, tri_rgb)

    # per-face RGB (nF,3)
    elif colors.ndim == 2 and colors.shape[0] == faces.shape[0] and colors.shape[1] == 3:
        tri_rgb = colors.astype(np.float64)
        mesh = _apply_triangle_colors(mesh, verts, faces, tri_rgb)

    elif colors.size == verts.shape[0]:
        # colors is (N,) per-vertex scalar; map to RGB and normalize for Open3D
        rgb = get_heatmap_color(colors.flatten(), mapname=colormap, caxis=caxis).astype(np.float64)
        if np.nanmax(rgb) > 1.0:  # defensive: handle 0–255 input
            rgb /= 255.0
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)

    elif colors.shape == verts.shape:
        colors = colors.astype(np.float64)
        if np.nanmax(colors) > 1.0:
            colors /= 255.0
        colors = np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([.1, .1, .1])

    if render:
        o3d.visualization.draw_geometries([mesh])

    return mesh


def overhead_camera(v):
    """Camera looking down S-axis (superior -> inferior).

    Returns (lookat, eye, up) for use with render_mesh_to_image.
    """
    center = v.mean(axis=0)
    extent = v.ptp(axis=0)
    eye = center + np.array([0, 0, extent[2] * 2])
    up = np.array([0, 1, 0])
    return center, eye, up


def anterior_camera(v):
    """Camera looking from anterior toward posterior (A-axis).

    Returns (lookat, eye, up) for use with render_mesh_to_image.
    """
    center = v.mean(axis=0)
    extent = v.ptp(axis=0)
    eye = center + np.array([0, extent[1] * 2, 0])
    up = np.array([0, 0, 1])
    return center, eye, up


def render_mesh_to_image(geom, lookat, eye, up, width=400, height=400, fov=45.0):
    """Render an Open3D geometry to a numpy RGB uint8 image.

    Tries OffscreenRenderer (headless) first, falls back to legacy Visualizer.
    """
    try:
        return _render_offscreen(geom, lookat, eye, up, width, height, fov)
    except Exception:
        return _render_legacy(geom, lookat, eye, up, width, height)


_offscreen_cache = {}  # (width, height) -> OffscreenRenderer


def _render_offscreen(geom, lookat, eye, up, width, height, fov):
    import open3d.visualization.rendering as rendering
    key = (width, height)
    if key not in _offscreen_cache:
        _offscreen_cache[key] = rendering.OffscreenRenderer(width, height)
    renderer = _offscreen_cache[key]
    renderer.scene.set_background(np.array([1, 1, 1, 1], dtype=np.float32))
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", geom, mat)
    renderer.setup_camera(fov, lookat, eye, up)
    img = np.asarray(renderer.render_to_image()).copy()
    renderer.scene.clear_geometry()
    return img


def _render_legacy(geom, lookat, eye, up, width, height):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.add_geometry(geom)
    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(eye - lookat)
    ctr.set_up(up)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def mesh_to_wireframe(
    vertices: np.ndarray,
    faces: np.ndarray,
    color: Tuple[float, float, float] = (0.7, 0.7, 0.9)
) -> o3d.geometry.LineSet:
    """
    Convert mesh vertices and faces to a LineSet wireframe.

    Parameters
    ----------
    vertices : (V, 3) ndarray
        Mesh vertices.
    faces : (F, 3) ndarray
        Triangle face indices.
    color : tuple
        RGB color for the wireframe lines.

    Returns
    -------
    wireframe : o3d.geometry.LineSet
        Wireframe representation of the mesh.
    """
    # Extract all half-edges and deduplicate via sorting + np.unique
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    lines = np.unique(e, axis=0).astype(np.int32)

    wireframe = o3d.geometry.LineSet()
    wireframe.points = o3d.utility.Vector3dVector(vertices)
    wireframe.lines = o3d.utility.Vector2iVector(lines)
    wireframe.paint_uniform_color(color)

    return wireframe



