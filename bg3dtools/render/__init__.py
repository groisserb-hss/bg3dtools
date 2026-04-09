"""
Rendering utilities for mesh and point cloud visualization.

This package provides wrappers for multiple visualization backends.
Colors and colormaps are exported at the top level. For visualization
functions, import directly from submodules:

    from bg3dtools.render.trimesh import scatt, trisurfsm
    from bg3dtools.render.o3d import scatt, trisurfsm

Submodules
----------
trimesh
    Trimesh-based visualization (scatt, trisurfsm, draw_geometries).
o3d
    Open3D-based visualization (scatt, trisurfsm, draw_line).
matplot
    Matplotlib-based visualization.
colors
    Color utilities and colormaps.
"""

# Re-export from colors module for backward compatibility
from .colors import default_colors, get_heatmap_color, xyz_to_rgb
from .o3d import render_mesh_to_image, overhead_camera, anterior_camera

__all__ = [
    "default_colors",
    "get_heatmap_color",
    "xyz_to_rgb",
    "render_mesh_to_image",
    "overhead_camera",
    "anterior_camera",
]
