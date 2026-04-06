"""
Point cloud processing utilities.

Provides I/O, quantization, fitting geometric primitives, registration,
and reconstruction from depth images.

Example
-------
>>> from bg3dtools.pointclouds import read_pcloud, fit_plane_to_points
>>> pts = read_pcloud("cloud.ply")
>>> plane = fit_plane_to_points(pts)
"""

__all__ = []

# Quantization and voxelization
try:
    from .quantize import (
        convert_to_points,
        voxelize,
        sparse_quantize,
    )
    __all__.extend([
        "convert_to_points",
        "voxelize",
        "sparse_quantize",
    ])
except ImportError:
    pass

# Fitting geometric primitives (requires scipy)
try:
    from .fitting import (
        isect_plane_line,
        fit_plane_to_points,
        fit_plane_to_noisy_points,
        fit_line_to_points,
        fit_line_to_noisy_points,
        fit_sphere_to_noisy_points_known_rad,
        fit_ellipse_2d,
        project_to_plane,
        project_to_line,
        align_axes,
        smooth_curve,
        redistribute_loop,
        clean_ring_points,
        naive_fps,
        reconstruct_from_distance,
    )
    __all__.extend([
        "isect_plane_line",
        "fit_plane_to_points",
        "fit_plane_to_noisy_points",
        "fit_line_to_points",
        "fit_line_to_noisy_points",
        "fit_sphere_to_noisy_points_known_rad",
        "fit_ellipse_2d",
        "project_to_plane",
        "project_to_line",
        "align_axes",
        "smooth_curve",
        "redistribute_loop",
        "clean_ring_points",
        "naive_fps",
        "reconstruct_from_distance",
    ])
except ImportError:
    pass

# I/O (requires plyfile)
try:
    from .pc_io import read_pcloud, write_pcloud
    __all__.extend([
        "read_pcloud",
        "write_pcloud",
    ])
except ImportError:
    pass

# Registration (requires scipy)
try:
    from .registration import pc_icp
    __all__.append("pc_icp")
except ImportError:
    pass

# Optional: Depth reconstruction (requires Open3D)
try:
    from .reconstruction import depth_to_pc, scale_intrinsics
    __all__.extend([
        "depth_to_pc",
        "scale_intrinsics",
    ])
except ImportError:
    pass
