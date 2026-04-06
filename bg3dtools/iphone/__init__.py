"""
iPhone depth scanning I/O.

Provides unified access to Stray Scanner and Record3D scan data,
including depth frames, camera intrinsics, and point cloud reconstruction.
"""

from .data_io import (read_data,
                      reconstruct_point_clouds,
                      load_depth,
                      save_data,
                      depth_to_pc)
from .record3d_io import (is_record3d,
                          read_record3d,
                          load_record3d_depth)


def read_iphone_scan(folder: str) -> dict:
    """Auto-detect recording format and load scan data.

    Returns the same dict shape regardless of source (Stray Scanner or
    Record3D).  The ``source`` key identifies the format.
    """
    if is_record3d(folder):
        return read_record3d(folder)
    return read_data(folder)


def load_iphone_depth(data: dict) -> "np.ndarray":
    """Load depth frames using the appropriate loader for *data*'s source."""
    if data.get('source') == 'record3d':
        return load_record3d_depth(data['depth_frames'], data['depth_size'])
    return load_depth(data['depth_frames'])


__all__ = [
    "read_data",
    "reconstruct_point_clouds",
    "load_depth",
    "save_data",
    "depth_to_pc",
    "read_iphone_scan",
    "load_iphone_depth",
]
