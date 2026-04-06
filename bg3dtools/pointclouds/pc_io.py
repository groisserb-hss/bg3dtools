"""
Point cloud I/O utilities.

This module provides functions for reading and writing point clouds
in PLY format with optional color attributes.
"""

from typing import Optional, Tuple, Union
import numpy as np

__all__ = [
    "write_pcloud",
    "read_pcloud",
]


def write_pcloud(
    filename: str,
    pts: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    ascii: bool = False
) -> None:
    """
    Write a point cloud to PLY file.

    Parameters
    ----------
    filename : str
        Output file path.
    pts : (N, 3) ndarray
        Point coordinates.
    rgb : (N, 3) ndarray, optional
        RGB colors as uint8 [0-255]. If None, writes XYZ only.
    ascii : bool, optional
        If True, write ASCII format. Default is False (binary).
    """
    from plyfile import PlyData, PlyElement

    if rgb is None:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    else:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')]

    vertex = np.empty(pts.shape[0], dtype=dtype)
    vertex['x'] = pts[:, 0]
    vertex['y'] = pts[:, 1]
    vertex['z'] = pts[:, 2]

    if rgb is not None:
        vertex['red'] = rgb[:, 0]
        vertex['green'] = rgb[:, 1]
        vertex['blue'] = rgb[:, 2]
        vertex['alpha'] = 1

    el_v = PlyElement.describe(vertex, 'vertex')

    PlyData([el_v], text=ascii, comments=["written by write_pcloud"]).write(filename)


def read_pcloud(
    filename: str,
    return_color: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Read a point cloud from PLY file.

    Parameters
    ----------
    filename : str
        Input PLY file path.
    return_color : bool, optional
        If True, also return RGB colors. Default is False.

    Returns
    -------
    points : (N, 3) ndarray, float32
        Point coordinates.
    rgb : (N, 3) ndarray, uint8
        RGB colors. Only returned if return_color is True.
    """
    from plyfile import PlyData

    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)

    vdata = plydata['vertex'].data
    verts = np.array([vdata[i] for i in ['x', 'y', 'z']], dtype=np.float32).T

    if return_color:
        rgb = np.array([vdata[i] for i in ['red', 'green', 'blue']], dtype=np.uint8).T
        return verts, rgb
    else:
        return verts

