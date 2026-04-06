"""
Point cloud quantization and voxelization utilities.

This module provides functions for converting between point clouds
and volumetric (voxel) representations.
"""

from typing import Tuple, Optional, Union
import numpy as np

__all__ = [
    "convert_to_points",
    "voxelize",
    "sparse_quantize",
]


def convert_to_points(
    mask: np.ndarray,
    affine: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert a binary mask to a point cloud.

    Extracts the coordinates of all non-zero voxels in a 3D mask
    and optionally transforms them using an affine matrix.

    Parameters
    ----------
    mask : (X, Y, Z) ndarray
        3D binary mask where True/non-zero indicates occupied voxels.
    affine : (4, 4) ndarray, optional
        Affine transformation matrix to apply to extracted points.
        If None, returns voxel indices as coordinates.

    Returns
    -------
    pts : (N, 3) ndarray
        Point cloud coordinates (float64).
    """
    pts = np.argwhere(mask).astype(np.float64)

    if affine is not None:
        from bg3dtools.transforms_unified import transform_points_forward
        pts = transform_points_forward(affine, pts)

    return pts


def voxelize(
    pts: np.ndarray,
    shape: Tuple[int, int, int],
    affine: Optional[np.ndarray] = None,
    allow_oob: bool = False
) -> np.ndarray:
    """
    Convert a point cloud to a binary voxel grid.

    Inverse operation of convert_to_points. Points are quantized to
    integer voxel coordinates.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Point cloud coordinates.
    shape : tuple of int
        Output volume shape (X, Y, Z).
    affine : (4, 4) ndarray, optional
        Inverse affine to transform points before quantization.
        If None, points are used directly.
    allow_oob : bool, optional
        If True, clip out-of-bounds points to volume edges.
        If False (default), raise AssertionError for OOB points.

    Returns
    -------
    seg : (X, Y, Z) ndarray
        Boolean voxel grid with True at occupied voxels.

    Raises
    ------
    AssertionError
        If allow_oob=False and any point falls outside the volume.
    """
    xmax, ymax, zmax = shape
    # indices must be integers within the field of view; we should never have an issue being on the edge
    if affine is None:
        coords = pts
    else:
        from bg3dtools.transforms_unified import transform_points_inverse
        coords = transform_points_inverse(affine, pts)

    coords = (coords + .5).astype(np.int64)
    assert np.min(coords) >= 0
    if allow_oob:
        coords = np.clip(coords, 0, [xmax-1, ymax-1, zmax-1])
    else:
        assert np.max(coords[:,0]) < xmax, 'max x %d >= %d' % (np.max(coords[:,0]), xmax)
        assert np.max(coords[:,1]) < ymax, 'max y %d >= %d' % (np.max(coords[:,1]), ymax)
        assert np.max(coords[:,2]) < zmax, 'max z %d >= %d' % (np.max(coords[:,2]), zmax)

    seg = np.zeros(shape, dtype=bool)
    seg[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    return seg


def sparse_quantize(
    pts: np.ndarray,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    return_index: bool = False,
    return_inverse: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    Quantize a point cloud to unique voxel positions.

    Bins points to integer voxel positions using floor operation,
    then returns unique voxels. Wrapper around numpy.unique with axis=0.

    Parameters
    ----------
    pts : (N, D) ndarray
        Point cloud coordinates.
    features : (N, F) ndarray, optional
        Per-point features. If provided, returns features at unique indices.
    labels : (N, L) ndarray, optional
        Per-point labels. If provided, returns labels at unique indices.
    return_index : bool, optional
        If True, return indices of first occurrences. Default is False.
    return_inverse : bool, optional
        If True, return inverse mapping. Default is False.

    Returns
    -------
    quantized : (M, D) ndarray
        Unique voxel positions (int32).
    features : (M, F) ndarray, optional
        Features at unique voxels. Only if features was provided.
    labels : (M, L) ndarray, optional
        Labels at unique voxels. Only if labels was provided.
    idx : (M,) ndarray, optional
        Indices into original array. Only if return_index=True.
    inv : (N,) ndarray, optional
        Inverse mapping from original to unique. Only if return_inverse=True.

    Examples
    --------
    >>> pts = np.array([[0.1, 0.2], [0.9, 0.8], [0.1, 0.3]])
    >>> sparse_quantize(pts)
    array([[0, 0],
           [0, 0],  # same voxel as first point
           ...])
    """

    floored = np.floor(pts).astype(np.int32)
    quantized, idx, inv = np.unique(floored, axis=0, return_index=True, return_inverse=True)

    outputs = [quantized]
    if features is not None:
        assert len(features) == len(pts)
        outputs.append(features[idx])
    if labels is not None:
        assert len(labels) == len(pts)
        outputs.append(labels[idx])
    if return_index:
        outputs.append(idx)
    if return_inverse:
        outputs.append(inv)

    if len(outputs) == 1:
        return quantized
    else:
        return tuple(outputs)
