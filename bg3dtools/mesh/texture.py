"""
Texture sampling utilities for triangle meshes.

This module provides functions for sampling texture data at arbitrary
locations on a mesh surface using barycentric coordinates.
"""

from typing import Optional
import numpy as np

__all__ = [
    "sample_texture_at_points",
]


def sample_texture_at_points(
    img: np.ndarray,
    tc: np.ndarray,
    ftc: np.ndarray,
    face_idx: np.ndarray,
    bary: np.ndarray,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Sample texture at arbitrary mesh surface locations.

    Given face indices and barycentric coordinates, interpolate UV coordinates
    on each face's texture corners, then sample the texture image.

    Parameters
    ----------
    img : (H, W, C) ndarray
        Texture image (typically uint8 RGB).
    tc : (nT, 2) ndarray
        UV texture coordinates, values in [0, 1].
    ftc : (nF, 3) ndarray
        Per-face texture coordinate indices into ``tc``.
    face_idx : (P,) ndarray
        Face index for each query point.
    bary : (P, 3) ndarray
        Barycentric coordinates for each query point.
    interpolation : {"bilinear", "nearest"}, optional
        Interpolation method for texture sampling. Default is "bilinear".

    Returns
    -------
    colors : (P, C) ndarray
        Sampled texture values, same dtype as ``img``.

    Examples
    --------
    >>> from bg3dtools.mesh import read_obj, sample_texture_at_points
    >>> v, tc, n, f, ftc, fn = read_obj("mesh.obj")
    >>> img = np.array(Image.open("texture.jpg"))
    >>> colors = sample_texture_at_points(img, tc, ftc, face_idx, bary)
    """
    H, W = img.shape[:2]
    C = img.shape[2] if img.ndim == 3 else 1
    face_idx = np.asarray(face_idx, dtype=np.intp)
    bary = np.asarray(bary, dtype=np.float64)

    # Gather UV coords for the three corners of each queried face
    # ftc[face_idx] -> (P, 3) indices into tc
    corner_uv = tc[ftc[face_idx]]  # (P, 3, 2)

    # Interpolate UV using barycentric weights
    uv = np.einsum("pi,pij->pj", bary, corner_uv)  # (P, 2)

    # UV -> pixel coordinates
    # Matches sample_obj_vtex convention: col = u * W, row = (1-v) * H
    col = uv[:, 0] * W
    row = (1.0 - uv[:, 1]) * H

    if interpolation == "bilinear":
        from scipy.ndimage import map_coordinates

        # map_coordinates expects (ndim, npoints) with row, col ordering
        if img.ndim == 3:
            colors = np.empty((len(face_idx), C), dtype=np.float64)
            for c in range(C):
                colors[:, c] = map_coordinates(
                    img[:, :, c].astype(np.float64),
                    [row, col],
                    order=1,
                    mode="nearest",
                )
        else:
            colors = map_coordinates(
                img.astype(np.float64), [row, col], order=1, mode="nearest"
            )[:, None]
        colors = np.clip(colors, 0, 255).astype(img.dtype) if img.dtype == np.uint8 else colors.astype(img.dtype)
    else:
        # Nearest-neighbor
        r = np.clip(np.floor(row).astype(np.intp), 0, H - 1)
        c = np.clip(np.floor(col).astype(np.intp), 0, W - 1)
        if img.ndim == 3:
            colors = img[r, c]  # (P, C)
        else:
            colors = img[r, c, None]  # (P, 1)

    return colors
