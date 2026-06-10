"""
Point cloud reconstruction from depth images.

This module provides functions for converting depth images to 3D point
clouds with optional normal estimation.
"""

from __future__ import annotations

from typing import Tuple, Optional, Union
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
from bg3dtools.image_tools import normal_edges

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False


def _require_open3d():
    if not HAS_OPEN3D:
        raise ImportError(
            "open3d is required for depth_to_pc. "
            "Install with: pip install 'bg3dtools[viz]'"
        )


__all__ = [
    "scale_intrinsics",
    "depth_to_pc",
]


def _compute_fast_normals(
    P: np.ndarray,
    ksize: int = 3,
    sigma: float = 1.2
) -> np.ndarray:
    """
    Compute normals from a gridded point cloud using gradient-based method.

    Parameters
    ----------
    P : (nR, nC, 3) ndarray
        Gridded XYZ coordinates from a depth image.
    ksize : int, optional
        Kernel size for gradient computation. Default is 3.
    sigma : float, optional
        Gaussian smoothing sigma. Set 0 to disable. Default is 1.2.

    Returns
    -------
    N : (nR, nC, 3) ndarray
        Unit normals oriented toward +Z hemisphere.
    """
    # 1. separable Gaussian to knock down depth speckle
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        P = gaussian_filter(P, sigma=(sigma, sigma, 0), mode='nearest')

    # 2. first-order derivatives with a larger-support Sobel/Scharr
    #    Here we build a size=ksize central-difference kernel (odd size 3,5,7…)
    half = ksize // 2
    g = np.arange(-half, half + 1, dtype=np.float32)
    gx = g[None, :]
    gy = g[:, None]

    # 3. apply to each coord
    dPu = np.stack([convolve(P[...,i], gx, mode='nearest') for i in range(3)], axis=-1)
    dPv = np.stack([convolve(P[...,i], gy, mode='nearest') for i in range(3)], axis=-1)

    # 4. normal = cross(∂P/∂v , ∂P/∂u)
    N = np.cross(dPv, dPu)
    N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-12  # unit length

    # 5. orient consistently (e.g. +Z hemisphere)
    flip = N[...,2] < 0
    N[flip] *= -1
    return N


def _normals_from_depth_grid(
    P: np.ndarray,
    invalid_mask: Optional[np.ndarray] = None,
    rho: float = 0.02,
    k: int = 2,
    spatial_sigma: float = 2.0,
    W: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute normals using local PCA on depth grid neighborhoods.

    More robust than gradient-based method for noisy regions.

    Parameters
    ----------
    P : (nR, nC, 3) ndarray
        XYZ coordinates in metres.
    invalid_mask : (nR, nC) ndarray of bool, optional
        True where depth is missing/invalid.
    rho : float, optional
        Depth discontinuity threshold in metres. Default is 0.02.
    k : int, optional
        Window half-size (window is 2k+1 x 2k+1). Default is 2.
    W : (2k+1, 2k+1) ndarray, optional
        Precomputed Gaussian weights.

    Returns
    -------
    N : (nR, nC, 3) ndarray
        Unit normals (NaN where undetermined).
    """
    if W is None:
        sigma = spatial_sigma  # pixels; tune > noise but < corner scale
        grid = np.arange(-k, k + 1)
        G = np.exp(-(grid ** 2) / (2 * sigma ** 2))
        W = (G[:, None] * G[None, :]).astype(np.float32)

    nR, nC, _ = P.shape
    L = (2 * k + 1) ** 2
    w_flat = W.reshape(-1).astype(np.float32)

    # Pad XYZ + validity for border handling
    Pp = np.pad(P, ((k, k), (k, k), (0, 0)), mode='edge')          # (nR+2k, nC+2k, 3)
    if invalid_mask is None:
        Mp = np.zeros((nR + 2 * k, nC + 2 * k), dtype=bool)
    else:
        Mp = np.pad(invalid_mask, ((k, k), (k, k)), constant_values=True)

    # Gather each pixel's (2k+1)^2 neighbourhood -> (nR,nC,L,3) + neighbour-invalid -> (nR,nC,L).
    # Offset order (di outer, dj inner) matches W.reshape(-1).
    Q = np.empty((nR, nC, L, 3), dtype=np.float32)
    Mn = np.empty((nR, nC, L), dtype=bool)
    idx = 0
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            Q[:, :, idx, :] = Pp[k + di:k + di + nR, k + dj:k + dj + nC, :]
            Mn[:, :, idx] = Mp[k + di:k + di + nR, k + dj:k + dj + nC]
            idx += 1

    # neighbour validity: within rho in depth AND not invalid (NaN depths -> False)
    valid_nb = (np.abs(Q[:, :, :, 2] - P[:, :, 2][:, :, None]) < rho) & (~Mn)
    cnt = valid_nb.sum(-1)
    compute = (~Mp[k:k + nR, k:k + nC]) & (cnt >= 3)               # valid centre with >=3 neighbours

    N = np.full((nR, nC, 3), np.nan, dtype=np.float32)
    if not compute.any():
        return N

    # local weighted-PCA on the compute-subset only (M pixels) -> one batched eigh
    Qs = np.nan_to_num(Q[compute], nan=0.0)                        # (M,L,3); invalid nbrs carry weight 0
    w = w_flat[None, :] * valid_nb[compute]                        # (M,L)
    C = (w[..., None] * Qs).sum(1) / np.maximum(w.sum(1)[:, None], 1e-12)
    Qc = Qs - C[:, None, :]
    Cov = np.einsum('mli,mlj->mij', w[..., None] * Qc, Qc)        # (M,3,3) weighted covariance
    _, evec = np.linalg.eigh(Cov)
    n = evec[:, :, 0]                                              # smallest-eigenvalue eigenvector
    n = np.where((n[:, 2] < 0)[:, None], -n, n)                   # orient z >= 0
    N[compute] = n
    return N


def scale_intrinsics(
    src_intrinsics: np.ndarray,
    src_image: Union[np.ndarray, Tuple[int, int]],
    dst_image: Union[np.ndarray, Tuple[int, int]],
) -> np.ndarray:
    """
    Scale camera intrinsics for image resize.

    Adjusts focal lengths, principal point, and skew when resizing
    an image (no crop/pad).

    Parameters
    ----------
    src_intrinsics : (3, 3) ndarray
        Original camera intrinsics matrix K.
    src_image : (H, W, ...) ndarray or (H, W) tuple
        Source image or its dimensions.
    dst_image : (H, W, ...) ndarray or (H, W) tuple
        Destination image or its dimensions.

    Returns
    -------
    K : (3, 3) ndarray
        Scaled intrinsics matrix (float64).
    """
    if isinstance(src_image, np.ndarray):
        src_h, src_w = src_image.shape[:2]
    else:
        src_h, src_w = src_image

    if isinstance(dst_image, np.ndarray):
        dst_h, dst_w = dst_image.shape[:2]
    else:
        dst_h, dst_w = dst_image

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    K = src_intrinsics.astype(np.float64).copy()

    # Scale focal lengths
    K[0, 0] *= sx  # fx
    K[1, 1] *= sy  # fy

    # Scale skew (pixel units, affects x)
    K[0, 1] *= sx  # s

    # Scale principal point
    K[0, 2] *= sx  # cx
    K[1, 2] *= sy  # cy

    # Keep homogeneous bottom row as-is (typically [0, 0, 1])
    return K


def depth_to_pc(
    depth: np.ndarray,
    depth_intrinsics: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic],
    rgb: Optional[np.ndarray],
    compute_normals: bool = True,
    orient_normals: str = 'camera',
    remove_edges: bool = False,
    edge_z_thresh: float = 0.03,
    depth_range: Tuple[float, float] = (0.3, np.inf),
    depth_scale: float = 1.0,
    project_valid_depth_only: bool = False
) -> o3d.geometry.PointCloud:
    """
    Convert a depth image to an Open3D point cloud.

    Parameters
    ----------
    depth : (H, W) ndarray
        Depth image.
    depth_intrinsics : (3, 3) ndarray or o3d.camera.PinholeCameraIntrinsic
        Camera intrinsics matrix or Open3D intrinsics object.
    rgb : (H, W, 3) ndarray or None
        RGB image for coloring points. Resized to match depth if needed.
    compute_normals : bool, optional
        If True, estimate surface normals. Default is True.
    orient_normals : str, optional
        'camera' (default) orients normals toward the camera (a visible surface's outward normal,
        n·P <= 0); 'raw' keeps the underlying +Z-hemisphere convention. Default 'camera'.
    remove_edges : bool, optional
        If True, remove points at depth discontinuities. Default is False.
    edge_z_thresh : float, optional
        Sobel depth-gradient threshold (metres) for ``remove_edges``. Default is 0.03.
    depth_range : (min, max) tuple, optional
        Valid depth range. Default is (0.3, inf).
    depth_scale : float, optional
        Scale factor for depth values. Default is 1.0.
    project_valid_depth_only : bool, optional
        If True, only include points with valid depth. Default is False.

    Returns
    -------
    pc : o3d.geometry.PointCloud
        Point cloud with optional colors and normals.
    """
    _require_open3d()
    DEPTH_HEIGHT, DEPTH_WIDTH = depth.shape
    MIN_DEPTH, MAX_DEPTH = depth_range

    if isinstance(depth_intrinsics, np.ndarray):
        depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=DEPTH_WIDTH, height=DEPTH_HEIGHT,
                                                             fx=depth_intrinsics[0, 0], fy=depth_intrinsics[1, 1],
                                                             cx=depth_intrinsics[0, 2], cy=depth_intrinsics[1, 2])

    if remove_edges:
        z_edges = normal_edges(np.asarray(depth), z_thresh=edge_z_thresh)
        depth = depth.copy()
        depth[z_edges] = 0

    bad_mask = (depth < MIN_DEPTH) | (depth >= MAX_DEPTH)

    depth_o3d = o3d.geometry.Image(depth)
    if rgb is None:
        pc = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, depth_intrinsics, depth_scale=depth_scale,
            depth_trunc=MAX_DEPTH, project_valid_depth_only=False
        )

    else:
        # scale rgb image to match depth size
        if rgb.shape[:2] != depth.shape[:2]:
            rgb = Image.fromarray(rgb)
            rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
            rgb = np.array(rgb)  # [DEPTH_HEIGHT x DEPTH_WIDTH]
        rgb_o3d = o3d.geometry.Image(rgb)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=depth_scale, depth_trunc=MAX_DEPTH,
            convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, depth_intrinsics, project_valid_depth_only=False)

    if compute_normals:
        point_grid = np.asarray(pc.points).reshape([DEPTH_HEIGHT, DEPTH_WIDTH, 3])
        normals = _compute_fast_normals(point_grid)
        ignore_mask = bad_mask | np.isfinite(normals).all(axis=-1)
        slow_normals = _normals_from_depth_grid(point_grid, ignore_mask)
        normals, slow_normals = normals.reshape([-1, 3]), slow_normals.reshape([-1, 3])
        normals[~ignore_mask.flatten()] = slow_normals[~ignore_mask.flatten()]
        if orient_normals == 'camera':
            # Orient toward the camera (origin, +Z forward): a visible surface's outward normal points
            # back toward the camera, i.e. n·P <= 0. Flip the rest. Method-agnostic; supersedes the
            # +Z-hemisphere convention above so callers don't need a downstream flip.
            pts_flat = np.asarray(pc.points)
            flip = (normals * pts_flat).sum(axis=-1) > 0
            normals[flip] = -normals[flip]
        pc.normals = o3d.utility.Vector3dVector(normals.reshape([-1, 3]))

    if project_valid_depth_only:
        keep_mask = ~bad_mask.flatten()
        pc = pc.select_by_index(np.where(keep_mask)[0])

    return pc

