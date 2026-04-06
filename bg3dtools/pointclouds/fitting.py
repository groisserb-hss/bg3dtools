"""
Point cloud and curve fitting utilities.

This module provides functions for fitting geometric primitives (planes, lines,
spheres, ellipses) to point clouds, curve smoothing, and RANSAC-based robust
fitting algorithms.
"""

from typing import Optional, Tuple, Union
import random
import numpy as np
from scipy.interpolate import UnivariateSpline

__all__ = [
    "isect_plane_line",
    "redistribute_loop",
    "clean_ring_points",
    "fit_ellipse_2d",
    "align_axes",
    "project_to_line",
    "project_to_plane",
    "fit_plane_to_points",
    "fit_plane_to_noisy_points",
    "fit_sphere_to_noisy_points_known_rad",
    "fit_line_to_points",
    "fit_line_to_noisy_points",
    "reconstruct_from_distance",
    "naive_fps",
    "smooth_curve",
]


def isect_plane_line(
    plane: np.ndarray,
    line: Tuple[np.ndarray, np.ndarray]
) -> Optional[np.ndarray]:
    """
    Compute the intersection point of a plane and a line.

    Parameters
    ----------
    plane : (4,) ndarray
        Plane coefficients [a, b, c, d] where ax + by + cz = d.
        Will be normalized internally.
    line : tuple of (point, direction)
        Line defined by a point (3,) and direction vector (3,).

    Returns
    -------
    intersection : (3,) ndarray or None
        Intersection point, or None if line is parallel to plane.
    """
    plane = plane / np.linalg.norm(plane[:-1])
    plane_normal = plane[:-1].reshape(1, 3)
    plane_pt = plane[-1] * plane_normal

    pt0, vec = line
    vec = vec.reshape(1, 3) / np.linalg.norm(vec)
    ndotu = plane_normal @ vec.T

    if abs(ndotu) < .000001:
        return None

    w = pt0 - plane_pt
    s = -(plane_normal @ w) / ndotu
    return pt0 + s * vec


def redistribute_loop(
    loop: np.ndarray,
    N: Optional[int] = None
) -> np.ndarray:
    """
    Redistribute points evenly along a closed loop.

    Parameters
    ----------
    loop : (M, D) ndarray
        Input loop coordinates.
    N : int, optional
        Number of output points. Default is len(loop).

    Returns
    -------
    evenly_spaced : (N, D) ndarray
        Points evenly distributed along the loop.
    """
    N = len(loop) if N is None else N
    # Compute total loop length
    distances = np.linalg.norm(loop - np.roll(loop, -1, axis=0), axis=1)
    target_spacing = np.sum(distances) / N

    # Evenly space the points along the smoothed loop
    evenly_spaced_points = np.empty((N, loop.shape[1]), dtype=loop.dtype)
    evenly_spaced_points[0] = loop[0]
    current_distance = 0.0
    current_idx = 0

    for i in range(1, N):
        while current_distance + distances[current_idx] < target_spacing:
            current_distance += distances[current_idx]
            current_idx = (current_idx + 1) % N

        overhang = current_distance + distances[current_idx] - target_spacing
        fraction = 1 - overhang / distances[current_idx]
        evenly_spaced_points[i] = loop[current_idx] * (1 - fraction) + loop[(current_idx + 1) % N] * fraction
        current_distance = 0
        distances[current_idx] = overhang

    return evenly_spaced_points


def clean_ring_points(
    points: np.ndarray,
    s: Optional[float] = None
) -> np.ndarray:
    """
    Smooth noisy ring points using spline fitting in cylindrical coordinates.

    Parameters
    ----------
    points : (N, 3) ndarray
        Input ring points.
    s : float, optional
        Spline smoothing factor. Default is 0.001.

    Returns
    -------
    new_points : (N, 3) ndarray
        Smoothed ring points.
    """
    n = len(points)

    # Center the points
    centered_points = points - np.mean(points, axis=0)
    # Perform Singular Value Decomposition
    _, _, vh = np.linalg.svd(centered_points)

    # Compute the projection matrix
    projected_points = centered_points @ vh.T

    # Convert to cylindrical coordinates
    rho = np.linalg.norm(projected_points[:, :2], axis=1)
    phi = np.arctan2(projected_points[:, 1], projected_points[:, 0])
    z = projected_points[:, 2]

    # pad each coordinate on either side to avoid edge effects
    w = n // 2
    rho_padded = np.pad(rho, w, mode='wrap')
    phi_padded = np.pad(phi, w, mode='wrap')
    phi_padded = np.unwrap(phi_padded)
    z_padded = np.pad(z, w, mode='wrap')

    # Fit a univariate spline to each coordinate as a function of index
    idx = np.arange(len(rho_padded))
    s = 0.001 if s is None else s
    spline_rho = UnivariateSpline(idx, rho_padded, s=s)
    spline_phi = UnivariateSpline(idx, phi_padded, s=s)
    spline_z = UnivariateSpline(idx, z_padded, s=s)

    # Sample the fitted curve to get new rho values
    rho_sampled = spline_rho(idx)[w:-w]
    phi_sampled = spline_phi(idx)[w:-w]
    z_sampled = spline_z(idx)[w:-w]

    # Step 7: Convert back to Cartesian coordinates in the plane
    x_plane = rho_sampled * np.cos(phi_sampled)
    y_plane = rho_sampled * np.sin(phi_sampled)
    sampled_pts = np.column_stack((x_plane, y_plane, z_sampled))

    # Convert plane coordinates back to 3D coordinates
    new_points = sampled_pts @ vh + np.mean(points, axis=0)

    return new_points


def fit_ellipse_2d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fit a 2-D ellipse to points via least-squares.

    Parameters
    ----------
    X, Y : (N, 1) ndarray
        Point coordinates.

    Returns
    -------
    coeffs : (5,) ndarray
        Coefficients [a0..a4] of  a0*x² + a1*xy + a2*y² + a3*x + a4*y = 1.
    """
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    a = np.linalg.lstsq(A, b)[0].squeeze()
    return a


def align_axes(pt0: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """Build a 4x4 affine aligning pt0→origin, pt0→pt1 to X, pt0→pt2 to Y."""
    pt0 = pt0.reshape([1, 3])
    pt1 = pt1.reshape([1, 3])
    pt2 = pt2.reshape([1, 3])

    v1 = pt1 - pt0
    v1 /= np.linalg.norm(v1)
    v2 = pt2 - pt0
    v2 -= (v2 @ v1.T) * v1
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)

    tform = np.row_stack((v1, v2, v3)).T
    tform = np.column_stack((tform, pt0.T))
    tform = np.pad(tform, [[0,1],[0,0]])
    tform[3,3] = 1

    return tform


def project_to_line(
    line: Tuple[np.ndarray, np.ndarray], pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project points onto a line.

    Parameters
    ----------
    line : (point, direction)
        Line defined by a point and direction vector.
    pts : (N, D) ndarray
        Points to project.

    Returns
    -------
    projected : (N, D) ndarray
        Closest points on the line.
    distance : (N,) ndarray
        Distance from each point to the line (signed in 2-D).
    """
    pt0, vec = line
    mag = np.maximum(0.00000001, np.linalg.norm(vec))
    vec = vec.reshape(1, -1) / mag
    pt0 = pt0.reshape(1, -1)

    pts = pts.reshape([-1, pt0.shape[1]])
    pts = pts - pt0
    projected = (pts @ vec.T) * vec
    residual = pts - projected
    dist = np.linalg.norm(residual, axis=1)

    # for 2D projections, return signed distance
    if pt0.shape[1] == 2:
        normal = np.array([[vec[0, 1]], [-vec[0, 0]]])
        s = 1 - 2*(residual @ normal > 0)
        dist *= s[:, 0]
    return pt0 + projected, dist


def project_to_plane(
    plane: np.ndarray,
    pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project points onto a plane.

    Parameters
    ----------
    plane : (4,) ndarray
        Plane coefficients [a, b, c, d] where ax + by + cz + d = 0.
    pts : (N, 3) ndarray
        Points to project.

    Returns
    -------
    projected : (N, 3) ndarray
        Projected point coordinates.
    distance : (N,) ndarray
        Signed distance from each point to the plane.
    """
    plane = np.array(plane, dtype=pts.dtype).reshape(4)
    plane /= np.linalg.norm(plane[:3])  # normalize plane
    normal = plane[:3]
    d = plane[-1]

    offset = pts @ normal.reshape(3, 1) + d

    return pts - offset * normal, offset.flatten()


def fit_plane_to_points(pts: np.ndarray) -> np.ndarray:
    """
    Fit a plane to points using SVD.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Point coordinates.

    Returns
    -------
    plane : (4,) ndarray
        Plane coefficients [a, b, c, d] where ax + by + cz + d = 0.
    """
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    n = vh[2:3]                       # last row = normal
    d = -n @ centroid.reshape(3, 1)
    return np.concatenate((n, d), axis=1).flatten()


def fit_plane_to_noisy_points(
        pts: np.ndarray,
        threshold: float = 5.,
        iterations: int = 1000,
        target_vec: Optional[np.ndarray] = None,
        ang_thresh: float = 0.3,
        ) -> Tuple[np.ndarray, np.ndarray]:

    import open3d as o3d
    if target_vec is not None:
        target_vec = np.asarray(target_vec, dtype=np.float64)
        target_vec /= np.linalg.norm(target_vec)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    # Fast path: no angle constraint — single high-iteration RANSAC call
    if target_vec is None:
        plane_model, inlier_idx = pc.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=iterations * 5,
        )
        plane_model = np.asarray(plane_model)
        inlier_mask = np.zeros(len(pts), dtype=bool)
        inlier_mask[inlier_idx] = True
        return plane_model, inlier_mask

    best_plane = np.full(4, np.nan)
    best_qual = -1
    inlier_mask = np.zeros(len(pts), dtype=bool)

    for _ in range(iterations):
        plane_model, _ = pc.segment_plane(
            distance_threshold=threshold / 3,
            ransac_n=3,
            num_iterations=5,
        )
        plane_model = np.asarray(plane_model)
        normal = plane_model[:3]

        # enforce angle threshold
        theta = np.arccos(np.clip(np.dot(normal, target_vec), -1, 1))
        if theta > ang_thresh:
            continue

        # compute distance to test plane for all points, find inliers
        dist = np.abs(project_to_plane(plane_model, pts)[1])
        inliers = dist < threshold

        quality = np.sum(inliers) * np.exp(-np.mean(dist) / threshold)
        if quality > best_qual:
            best_qual = quality
            inlier_mask = inliers
            best_plane = plane_model

    return best_plane, inlier_mask


def _fit_sphere_4pts(p):
    """Fit sphere through 4 non-coplanar points via linear system."""
    A = 2 * (p[1:] - p[0])
    b = np.sum(p[1:]**2 - p[0]**2, axis=1)
    try:
        center = np.linalg.solve(A, b)
        radius = np.linalg.norm(p[0] - center)
        return center, radius
    except np.linalg.LinAlgError:
        return None, None


def fit_sphere_to_noisy_points_known_rad(
    pts: np.ndarray,
    target_radius: float,
    threshold: float = 15,
    iterations: int = 500,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """RANSAC sphere fit with a known target radius.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Point cloud.
    target_radius : float
        Expected sphere radius.
    threshold : float
        Inlier distance threshold.
    iterations : int
        Number of RANSAC iterations.

    Returns
    -------
    center : (3,) ndarray
        Best-fit sphere centre.
    radius : float
        Fitted radius.
    inliers : (N,) bool ndarray
        Inlier mask.
    """
    n_pts = len(pts)
    best_center, best_radius, best_qual, best_inliers = 0, 0, 0, np.zeros(0)
    for _ in range(iterations):
        idx = random.sample(range(n_pts), 4)
        center, radius = _fit_sphere_4pts(pts[idx])
        if center is None:
            continue

        pt_rad = np.linalg.norm(pts - center, axis=-1)
        inliers = pt_rad - target_radius < threshold
        mae = np.mean(np.abs(pt_rad - target_radius))

        quality = np.sum(inliers) * np.exp(-mae / threshold)
        if quality > best_qual:
            best_qual = quality
            best_center = center
            best_radius = radius
            best_inliers = inliers

    return best_center, best_radius, best_inliers


def fit_line_to_points(
    pts: np.ndarray, w: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a line through *pts* using SVD, optionally weighted.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Point cloud.
    w : (N,) ndarray, optional
        Per-point importance weights.

    Returns
    -------
    center : (1, 3) ndarray
        Centre of gravity along the line.
    direction : (1, 3) ndarray
        Unit direction vector.
    """
    if w is None:
        cog_w = np.mean(pts, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(pts - cog_w, full_matrices=False)
    else:
        # Weighted covariance via sqrt-weight scaling (avoids point duplication)
        w = w - np.min(w)
        w = w / (np.max(w) + 1e-12)
        w = w + 0.1  # ensure all points contribute at least a little
        sqrt_w = np.sqrt(w)[:, None]  # (N, 1)
        cog_w = np.average(pts, axis=0, weights=w).reshape(1, 3)
        _, _, vh = np.linalg.svd(sqrt_w * (pts - cog_w), full_matrices=False)
    major_axis = vh[0:1]              # first row = dominant direction

    return cog_w, major_axis


def fit_line_to_noisy_points(
    pts: np.ndarray,
    threshold: float = 2.,
    iterations: int = 500,
    angle: Optional[np.ndarray] = None,
    ang_thresh: float = 0.3,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """RANSAC line fit to noisy 3-D points.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Point cloud.
    threshold : float
        Radial inlier threshold.
    iterations : int
        Number of RANSAC iterations.
    angle : (3,) ndarray, optional
        Reference direction to constrain alignment.
    ang_thresh : float
        Maximum angular deviation (rad) from *angle*.

    Returns
    -------
    line : (point, direction)
        Best-fit line as (midpoint, unit direction).
    inliers : (N,) bool ndarray
        Inlier mask.
    """
    indexes = range(pts.shape[0])
    angle = None if angle is None else angle / np.linalg.norm(angle)

    max_sum = 0
    best_line = ([0, 0, 0], [0,0,0])
    best_inliers = [None]
    ii = 0
    while ii < iterations:

        idx_samples = random.sample(indexes, 2)
        ptA = pts[idx_samples[0]].reshape(1, 3)
        ptB = pts[idx_samples[1]].reshape(1, 3)
        vec = (ptB - ptA) / np.linalg.norm(ptB - ptA)

        # enforce angle threshold (if applicable)
        theta = None if angle is None else np.arccos(np.dot(vec, angle).clip(-1, 1))
        if theta is not None and theta > ang_thresh:
            continue

        # compute distance to test plane for all points, find inliers
        d = project_to_line((ptA, vec), pts)[1]
        inliers = d < threshold
        w = (1 - (d / threshold)).clip(0)
        weighted_sum = np.sum(w)

        if weighted_sum > max_sum:
            max_sum = weighted_sum
            best_inliers = inliers
            midpt = np.mean(pts[inliers], axis=0, keepdims=True)
            best_line = (midpt, vec)
        ii += 1

    return best_line, best_inliers


def reconstruct_from_distance(
    anchors: np.ndarray, distances: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    assert len(anchors) == len(distances)
    assert len(anchors) >= 2 * anchors.shape[1]

    pts_a, dist_a = anchors[::2], distances[::2]
    pts_b, dist_b = anchors[1::2], distances[1::2]
    A = pts_a - pts_b
    b = (dist_b**2 - dist_a**2 + np.sum(pts_a**2, axis=1) - np.sum(pts_b**2, axis=1)) / 2
    if weights is not None:
        w = weights.flatten()
        w = np.mean( np.c_[w[::2], w[1::2]], axis=-1)
        A = A * w.reshape(-1, 1)
        b = b * w
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x


def naive_fps(points: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    from bg3dtools.utils import farthest_point_sampling
    k = len(points) if k is None else k
    idx = farthest_point_sampling(points, k)
    return points[idx], idx


def smooth_curve(P: np.ndarray, s: float = 0.01, n_out: int = 1000) -> np.ndarray:
    """
    Smooth a 3-D polyline and resample it.

    Parameters
    ----------
    P : (17, 3) array_like
        Original points ordered along the curve.
    s : float, optional
        Smoothing factor (larger ⇒ smoother).  Rough guide:
            s ≈ 0      : exact interpolant (through all points)
            s ≈ 0.01   : slight smoothing (good default)
            s ≥ 0.1    : strong smoothing
    n_out : int, optional
        Number of output points on the smoothed curve.

    Returns
    -------
    C  : (n_out,3) – smoothed coordinates at those positions
    """
    P = np.asarray(P, dtype=float)
    nP = len(P)

    # 1. arclength parameter -----------------------------------------------
    seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1)
    S_raw   = np.concatenate(([0.0], np.cumsum(seg_len)))        # 17 values
    L       = S_raw[-1]                                          # total length

    # 2. smoothing splines ---------------------------------------------------
    # Scale s by total length so 's' is roughly dimensionless w.r.t. the data
    s_scaled = s * L
    spl_x = UnivariateSpline(S_raw, P[:, 0], s=s_scaled, k=3)
    spl_y = UnivariateSpline(S_raw, P[:, 1], s=s_scaled, k=3)
    spl_z = UnivariateSpline(S_raw, P[:, 2], s=s_scaled, k=3)

    # 3. resample ------------------------------------------------------------
    T = UnivariateSpline(np.arange(nP), S_raw, s=0.0, k=2)  # linear interpolation
    s = np.linspace(0.0, nP, n_out)                # 1-D grid
    S = T(s)
    C = np.column_stack((spl_x(S), spl_y(S), spl_z(S)))  # (n_out, 3)

    return C