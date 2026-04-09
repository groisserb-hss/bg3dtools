"""
Mesh registration algorithms.

This module provides iterative closest point (ICP) algorithms for
registering point clouds to triangle meshes, including both rigid
and non-rigid variants.
"""

import logging
import time
import random
from typing import Tuple, Optional

import numpy as np
import igl
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized
from scipy.spatial import KDTree

from bg3dtools.mesh.utils import (per_face_normals, per_vertex_normals,
                                     surface_sample, sample_E2V, row_normalize_csr,
                                     ordered_edges)
from bg3dtools.mesh.laplace import cotangent_weights
from bg3dtools.mesh.barycentric import points_to_barycentric, bc2sparse
from bg3dtools.mesh.distortion import normal_fold_score
from bg3dtools.utils import row_normalize, ConvergenceScheduler
from bg3dtools.transforms_unified import rigid_reg, affine_reg, transform_points_forward
from scipy.sparse import csr_matrix

__all__ = [
    "nonrigid_ICP",
    "discrete_match",
    "surface_match",
    "fit_vertices",
    "affine_ICP",
]

EPS = 0.0000001


def nonrigid_ICP(
    points: np.ndarray,
    faces: np.ndarray,
    modelV: np.ndarray,
    initV: Optional[np.ndarray] = None,
    pt_normals: Optional[np.ndarray] = None,
    smooth: bool = True,
    converge_thresh: float = 0.01,
    rad: float = 0.05,
    model_weight: float = 0.2,
    step_size: float = 1.0,
    discrete: bool = False,
    edges: Optional[np.ndarray] = None,
    landmark_targets: Optional[np.ndarray] = None,
    landmark_regressor: Optional[sparse.spmatrix] = None,
    landmark_weight: float = 0.0,
    landmark_confidence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Non-rigid ICP registration of a point cloud to a template mesh.

    Iteratively deforms the template mesh to match the point cloud while
    preserving mesh structure through edge-length regularization.

    Parameters
    ----------
    points : (nS, 3) ndarray
        Point cloud to register to.
    faces : (nF, 3) ndarray
        Triangle indices of the template mesh.
    modelV : (nV, 3) ndarray
        Template mesh vertex positions (regularization target).
    initV : (nV, 3) ndarray, optional
        Initial vertex positions. Default is modelV.
    pt_normals : (nS, 3) ndarray, optional
        Surface normals for the point cloud. Required if discrete=True.
    smooth : bool, optional
        If True, apply Laplacian smoothing each iteration. Default is True.
    converge_thresh : float, optional
        RMS error threshold for convergence. Default is 0.05.
    rad : float, optional
        Robust error kernel radius. Default is 0.05.
    model_weight : float, optional
        Weight for edge regularization term. Default is 1.0.
    discrete : bool, optional
        If True, use discrete point matching with normal similarity.
        Default is False.
    edges : (nE, 2) ndarray, optional
        Mesh edge indices. Computed from faces if not provided.
    landmark_targets : (nL, 3) ndarray, optional
        Target 3D landmark positions for sparse correspondences.
    landmark_regressor : (nL, nV) sparse matrix, optional
        Maps mesh vertices to landmark positions (e.g. SMPL-to-BlazePose).
    landmark_weight : float, optional
        Weight for landmark correspondence term. Default is 0.0 (disabled).
        At ``landmark_weight=1.0``, total landmark weight sums to ~0.5
        (scan weights sum to ~1.0).
    landmark_confidence : (nL,) ndarray, optional
        Per-landmark confidence in [0, 1]. Defaults to 1 for all landmarks.

    Returns
    -------
    fittedV : (nV, 3) ndarray
        Optimized vertex positions.
    err : float
        Final RMS registration error.

    Notes
    -----
    Uses a robust Geman-McClure-like weighting to handle outliers.
    The template mesh should have ~1.8x more faces than vertices
    for stable optimization.
    """
    log = logging.getLogger('nonrigid_reg')
    assert len(faces) > 1.8 * len(modelV)
    faces = faces.astype(np.int32)
    modelV = modelV.astype(np.float64)

    start_t = time.time()

    if discrete:
        assert pt_normals is not None
    if pt_normals is not None:
        assert np.all(pt_normals.shape == points.shape)
        pt_normals = row_normalize(pt_normals).astype(np.float64)
    if initV is not None:
        assert np.all(initV.shape == modelV.shape)
        initV = initV.astype(np.float64)

    if edges is None:
        edges = ordered_edges(faces)
    # edge_len = np.mean(np.linalg.norm(modelV[edges[:, 0]] - modelV[edges[:, 1]], axis=-1))
    nE = edges.shape[0]
    K = faces.shape[0] * 3
    nS = points.shape[0]

    # Precompute landmark base weights (GM modulation applied per iteration)
    use_landmarks = (landmark_weight > 0 and landmark_targets is not None
                     and landmark_regressor is not None)
    if use_landmarks:
        nL = landmark_targets.shape[0]
        if landmark_confidence is None:
            landmark_confidence = np.ones(nL, dtype=np.float64)
        # At landmark_weight=1, total landmark weight sums to ~0.5
        # (scan sums to ~1.0).  GM factor applied per iteration.
        lm_w_base = (landmark_weight * landmark_confidence / (2 * nL)).astype(np.float64)
        lm_regressor = landmark_regressor.astype(np.float64)
        lm_targets = landmark_targets.astype(np.float64)
    else:
        lm_w = lm_w_base = lm_regressor = lm_targets = None

    # print header
    log.debug('Starting nonrigid registration')

    fittedV = modelV if initV is None else initV
    face_normals = per_face_normals(modelV, faces)

    # Cotangent Laplacian on the undeformed model (fixed throughout).
    # (igl.cotmatrix is broken in igl 2.5.1, returns all zeros)
    model_cot_L = csr_matrix(-cotangent_weights(modelV, faces))

    # Implicit smoothing for edge targets: (I - t*L) @ smooth = disp.
    # Smoothed displacement captures body-part rotations but washes out
    # local distortion.
    nV = modelV.shape[0]
    # heavy-duty smoothing for edge weights
    heavy_smooth = factorized((sparse.eye(nV) - 1000 * model_cot_L).tocsc())
    if smooth:
        light_t = np.clip((1 / model_weight), 1.0, 100.)
        light_smooth = factorized((sparse.eye(nV) - light_t * model_cot_L).tocsc())

    assert 0 < step_size <= 1.0, 'Step size must be bewtween 0 and 1'

    reg_converge = ConvergenceScheduler(thresh=converge_thresh, window=3)
    while not reg_converge.complete:
        assert np.all(np.isfinite(fittedV)), 'non-finite vertex positions on iteration %d' % reg_converge.steps
        step_t = time.time()

        # subsample scan
        idx = np.arange(nS) if nS < K else np.array(random.sample(range(nS), K))
        subscan = points[idx]
        subnormals = None if pt_normals is None else pt_normals[idx]

        # match
        if discrete:
            d2, fidx, bc = discrete_match(subscan, subnormals, fittedV, faces, face_normals, radius=rad)
        else:
            d2, fidx, bc = surface_match(subscan, fittedV, faces)
        assert np.all(np.isfinite(d2)), 'distance calculation failed'

        # rms error for this step
        err = np.sqrt(np.mean(d2))

        # Smooth the displacement field to get edge targets that follow
        # body-part rotations but not local distortion (folds, stretching).
        disp = fittedV - modelV
        smooth_disp = np.column_stack([heavy_smooth(disp[:, i]) for i in range(3)])
        smoothV = modelV + smooth_disp

        # Uniform edge weight — smoothed targets handle rotation correctly,
        # so no need for distortion-adaptive weighting.
        edge_weight = np.full(nE, model_weight / nE)

        if discrete or pt_normals is None:
            # for discrete scan matching, similarity of normals has already been accounted for
            # in the nearest neighbor search
            nd = np.ones_like(d2)
        else:
            # weight scan distance by cosine similarity (between scan pts and matching surface points)
            nd = np.sum(subnormals * face_normals[fidx], axis=1) / 4 + .75

        # Geman-McClure robust weighting: w² * d² = d²/(rad²+d²).
        w = np.sqrt((nd / (rad ** 2 + d2)))
        scan_w = w * rad / K
        bcmap = bc2sparse(faces, fidx, bc)
        # scan_w *= bcmap @ vert_w

        # GM-modulate landmark weights (sigma = 2*rad, frozen denominator)
        if use_landmarks:
            current_lm = lm_regressor @ fittedV
            lm_d2 = np.sum((current_lm - lm_targets) ** 2, axis=1)
            lm_w = lm_w_base * np.sqrt(rad**2 / (rad**2 + lm_d2))

        # build new mesh based on closest points from scan to mesh
        newV = fit_vertices(fittedV, faces, smoothV, fidx, bc, subscan, edges, scan_w, edge_weight,
                            landmark_regressor=lm_regressor, landmark_targets=lm_targets, landmark_w=lm_w)
        fittedV = step_size * newV + (1 - step_size) * fittedV

        # Smooth the deformation from modelV (preserves volume — fingers
        # don't shrink); uniform rotations (arm pose change) are left alone.
        if smooth and reg_converge.steps > 0:
            disp = fittedV - modelV
            smooth_disp = np.column_stack([light_smooth(disp[:, i]) for i in range(3)])
            fittedV = modelV + smooth_disp

        log.debug('    step %d - time: %.2f , rms error = %.5f' % (reg_converge.steps, time.time() - step_t, err))
        reg_converge.push(err)

    log.debug('- Total time    : %.2f sec' % (time.time() - start_t))
    # scan = scatt(points, render=False)
    # mesh = trisurfsm(fittedV, faces, render=False)
    # draw_geometries([scan, mesh])

    return fittedV, err


def discrete_match(
    pts: np.ndarray,
    pt_normals: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    face_normals: np.ndarray,
    normal_weight: float = .5,
    radius: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match points to mesh using joint position-normal distance.

    Builds a 6D KDTree over ``[x, y, z, α*nx, α*ny, α*nz]`` where
    ``α = normal_weight * mean_edge_length`` so that a 90° normal
    mismatch costs roughly ``α`` in Euclidean distance.  A single
    nearest-neighbor query finds the best combined match.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Query points.
    pt_normals : (N, 3) ndarray
        Surface normals at query points.
    mesh_verts : (nV, 3) ndarray
        Mesh vertex positions.
    mesh_faces : (nF, 3) ndarray
        Mesh triangle indices.
    face_normals : (nF, 3) ndarray
        Per-face normals of the mesh.
    normal_weight : float
        Relative importance of normal alignment vs position.
        At 1.0, a 90° normal mismatch costs ~1 mean edge length.

    Returns
    -------
    d2 : (N,) ndarray
        Squared Euclidean distances to matched points (3D, not 6D).
    fidx : (N,) ndarray
        Face indices of matched triangles.
    bc : (N, 3) ndarray
        Barycentric coordinates within matched triangles.
    """
    res = 400 / np.linalg.norm(np.max(mesh_verts, axis=0) - np.min(mesh_verts, axis=0))
    sample_map, sampled_fidx, sampled_bc = surface_sample(
        mesh_verts, mesh_faces, N=9*len(pts), res=res
    )
    sampled_pts = sample_map @ mesh_verts
    sampled_normals = face_normals[sampled_fidx]

    # Scale factor:
    alpha = normal_weight * radius

    # Build 6D points: [x, y, z, α*nx, α*ny, α*nz]
    mesh_6d = np.hstack([sampled_pts, alpha * sampled_normals])
    query_6d = np.hstack([pts, alpha * pt_normals])

    tree = KDTree(mesh_6d)
    _, idx = tree.query(query_6d, 1)
    idx = idx.ravel()

    # Return 3D squared distance (not 6D)
    d2 = np.sum((pts - sampled_pts[idx]) ** 2, axis=-1)
    fidx = sampled_fidx[idx]
    bc = sampled_bc[idx]

    return d2, fidx, bc


def surface_match(
    pts: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find closest points on mesh surface.

    Uses libigl's point-to-mesh distance for efficient nearest-point queries.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Query points.
    mesh_verts : (nV, 3) ndarray
        Mesh vertex positions.
    mesh_faces : (nF, 3) ndarray
        Mesh triangle indices.

    Returns
    -------
    d2 : (N,) ndarray
        Squared distances to closest surface points.
    fidx : (N,) ndarray
        Face indices containing closest points.
    bc : (N, 3) ndarray
        Barycentric coordinates of closest points.

    Raises
    ------
    ValueError
        If distance calculation produces non-finite values.
    """
    # find the closest points on mesh
    d2, fidx, proj = igl.point_mesh_squared_distance(pts, mesh_verts, mesh_faces)
    if np.any(np.isfinite(d2) == False):
        raise ValueError('distance calculation failed')
    matched_faces = mesh_faces[fidx, :]
    bc = points_to_barycentric(mesh_verts[matched_faces, :], proj)

    return d2, fidx, bc


def fit_vertices(
    initV: np.ndarray,
    faces: np.ndarray,
    modelV: np.ndarray,
    fidx: np.ndarray,
    bc: np.ndarray,
    scanpts: np.ndarray,
    edges: np.ndarray,
    scan_w: np.ndarray,
    edge_w: np.ndarray,
    landmark_regressor: Optional[sparse.spmatrix] = None,
    landmark_targets: Optional[np.ndarray] = None,
    landmark_w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fit mesh vertices via weighted least squares.

    Solves for vertex positions that minimize a combination of:
    1. Point-to-surface distances (scan points to mesh)
    2. Edge length regularization (preserve template structure)
    3. Sparse landmark correspondences (optional)

    Parameters
    ----------
    initV : (nV, 3) ndarray
        Initial vertex positions for iterative solver.
    faces : (nF, 3) ndarray
        Triangle indices.
    modelV : (nV, 3) ndarray
        Template vertex positions (edge regularization target).
    fidx : (nS,) ndarray
        Face indices for scan point correspondences.
    bc : (nS, 3) ndarray
        Barycentric coordinates of correspondences.
    scanpts : (nS, 3) ndarray
        Target scan point positions.
    edges : (nE, 2) ndarray
        Mesh edge vertex indices.
    scan_w : (nS,) ndarray
        Per-point weights for data term.
    edge_w : (nE,) ndarray
        Per-edge weights for regularization term.
    landmark_regressor : (nL, nV) sparse matrix, optional
        Maps mesh vertices to landmark positions.
    landmark_targets : (nL, 3) ndarray, optional
        Target 3D landmark positions.
    landmark_w : (nL,) ndarray, optional
        Per-landmark weights.

    Returns
    -------
    fittedV : (nV, 3) ndarray
        Optimized vertex positions.

    Notes
    -----
    Uses sparse LSQR solver. On failure, saves debug data to /tmp/crashdump.npz.
    """

    nS = scanpts.shape[0]
    nV = modelV.shape[0]
    nE = edges.shape[0]

    # cartesian targets are weighted scan points
    weighted_targets = scanpts * scan_w[:, None]

    # construct sparse matrix mapping barycentric to cartesian coordinates
    sidx = np.tile(np.arange(nS), (3, 1)).T.flatten()
    vidx = faces[fidx, :].flatten()
    weighted_bc = (bc * scan_w[:,None]).flatten()
    vert_2_bc = sparse.csr_matrix((weighted_bc, (sidx, vidx)), (nS, nV))

    # target edge vectors
    weighted_edges = (modelV[edges[:, 0], :] - modelV[edges[:, 1], :]) * edge_w[:, None]

    # construct sparse matrix mapping edge vertices to edge vectors
    eidx = np.tile(np.arange(nE), (2,1)).T.flatten()
    vw = np.column_stack((edge_w, -edge_w)).flatten()
    vert_2_edge = sparse.csr_matrix((vw, (eidx, edges.flatten())), (nE, nV))

    # build sparse system
    blocks_A = [vert_2_bc, vert_2_edge]
    blocks_b = [weighted_targets, weighted_edges]

    # optional landmark rows: regressor @ V ≈ target
    if landmark_regressor is not None and landmark_targets is not None and landmark_w is not None:
        weighted_lm_targets = landmark_targets * landmark_w[:, None]
        weighted_regressor = sparse.diags(landmark_w) @ landmark_regressor
        blocks_A.append(weighted_regressor)
        blocks_b.append(weighted_lm_targets)

    # solve sparse linear equations
    A = sparse.vstack(blocks_A).tocsc()  # strangely, csc seems fastest
    b = np.row_stack(blocks_b)

    try:
        x = sparse.linalg.lsqr(A, b[:, 0], x0=initV[:, 0])[0]
        y = sparse.linalg.lsqr(A, b[:, 1], x0=initV[:, 1])[0]
        z = sparse.linalg.lsqr(A, b[:, 2], x0=initV[:, 2])[0]
        fittedV = np.column_stack((x, y, z))
        assert np.all(np.isfinite(fittedV))
    except Exception as e:
        np.savez('/tmp/crashdump.npz', initV=initV, faces=faces, modelV=modelV,
                 fidx=fidx, bc=bc, scanpts=scanpts, edges=edges, scan_w=scan_w, edge_w=edge_w)
        raise e
    
    return fittedV


def affine_ICP(
    points: np.ndarray,
    faces: np.ndarray,
    verts: np.ndarray,
    init_tform: Optional[np.ndarray] = None,
    pthresh: float = 95,
    max_iters: int = 100,
    scale: bool = False,
    affine: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ICP registration of a point cloud to a mesh.

    Iteratively finds the transformation (rigid, similarity, or affine)
    that aligns the point cloud to the mesh surface.

    Parameters
    ----------
    points : (N, 3) ndarray
        Point cloud to register.
    faces : (nF, 3) ndarray
        Mesh triangle indices.
    verts : (nV, 3) ndarray
        Mesh vertex positions.
    init_tform : (4, 4) ndarray, optional
        Initial transformation. Default is identity.
    pthresh : float, optional
        Final percentile threshold for outlier rejection (0-100).
        Threshold anneals from 100 to pthresh. Default is 95.
    max_iters : int, optional
        Maximum iterations. Default is 100.
    scale : bool, optional
        If True, allow uniform scaling (7-DOF similarity). Default is False.
        Ignored when *affine* is True.
    affine : bool, optional
        If True, solve full 9-DOF affine (per-axis scale + shear).
        Default is False.

    Returns
    -------
    tform : (4, 4) ndarray
        Optimal transformation matrix.
    regpoints : (N, 3) ndarray
        Transformed point cloud.

    Notes
    -----
    Uses adaptive outlier rejection that starts permissive and tightens
    to pthresh as iterations progress.
    """
    log = logging.getLogger('affine_ICP')
    assert 0 < pthresh < 100

    # initialize transform
    opt_tform = np.eye(4) if init_tform is None else init_tform
    regpoints = transform_points_forward(opt_tform, points)

    scheduler = ConvergenceScheduler(thresh=0.01, window=(max_iters//20), max_iter=max_iters)
    log.debug('Beginning rigid registration of point cloud TO surface mesh')
    log.debug('Iteration    rmsErr    robustD    inPoints')

    while not scheduler.complete:
        ii = scheduler.steps
        # find closest points on mesh using moved points
        d2, _, surf_points = igl.point_mesh_squared_distance(regpoints, verts, faces)

        # filter out points not close to surface
        prct = 100 * (1 - ii / max_iters) + pthresh * (ii / max_iters)
        d_thresh = max(np.percentile(d2, prct), EPS)
        idx = d2 < d_thresh
        surf_match = surf_points[idx]
        cloud_match = points[idx]
        err = np.sqrt(np.mean(d2[idx]))

        # find registration to close points
        if affine:
            opt_tform = affine_reg(cloud_match, surf_match)
        else:
            opt_tform = rigid_reg(cloud_match, surf_match, scale=scale)
        regpoints = transform_points_forward(opt_tform, points)

        # update scheduler
        scheduler.push(err)
        log.debug(' %d       %f     %f      % 9d ' % (ii, err, np.sqrt(d_thresh), len(surf_match)))

    return opt_tform, regpoints
