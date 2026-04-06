"""
Point cloud registration algorithms.

This module provides iterative closest point (ICP) registration for aligning
point clouds using rigid transformations.
"""

import logging
import numpy as np
from bg3dtools.transforms_unified import transform_points_forward, rigid_reg
from bg3dtools.utils import ConvergenceScheduler
from scipy.spatial import KDTree

__all__ = [
    "pc_icp",
]


def pc_icp(dst_pts, src_pts, dst_normals=None, src_normals=None, init_tform=None,
              normal_weight=1.0, pthresh=None, dthresh=None, max_iters=100, return_transformed=False):
    """
    Rigid ICP registration of source points to destination points.

    Iteratively finds correspondences and computes optimal rigid transforms
    until convergence. Optionally uses surface normals for improved matching.

    Parameters
    ----------
    dst_pts : (N, 3) ndarray
        Target point cloud coordinates.
    src_pts : (M, 3) ndarray
        Source point cloud coordinates to transform.
    dst_normals : (N, 3) ndarray, optional
        Surface normals for target points.
    src_normals : (M, 3) ndarray, optional
        Surface normals for source points.
    init_tform : (4, 4) ndarray, optional
        Initial transformation matrix. Default is identity.
    normal_weight : float, optional
        Weight for normal matching in correspondence search. Default is 1.0.
    pthresh : float, optional
        Percentile threshold (0-100) for outlier rejection. Cannot be used
        with dthresh.
    dthresh : float, optional
        Distance threshold for outlier rejection. Cannot be used with pthresh.
    max_iters : int, optional
        Maximum number of iterations. Default is 100.
    return_transformed : bool, optional
        If True, also return transformed source points. Default is False.

    Returns
    -------
    tform : (4, 4) ndarray
        Optimal rigid transformation matrix.
    regpoints : (M, 3) ndarray, optional
        Transformed source points. Only returned if return_transformed=True.
    """
    log = logging.getLogger('rigid_ICP')
    assert (pthresh is None) or (dthresh is None), 'Cannot specify both pthresh and dthresh'
    assert pthresh is None or (0 < pthresh <= 100)
    assert dthresh is None or dthresh > 0

    use_normals = (dst_normals is not None) and (src_normals is not None)

    # initialize transform
    opt_tform = np.eye(4) if init_tform is None else init_tform
    regpoints = transform_points_forward(opt_tform, src_pts)
    if use_normals:
        regnorms = src_normals @ opt_tform[:3, :3].T  # (M @ normals.T).T == (normals @ M.T)

    # construct KDTree
    target = np.column_stack([dst_pts, normal_weight * dst_normals]) if use_normals else dst_pts
    target = KDTree(target)

    scheduler = ConvergenceScheduler(thresh=0.01, window=(max_iters//20), max_iter=max_iters)
    log.debug('Beginning rigid registration of point cloud TO surface mesh')
    log.debug('Iteration    rmsErr    robustD    inPoints')

    while not scheduler.complete:
        ii = scheduler.steps
        # find closest points on mesh using moved points
        moving = np.column_stack([regpoints, normal_weight * regnorms]) if use_normals else regpoints
        _, idx = target.query(moving, k=1)
        targ_match = dst_pts[idx]
        d2 = np.sum((targ_match - regpoints)**2, axis=1)

        # filter out points not close to surface
        # prct = 100 * (1 - ii / max_iters) + pthresh * (ii / max_iters)
        d_thresh = np.percentile(d2, pthresh) if dthresh is None else dthresh
        mask = d2 < (d_thresh + 1e-6)
        targ_match = targ_match[mask]
        cloud_match = src_pts[mask]
        err = np.sqrt(np.mean(d2[mask]))

        # find rigid registration to close points
        opt_tform = rigid_reg(cloud_match, targ_match)
        regpoints = transform_points_forward(opt_tform, src_pts)
        if use_normals:
            regnorms = src_normals @ opt_tform[:3, :3].T  # (M @ normals.T).T == (normals @ M.T)

        # update scheduler
        scheduler.push(err)
        log.debug(' %d       %f     %f      % 9d ' %
                  (ii, err, np.sqrt(d_thresh), np.count_nonzero(mask)))

    if return_transformed:
        return opt_tform, regpoints

    return opt_tform