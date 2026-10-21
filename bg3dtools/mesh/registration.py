"""
Mesh registration algorithms.

This module provides iterative closest point (ICP) algorithms for
registering point clouds to triangle meshes, including both rigid
and non-rigid variants.
"""

import logging
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized, splu
from scipy.spatial import KDTree

from bg3dtools.mesh.utils import (per_face_normals, per_vertex_normals,
                                     surface_sample, stratified_surface_sample, sample_E2V,
                                     row_normalize_csr, ordered_edges)
from bg3dtools.igl_compat import point_mesh_squared_distance
from bg3dtools.mesh.laplace import cotangent_weights
from bg3dtools.mesh.barycentric import points_to_barycentric, bc2sparse
from bg3dtools.mesh.distortion import normal_fold_score
from bg3dtools.utils import row_normalize, ConvergenceScheduler
from bg3dtools.transforms_unified import rigid_reg, affine_reg, transform_points_forward
from scipy.sparse import csr_matrix

__all__ = [
    "NRICPResult",
    "nonrigid_ICP",
    "discrete_match",
    "stratified_candidate_count",
    "surface_match",
    "transfer_vertex_field",
    "fit_vertices",
    "affine_ICP",
]

EPS = 0.0000001

_SOLVERS = ('lsqr', 'direct')
_SAMPLERS = ('voxel', 'stratified')


@dataclass(frozen=True)
class NRICPResult:
    """What :func:`nonrigid_ICP` returns with ``return_stats=True``: the fit and its iteration record.

    ``verts`` and ``err`` are exactly the two values of the plain return; ``n_iters`` is the number of
    vertex solves performed and the arrays hold one entry per iteration — the RMS of the matched
    distances (measured before that iteration's solve, so ``errors[-1] == err``), the wall seconds of
    the iteration, of its correspondence phase (candidate sampling, tree, query) and of its vertex
    solve.
    """
    verts: np.ndarray
    err: float
    n_iters: int
    errors: np.ndarray
    iter_s: np.ndarray
    match_s: np.ndarray
    solve_s: np.ndarray
    total_s: float


def _surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    """The mesh's total area -- the sum of its lumped vertex areas, Voronoi or barycentric alike."""
    v = np.asarray(verts, np.float64)
    e1, e2 = v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())


def equivalent_membrane(verts: np.ndarray, faces: np.ndarray, time: float) -> float:
    """The :func:`nonrigid_ICP` membrane width (in the mesh's units) as stiff as the post-solve Laplace filter
    ``(I - time * L)^-1`` it replaced on 2026-10-20 (``L`` the cotangent Laplacian, unnormalised, so *time* is in
    mesh rings): that filter is the minimiser of ``|d - d0|^2 + time * d' (-L) d``, a membrane of width^2 =
    ``time`` times the vertex area against data of unit weight per vertex, so the width is
    ``sqrt(time * mean vertex area)``, the mean being the surface area over the vertex count. For a caller
    migrating from a filter time it had no data to tune: on STAR (m) time 1, 4, 5 give about 16, 32, 36 mm."""
    return float(np.sqrt(time * _surface_area(verts, faces) / len(verts)))


def nonrigid_ICP(
    points: np.ndarray,
    faces: np.ndarray,
    modelV: np.ndarray,
    initV: Optional[np.ndarray] = None,
    pt_normals: Optional[np.ndarray] = None,
    converge_thresh: float = 0.01,
    rad: float = 0.05,
    model_weight: float = 0.2,
    step_size: float = 1.0,
    discrete: bool = False,
    edges: Optional[np.ndarray] = None,
    edge_weights: Optional[np.ndarray] = None,
    landmark_targets: Optional[np.ndarray] = None,
    landmark_regressor: Optional[sparse.spmatrix] = None,
    landmark_weight: float = 0.0,
    landmark_confidence: Optional[np.ndarray] = None,
    cosine_data_weight: bool = False,
    *,
    window: int = 3,
    max_iter: Optional[int] = None,
    solver: str = 'direct',
    sampler: str = 'stratified',
    sample_scale: float = 1.0,
    rng: Optional[Union[int, np.random.Generator, np.random.SeedSequence]] = None,
    return_stats: bool = False,
    membrane: Optional[float] = None,
) -> Union[Tuple[np.ndarray, float], NRICPResult]:
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
    membrane : float, optional
        A membrane term in each solve's energy, ``lam * d' (-L) d`` on the displacement ``d`` from the model
        (the squared gradient of the displacement, integrated; ``L`` the model's cotangent Laplacian), with
        ``lam = membrane^2 * rho`` and ``rho`` the data term's nominal weight per unit area (the subsample's
        points at weight ``1/K`` each, over the model's area). *membrane* is a physical width IN THE MESH'S
        UNITS (m for a body model, mm for a mesh in mm): a fully observed offset at wavenumber k is fitted by
        ``1 / (1 + membrane^2 k^2)``, so offsets much wider than it (a rib hump) are taken up and jags much
        narrower are not. Every iterate -- the result too -- minimises data + edges + membrane: there is no
        smoothing between the solves, and the result is the last solve. The membrane charges any gradient of
        the displacement, a rotation or shear of the registration against the model too, so it holds vertices
        the scan cannot pin (a slide around a skull under hair; the edge term's targets follow such
        rotations). Where the data is weak (holes, matches far off under the robust kernel) it takes over.
        None (default): edges only. Needs ``solver='direct'``. Until 2026-10-20 the regularisation was a
        Laplace filter applied after each solve and to the result (time ``clip(1 / model_weight, 1, 100)``,
        ``smooth=True``); :func:`equivalent_membrane` converts such a time to a width.
    converge_thresh : float, optional
        Relative change of the windowed RMS error below which the iteration stops. Default is 0.01.
    rad : float, optional
        Robust error kernel radius. Default is 0.05.
    model_weight : float, optional
        Weight for edge regularization term. Default is 0.2.
    discrete : bool, optional
        If True, use discrete point matching with normal similarity.
        Default is False.
    edges : (nE, 2) ndarray, optional
        Mesh edge indices. Computed from faces if not provided.
    edge_weights : (nE,) ndarray, optional
        Per-edge regularization weights. If given, used as-is (length must
        match ``edges``), letting the caller control the relative strength of
        different edge groups (e.g. face edges vs. internal struts). If None
        (default), every edge gets the uniform weight ``model_weight / nE`` —
        reproducing the original behavior exactly.
    landmark_targets : (nL, 3) ndarray, optional
        Target 3D positions to anchor toward. Treated as known anchors, so the
        term is a plain **L2** pull (constant weight, no robust down-weighting).
    landmark_regressor : (nL, nV) sparse matrix, optional
        Maps mesh vertices to landmark positions (e.g. SMPL-to-BlazePose, or an
        identity to anchor each vertex to its own target position).
    landmark_weight : float, optional
        Weight for the L2 anchor term. Default is 0.0 (disabled).
        At ``landmark_weight=1.0``, total landmark weight sums to ~0.5
        (scan weights sum to ~1.0); per-anchor weight scales as ``1/(2*nL)``.
    landmark_confidence : (nL,) ndarray, optional
        Per-landmark confidence in [0, 1]. Defaults to 1 for all landmarks.
    window : int, optional
        Window of the convergence test (:class:`ConvergenceScheduler`): the last ``window`` errors
        against the ``window`` before them, so at least ``2 * window`` iterations run. Default 3.
    max_iter : int, optional
        At most this many vertex solves; None (default) runs until the window converges.
    solver : {'direct', 'lsqr'}, optional
        How :func:`fit_vertices` solves the vertex system: the normal equations with one sparse
        factorisation (default since 2026-10-16: exact, 20-30x faster on a body-sized mesh) or three
        LSQR runs (``'lsqr'``, the behaviour before).
    sampler : {'stratified', 'voxel'}, optional
        How :func:`discrete_match` draws the candidate points on the mesh (``discrete=True`` only):
        an area-stratified draw (default since 2026-10-16) or a random draw thinned to a voxel grid
        (``'voxel'``, the behaviour before, ~10-30 % more candidates); both are redrawn every iteration.
    sample_scale : float, optional
        The stratified sampler's candidate count relative to one per ``1/res``-sided surface cell
        (``res`` as in :func:`discrete_match`); the voxel sampler keeps 10-30 % more. Default 1.
    rng : int, Generator or SeedSequence, optional
        The generator behind the scan subsample and the stratified candidate draw. None (default)
        draws fresh entropy. A run is reproducible from a seed only with ``sampler='stratified'``:
        the voxel sampler draws inside libigl, which cannot be seeded from here.
    return_stats : bool, optional
        Return an :class:`NRICPResult` (the fit plus its iteration record) instead of the tuple.

    Returns
    -------
    fittedV : (nV, 3) ndarray
        Optimized vertex positions.
    err : float
        Final RMS registration error.
    With ``return_stats=True`` a single :class:`NRICPResult` instead, whose ``verts`` and ``err``
    are these two values.

    Notes
    -----
    Uses a robust Geman-McClure-like weighting to handle outliers.
    The template mesh should have ~1.8x more faces than vertices
    for stable optimization.
    """
    log = logging.getLogger('nonrigid_reg')
    if solver not in _SOLVERS:
        raise ValueError(f"solver must be one of {_SOLVERS}, got {solver!r}")
    if sampler not in _SAMPLERS:
        raise ValueError(f"sampler must be one of {_SAMPLERS}, got {sampler!r}")
    if window < 1:
        raise ValueError(f'window must be >= 1, got {window}')
    if max_iter is not None and max_iter < 1:
        raise ValueError(f'max_iter must be >= 1 or None, got {max_iter}')
    if membrane is not None and not membrane > 0:
        raise ValueError(f'membrane must be a width > 0 or None, got {membrane}')
    if membrane is not None and solver != 'direct':
        raise ValueError(f"membrane needs solver='direct', got {solver!r}")
    rng = np.random.default_rng(rng)
    assert len(faces) > 1.8 * len(modelV)
    faces = faces.astype(np.int32)
    modelV = modelV.astype(np.float64)

    start_t = time.perf_counter()
    errors, iter_s, match_s, solve_s = [], [], [], []

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
    if edge_weights is not None:
        edge_weights = np.asarray(edge_weights, dtype=np.float64)
        assert edge_weights.shape == (nE,), \
            'edge_weights must have one weight per edge (%d), got %s' % (nE, edge_weights.shape)
    K = faces.shape[0] * 3
    nS = points.shape[0]

    # Precompute landmark weights.  Landmarks are ANCHORS — the target is a position you KNOW
    # (e.g. a boundary vertex pinned to a tidal-average location, or a vertex held near its init),
    # so the loss is plain L2 with a CONSTANT weight (no robust/Geman-McClure down-weighting; that
    # would weaken exactly the anchors that drift, defeating the purpose).  The robust kernel stays on
    # the *scan* data term only.
    use_landmarks = (landmark_weight > 0 and landmark_targets is not None
                     and landmark_regressor is not None)
    if use_landmarks:
        nL = landmark_targets.shape[0]
        if landmark_confidence is None:
            landmark_confidence = np.ones(nL, dtype=np.float64)
        # At landmark_weight=1, total landmark weight sums to ~0.5 (scan sums to ~1.0).
        lm_w = lm_w_base = (landmark_weight * landmark_confidence / (2 * nL)).astype(np.float64)
        lm_regressor = landmark_regressor.astype(np.float64)
        lm_targets = landmark_targets.astype(np.float64)
    else:
        lm_w = lm_w_base = lm_regressor = lm_targets = None

    # print header
    log.debug('Starting nonrigid registration')

    fittedV = modelV if initV is None else initV
    face_normals = per_face_normals(modelV, faces)

    # Cotangent Laplacian on the undeformed model (fixed throughout).
    # Negated: cotangent_weights uses the opposite sign convention, so this equals
    # igl_compat.cotmatrix (to 1e-15) where that works: libigl 2.5.1 under numpy 2 returns
    # EMPTY sparse matrices (cotmatrix, massmatrix; seen 2026-10-20 in the screws310 env), so
    # nothing here takes a sparse matrix from igl -- see igl_compat.massmatrix's Notes.
    model_cot_L = csr_matrix(-cotangent_weights(modelV, faces))

    # Implicit smoothing for edge targets: (I - t*L) @ smooth = disp.
    # Smoothed displacement captures body-part rotations but washes out
    # local distortion.
    nV = modelV.shape[0]
    # heavy-duty smoothing for edge weights
    heavy_smooth = factorized((sparse.eye(nV) - 1000 * model_cot_L).tocsc())
    # The membrane per scan point of the subsample: each point weighs 1/K^2 nominally, so the data's weight per
    # area is n / (K^2 * area) and the membrane's weight membrane^2 times that.
    membrane_unit = None
    if membrane is not None:
        membrane_unit = (-membrane ** 2 / (K * K * _surface_area(modelV, faces)) * model_cot_L).tocsr()

    assert 0 < step_size <= 1.0, 'Step size must be bewtween 0 and 1'

    reg_converge = ConvergenceScheduler(thresh=converge_thresh, window=window)
    while not reg_converge.complete and (max_iter is None or reg_converge.steps < max_iter):
        assert np.all(np.isfinite(fittedV)), 'non-finite vertex positions on iteration %d' % reg_converge.steps
        step_t = time.perf_counter()

        # subsample scan
        idx = np.arange(nS) if nS < K else rng.choice(nS, K, replace=False)
        subscan = points[idx]
        subnormals = None if pt_normals is None else pt_normals[idx]

        # match
        if discrete:
            d2, fidx, bc = discrete_match(subscan, subnormals, fittedV, faces, face_normals, radius=rad,
                                          sampler=sampler, sample_scale=sample_scale, rng=rng)
        else:
            d2, fidx, bc = surface_match(subscan, fittedV, faces)
        assert np.all(np.isfinite(d2)), 'distance calculation failed'
        match_s.append(time.perf_counter() - step_t)

        # rms error for this step
        err = np.sqrt(np.mean(d2))

        # Smooth the displacement field to get edge targets that follow
        # body-part rotations but not local distortion (folds, stretching).
        disp = fittedV - modelV
        smooth_disp = np.column_stack([heavy_smooth(disp[:, i]) for i in range(3)])
        smoothV = modelV + smooth_disp

        # Uniform edge weight — smoothed targets handle rotation correctly,
        # so no need for distortion-adaptive weighting.  A caller-supplied
        # edge_weights overrides this (e.g. to weight internal struts
        # separately from face edges without diluting either group).
        edge_weight = np.full(nE, model_weight / nE) if edge_weights is None else edge_weights

        if pt_normals is None:
            nd = np.ones_like(d2)
        elif discrete and not cosine_data_weight:
            # default discrete path: normal similarity already used in the 6D NN search; no extra weight.
            nd = np.ones_like(d2)
        elif discrete and cosine_data_weight:
            # OPT-IN: even after the 6D match, down-weight correspondences whose scan normal disagrees
            # with the matched surface — a "toxic" point (e.g. arm/background forced onto the wrong
            # surface) contributes little no matter how good its least-bad match was. Raised-cosine
            # (Hann) taper over the normal-disagreement angle θ ∈ [0°, 135°]: ½(1+cos(θ·4/3)),
            # flat-zero past 135°. → 1 at 0°, 0.75 at 45°, 0.25 at 90°, ~0 at 120°.
            cos = np.clip(np.sum(subnormals * face_normals[fidx], axis=1), -1.0, 1.0)
            theta = np.arccos(cos)
            nd = 0.5 * (1.0 + np.cos(np.minimum(theta * (4.0 / 3.0), np.pi)))
        else:
            # continuous (surface_match) path with normals — original gentle cosine weight.
            nd = np.sum(subnormals * face_normals[fidx], axis=1) / 4 + .75

        # Geman-McClure robust weighting: w² * d² = d²/(rad²+d²).
        w = np.sqrt((nd / (rad ** 2 + d2)))
        scan_w = w * rad / K
        bcmap = bc2sparse(faces, fidx, bc)
        # scan_w *= bcmap @ vert_w

        # Landmark weight is constant L2 (lm_w = lm_w_base, set once above) — no per-iteration
        # robust re-weighting; the anchor pull grows with displacement so anchors actually hold.

        # build new mesh based on closest points from scan to mesh
        solve_t = time.perf_counter()
        newV = fit_vertices(fittedV, faces, smoothV, fidx, bc, subscan, edges, scan_w, edge_weight,
                            landmark_regressor=lm_regressor, landmark_targets=lm_targets, landmark_w=lm_w,
                            solver=solver, reg_op=None if membrane_unit is None else len(subscan) * membrane_unit,
                            reg_target=modelV)
        solve_s.append(time.perf_counter() - solve_t)
        fittedV = step_size * newV + (1 - step_size) * fittedV

        iter_s.append(time.perf_counter() - step_t)
        errors.append(err)
        log.debug('    step %d - time: %.2f (match %.2f, solve %.2f), rms error = %.5f'
                  % (reg_converge.steps, iter_s[-1], match_s[-1], solve_s[-1], err))
        reg_converge.push(err)

    log.debug('- Total time    : %.2f sec' % (time.perf_counter() - start_t))

    result = NRICPResult(verts=fittedV, err=err, n_iters=len(errors), errors=np.asarray(errors),
                         iter_s=np.asarray(iter_s), match_s=np.asarray(match_s), solve_s=np.asarray(solve_s),
                         total_s=time.perf_counter() - start_t)
    return result if return_stats else (result.verts, result.err)


def stratified_candidate_count(verts: np.ndarray, faces: np.ndarray, res: float,
                               sample_scale: float = 1.0) -> int:
    """How many candidates the stratified sampler draws for a mesh at resolution ``res``: one per
    ``1/res``-sided cell of surface area, ``sample_scale`` times that, at least 1."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces)
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()
    return max(1, int(round(sample_scale * area * res ** 2)))


def discrete_match(
    pts: np.ndarray,
    pt_normals: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    face_normals: np.ndarray,
    normal_weight: float = .5,
    radius: float = 1.0,
    *,
    sampler: str = 'stratified',
    sample_scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match points to mesh using joint position-normal distance.

    Builds a 6D KDTree over ``[x, y, z, α*nx, α*ny, α*nz]`` where
    ``α = normal_weight * radius`` so that a 90° normal mismatch costs
    roughly ``α`` in Euclidean distance.  A single nearest-neighbor query
    finds the best combined match. The candidates are points drawn on the
    mesh surface afresh on every call, about one per ``1/res``-sided
    surface cell with ``res = 400 / bbox_diagonal``.

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
        At 1.0, a 90° normal mismatch costs ~1 ``radius``.
    radius : float
        The robust radius of the registration this match serves; it scales
        the normal term (see ``normal_weight``).
    sampler : {'stratified', 'voxel'}
        ``'stratified'`` (default since 2026-10-16): ``stratified_surface_sample``
        draws :func:`stratified_candidate_count` candidates with per-face counts
        proportional to face area, seedable through ``rng``.
        ``'voxel'`` (the behaviour before): ``surface_sample`` draws 9 candidates
        per query point at random and thins them to one per voxel of side
        ``1/res`` — the same density, with a ``np.unique`` that was the cost.
    sample_scale : float
        Multiplier on the stratified candidate count (the voxel sampler keeps
        10-30 % more than one per cell, oblique surfaces cutting more voxels).
    rng : numpy Generator, optional
        The stratified draw's generator (None: fresh entropy).

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
    if sampler == 'voxel':
        sample_map, sampled_fidx, sampled_bc = surface_sample(
            mesh_verts, mesh_faces, N=9*len(pts), res=res
        )
    elif sampler == 'stratified':
        n = stratified_candidate_count(mesh_verts, mesh_faces, res, sample_scale)
        sample_map, sampled_fidx, sampled_bc = stratified_surface_sample(
            mesh_verts, mesh_faces, n, np.random.default_rng(rng)
        )
    else:
        raise ValueError(f"sampler must be one of {_SAMPLERS}, got {sampler!r}")
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
    d2, fidx, proj = point_mesh_squared_distance(pts, mesh_verts, mesh_faces)
    if np.any(np.isfinite(d2) == False):
        raise ValueError('distance calculation failed')
    matched_faces = mesh_faces[fidx, :]
    bc = points_to_barycentric(mesh_verts[matched_faces, :], proj)

    return d2, fidx, bc


def transfer_vertex_field(
    src_verts: np.ndarray,
    src_field: np.ndarray,
    dst_pts: np.ndarray,
    max_dist: Optional[float] = None,
    fill=0,
) -> np.ndarray:
    """Carry a per-vertex field from one mesh onto the points of another.

    Each destination point takes the value of its nearest *source vertex*.
    Correspondence is by position alone, so the two meshes must already be in a
    common frame and in the same units; nothing here aligns them.

    Nearest-vertex, not nearest-surface-point: it needs no source topology and
    is what a discrete label or mask wants, since interpolating a label across a
    triangle is not meaningful. For a continuous field where the source
    tessellation is coarse relative to the destination, prefer `surface_match`
    plus barycentric interpolation, which does not quantise to source vertices.

    Parameters
    ----------
    src_verts : (nS, 3) ndarray
        Source vertex positions.
    src_field : (nS,) or (nS, C) ndarray
        Values carried by the source vertices. Any dtype, including bool.
    dst_pts : (nD, 3) ndarray
        Destination points — typically another mesh's vertices.
    max_dist : float, optional
        Destination points further than this from every source vertex take
        `fill` instead of a transferred value. Without it every point takes a
        value however far away its match is, which silently paints geometry the
        source never covered. Same units as the coordinates.
    fill : scalar
        Value for points beyond `max_dist`. Defaults to 0, which is False for a
        boolean field.

    Returns
    -------
    values : (nD,) or (nD, C) ndarray
        Same trailing shape and dtype as `src_field`.

    Examples
    --------
    Move a boolean region from a registration onto a raw scan::

        scan_mask = transfer_vertex_field(reg_verts, reg_mask, scan_verts,
                                          max_dist=0.08, fill=False)
    """
    src_verts = np.asarray(src_verts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    field = np.asarray(src_field)
    if len(field) != len(src_verts):
        raise ValueError(f"src_field has {len(field)} rows, src_verts has {len(src_verts)}")

    dist, idx = KDTree(src_verts).query(dst_pts, k=1)
    out = field[idx]
    if max_dist is not None:
        too_far = dist > max_dist
        if field.ndim > 1:
            too_far = too_far[:, None]
        out = np.where(too_far, np.asarray(fill, dtype=field.dtype), out)
    return out.astype(field.dtype, copy=False)


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
    *,
    solver: str = 'direct',
    reg_op: Optional[sparse.spmatrix] = None,
    reg_target: Optional[np.ndarray] = None,
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
    solver : {'direct', 'lsqr'}
        ``'lsqr'`` (the behaviour before 2026-10-16): three LSQR runs on the same ``A`` (one per
        coordinate), each started from ``initV`` and stopped at LSQR's default tolerances.
        ``'direct'`` (default): the normal equations
        ``(AᵀA + λI) x = Aᵀb + λ initV``, one sparse LU factorisation (SuperLU) solving the three
        right-hand sides at once, with ``λ`` a ridge of 1e-9 of the mean diagonal — far below the
        rounding of the solve on every coordinate the data or an edge determines, and the tie-break
        on any it does not (a vertex of an edge-graph component with no scan point and no landmark):
        such a vertex stays at ``initV``, which is what LSQR's warm start does there. The exact
        least-squares minimiser otherwise (the two agree to ~0.01-0.04 mm on a body-sized mesh, LSQR
        being the approximate one) at 20-30x less time.
    reg_op : (nV, nV) sparse matrix, optional
        A quadratic term ``(V - reg_target)' reg_op (V - reg_target)`` per coordinate, added to the normal
        equations (:func:`nonrigid_ICP`'s membrane: the weighted ``-L``).
        Direct solver only.
    reg_target : (nV, 3) ndarray, optional
        What *reg_op* measures the displacement from (required with it).

    Returns
    -------
    fittedV : (nV, 3) ndarray
        Optimized vertex positions.

    Notes
    -----
    On failure, saves debug data to /tmp/crashdump.npz and re-raises. An unknown ``solver`` raises
    ``ValueError`` before anything is built.
    """
    if solver not in _SOLVERS:
        raise ValueError(f"solver must be one of {_SOLVERS}, got {solver!r}")
    if reg_op is not None and solver != 'direct':
        raise ValueError(f"reg_op needs solver='direct', got {solver!r}")
    if reg_op is not None and reg_target is None:
        raise ValueError('reg_op needs reg_target')

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
    b = np.vstack(blocks_b)

    try:
        if solver == 'lsqr':
            x = sparse.linalg.lsqr(A, b[:, 0], x0=initV[:, 0])[0]
            y = sparse.linalg.lsqr(A, b[:, 1], x0=initV[:, 1])[0]
            z = sparse.linalg.lsqr(A, b[:, 2], x0=initV[:, 2])[0]
            fittedV = np.column_stack((x, y, z))
        else:
            # the normal equations: one factorisation of the (nV x nV) sparse matrix, three
            # right-hand sides; A changes with the correspondences every iteration, so nothing is
            # kept between calls. The ridge toward initV decides only the coordinates nothing else
            # determines (see the docstring); it is 1e-9 of the diagonal everywhere else.
            AtA = (A.T @ A).tocsc()
            Atb = A.T @ b
            if reg_op is not None:
                AtA = (AtA + reg_op).tocsc()
                Atb = Atb + reg_op @ reg_target
            lam = 1e-9 * AtA.diagonal().mean()
            fittedV = splu((AtA + lam * sparse.eye(nV, format='csc')).tocsc()).solve(Atb + lam * initV)
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
        d2, _, surf_points = point_mesh_squared_distance(regpoints, verts, faces)

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
