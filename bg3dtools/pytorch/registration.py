"""
Non-rigid ICP registration using PyTorch.

This module provides differentiable non-rigid iterative closest point (NRICP)
registration for deforming template meshes to match point cloud data. Uses
PyTorch for gradient-based optimization with LBFGS.

Functions
---------
nonrigid_ICP
    Batch non-rigid registration of template mesh to point clouds.
batch_nricp
    Single batch optimization step for NRICP.
"""

import logging

import torch
from torch import Tensor
import numpy as np
import scipy.sparse as sparse
from scipy.spatial import KDTree
from typing import Optional, Tuple

from torch.optim import LBFGS
from tqdm import tqdm

from bg3dtools.utils import row_normalize, ConvergenceScheduler
from bg3dtools.mesh.utils import face_2_vertex_map, per_vertex_normals, per_vertex_smoothing, sample_E2V, ordered_edges
from bg3dtools.mesh.laplace import cotangent_weights
from bg3dtools.pytorch.utils import sparse_to_tensor
from bg3dtools.pytorch import device
from bg3dtools.pytorch.mesh import per_vertex_normals as torch_per_vertex_normals
from bg3dtools.render.trimesh import trisurfsm, scatt, draw_geometries, scatts


log = logging.getLogger('torch_NRICP')
search_w = 0.5  # weight for normals


def nonrigid_ICP(
    pts: list, faces: np.ndarray, modelV: np.ndarray, timepoints: list,
    initV: Optional[np.ndarray] = None, smooth_weight: float = 1.0,
    pt_normals: Optional[list] = None, converge_thresh: float = 0.05,
    rad: float = 0.025, E2V: Optional[sparse.spmatrix] = None,
    batch_size: int = 48,
) -> np.ndarray:

    # print header
    log.debug('Starting batch nonrigid registration')
    # check sizes
    nS, nV, dims = initV.shape
    assert len(pts) == len(pt_normals) == len(timepoints) == nS
    if np.isscalar(smooth_weight):
        smooth_weight = np.ones(nV) * smooth_weight
    assert len(modelV) == len(smooth_weight) == nV

    # convert to float32
    nS = len(pts)
    # convert to float32
    pts = [p.astype(np.float32) for p in pts]
    lowres = sum([len(p) < 4 * len(modelV) for p in pts])
    if lowres:
        log.warning('%d/%d low resolution point clouds' % (lowres, len(pts)))
    pt_normals = [row_normalize(n.astype(np.float32)) for n in pt_normals]
    scan_trees = [KDTree(np.column_stack([p, search_w * n])) for p, n in zip(pts, pt_normals)]

    fittedV = np.array(initV, dtype=np.float32)  # make copy

    # convert to torch tensors
    templateV = torch.tensor(modelV, dtype=torch.float32, device=device)
    # laplacian matrix used to enforce smoothness
    # (igl.cotmatrix is broken in igl 2.5.1, returns all zeros)
    L = sparse_to_tensor(-cotangent_weights(modelV, faces), device, dtype=torch.float32).to_sparse_csr()
    # sparse mapping from faces to vertices
    F2V = sparse_to_tensor(face_2_vertex_map(modelV, faces), device, dtype=torch.float32).to_sparse_csr()
    # sparse mapping from edges to vertices
    if E2V is None:
        edges = ordered_edges(faces)
        E2V = sample_E2V(edges, modelV)[0]
    E2V = sparse_to_tensor(E2V, device, dtype=torch.float32).to_sparse_csr()

    window_size = (7 * batch_size) // 8
    for ii in tqdm(range(0, nS, window_size), 'Batch ICP; window size %d' % window_size):
        batch_idx = np.arange(ii, min(ii+batch_size, nS))
        batch_trees = [scan_trees[i] for i in batch_idx]
        batch_timepoints = [timepoints[i] for i in batch_idx]
        batch_fittedV = fittedV[batch_idx]
        fittedV[batch_idx], _ = batch_nricp(faces, templateV, L, F2V, smooth_weight,
                                            batch_trees, batch_fittedV, batch_timepoints,
                                            rad, converge_thresh, E2V=E2V)

    return fittedV


def batch_nricp(faces: np.ndarray, templateV: Tensor,
                L: Tensor, F2V: Tensor, smooth_weight: np.ndarray,
                scan_trees: list, fittedV: np.ndarray, timepoints: list,
                rad: float, thresh: float, E2V: Tensor) -> Tuple[np.ndarray, float]:

    nS, nV, dims = fittedV.shape
    # convert to torch tensors
    fittedV = torch.tensor(fittedV, dtype=torch.float32, device=device, requires_grad=True)
    timepoints = torch.tensor(timepoints, dtype=torch.float32, device=device)
    smooth_weight = torch.tensor(smooth_weight, dtype=torch.float32, device=F2V.device)  # nV
    f_tensor = torch.from_numpy(faces)

    # optimizer = Adam([fittedV], lr=0.01)
    optimizer = LBFGS([fittedV], lr=5., max_iter=30, max_eval=1000, line_search_fn="strong_wolfe")
    reg_converge = ConvergenceScheduler(thresh=thresh, window=3)
    while not reg_converge.complete:

        # point matching performed on the CPU
        # mostly because I don't know a good library for this in pytorch
        posed_verts = fittedV.detach().cpu().numpy()
        # if reg_converge.steps % 10 == 0:
        smooth_verts = [per_vertex_smoothing(v, faces) for v in posed_verts]
        posed_normals = np.stack([per_vertex_normals(v, faces) for v in smooth_verts], axis=0)

        match_pts = np.empty([nS, nV, 3], dtype=np.float32)
        match_weights = np.empty([nS, nV], dtype=np.float32)
        # get closest points on the mesh
        for ss, (tree, verts, normals) in enumerate(zip(scan_trees, posed_verts, posed_normals)):
            m = np.column_stack((verts, search_w * normals))
            d, idx = tree.query(m, k=5)  # outputs are [nV x 5]
            match_m = tree.data[idx]  # [nV x 5 x 3]
            # softmax weights
            match_w = np.exp(-d / rad)
            match_w /= np.sum(match_w, axis=1, keepdims=True)
            # weighted average
            match_pts[ss] = np.sum(match_w[:, :, None] * match_m[:, :, :3], axis=1)  # [nV x 3]
            match_weights[ss] = np.sum(d**2, axis=1)

        # convert to torch tensors
        match_pts = torch.tensor(match_pts, dtype=torch.float32, device=device)
        match_weights = np.exp(-match_weights / rad)
        match_weights = torch.tensor(match_weights, dtype=torch.float32, device=device)

        # Following function is for pytorch optimization; gradients required
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            offset = match_pts - fittedV
            posed_normals = torch_per_vertex_normals(fittedV, f_tensor, F2V)

            dist_loss = 0.1 * get_distance_loss(offset, match_weights, rad)
            plane_loss = 10. * get_projected_loss(offset, posed_normals, match_weights, rad)
            smooth_loss = 1. * get_smoothness_loss(100*fittedV, 100*templateV, L, smooth_weight)
            continuity_loss = 100. * get_continuity_loss(fittedV, timepoints) if nS > 2 else 0

            loss = dist_loss + plane_loss + smooth_loss + continuity_loss #+ normal_loss

            if loss.requires_grad:
                loss.backward()
            return loss

        # a single step requires several calls to closure
        loss = optimizer.step(closure)
        reg_converge.push(loss.item())

    return fittedV.detach().cpu().numpy(), reg_converge.current


# --- Loss functions ---


def get_distance_loss(offset: Tensor, w: Tensor, sigma: float) -> Tensor:
    """Geman-McClure robust distance loss."""
    d2 = torch.sum(offset**2, dim=-1)
    gm = d2 / (sigma**2 + d2)
    return torch.mean(gm * w)


def get_projected_loss(offset: Tensor, v_normal: Tensor, w: Tensor, sigma: float) -> Tensor:
    """Robust distance projected along vertex normals."""
    d2 = torch.sum(offset * v_normal, dim=-1)**2  # distance (squared) along normal
    gm = d2 / (sigma**2 + d2)  # robust distance measure
    return torch.mean(gm * w)


def get_normal_loss(posed_normals: Tensor, match_normals: Tensor) -> Tensor:
    """Robust normal alignment loss (1 - dot product)."""
    c = 1 - torch.sum(posed_normals * match_normals, dim=-1)  # ranges from 0 to 2
    gm = c / (0.1 + c)
    return torch.mean(gm)


def get_smoothness_loss(verts: Tensor, modelV: Tensor, L: Tensor, w: Tensor) -> Tensor:
    """Laplacian smoothness penalty on vertex displacements."""
    if verts.ndim > modelV.ndim:
        assert verts.ndim == modelV.ndim + 1
        modelV = modelV.unsqueeze(0)
    trans = verts - modelV

    trans = trans.to(L.device)
    # Batch sparse matmul: (nV, nV) @ (nS, nV, 3).permute -> (nS, nV, 3)
    # torch sparse @ dense broadcasts over leading dims when transposed
    Lt = (L @ trans.permute(1, 0, 2).reshape(trans.shape[1], -1)).reshape(trans.shape[1], trans.shape[0], trans.shape[2]).permute(1, 0, 2)
    diff_v = w[:, None] * Lt

    loss_sm = torch.mean(diff_v**2) + torch.mean(diff_v**4)
    return loss_sm.to(verts.device)


def get_continuity_loss(verts: Tensor, timepoints: Tensor) -> Tensor:
    """Temporal continuity loss penalising deviation from linear interpolation."""

    # enforce smooth transformation in time domain
    a = timepoints[2:] - timepoints[:-2]
    b = timepoints[1:-1] - timepoints[:-2]

    p = (b / a).reshape([-1, 1, 1])  # interpolation factor; higher means closer to second timepoint
    v_in = (1 - p) * verts[:-2] + p * verts[2:]

    diff_v = verts[1:-1] - v_in
    d = torch.minimum(b, a-b)  # weight based on temporal proximity to neighboring scans
    d = torch.sigmoid(4 - d/50).reshape([-1, 1, 1])  # sigmoid to ignore scans more than 200 ms apart
    return torch.mean((d * diff_v)**2)
