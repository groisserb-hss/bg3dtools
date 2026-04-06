"""
PyTorch mesh operations.

This module provides differentiable mesh operations for PyTorch tensors,
including volume computation and normal calculations.
"""

import torch


def mesh_volume(verts: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Compute the signed volume of a closed triangle mesh.

    Uses Gauss' theorem (divergence theorem) to compute volume from
    surface normals and face centroids.

    Parameters
    ----------
    verts : (nV, 3) or (B, nV, 3) torch.Tensor
        Vertex coordinates, optionally batched.
    F : (nF, 3) torch.Tensor
        Triangle face indices.

    Returns
    -------
    volume : () or (B,) torch.Tensor
        Mesh volume(s). Scalar if unbatched, (B,) if batched.

    Notes
    -----
    The mesh should be watertight for accurate volume computation.
    Negative values indicate inverted normals.
    """
    batched = verts.ndim == 3

    if not batched:
        assert verts.ndim == 2
        verts = verts[None, :, :]

    # 3d coordinates of vertices for each face
    v0 = verts[:, F[:, 0], :]
    v1 = verts[:, F[:, 1], :]
    v2 = verts[:, F[:, 2], :]

    # compute barycenters
    center = (v0 + v1 + v2) / 3

    # normal for each face, scaled to area size
    FNdA = torch.cross(v1 - v0, v2 - v0, dim=-1) / 2

    # apply gauss' theorem
    volXYZ = torch.sum(center * FNdA, dim=1)

    volume = torch.mean(volXYZ, dim=-1)
    if not batched:
        volume = volume[0]

    return volume


def per_face_normals(
    verts: torch.Tensor,
    faces: torch.Tensor,
    eps: float = 0.000001,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute per-face normals for a triangle mesh.

    Parameters
    ----------
    verts : (..., nV, 3) torch.Tensor
        Vertex coordinates, optionally batched.
    faces : (nF, 3) torch.Tensor
        Triangle face indices.
    eps : float, optional
        Epsilon for numerical stability in normalization. Default is 1e-6.
    normalize : bool, optional
        If True, return unit normals. If False, return area-weighted normals.
        Default is True.

    Returns
    -------
    normals : (..., nF, 3) torch.Tensor
        Face normals with same batch dimensions as verts.

    Notes
    -----
    MPS devices use CPU fallback for cross product due to limitations.
    """
    v0 = verts[..., faces[:, 0], :]
    v1 = verts[..., faces[:, 1], :]
    v2 = verts[..., faces[:, 2], :]

    e1 = v1 - v0
    e2 = v2 - v0

    if verts.device.type == 'mps':
        # workaround for mps
        e1, e2 = e1.cpu(), e2.cpu()

    normals = torch.cross(e1, e2, dim=-1)

    if normalize:
        l = torch.linalg.norm(normals, ord=2, dim=-1, keepdim=True).clip(eps)
        normals = normals / l

    return normals.to(verts.device)


def per_vertex_normals(
    verts: torch.Tensor,
    faces: torch.Tensor,
    F2V: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-vertex normals by averaging face normals.

    Parameters
    ----------
    verts : (nV, 3) or (B, nV, 3) torch.Tensor
        Vertex coordinates, optionally batched.
    faces : (nF, 3) torch.Tensor
        Triangle face indices.
    F2V : (nV, nF) torch.Tensor
        Face-to-vertex adjacency matrix for averaging face normals
        onto vertices.

    Returns
    -------
    normals : (nV, 3) or (B, nV, 3) torch.Tensor
        Unit vertex normals with same batch dimensions as verts.
    """
    batched = verts.ndim > faces.ndim
    if not batched:
        verts = verts.unsqueeze(0)

    face_normals = per_face_normals(verts, faces)
    face_normals = face_normals.to(F2V.device)

    # average onto vertices
    posed_normals = torch.einsum('vf,bfd->bvd', F2V, face_normals)
    # normalize normals
    posed_normals = posed_normals / torch.linalg.norm(posed_normals, ord=2, dim=-1, keepdim=True)

    if not batched:
        posed_normals = posed_normals[0]

    return posed_normals.to(verts.device)

