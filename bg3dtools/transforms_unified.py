"""
Unified transforms module supporting both numpy and pytorch backends.

All functions handle arbitrary batch dimensions using [..., N] notation.
For example, twist_to_R accepts [..., 3] and returns [..., 3, 3].

Usage:
    import numpy as np
    from bg3dtools.transforms_unified import twist_to_R, make_aff

    # Numpy (default)
    R = twist_to_R(twist)  # [..., 3] -> [..., 3, 3]

    # PyTorch
    from bg3dtools.pytorch.backend import TorchBackend
    bk = TorchBackend()
    R = twist_to_R(twist, bk=bk)
"""
from __future__ import annotations
from typing import Union, Optional
import numpy as np

from bg3dtools.pytorch.backend import ArrayLike, infer_backend
EPS = 1e-7

__all__ = [
    "twist_to_R", "R_to_twist", "twist_to_quat",
    "quat_to_twist", "quat_to_R", "R_to_quat",
    "make_aff", "inverse", "inverse_rigid",
    "extract_R", "extract_twist", "extract_trans", "extract_params",
    "transform_points_forward", "transform_points_inverse",
    "rel_params_to_aff", "aff_to_rel_params",
    "rigid_reg",
    "affine_reg",
    "spherical_to_cartesian", "cartesian_to_spherical",
]


def twist_to_R(twist: ArrayLike, bk=None) -> ArrayLike:
    """
    Convert twist (axis-angle * magnitude) to rotation matrix.

    Args:
        twist: [..., 3] twist vectors
        bk: Backend (numpy or TorchBackend). Inferred from twist if None.

    Returns:
        R: [..., 3, 3] rotation matrices
    """
    if bk is None:
        bk = infer_backend(twist)

    orig_shape = twist.shape
    assert orig_shape[-1] == 3, f"twist must have shape [..., 3], got {orig_shape}"

    # Flatten batch dimensions
    twist = bk.reshape(twist, (-1, 3))
    n = twist.shape[0]

    # Compute angle (magnitude of twist)
    theta = bk.linalg.norm(twist, axis=-1, keepdims=True)  # [n, 1]
    theta = bk.clip(theta, EPS, None)  # Avoid division by zero

    # Normalize to get axis
    axis = twist / theta  # [n, 3]

    cos_t = bk.cos(theta)  # [n, 1]
    sin_t = bk.sin(theta)  # [n, 1]
    one_minus_cos = 1 - cos_t  # [n, 1]

    # Extract axis components
    vx = axis[:, 0:1]  # [n, 1]
    vy = axis[:, 1:2]
    vz = axis[:, 2:3]

    # Build rotation matrix using Rodrigues' formula
    # R = I + sin(t)*K + (1-cos(t))*K^2
    # where K is the skew-symmetric matrix of the axis
    R = bk.stack([
        vx*vx*one_minus_cos + cos_t,      vx*vy*one_minus_cos - vz*sin_t, vx*vz*one_minus_cos + vy*sin_t,
        vy*vx*one_minus_cos + vz*sin_t,   vy*vy*one_minus_cos + cos_t,    vy*vz*one_minus_cos - vx*sin_t,
        vz*vx*one_minus_cos - vy*sin_t,   vz*vy*one_minus_cos + vx*sin_t, vz*vz*one_minus_cos + cos_t
    ], axis=-1)  # [n, 9]

    R = bk.reshape(R, (-1, 3, 3))  # [n, 3, 3]

    # Restore original batch shape
    return bk.reshape(R, orig_shape[:-1] + (3, 3))


def R_to_twist(R: ArrayLike, bk=None) -> ArrayLike:
    """
    Convert rotation matrix to twist (axis-angle * magnitude).

    Args:
        R: [..., 3, 3] rotation matrices
        bk: Backend (numpy or TorchBackend). Inferred from R if None.

    Returns:
        twist: [..., 3] twist vectors
    """
    if bk is None:
        bk = infer_backend(R)

    orig_shape = R.shape
    assert orig_shape[-2:] == (3, 3), f"R must have shape [..., 3, 3], got {orig_shape}"

    # Flatten batch dimensions
    R = bk.reshape(R, (-1, 3, 3))

    # trace = 1 + 2*cos(theta) -> theta = arccos((trace-1)/2)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = bk.clip((trace - 1) / 2, -1.0, 1.0)
    theta = bk.arccos(cos_theta)  # [n]

    # Extract axis from skew-symmetric part: (R - R^T) / (2*sin(theta))
    sin_theta = bk.sin(theta)
    sin_theta = bk.clip(bk.abs(sin_theta), EPS, None)  # Avoid division by zero

    axis = bk.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], axis=-1) / (2 * sin_theta[:, None])  # [n, 3]

    # For small angles, use first-order approximation
    small_angle = theta < 1e-4
    if bk is np:
        if np.any(small_angle):
            axis[small_angle] = bk.stack([
                (R[small_angle, 2, 1] - R[small_angle, 1, 2]) / 2,
                (R[small_angle, 0, 2] - R[small_angle, 2, 0]) / 2,
                (R[small_angle, 1, 0] - R[small_angle, 0, 1]) / 2
            ], axis=-1)
    else:
        # Torch: use where
        axis_small = bk.stack([
            (R[:, 2, 1] - R[:, 1, 2]) / 2,
            (R[:, 0, 2] - R[:, 2, 0]) / 2,
            (R[:, 1, 0] - R[:, 0, 1]) / 2
        ], axis=-1)
        import torch
        axis = torch.where(small_angle[:, None], axis_small, axis)

    twist = axis * theta[:, None]

    # Restore original batch shape
    return bk.reshape(twist, orig_shape[:-2] + (3,))


def twist_to_quat(twist: ArrayLike, bk=None, center: bool = False) -> ArrayLike:
    """
    Convert twist (axis-angle * magnitude) to quaternion [x, y, z, w].

    Args:
        twist: [..., 3] twist vectors
        bk: Backend (numpy or TorchBackend). Inferred from twist if None.
        center: If True, subtract 1 from w component (centered quaternion).

    Returns:
        quat: [..., 4] quaternions in [x, y, z, w] format
    """
    if bk is None:
        bk = infer_backend(twist)

    orig_shape = twist.shape
    assert orig_shape[-1] == 3

    # Flatten batch dimensions
    twist = bk.reshape(twist, (-1, 3))

    theta = bk.linalg.norm(twist, axis=-1, keepdims=True)
    theta = bk.clip(theta, EPS, None)
    axis = twist / theta

    half_theta = theta / 2
    w = bk.cos(half_theta)
    if center:
        w = w - 1
    xyz = axis * bk.sin(half_theta)

    quat = bk.concatenate([xyz, w], axis=-1)  # [n, 4] in [x,y,z,w] format

    return bk.reshape(quat, orig_shape[:-1] + (4,))


def quat_to_twist(quat: ArrayLike, bk=None) -> ArrayLike:
    """
    Convert quaternion [x, y, z, w] to twist (axis-angle * magnitude).

    Args:
        quat: [..., 4] quaternions in [x, y, z, w] format (scalar-last)
        bk: Backend (numpy or TorchBackend). Inferred from quat if None.

    Returns:
        twist: [..., 3] twist vectors
    """
    if bk is None:
        bk = infer_backend(quat)

    orig_shape = quat.shape
    assert orig_shape[-1] == 4, f"quat must have shape [..., 4], got {orig_shape}"

    quat = bk.reshape(quat, (-1, 4))

    # Normalize
    qnorm = bk.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / bk.clip(qnorm, EPS, None)

    xyz = quat[:, :3]
    w = quat[:, 3:4]  # [n, 1]

    # Canonical form: ensure w >= 0 (q and -q represent same rotation)
    sign = bk.sign(w)
    # When w == 0, sign returns 0; treat as positive
    if bk is np:
        sign = np.where(sign == 0, 1.0, sign)
    else:
        import torch
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    xyz = xyz * sign
    w = w * sign

    half_theta = bk.arccos(bk.clip(w, -1.0, 1.0))  # [n, 1]
    theta = 2 * half_theta

    sin_half = bk.sin(half_theta)
    axis = xyz / bk.clip(sin_half, EPS, None)
    twist = axis * theta

    # Small-angle fallback: twist ≈ 2 * xyz (first-order Taylor)
    twist_small = 2 * xyz
    small_angle = (theta < 1e-4)  # [n, 1]

    if bk is np:
        twist = np.where(small_angle, twist_small, twist)
    else:
        import torch
        twist = torch.where(small_angle, twist_small, twist)

    return bk.reshape(twist, orig_shape[:-1] + (3,))


def quat_to_R(quat: ArrayLike, bk=None) -> ArrayLike:
    """
    Convert quaternion [x, y, z, w] to rotation matrix.

    Args:
        quat: [..., 4] quaternions in [x, y, z, w] format (scalar-last)
        bk: Backend (numpy or TorchBackend). Inferred from quat if None.

    Returns:
        R: [..., 3, 3] rotation matrices
    """
    if bk is None:
        bk = infer_backend(quat)

    orig_shape = quat.shape
    assert orig_shape[-1] == 4, f"quat must have shape [..., 4], got {orig_shape}"

    quat = bk.reshape(quat, (-1, 4))

    # Normalize
    qnorm = bk.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / bk.clip(qnorm, EPS, None)

    x = quat[:, 0:1]
    y = quat[:, 1:2]
    z = quat[:, 2:3]
    w = quat[:, 3:4]

    # Direct quaternion-to-rotation formula
    R = bk.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], axis=-1)  # [n, 9]

    R = bk.reshape(R, (-1, 3, 3))

    return bk.reshape(R, orig_shape[:-1] + (3, 3))


def R_to_quat(R: ArrayLike, bk=None) -> ArrayLike:
    """
    Convert rotation matrix to quaternion [x, y, z, w] using Shepperd's method.

    Numerically stable: always takes the sqrt of the largest discriminant.

    Args:
        R: [..., 3, 3] rotation matrices
        bk: Backend (numpy or TorchBackend). Inferred from R if None.

    Returns:
        quat: [..., 4] quaternions in [x, y, z, w] format, canonical (w >= 0)
    """
    if bk is None:
        bk = infer_backend(R)

    orig_shape = R.shape
    assert orig_shape[-2:] == (3, 3), f"R must have shape [..., 3, 3], got {orig_shape}"

    R = bk.reshape(R, (-1, 3, 3))
    n = R.shape[0]

    R00 = R[:, 0, 0]; R01 = R[:, 0, 1]; R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]; R11 = R[:, 1, 1]; R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]; R21 = R[:, 2, 1]; R22 = R[:, 2, 2]

    trace = R00 + R11 + R22

    # Four discriminants
    d_w = trace           # 1 + trace => w = sqrt(1+trace)/2
    d_x = 2*R00 - trace   # 1 + 2*R00 - trace => x = sqrt(1+2R00-trace)/2
    d_y = 2*R11 - trace
    d_z = 2*R22 - trace

    # Stack discriminants: [n, 4] in order [w, x, y, z]
    discs = bk.stack([d_w, d_x, d_y, d_z], axis=-1)  # [n, 4]

    # Compute all 4 candidate quaternions unconditionally.
    # Clamp discriminants >= 0 to avoid NaN in sqrt for unselected branches.
    # s = 2*sqrt(discriminant) = 4 * largest_component
    d_w_safe = bk.clip(1 + d_w, 0.0, None)
    d_x_safe = bk.clip(1 + d_x, 0.0, None)
    d_y_safe = bk.clip(1 + d_y, 0.0, None)
    d_z_safe = bk.clip(1 + d_z, 0.0, None)

    # Candidate 0: w is largest, s_w = 4w
    s_w = 2 * bk.sqrt(d_w_safe)
    s_w_safe = bk.clip(s_w, EPS, None)
    q_w = bk.stack([
        (R21 - R12) / s_w_safe,
        (R02 - R20) / s_w_safe,
        (R10 - R01) / s_w_safe,
        s_w / 4,
    ], axis=-1)  # [n, 4]

    # Candidate 1: x is largest, s_x = 4x
    s_x = 2 * bk.sqrt(d_x_safe)
    s_x_safe = bk.clip(s_x, EPS, None)
    q_x = bk.stack([
        s_x / 4,
        (R01 + R10) / s_x_safe,
        (R02 + R20) / s_x_safe,
        (R21 - R12) / s_x_safe,
    ], axis=-1)

    # Candidate 2: y is largest, s_y = 4y
    s_y = 2 * bk.sqrt(d_y_safe)
    s_y_safe = bk.clip(s_y, EPS, None)
    q_y = bk.stack([
        (R01 + R10) / s_y_safe,
        s_y / 4,
        (R12 + R21) / s_y_safe,
        (R02 - R20) / s_y_safe,
    ], axis=-1)

    # Candidate 3: z is largest, s_z = 4z
    s_z = 2 * bk.sqrt(d_z_safe)
    s_z_safe = bk.clip(s_z, EPS, None)
    q_z = bk.stack([
        (R02 + R20) / s_z_safe,
        (R12 + R21) / s_z_safe,
        s_z / 4,
        (R10 - R01) / s_z_safe,
    ], axis=-1)

    if bk is np:
        # Pick candidate with largest discriminant
        best = np.argmax(discs, axis=-1)  # [n]
        candidates = np.stack([q_w, q_x, q_y, q_z], axis=0)  # [4, n, 4]
        quat = candidates[best, np.arange(n)]  # [n, 4]
    else:
        import torch
        # torch.where chain for differentiability
        best = torch.argmax(discs, dim=-1)  # [n]
        mask_w = (best == 0).unsqueeze(-1)  # [n, 1]
        mask_x = (best == 1).unsqueeze(-1)
        mask_y = (best == 2).unsqueeze(-1)
        quat = torch.where(mask_w, q_w,
               torch.where(mask_x, q_x,
               torch.where(mask_y, q_y, q_z)))

    # Normalize
    qnorm = bk.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / bk.clip(qnorm, EPS, None)

    # Canonical form: w >= 0
    w = quat[:, 3:4]
    sign = bk.sign(w)
    if bk is np:
        sign = np.where(sign == 0, 1.0, sign)
    else:
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    quat = quat * sign

    return bk.reshape(quat, orig_shape[:-2] + (4,))


def make_aff(twist: Optional[ArrayLike], trans: Optional[ArrayLike], bk=None) -> ArrayLike:
    """
    Create homogeneous affine transformation matrix from twist and translation.

    Args:
        twist: [..., 3] twist vectors (can be None, defaults to zeros)
        trans: [..., 3] translation vectors (can be None, defaults to zeros)
        bk: Backend (numpy or TorchBackend). Inferred from inputs if None.

    Returns:
        aff: [..., 4, 4] homogeneous affine transformation matrices
    """
    if twist is None and trans is None:
        raise ValueError("At least one of twist or trans must be provided")

    if bk is None:
        bk = infer_backend(twist if twist is not None else trans)

    if twist is None:
        twist = bk.zeros_like(trans)
    if trans is None:
        trans = bk.zeros_like(twist)

    orig_shape = twist.shape
    assert orig_shape == trans.shape, f"twist and trans must have same shape, got {twist.shape} and {trans.shape}"
    assert orig_shape[-1] == 3

    # Get rotation matrices
    R = twist_to_R(twist, bk)  # [..., 3, 3]

    # Build 4x4 affine
    # Flatten for easier manipulation
    R_flat = bk.reshape(R, (-1, 3, 3))
    trans_flat = bk.reshape(trans, (-1, 3))
    n = R_flat.shape[0]

    # Create [R | t; 0 0 0 1] matrix
    if bk is np:
        aff = np.zeros((n, 4, 4), dtype=R.dtype)
        aff[:, :3, :3] = R_flat
        aff[:, :3, 3] = trans_flat
        aff[:, 3, 3] = 1
    else:
        import torch
        aff = torch.zeros((n, 4, 4), dtype=R.dtype, device=R.device)
        aff[:, :3, :3] = R_flat
        aff[:, :3, 3] = trans_flat
        aff[:, 3, 3] = 1

    return bk.reshape(aff, orig_shape[:-1] + (4, 4))


def inverse(aff: ArrayLike, bk=None) -> ArrayLike:
    """
    Compute inverse of affine transformation matrix.

    Args:
        aff: [..., 4, 4] affine transformation matrices
        bk: Backend (numpy or TorchBackend). Inferred from aff if None.

    Returns:
        aff_inv: [..., 4, 4] inverse affine transformation matrices
    """
    if bk is None:
        bk = infer_backend(aff)

    if bk is np:
        return np.linalg.inv(aff)
    else:
        import torch
        return torch.linalg.inv(aff)


def inverse_rigid(aff: ArrayLike, bk=None) -> ArrayLike:
    """
    Fast inverse of rigid affine transformation: [R|t] -> [R^T | -R^T @ t].

    Args:
        aff: [..., 4, 4] rigid affine transformation matrices
        bk: Backend (numpy or TorchBackend). Inferred from aff if None.

    Returns:
        aff_inv: [..., 4, 4] inverse affine transformation matrices
    """
    if bk is None:
        bk = infer_backend(aff)

    R = aff[..., :3, :3]
    t = aff[..., :3, 3:4]  # [..., 3, 1]

    R_T = bk.swapaxes(R, -1, -2)
    t_new = -R_T @ t

    if bk is np:
        last_row = np.array([0, 0, 0, 1], dtype=aff.dtype)
        last_row = np.broadcast_to(last_row, aff.shape[:-2] + (1, 4))
    else:
        import torch
        last_row = torch.tensor([0, 0, 0, 1], dtype=aff.dtype, device=aff.device)
        last_row = last_row.expand(aff.shape[:-2] + (1, 4))

    upper = bk.concatenate([R_T, t_new], axis=-1)
    return bk.concatenate([upper, last_row], axis=-2)


def extract_R(aff: ArrayLike) -> ArrayLike:
    """
    Extract rotation matrix from affine transformation.

    Args:
        aff: [..., 4, 4] affine transformation matrices
        bk: Backend. Inferred from aff if None.

    Returns:
        R: [..., 3, 3] rotation matrices
    """
    return aff[..., :3, :3]


def extract_twist(aff: ArrayLike, bk=None) -> ArrayLike:
    """
    Extract twist (axis-angle * magnitude) from affine transformation.

    Args:
        aff: [..., 4, 4] affine transformation matrices
        bk: Backend. Inferred from aff if None.

    Returns:
        twist: [..., 3] twist vectors
    """
    if bk is None:
        bk = infer_backend(aff)
    R = extract_R(aff)

    return R_to_twist(R, bk)


def extract_trans(aff: ArrayLike, bk=None) -> ArrayLike:
    """
    Extract translation from affine transformation.

    Args:
        aff: [..., 4, 4] affine transformation matrices
        bk: Backend. Inferred from aff if None.

    Returns:
        trans: [..., 3] translation vectors
    """
    if bk is None:
        bk = infer_backend(aff)
    return bk.copy(aff[..., :3, 3])


def extract_params(aff: ArrayLike, bk=None):
    """
    Extract twist and translation from affine transformation.

    Args:
        aff: [..., 4, 4] affine transformation matrices
        bk: Backend. Inferred from aff if None.

    Returns:
        (twist, trans): Tuple of [..., 3] arrays
    """
    if bk is None:
        bk = infer_backend(aff)
    return extract_twist(aff, bk), extract_trans(aff, bk)


def transform_points_forward(aff: ArrayLike, pts: ArrayLike, bk=None) -> ArrayLike:
    """
    Apply affine transformation to points.

    Args:
        aff: [..., 4, 4] affine transformation matrix
        pts: [..., N, 3] or [..., N, 4] points
        bk: Backend. Inferred from inputs if None.

    Returns:
        pts_transformed: [..., N, 3] or [..., N, 4] transformed points
    """
    if bk is None:
        bk = infer_backend(aff)

    if pts.shape[-1] == 4:
        # Already homogeneous: full matmul
        return pts @ bk.swapaxes(aff, -1, -2)

    # Fast path for 3D points: pts @ R.T + t  (avoids homogeneous padding)
    R = aff[..., :3, :3]
    t = aff[..., :3, 3]
    # Unsqueeze t so (B,3) becomes (B,1,3) — broadcasts with (B,N,3)
    return pts @ bk.swapaxes(R, -1, -2) + bk.expand_dims(t, axis=-2)


def transform_points_inverse(aff: ArrayLike, pts: ArrayLike, bk=None) -> ArrayLike:
    """
    Apply inverse affine transformation to points.

    Args:
        aff: [..., 4, 4] affine transformation matrix
        pts: [..., N, 3] or [..., N, 4] points
        bk: Backend. Inferred from inputs if None.

    Returns:
        pts_transformed: [..., N, 3] or [..., N, 4] transformed points
    """
    if bk is None:
        bk = infer_backend(aff)
    return transform_points_forward(inverse_rigid(aff, bk), pts, bk)


def rel_params_to_aff(trunk: list, rel_twist: ArrayLike, rel_trans: ArrayLike, bk=None):
    """
    Convert relative twist/translation parameters to absolute affine transforms
    for an articulated chain.

    Args:
        trunk: List of parent indices (length nJ), trunk[0] = -1 for root
        rel_twist: [..., nJ, 3] relative twist parameters (or None for zeros)
        rel_trans: [..., nJ, 3] relative translation parameters (or None for zeros)
        bk: Backend. Inferred from inputs if None.

    Returns:
        abs_affs: [..., nJ, 4, 4] absolute affine transforms
    """
    # Infer backend from whichever input is not None
    if bk is None:
        ref = rel_twist if rel_twist is not None else rel_trans
        bk = infer_backend(ref)

    nJ = len(trunk)

    # Handle None inputs by creating zeros with matching shape/dtype/device
    if rel_twist is None and rel_trans is None:
        raise ValueError("At least one of rel_twist or rel_trans must be provided")

    if rel_twist is not None:
        ref_array = rel_twist
        batch_shape = rel_twist.shape[:-2]
    else:
        ref_array = rel_trans
        batch_shape = rel_trans.shape[:-2]

    if rel_twist is None:
        if bk is np:
            rel_twist = np.zeros(batch_shape + (nJ, 3), dtype=ref_array.dtype)
        else:
            import torch
            rel_twist = torch.zeros(batch_shape + (nJ, 3), dtype=ref_array.dtype, device=ref_array.device)

    if rel_trans is None:
        if bk is np:
            rel_trans = np.zeros(batch_shape + (nJ, 3), dtype=ref_array.dtype)
        else:
            import torch
            rel_trans = torch.zeros(batch_shape + (nJ, 3), dtype=ref_array.dtype, device=ref_array.device)

    # Build affine transforms for each joint
    rel_affs = make_aff(rel_twist, rel_trans, bk)  # [..., nJ, 4, 4]

    # Compose transforms along kinematic chain (no in-place ops for autograd)
    if bk is np:
        abs_affs = rel_affs.copy()
        for jj in range(1, nJ):
            parent = trunk[jj]
            abs_affs[..., jj, :, :] = abs_affs[..., parent, :, :] @ rel_affs[..., jj, :, :]
    else:
        # Collect per-joint absolute transforms as a list, then stack
        abs_list = [rel_affs[..., 0, :, :]]  # root = identity compose
        for jj in range(1, nJ):
            parent = trunk[jj]
            abs_list.append(abs_list[parent] @ rel_affs[..., jj, :, :])
        import torch
        abs_affs = torch.stack(abs_list, dim=-3)  # [..., nJ, 4, 4]

    return abs_affs


def aff_to_rel_params(trunk: list, abs_affs: ArrayLike, bk=None):
    """
    Convert absolute affine transforms to relative twist/translation parameters.

    Args:
        trunk: List of parent indices (length nJ), trunk[0] = -1 for root
        abs_affs: [..., nJ, 4, 4] absolute affine transforms
        bk: Backend. Inferred from inputs if None.

    Returns:
        rel_twist: [..., nJ, 3] relative twist parameters
        rel_trans: [..., nJ, 3] relative translation parameters
    """
    if bk is None:
        bk = infer_backend(abs_affs)

    nJ = len(trunk)
    batch_shape = abs_affs.shape[:-3]

    # Initialize output arrays
    if bk is np:
        rel_twist = np.zeros(batch_shape + (nJ, 3), dtype=abs_affs.dtype)
        rel_trans = np.zeros(batch_shape + (nJ, 3), dtype=abs_affs.dtype)
    else:
        import torch
        rel_twist = torch.zeros(batch_shape + (nJ, 3), dtype=abs_affs.dtype, device=abs_affs.device)
        rel_trans = torch.zeros(batch_shape + (nJ, 3), dtype=abs_affs.dtype, device=abs_affs.device)

    # Root joint: relative = absolute
    rel_twist[..., 0, :], rel_trans[..., 0, :] = extract_params(abs_affs[..., 0, :, :], bk)

    # Other joints: relative = parent_inv @ absolute
    for jj in range(1, nJ):
        parent = trunk[jj]
        parent_inv = inverse_rigid(abs_affs[..., parent, :, :], bk)
        rel_aff = parent_inv @ abs_affs[..., jj, :, :]
        rel_twist[..., jj, :], rel_trans[..., jj, :] = extract_params(rel_aff, bk)

    return rel_twist, rel_trans


def rigid_reg(
    source: ArrayLike,
    dest: ArrayLike,
    scale: bool = False,
    return_aligned: bool = False,
    bk=None,
) -> Union[ArrayLike, tuple]:
    """
    Rigid registration (SVD-based point set alignment).

    Finds the rigid (or similarity) transform that best maps *source* onto
    *dest* in the least-squares sense.  Based on the algorithm described in:

        Arun, Huang, Blostein, "Least-Squares Fitting of Two 3-D Point Sets",
        IEEE TPAMI 9(5), 1987.

    Parameters
    ----------
    source : (N, 3) array
        Source points to transform.
    dest : (N, 3) array
        Destination points to align to.
    scale : bool, optional
        If True, allow uniform scaling. Default is False.
    return_aligned : bool, optional
        If True, also return the aligned source points. Default is False.
    bk : optional
        Backend (numpy or TorchBackend). Inferred from *source* if None.

    Returns
    -------
    aff : (4, 4) array
        Rigid (or similarity) affine transformation matrix.
    aligned : (N, 3) array, optional
        Aligned source points. Only returned when *return_aligned* is True.
    """
    if bk is None:
        bk = infer_backend(source)

    assert source.shape == dest.shape

    # Filter out rows containing NaN / inf in either set
    valid = bk.all(bk.isfinite(source), axis=1) & bk.all(bk.isfinite(dest), axis=1)
    src = source[valid]
    dst = dest[valid]

    assert len(src) >= 3, "Need at least 3 valid point pairs"

    # Shortcut: if points are already coincident, return identity
    if bk.all(bk.abs(src - dst) < EPS):
        if bk is np:
            aff = np.eye(4)
        else:
            import torch
            aff = torch.eye(4, dtype=source.dtype, device=source.device)
        if return_aligned:
            return aff, bk.copy(source)
        return aff

    # 1. Compute centroids and centre the clouds
    mu_src = bk.mean(src, axis=0)
    mu_dst = bk.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # 2. Cross-covariance matrix
    H = src_c.T @ dst_c  # (3, 3)

    # 3. SVD of H
    U, S, Vt = bk.linalg.svd(H)

    # 4. Optimal rotation  R = V @ diag(1,1,d) @ U^T
    #    where d = sign(det(V @ U^T)) ensures a proper rotation (det R = +1)
    d = bk.linalg.det(Vt.T @ U.T)
    if bk is np:
        sign_d = np.array([1.0, 1.0, np.sign(d)])
    else:
        import torch
        sign_d = torch.tensor([1.0, 1.0, torch.sign(d).item()],
                              dtype=source.dtype, device=source.device)
    R = (Vt.T * sign_d) @ U.T  # broadcasting the diagonal multiply

    # 5. Optional uniform scale factor  s = trace(R @ H) / ||src_c||^2
    if scale:
        s = bk.sum(S * sign_d) / bk.sum(src_c ** 2)
        R = s * R

    # 6. Translation  t = mu_dst - R @ mu_src
    t = mu_dst - R @ mu_src

    # 7. Assemble 4x4 affine
    if bk is np:
        aff = np.eye(4)
    else:
        import torch
        aff = torch.eye(4, dtype=source.dtype, device=source.device)
    aff[:3, :3] = R
    aff[:3, 3] = t

    if return_aligned:
        aligned = transform_points_forward(aff, source, bk=bk)
        return aff, aligned
    return aff


def affine_reg(
    source: ArrayLike,
    dest: ArrayLike,
    reg: float = 1e-6,
    return_aligned: bool = False,
    bk=None,
) -> Union[ArrayLike, tuple]:
    """
    Affine registration (least-squares, 9-DOF + translation).

    Finds the affine transform (per-axis scale, shear, rotation, translation)
    that best maps *source* onto *dest* in the least-squares sense via
    normal-equation solve.

    Parameters
    ----------
    source : (N, 3) array
        Source points to transform.
    dest : (N, 3) array
        Destination points to align to.
    reg : float, optional
        Tikhonov regularization strength. Default is 1e-6.
    return_aligned : bool, optional
        If True, also return the aligned source points. Default is False.
    bk : optional
        Backend (numpy or TorchBackend). Inferred from *source* if None.

    Returns
    -------
    aff : (4, 4) array
        Affine transformation matrix.
    aligned : (N, 3) array, optional
        Aligned source points. Only returned when *return_aligned* is True.
    """
    if bk is None:
        bk = infer_backend(source)

    assert source.shape == dest.shape

    # Filter out rows containing NaN / inf in either set
    valid = bk.all(bk.isfinite(source), axis=1) & bk.all(bk.isfinite(dest), axis=1)
    src = source[valid]
    dst = dest[valid]

    assert len(src) >= 4, "Need at least 4 valid point pairs"

    # Shortcut: if points are already coincident, return identity
    if bk.all(bk.abs(src - dst) < EPS):
        if bk is np:
            aff = np.eye(4)
        else:
            import torch
            aff = torch.eye(4, dtype=source.dtype, device=source.device)
        if return_aligned:
            return aff, bk.copy(source)
        return aff

    # 1. Compute centroids and centre the clouds
    mu_src = bk.mean(src, axis=0)
    mu_dst = bk.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # 2. Solve normal equations: (src_c.T @ src_c + reg*I) @ A.T = src_c.T @ dst_c
    if bk is np:
        lhs = src_c.T @ src_c + reg * np.eye(3)
    else:
        import torch
        lhs = src_c.T @ src_c + reg * torch.eye(3, dtype=source.dtype, device=source.device)
    rhs = src_c.T @ dst_c
    A = bk.linalg.solve(lhs, rhs).T  # (3, 3)

    # 3. Translation  t = mu_dst - A @ mu_src
    t = mu_dst - A @ mu_src

    # 4. Assemble 4x4 affine
    if bk is np:
        aff = np.eye(4)
    else:
        import torch
        aff = torch.eye(4, dtype=source.dtype, device=source.device)
    aff[:3, :3] = A
    aff[:3, 3] = t

    if return_aligned:
        aligned = transform_points_forward(aff, source, bk=bk)
        return aff, aligned
    return aff


def spherical_to_cartesian(spherical: ArrayLike, bk=None) -> ArrayLike:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    spherical : (2,) or (3,) array
        Spherical coordinates ``[theta, phi]`` or ``[theta, phi, rho]``.

        * *theta* — azimuthal angle (longitude, in the x-y plane from +x)
        * *phi* — polar angle (colatitude, from +z)
        * *rho* — radius (default 1)
    bk : optional
        Backend (numpy or TorchBackend). Inferred from *spherical* if None.

    Returns
    -------
    cartesian : (3,) array
        Cartesian coordinates ``[x, y, z]``.
    """
    if bk is None:
        bk = infer_backend(spherical)

    spherical = bk.reshape(spherical, (-1,))
    theta, phi = spherical[0], spherical[1]
    rho = spherical[2] if len(spherical) >= 3 else 1.0

    x = rho * bk.sin(phi) * bk.cos(theta)
    y = rho * bk.sin(phi) * bk.sin(theta)
    z = rho * bk.cos(phi)
    return bk.stack([x, y, z])


def cartesian_to_spherical(vec: ArrayLike, bk=None) -> ArrayLike:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    vec : (3,) or (N, 3) array
        Cartesian coordinates ``[x, y, z]``.
    bk : optional
        Backend (numpy or TorchBackend). Inferred from *vec* if None.

    Returns
    -------
    spherical : (3,) or (N, 3) array
        Spherical coordinates ``[theta, phi, rho]``.

        * *theta* — azimuthal angle (longitude)
        * *phi* — polar angle (colatitude)
        * *rho* — radius
    """
    if bk is None:
        bk = infer_backend(vec)

    squeeze = vec.ndim == 1
    vec = bk.reshape(vec, (-1, 3))

    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    rho = bk.linalg.norm(vec, axis=1)
    theta = bk.arctan2(y, x)
    phi = bk.arccos(bk.clip(z / bk.clip(rho, 1e-30, None), -1.0, 1.0))

    out = bk.column_stack([theta, phi, rho])
    if squeeze:
        out = out.ravel()
    return out
