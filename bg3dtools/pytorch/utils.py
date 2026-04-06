"""
PyTorch utility functions.

This module provides helper functions for sparse matrix conversion
and numerical linear algebra operations in PyTorch.
"""

import logging
from typing import Optional
import numpy as np
import scipy.sparse
import torch


def sparse_to_tensor(
    array_sp: scipy.sparse.spmatrix,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    add_batch: bool = False
) -> torch.Tensor:
    """
    Convert a scipy sparse matrix to a PyTorch sparse COO tensor.

    Parameters
    ----------
    array_sp : scipy.sparse.spmatrix
        2D sparse matrix in any scipy sparse format.
    device : torch.device
        Target device for the tensor.
    dtype : torch.dtype, optional
        Data type for the tensor values.
    add_batch : bool, optional
        If True, add a batch dimension of size 1. Default is False.

    Returns
    -------
    tensor_sp : torch.sparse_coo_tensor
        PyTorch sparse COO tensor.

    Notes
    -----
    For MPS devices, the tensor is created on CPU due to sparse support limitations.
    """
    assert array_sp.ndim == 2

    coo = array_sp.tocoo()

    values = coo.data
    if add_batch:
        indices = np.vstack((np.zeros_like(coo.row), coo.row, coo.col))
    else:
        indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    #tensor_sp = torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))
    if device.type == 'mps':
        tensor_sp = torch.sparse_coo_tensor(i, v, array_sp.shape, device=torch.device('cpu'), dtype=dtype)
    else:
        tensor_sp = torch.sparse_coo_tensor(i, v, array_sp.shape, device=device, dtype=dtype)

    return tensor_sp


def solve_with_regularization(
    A: torch.Tensor,
    B: torch.Tensor,
    tolerance: float = 1e-8,
    left: bool = True
) -> torch.Tensor:
    """
    Solve linear system A * X = B with adaptive regularization.

    Automatically adds regularization (identity matrix scaling) when the
    condition number is too high, ensuring numerical stability.

    Parameters
    ----------
    A : (..., N, N) torch.Tensor
        Coefficient matrix. Last two dimensions must be square.
    B : (..., N, M) or (..., N) torch.Tensor
        Right-hand side matrix or vector.
    tolerance : float, optional
        Condition number threshold. Systems with condition number >= 1/tolerance
        are regularized. Default is 1e-8.
    left : bool, optional
        If True, solve A @ X = B. If False, solve X @ A = B. Default is True.

    Returns
    -------
    X : torch.Tensor
        Solution with same batch dimensions as inputs.

    Raises
    ------
    ValueError
        If A is not at least 2D or not square in last two dimensions.
        If regularization fails to reduce condition number.

    Notes
    -----
    Uses adaptive epsilon starting at 1e-5 and growing exponentially until
    the condition number is acceptable.
    """
    log = logging.getLogger('solve_with_regularization')

    # inputs must be on the same device
    assert A.device == B.device
    # inputs must be finite
    assert torch.all(torch.isfinite(A))
    assert torch.all(torch.isfinite(B))
    assert tolerance > 0

    # Ensure A is at least 2D
    if A.dim() < 2:
        raise ValueError("A must be at least 2D")

    # Ensure A is square in the last two dimensions
    if A.size(-2) != A.size(-1):
        raise ValueError("Last two dimensions of A must be square")

    # Calculate the condition number of A in the batched manner
    # Apply regularization
    q99 = torch.quantile(torch.abs(A), 0.99)
    identity_matrix = torch.eye(A.size(-2), dtype=A.dtype, device=A.device).expand_as(A)
    cond_number = torch.tensor(float('inf'), device=A.device, dtype=A.dtype)
    epsilon, step = 0., 0

    while torch.any(cond_number >= 1/tolerance):
        if epsilon > 1e-3:
            log.warning(f"Step {step}: epsilon is {epsilon}")
            A_debug = torch.reshape(A_regularized, (-1, A.size(-2), A.size(-1)))
            for i in range(len(A_debug)):
                try:
                    # torch.cuda.synchronize()
                    cond_i = torch.linalg.cond(A_debug[i])
                    # torch.cuda.synchronize()
                except torch._C._LinAlgError:
                    cond_i = torch.tensor(float('inf'), device=A.device, dtype=A.dtype)
                    # torch.cuda.synchronize()
                if cond_i >= 1/tolerance:
                    log.warning(f"Condition number {cond_i} at index {i}")
                    log.warning(A_debug[i].detach().cpu().numpy())
            raise ValueError("Regularization failed to reduce condition number")

        # only use regularization where cond_number >= 1/tolerance
        A_regularized = torch.clip(A, -2 * q99, 2 * q99)
        A_regularized = (A_regularized + epsilon * identity_matrix) / (1 + epsilon)
        good_A = cond_number < 1/tolerance
        A_regularized = torch.where(good_A.expand_as(A), A, A_regularized)

        assert torch.all(torch.isfinite(A_regularized)), "step %d: Regularized matrix has %d NaNs" % (step, torch.isnan(A_regularized).sum())
        try:
            # torch.cuda.synchronize()
            cond_number = torch.linalg.cond(A_regularized)
            # torch.cuda.synchronize()
        except torch._C._LinAlgError:
            cond_number = torch.tensor(float('inf'), device=A.device, dtype=A.dtype)
        epsilon = np.e * epsilon + 1e-5
        step += 1

    # If the condition number is not too large, try direct solve
    return torch.linalg.solve(A_regularized, B, left=left)

    # # Use SVD or pseudoinverse for a robust solution
    # if torch.all(cond_number < 1/tolerance):
    #     U, S, Vh = torch.linalg.svd(A_regularized)
    #     # Zero out very small singular values for stability
    #     S = torch.where(S < tolerance, torch.tensor(0.0, device=S.device, dtype=S.dtype), S)
    #     S_inv = torch.diag_embed(1.0 / S)
    #     A_inv = torch.matmul(torch.matmul(Vh.mH, S_inv), U.mH)
    #     X = torch.matmul(A_inv, B)
    # else:
    #     # If still ill-conditioned, use pseudoinverse
    #     X = torch.linalg.pinv(A_regularized) @ B

