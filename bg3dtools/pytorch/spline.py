"""
Differentiable natural cubic spline interpolation in PyTorch.

This module provides batched natural cubic spline interpolation with
autodiff support. Useful for smooth curve fitting through control points.

Classes
-------
NaturalCubicSpline
    Evaluator for pre-computed spline coefficients.

Functions
---------
natural_cubic_spline_coeffs
    Compute spline coefficients from knot positions and values.
"""

# ---- plain-PyTorch natural cubic spline (batched) -------------------------
# x: (B, J) strictly increasing parameter (e.g., normalized arclength in [0,1])
# y: (B, J, D) anchors (D=2 for your YZ use case)
# t: (L,) or (B, L) query parameters in [0,1]
import torch

def natural_cubic_spline_coeffs(x: torch.Tensor, y: torch.Tensor):
    """
    Returns (x, y, M) where M are second derivatives at knots, solving the
    natural spline system. All ops are differentiable.
    """
    assert x.ndim == 2 and y.ndim == 3 and x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]
    B, J, D = y.shape
    device, dtype = y.device, y.dtype

    # spacings
    h = (x[:, 1:] - x[:, :-1]).clamp_min(1e-8)  # (B, J-1)

    # Build banded system A m = rhs for second derivatives m (natural BCs: m0=m_{J-1}=0)
    A = torch.zeros(B, J, J, dtype=dtype, device=device)
    rhs = torch.zeros(B, J, D, dtype=dtype, device=device)
    A[:, 0, 0] = 1.0
    A[:, -1, -1] = 1.0

    # Vectorized tridiagonal fill for interior knots
    idx = torch.arange(1, J - 1, device=device)
    hi_1 = h[:, :-1]  # (B, J-2) — h[i-1] for i=1..J-2
    hi = h[:, 1:]      # (B, J-2) — h[i] for i=1..J-2
    A[:, idx, idx - 1] = hi_1
    A[:, idx, idx]     = 2.0 * (hi_1 + hi)
    A[:, idx, idx + 1] = hi
    d1 = (y[:, 2:] - y[:, 1:-1]) / hi.unsqueeze(-1)       # (B, J-2, D)
    d0 = (y[:, 1:-1] - y[:, :-2]) / hi_1.unsqueeze(-1)    # (B, J-2, D)
    rhs[:, 1:-1] = 6.0 * (d1 - d0)

    # Solve for M (B, J, D); torch.linalg.solve supports batch × rhs with extra dim
    # Fallback to CPU if a backend (e.g., some MPS builds) lacks a kernel.
    try:
        M = torch.linalg.solve(A, rhs)  # (B, J, D)
    except RuntimeError:
        M = torch.linalg.solve(A.cpu(), rhs.cpu()).to(device)

    return (x, y, M)


class NaturalCubicSpline:
    def __init__(self, coeffs):
        self.x, self.y, self.M = coeffs  # x: (B,J), y: (B,J,D), M: (B,J,D)

    def evaluate(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (L,) or (B,L). Returns (B, L, D)
        """
        x, y, M = self.x, self.y, self.M
        B, J, D = y.shape
        device, dtype = y.device, y.dtype

        if t.ndim == 1:
            t = t.unsqueeze(0).expand(B, -1)  # (B, L)
        else:
            assert t.shape[0] == B
        t = t.clamp(0.0, 1.0 - 1e-12)  # keep inside last interval

        # Find segment index i such that x_i <= t < x_{i+1}
        # Batched searchsorted (torch supports batched inputs)
        idx = torch.searchsorted(x, t, right=True).clamp(1, J - 1) - 1  # (B, L)

        # Gather per-query quantities
        i0 = idx
        i1 = idx + 1
        x0 = torch.gather(x, 1, i0)                      # (B, L)
        x1 = torch.gather(x, 1, i1)                      # (B, L)
        h  = (x1 - x0).clamp_min(1e-8)                   # (B, L)
        a  = ((x1 - t) / h).unsqueeze(-1)                # (B, L, 1)
        b  = (1.0 - a)                                   # (B, L, 1)
        h2 = (h ** 2).unsqueeze(-1)                      # (B, L, 1)

        # Expand idx for D
        gather_idx = i0.unsqueeze(-1).expand(-1, -1, D)  # (B, L, D)
        y0 = torch.gather(y, 1, gather_idx)              # (B, L, D)
        M0 = torch.gather(M, 1, gather_idx)              # (B, L, D)

        gather_idx = i1.unsqueeze(-1).expand(-1, -1, D)
        y1 = torch.gather(y, 1, gather_idx)              # (B, L, D)
        M1 = torch.gather(M, 1, gather_idx)              # (B, L, D)

        C = ((a ** 3 - a) * h2) / 6.0                    # (B, L, 1)
        Dcoef = ((b ** 3 - b) * h2) / 6.0                # (B, L, 1)

        S = a * y0 + b * y1 + C * M0 + Dcoef * M1        # (B, L, D)
        return S
