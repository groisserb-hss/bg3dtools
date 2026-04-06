"""
PyTorch backend that accepts numpy-style function signatures.

This module provides a TorchBackend class that wraps PyTorch to accept
numpy-style function signatures (e.g., axis instead of dim, keepdims instead
of keepdim). This allows writing code that works with both numpy and torch.

Usage:
    import numpy as np
    from bg3dtools.pytorch.backend import TorchBackend

    bk = np if use_numpy else TorchBackend()
    result = bk.sum(arr, axis=0, keepdims=True)  # works for both
"""
from __future__ import annotations
from typing import Optional, Union, Iterator
import numpy as np

# Optional torch import
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False

# Type alias for array-like objects
ArrayLike = Union[np.ndarray, 'torch.Tensor']


def _require_torch():
    """Raise ImportError if torch is not available."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this functionality. Install with: pip install torch")


_torch_backend_singleton = None

def infer_backend(arr: ArrayLike, as_str: bool = False):
    """Infer the backend from an array type."""
    global _torch_backend_singleton
    if HAS_TORCH and isinstance(arr, torch.Tensor):
        if as_str:
            return 'torch'
        if _torch_backend_singleton is None:
            _torch_backend_singleton = TorchBackend()
        return _torch_backend_singleton
    elif isinstance(arr, np.ndarray):
        return 'numpy' if as_str else np
    else:
        raise TypeError(f"'{type(arr).__name__}' is not supported")


def to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert array to numpy, handling torch tensors."""
    if arr is None:
        return None
    if HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def to_torch(arr: ArrayLike, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert array to torch tensor."""
    _require_torch()
    if arr is None:
        return None
    if dtype is None:
        dtype = torch.float32
    if not isinstance(arr, torch.Tensor):
        return torch.tensor(arr, dtype=dtype, device=device)
    return arr.to(dtype=dtype, device=device)


class _LinalgNamespace:
    """Wraps torch.linalg to accept numpy-style kwargs."""

    _kwarg_map = {
        'axis': 'dim',
        'keepdims': 'keepdim',
    }

    def __getattr__(self, name: str):
        if hasattr(torch.linalg, name):
            torch_func = getattr(torch.linalg, name)
            return _NumpyStyleWrapper(torch_func, False, self._kwarg_map)
        raise AttributeError(f"'torch.linalg' has no attribute '{name}'")


class TorchBackend:
    """
    Wraps torch to accept numpy-style function signatures.

    - Translates numpy kwarg names to torch (axis->dim, keepdims->keepdim)
    - Maps numpy function names to torch equivalents (concatenate->cat)
    - Automatically infers device from input tensors
    - Falls through to torch for anything not explicitly handled
    - Provides linalg namespace that mirrors np.linalg
    """

    def __init__(self):
        _require_torch()

    # Numpy function names -> torch function names
    _name_map = {
        'concatenate': 'cat',
        'expand_dims': 'unsqueeze',
        'clip': 'clamp',
        'arccos': 'acos',
        'arcsin': 'asin',
        'arctan': 'atan',
        'arctan2': 'atan2',
    }

    # Numpy kwarg names -> torch kwarg names
    _kwarg_map = {
        'axis': 'dim',
        'keepdims': 'keepdim',
    }

    # Functions that return namedtuples in torch but scalars/arrays in numpy
    _returns_namedtuple = {'max', 'min', 'sort'}

    # Linalg namespace
    linalg = _LinalgNamespace()

    def __getattr__(self, name: str):
        # Map numpy name to torch name
        torch_name = self._name_map.get(name, name)

        # Try to get from torch, then torch.linalg
        if hasattr(torch, torch_name):
            torch_func = getattr(torch, torch_name)
        elif hasattr(torch.linalg, torch_name):
            torch_func = getattr(torch.linalg, torch_name)
        else:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        return _NumpyStyleWrapper(torch_func, name in self._returns_namedtuple, self._kwarg_map)

    # Functions needing special handling (different APIs entirely)

    def copy(self, arr):
        """np.copy -> tensor.clone"""
        return arr.clone()

    def hstack(self, arrays):
        """np.hstack -> torch.cat with dim=-1"""
        return torch.cat(arrays, dim=-1)

    def vstack(self, arrays):
        """np.vstack -> torch.cat with dim=0"""
        return torch.cat(arrays, dim=0)

    def transpose(self, arr, axes=None):
        """np.transpose -> tensor.permute"""
        if axes is None:
            return arr.T
        return arr.permute(*axes)

    def flip(self, arr, axis=None):
        """np.flip(axis=int) -> torch.flip(dims=list)"""
        if axis is None:
            dims = list(range(arr.ndim))
        elif isinstance(axis, int):
            dims = [axis]
        else:
            dims = list(axis)
        return torch.flip(arr, dims=dims)

    def pad(self, arr, pad_width, mode='constant', constant_values=0):
        """
        np.pad -> F.pad with converted pad_width format.
        numpy: ((before_0, after_0), (before_1, after_1), ...)
        torch: (left, right, top, bottom, ...) starting from last dim
        """
        # Convert numpy-style to torch-style
        if isinstance(pad_width, (list, tuple)) and len(pad_width) > 0:
            if isinstance(pad_width[0], (list, tuple)):
                torch_pad = []
                for before, after in reversed(pad_width):
                    torch_pad.extend([before, after])
                pad_width = torch_pad

        if mode == 'constant':
            return F.pad(arr, pad_width, mode=mode, value=constant_values)
        return F.pad(arr, pad_width, mode=mode)

    def lstsq(self, a, b, rcond=None):
        """np.linalg.lstsq -> torch.linalg.lstsq (ignore rcond)"""
        return torch.linalg.lstsq(a, b)

    # Array creation with automatic device inference

    def array(self, data, dtype=None):
        """Create tensor, inferring device from data if it's already a tensor."""
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype) if dtype else data
        return torch.tensor(data, dtype=dtype)

    def zeros(self, shape, dtype=None, *, like=None):
        """Create zeros, optionally matching another tensor's device/dtype."""
        if like is not None:
            return torch.zeros(shape, dtype=dtype or like.dtype, device=like.device)
        return torch.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None, *, like=None):
        """Create ones, optionally matching another tensor's device/dtype."""
        if like is not None:
            return torch.ones(shape, dtype=dtype or like.dtype, device=like.device)
        return torch.ones(shape, dtype=dtype)

    def eye(self, n, dtype=None, *, like=None):
        """Create identity matrix, optionally matching another tensor's device/dtype."""
        if like is not None:
            return torch.eye(n, dtype=dtype or like.dtype, device=like.device)
        return torch.eye(n, dtype=dtype)

    def empty(self, shape, dtype=None, *, like=None):
        """Create uninitialized tensor, optionally matching another tensor's device/dtype."""
        if like is not None:
            return torch.empty(shape, dtype=dtype or like.dtype, device=like.device)
        return torch.empty(shape, dtype=dtype)

    def to_numpy(self, arr):
        """Convert tensor to numpy array."""
        if arr.requires_grad:
            return arr.detach().cpu().numpy()
        return arr.cpu().numpy()


class _NumpyStyleWrapper:
    """Wraps a torch function to accept numpy-style kwargs."""

    def __init__(self, func, returns_namedtuple: bool, kwarg_map: dict):
        self._func = func
        self._returns_namedtuple = returns_namedtuple
        self._kwarg_map = kwarg_map

    def __call__(self, *args, **kwargs):
        # Translate numpy-style kwargs to torch-style
        translated = {}
        for k, v in kwargs.items():
            new_k = self._kwarg_map.get(k, k)
            translated[new_k] = v

        result = self._func(*args, **translated)

        # Handle functions that return namedtuples in torch
        if self._returns_namedtuple and hasattr(result, 'values'):
            return result.values

        return result
