"""
PyTorch utilities for bg3dtools.

Provides backend abstraction for numpy/torch interoperability,
device management, and neural network utilities.

Optional: Requires PyTorch to be installed.

Example
-------
>>> from bg3dtools.pytorch import TorchBackend, infer_backend, to_numpy
>>> bk = infer_backend(tensor)  # Returns TorchBackend for tensors
>>> result = bk.sum(arr, axis=0, keepdims=True)  # numpy-style API
"""

__all__ = []

try:
    import torch

    if torch.cuda.is_available():
        default_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        default_device = torch.device("mps")
    else:
        default_device = torch.device("cpu")

    # Backend utilities for numpy/torch interoperability
    from .backend import (
        TorchBackend,
        infer_backend,
        to_numpy,
        to_torch,
        HAS_TORCH,
        ArrayLike,
    )

    __all__.extend([
        "default_device",
        "TorchBackend",
        "infer_backend",
        "to_numpy",
        "to_torch",
        "HAS_TORCH",
        "ArrayLike",
    ])

except ImportError:
    default_device = None
    TorchBackend = None
    infer_backend = None
    to_numpy = None
    to_torch = None
    HAS_TORCH = False
    ArrayLike = None

    __all__.extend([
        "default_device",
        "HAS_TORCH",
    ])
