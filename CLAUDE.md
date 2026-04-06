# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**bg3dtools** is a scientific computing toolkit for 3D geometry processing, computer vision, and parametric human body modeling. The primary codebase is in `bg3dtools/`.

Key domains:
- Triangle mesh processing and registration
- Point cloud analysis and reconstruction
- Human pose detection and landmarking (MediaPipe-based)
- Parametric body models (SMPL, STAR)
- PyTorch-based neural network utilities

## Commands

```bash
# Environment setup (conda)
conda env create -f environment.yml
conda activate bg3dtools

# Run tests (pytest)
pytest tests/ -v  # Run all tests
pytest tests/test_file.py -v  # Run specific test file
```

## Architecture

### Dual-Backend Design (NumPy + PyTorch)

The codebase supports code that works with both NumPy arrays and PyTorch tensors through `TorchBackend`:

```python
from bg3dtools.pytorch.backend import TorchBackend, infer_backend, to_numpy, to_torch

# Auto-detect backend from input
bk = infer_backend(arr)  # Returns TorchBackend() for tensors, np for arrays
result = bk.sum(arr, axis=0, keepdims=True)  # Works for both!

# TorchBackend translates numpy-style calls:
# - axis -> dim
# - keepdims -> keepdim
# - concatenate -> cat
# - expand_dims -> unsqueeze
```

**Key files:**
- `bg3dtools/pytorch/backend.py` - `TorchBackend` class, `infer_backend()`, `to_numpy()`, `to_torch()`
- `bg3dtools/transforms_unified.py` - Geometric transforms supporting both backends

### Batch-Dimension Agnostic Functions

Functions accept arbitrary leading batch dimensions using `[..., N]` notation:
```python
twist_to_R(twist)  # [..., 3] -> [..., 3, 3]
make_aff(twist, trans)  # [..., 3], [..., 3] -> [..., 4, 4]
```

### Dependencies

**Hard dependencies** (always required): numpy, scipy, libigl, sklearn, PIL/Pillow

**Soft dependencies** (lazy-loaded inside functions): trimesh, plyfile, open3d, pytorch, mediapipe, cv2

Soft dependencies use lazy imports inside functions to avoid import errors:
```python
def read_triangle_mesh(file, process=False):
    import trimesh  # Lazy import
    mesh = trimesh.load_mesh(file, process=process)
    ...
```

## Module Organization

All modules have explicit `__all__` exports. Import from submodules:
```python
from bg3dtools.mesh import read_triangle_mesh, per_vertex_normals
from bg3dtools.pointclouds import fit_plane_to_points
from bg3dtools.pytorch import TorchBackend, infer_backend  # if torch installed
```

| Module | Purpose |
|--------|---------|
| `mesh/` | Triangle mesh I/O, registration, cleaning, metrics, Laplacian ops |
| `pointclouds/` | Point cloud fitting (RANSAC), reconstruction, registration |
| `pytorch/` | Backend abstraction, transforms, mesh ops, neural net modules |
| `pose_landmarking/` | MediaPipe pose detection, joint mapping, segmentation |
| `image_tools/` | Image packing, filters, video I/O |
| `render/` | Colors/colormaps; visualization via `render.trimesh` or `render.o3d` |
| `utils/` | Timing, algorithms (FPS, PCA), filesystem, stats |
| `transforms_unified.py` | Rotation/affine transforms (twist/quaternion/matrix) |

## Code Style

- NumPy-style docstrings with Parameters/Returns/Examples sections
- Type hints from `typing` module
- Handle optional dependencies gracefully with try/except imports
