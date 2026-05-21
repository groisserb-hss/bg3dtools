# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**bg3dtools** is a scientific computing toolkit for 3D geometry processing, computer vision, and parametric human body modeling. The primary codebase is in `bg3dtools/`. A companion package `spectral_match/` provides spectral mesh correspondence via functional maps.

Key domains:
- Triangle mesh processing and registration
- Point cloud analysis and reconstruction
- Human pose detection and landmarking (MediaPipe-based)
- Parametric body models (SMPL, STAR)
- PyTorch-based neural network utilities
- Spectral mesh matching and correspondence

## Commands

```bash
# Environment setup (conda)
conda env create -f environment.yml
conda activate bg3dtools

# Install package in dev mode
pip install -e .

# Run tests (pytest)
pytest tests/ -v
pytest tests/test_transforms_unified.py -v        # specific file
pytest tests/test_transforms_unified.py::TestClass::test_method -v  # specific test
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
# - axis -> dim, keepdims -> keepdim
# - concatenate -> cat, expand_dims -> unsqueeze
# - clip -> clamp, arccos -> acos
# - Also provides bk.linalg namespace mirroring np.linalg
```

**Key files:**
- `bg3dtools/pytorch/backend.py` - `TorchBackend` class, `infer_backend()`, `to_numpy()`, `to_torch()`
- `bg3dtools/transforms_unified.py` - Geometric transforms supporting both backends

### Batch-Dimension Agnostic Functions

Functions in `transforms_unified.py` accept arbitrary leading batch dimensions using `[..., N]` notation. The implementation pattern is: flatten batch dims → process → reshape back.

```python
twist_to_R(twist)  # [..., 3] -> [..., 3, 3]
make_aff(twist, trans)  # [..., 3], [..., 3] -> [..., 4, 4]
```

### Dependencies

**Hard dependencies** (always required): numpy, scipy, libigl, Pillow, imageio

**Optional dependencies** are organized into extras in `pyproject.toml`:
- `mesh`   — trimesh, plyfile, triangle
- `learn`  — scikit-learn (only needed by `utils.load_pca`)
- `torch`  — torch
- `vision` — opencv-python, mediapipe
- `viz`    — open3d, matplotlib
- `graph`  — igraph (only needed by `bg3dtools.graphs`)
- `io`     — tenacity (only needed by `utils.cifs_wrappers`)
- `all`    — bundles all of the above

Install with: `pip install 'bg3dtools[all]'` or pick the extras you need.

Optional deps are lazy-loaded inside functions:
```python
def read_triangle_mesh(file, process=False):
    import trimesh  # Lazy import
    mesh = trimesh.load_mesh(file, process=process)
```

Subpackages that wrap optional-dep modules use try/except in `__init__.py` so the package still imports without the extra (the optional symbols become `None`).

## Module Organization

All modules have explicit `__all__` exports. Import from submodules:
```python
from bg3dtools.mesh import read_triangle_mesh, per_vertex_normals
from bg3dtools.pointclouds import fit_plane_to_points
from bg3dtools.pytorch import TorchBackend, infer_backend  # if torch installed
```

| Module | Purpose |
|--------|---------|
| `mesh/` | Triangle mesh I/O, registration, cleaning, metrics, Laplacian/spectral ops |
| `pointclouds/` | Point cloud fitting (RANSAC), reconstruction, registration |
| `pytorch/` | Backend abstraction, transforms, mesh ops, neural net modules |
| `pytorch/detection/` | Object detection training/eval utilities (COCO metrics) |
| `pytorch/autoencoders/` | VAE implementations (1D, 2D, conditional, categorical) |
| `pose_landmarking/` | MediaPipe pose detection, joint mapping, segmentation |
| `image_tools/` | Image packing, filters, video I/O |
| `iphone/` | iOS depth scanning data I/O (Stray Scanner & Record3D) |
| `render/` | Colors/colormaps; visualization via `render.trimesh` or `render.o3d`; `run_isolated()` subprocess wrapper |
| `utils/` | Timing, algorithms (FPS, PCA), filesystem, stats |
| `utils/cifs_wrappers/` | Network filesystem I/O with exponential backoff retry |
| `transforms_unified.py` | Rotation/affine transforms (twist/quaternion/matrix) |

### spectral_match package

Separate package (included in setuptools config) for dense mesh correspondence using functional maps:
- `spectral_match/pipeline.py` - `FunctionalMapper` orchestrates eigendecomposition → descriptors → functional map solving → product manifold filtering
- Uses bg3dtools mesh utilities for Laplacian computation and I/O
- Configured via `SigConfig` and `MatchConfig` namedtuples

## macOS Open3D Rendering Caveat

On macOS, Open3D's `Visualizer` creates Cocoa/OpenGL windows even with `visible=False`. Repeated create/destroy cycles accumulate window-server resources and eventually deadlock the process (low CPU, immune to Ctrl-C). The `OffscreenRenderer` (EGL-based) is not available on macOS.

**Fix**: Use `run_isolated()` from `bg3dtools.render` to run rendering functions in a spawned subprocess. All GPU handles are released when the child exits.

```python
from bg3dtools.render import run_isolated

# Each call gets a fresh GPU context — no resource accumulation
run_isolated(my_render_function, arg1, arg2, timeout=120)
```

The legacy `_render_legacy()` fallback uses `visible=False` to minimize window flashing, but subprocess isolation is still needed for batch pipelines that render many figures.

## Code Style

- NumPy-style docstrings with Parameters/Returns/Examples sections
- Type hints from `typing` module
- Handle optional dependencies gracefully with lazy imports inside functions

## Testing Patterns

Tests use these conventions:
- **Reference implementations**: Inline old/reference code for comparison testing
- **Batch testing**: Verify functions with arbitrary leading batch dimensions
- **Known-answer tests**: Apply known transform, recover parameters, assert roundtrip
- **Cross-backend testing**: Verify numpy and PyTorch produce equivalent results
- **Test mesh fixtures**: `_icosahedron()`, `_subdivided_icosahedron()`, `_unit_tetrahedron()`
- **Note**: igl.cotmatrix and igl.massmatrix are broken in igl 2.5.1; custom implementations in `mesh/laplace.py` are used instead
