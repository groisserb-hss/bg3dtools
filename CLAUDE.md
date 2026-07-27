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
| `igl_compat.py` | libigl version-compatibility wrappers — the only module that imports `igl` |

### spectral_match package

Separate package (included in setuptools config) for dense mesh correspondence using functional maps:
- `spectral_match/pipeline.py` - `FunctionalMapper` orchestrates eigendecomposition → descriptors → functional map solving → product manifold filtering
- Uses bg3dtools mesh utilities for Laplacian computation and I/O
- Configured via `SigConfig` and `MatchConfig` namedtuples

## libigl Version Compatibility — `igl_compat`

**igl 2.5.1 is canonical** (conda-forge; `environment.yml` pins it). igl 2.6 is a full
rewrite of the python bindings (pybind11 → nanobind) with breaking changes to names,
return arity, return order and dtypes. A machine that drifted to 2.6.1 took down a
downstream pipeline, so all libigl access is now funnelled through one module.

**`bg3dtools/igl_compat.py` is the only place in the package that does `import igl`.**
Every wrapper presents the 2.5.1 contract regardless of which binding is installed, and
each one documents its 2.6 behavior. Never call `igl.*` directly — add a wrapper instead.

```python
from bg3dtools.igl_compat import facet_components, doublearea, AVAILABLE

labels = facet_components(faces)     # bare (nF,) int64 array, never a tuple
if 'is_vertex_manifold' in AVAILABLE:   # use this, not hasattr(igl, ...)
    ...
```

Beyond per-function fixes the layer guarantees repo-wide: **int64 C-contiguous index
arrays, float64 C-contiguous coordinate arrays**, and a preserved leading dimension even
when it is 1 (the 2.5.1 binding squeezes those). The int64 pinning applies to *inputs*
too, which makes the index-dtype mismatch noted under Testing Patterns structurally
impossible through the layer. **Never feature-detect by version string** —
`igl.__version__` does not exist in the 2.6.1 conda build.

The nastiest 2.6 change is `exact_geodesic`: the positional signature went from
`(v, f, vs, vt)` to `(V, F, VS, FS, VT, FT)`, so a 4-positional call binds target
*vertices* to the source *faces* slot and returns an **empty array instead of raising**.
The wrapper always calls by keyword. Also note `igl.point_simplex_squared_distance` is
outright wrong in 2.5.1 (it reads vertices column-major); see its wrapper docstring.

`tests/test_igl_compat.py` pins every contract and is import-light on purpose, so it runs
in a bare scratch env. To check a candidate igl version:

```bash
conda create -y -n igl261 -c conda-forge python=3.12 igl=2.6.1 numpy scipy pytest
~/opt/anaconda3/envs/igl261/bin/python -m pytest tests/test_igl_compat.py -v
```

Functions absent from one binding are skipped there and flagged in `AVAILABLE`: 2.6
removed `extract_manifold_patches`, `collapse_small_triangles`,
`resolve_duplicated_faces` and `point_simplex_squared_distance`; 2.5.1 lacks
`is_vertex_manifold`. `spectral_match/` still imports `igl` directly and has **not** been
migrated.

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
- **Cross-version testing**: `tests/test_igl_compat.py` must pass on igl 2.5.1 *and* 2.6.1 — keep it import-light (numpy/scipy only) so it runs in a bare scratch env. See the libigl Version Compatibility section.
- **Note**: igl.cotmatrix and igl.massmatrix are broken in igl 2.5.1; custom implementations in `mesh/laplace.py` are used instead
- **Note**: libigl requires every *array* integer argument of a call (faces `f` + index arrays like `exact_geodesic`'s `vs`/`vt`) to share one integer dtype, else it raises `ValueError: Invalid type (int64 ...) ... Expected it to match argument 'f'`. NumPy's default int is int64 on Linux/macOS but int32 on Windows, so this only bites on some hosts. `igl_compat` now removes this hazard structurally by pinning every array index argument to int64 on the way in. Two repo-level helpers remain for the non-igl consumers: mesh readers in `mesh/mesh_io.py` canonicalize faces to **int64**, and `mesh/utils.as_igl_faces` / `match_index_dtype(f, *arrays)` keep hand-built face arrays canonical. Guarded by `tests/test_igl_index_dtype.py`. (Scalar index args, e.g. `point_simplex_squared_distance`'s face index, are exempt.)
