# bg3dtools

Python toolkit for 3D geometry processing, mesh registration, point cloud reconstruction, and human pose landmarking.

## Install

```bash
pip install -e .
```

The base install pulls in only `numpy`, `scipy`, `libigl`, `Pillow`, and `imageio`. Heavier features sit behind optional extras:

| Extra    | Adds                        | Needed for                                       |
|----------|-----------------------------|--------------------------------------------------|
| `mesh`   | trimesh, plyfile, triangle  | Mesh I/O (PLY/OBJ via trimesh), `fill_hole`      |
| `learn`  | scikit-learn                | `utils.load_pca`                                 |
| `torch`  | torch                       | Everything in `bg3dtools.pytorch`                |
| `vision` | opencv-python, mediapipe    | Pose detection, face de-identification           |
| `viz`    | open3d, matplotlib          | Visualization wrappers in `render/`              |
| `graph`  | igraph                      | `bg3dtools.graphs`                               |
| `io`     | tenacity                    | `utils.cifs_wrappers` retry logic                |
| `match`  | jax[cpu]>=0.4.18,<0.5       | `spectral_match` functional-map solver           |
| `all`    | all of the above            | One-shot install                                 |

```bash
pip install -e '.[mesh,viz]'      # pick what you need
pip install -e '.[all]'           # or grab the lot
```

Subpackages that wrap an optional dep import cleanly even without the extra installed — the optional symbols become `None` and only raise when called.

### libigl version

**igl 2.5.1 is the canonical version** — `environment.yml` pins it, from conda-forge (PyPI
has no macOS-arm64 wheels). igl 2.6 is a nanobind rewrite of the bindings with breaking
changes to names, return arity and dtypes, so all libigl access goes through
`bg3dtools.igl_compat` (`spectral_match` included), which presents the 2.5.1 contract on
either version. Import from there rather than calling `igl.*` directly. To check a
candidate igl version:

```bash
conda create -y -n igl261 -c conda-forge python=3.12 igl=2.6.1 numpy scipy pytest
~/opt/anaconda3/envs/igl261/bin/python -m pytest tests/test_igl_compat.py -v
```

## Modules

| Module | Purpose |
|--------|---------|
| `mesh/` | Triangle mesh I/O, registration, cleaning, Laplacian/spectral ops |
| `pointclouds/` | Point cloud fitting (RANSAC), reconstruction, registration |
| `pytorch/` | Backend abstraction (numpy/torch), transforms, neural-net modules |
| `pose_landmarking/` | MediaPipe pose detection, joint mapping, segmentation |
| `image_tools/` | Packing, filters, video I/O, volumetric similarity metrics |
| `iphone/` | iOS depth scanning data I/O (Stray Scanner & Record3D) |
| `igl_compat.py` | libigl version-compatibility wrappers (the only module importing `igl`) |
| `render/` | Visualization via trimesh, Open3D, matplotlib |
| `utils/` | Timing, FPS, PCA, filesystem, stats |
| `graphs.py` | Graph algorithms over mesh connectivity (needs the `graph` extra) |
| `transforms_unified.py` | Rotation/affine transforms (twist/quaternion/matrix), incl. SO(3) averaging |

A separate `spectral_match/` package (in the same repo) provides dense mesh correspondence via functional maps.

## Usage

```python
from bg3dtools.mesh import read_triangle_mesh, per_vertex_normals

V, F = read_triangle_mesh("mesh.ply")
N = per_vertex_normals(V, F)
```

NumPy and PyTorch share one API via `TorchBackend`, so the same code path runs on either:

```python
from bg3dtools.pytorch import infer_backend

bk = infer_backend(arr)                          # np or TorchBackend
result = bk.sum(arr, axis=0, keepdims=True)
```

Transforms in `transforms_unified.py` accept arbitrary leading batch dimensions:

```python
from bg3dtools.transforms_unified import twist_to_R, make_aff

R = twist_to_R(twist)            # [..., 3]    -> [..., 3, 3]
T = make_aff(twist, trans)       # [..., 3], [..., 3] -> [..., 4, 4]
```

## macOS + Open3D

Open3D's `Visualizer` accumulates Cocoa/OpenGL handles across repeated create/destroy cycles and eventually deadlocks the process. For batch rendering, run each call in a spawned subprocess:

```python
from bg3dtools.render import run_isolated

run_isolated(my_render_function, arg1, arg2, timeout=120)
```

## Tests

```bash
pytest tests/ -v
```
