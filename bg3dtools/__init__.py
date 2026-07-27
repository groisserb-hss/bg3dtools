"""
bg3dtools - Scientific computing utilities for mesh processing and computer vision.

Subpackages
-----------
mesh
    Triangle mesh processing utilities (I/O, registration, cleaning, metrics).
pointclouds
    Point cloud processing and quantization.
pytorch
    PyTorch utilities, neural network modules, and detection tools.
render
    Visualization with trimesh, Open3D, and matplotlib.
utils
    General utilities (file I/O, numpy helpers, scheduling).
image_tools
    Image processing and filtering utilities.
pose_landmarking
    Human pose detection and landmarking.
iphone
    iPhone depth/RGB data I/O utilities.

Modules
-------
graphs
    Graph algorithms over mesh connectivity (requires the ``graph`` extra).
igl_compat
    libigl version-compatibility wrappers; the only module that imports ``igl``.
transforms_unified
    Rotation and affine transforms (twist/quaternion/matrix), numpy or torch.

Examples
--------
>>> from bg3dtools.mesh import read_triangle_mesh, submesh
>>> from bg3dtools.render.trimesh import trisurfsm
>>> from bg3dtools.utils import Timer
"""

__version__ = "1.0.2"
