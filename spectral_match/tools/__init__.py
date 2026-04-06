"""
Spectral matching utility tools.

This package provides mesh classes and geometric utilities for spectral
shape matching algorithms.
"""

from .mesh_class import Mesh
from .geometric_utilities import (
    normalize_mesh,
    difference_matrix,
    centre_and_norm,
    orthogonal_procrustes,
    reorder_mesh,
    extract_edges,
    edge_lengths,
    boundary_vertices,
    mesh_neighbours,
    geodesic_matrix,
    biharmonic_matrix,
    area,
    face_areas,
    face_normals,
    metric_sampling,
    propogate_points,
    extrapolate_geodesic_matrix,
    extrapolate_scalars,
    sign,
    safe_divide,
    safe_inverse,
    safe_sqrt,
    first_n,
    arc_length,
)
from .util import double_plot, vcolor, get_platform

__all__ = [
    # mesh_class
    "Mesh",
    # geometric_utilities
    "normalize_mesh",
    "difference_matrix",
    "centre_and_norm",
    "orthogonal_procrustes",
    "reorder_mesh",
    "extract_edges",
    "edge_lengths",
    "boundary_vertices",
    "mesh_neighbours",
    "geodesic_matrix",
    "biharmonic_matrix",
    "area",
    "face_areas",
    "face_normals",
    "metric_sampling",
    "propogate_points",
    "extrapolate_geodesic_matrix",
    "extrapolate_scalars",
    "sign",
    "safe_divide",
    "safe_inverse",
    "safe_sqrt",
    "first_n",
    "arc_length",
    # util
    "double_plot",
    "vcolor",
    "get_platform",
]
