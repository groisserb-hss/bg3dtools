"""
Triangle mesh processing utilities.

Provides I/O, cleaning, registration, utilities, generation, and intersection
operations for triangle meshes.

Example
-------
>>> from bg3dtools.mesh import read_triangle_mesh, per_vertex_normals
>>> verts, faces = read_triangle_mesh('model.ply')
>>> normals = per_vertex_normals(verts, faces)
"""

# I/O operations (trimesh/plyfile are lazy-loaded inside functions)
from .mesh_io import (
    read_triangle_mesh,
    read_obj,
    load_textured_obj,
    write_colored_plyfile,
    read_colored_plyfile,
)

# Mesh utilities
from .utils import (
    extract_manifold_patches,
    join_meshes,
    submesh,
    per_vertex_normals,
    per_face_normals,
    adj_from_edges,
    row_normalize_csr,
    per_vertex_smoothing,
    laplace_vertex_smoothing,
    average_onto_vertices,
    face_2_vertex_map,
    edge_triangle_adjacency,
    surface_sample,
    mesh_volume,
    ordered_edges,
    internal_edges,
    sample_E2V,
    sparse_edge_map,
    sample_obj_vtex,
    get_genus,
    geodesic_submesh,
)

# Laplacian operators and spectral analysis
from .laplace import (
    cotangent_weights,
    lumped_vertex_areas,
    fem_mass_matrix,
    laplace_beltrami_operator,
    laplace_eigen_decomposition,
    laplacian_smoothing,
    laplacian_smoothing_batch,
    gaussian_curvature,
    biharmonic_embedding,
    laplacian_spectrum,
)

# Metrics
from .metrics import calc_geodesic

# Barycentric coordinate operations
from .barycentric import (
    points_to_barycentric,
    bc2sparse,
    project_to_bccoord,
    blend_vert_face,
)

# Registration
from .registration import (
    nonrigid_ICP,
    discrete_match,
    surface_match,
    transfer_vertex_field,
    fit_vertices,
    affine_ICP,
)

# Cleaning
from .clean import (
    bounding_box_diagonal,
    largest_patch,
    remove_ears,
    repair_with_model,
    remove_large_faces,
    fill_hole,
    fill_hole_fan,
    fill_hole_safe,
    smooth_face_mask,
    dilate_vertex_mask,
    largest_component_mask,
    close_end_caps,
    nonmanifold_edges,
    nonmanifold_verts,
    split_nonmanifold_verts,
    make_manifold,
)

# Generation
from .generate import (
    build_cylinder_capped,
    build_cube,
    build_plane,
    generate_icosahedron,
    pointcloud_to_splatted_mesh,
)

# Texture sampling
from .texture import sample_texture_at_points

# Flattening / rasterization
from .flatten import tangent_plane_project, rasterize_mesh_2d, has_flipped_triangles, mds_flatten, lscm_flatten

# Intersections (trimesh is lazy-loaded inside function)
from .intersections import boolean_slice

__all__ = [
    # I/O
    "read_triangle_mesh",
    "read_obj",
    "load_textured_obj",
    "write_colored_plyfile",
    "read_colored_plyfile",
    # Utils
    "extract_manifold_patches",
    "join_meshes",
    "submesh",
    "per_vertex_normals",
    "per_face_normals",
    "adj_from_edges",
    "row_normalize_csr",
    "per_vertex_smoothing",
    "laplace_vertex_smoothing",
    "average_onto_vertices",
    "face_2_vertex_map",
    "edge_triangle_adjacency",
    "surface_sample",
    "mesh_volume",
    "ordered_edges",
    "internal_edges",
    "sample_E2V",
    "sparse_edge_map",
    "sample_obj_vtex",
    "get_genus",
    "geodesic_submesh",
    # Texture sampling
    "sample_texture_at_points",
    # Flattening / rasterization
    "tangent_plane_project",
    "rasterize_mesh_2d",
    "has_flipped_triangles",
    "mds_flatten",
    "lscm_flatten",
    # Laplacian operators and spectral analysis
    "cotangent_weights",
    "lumped_vertex_areas",
    "fem_mass_matrix",
    "laplace_beltrami_operator",
    "laplace_eigen_decomposition",
    "laplacian_smoothing",
    "laplacian_smoothing_batch",
    "gaussian_curvature",
    "biharmonic_embedding",
    "laplacian_spectrum",
    # Metrics
    "calc_geodesic",
    # Barycentric
    "points_to_barycentric",
    "bc2sparse",
    "project_to_bccoord",
    "blend_vert_face",
    # Registration
    "nonrigid_ICP",
    "discrete_match",
    "surface_match",
    "transfer_vertex_field",
    "fit_vertices",
    "affine_ICP",
    # Clean
    "bounding_box_diagonal",
    "largest_patch",
    "remove_ears",
    "repair_with_model",
    "remove_large_faces",
    "fill_hole",
    "fill_hole_fan",
    "fill_hole_safe",
    "smooth_face_mask",
    "dilate_vertex_mask",
    "largest_component_mask",
    "close_end_caps",
    "nonmanifold_edges",
    "nonmanifold_verts",
    "split_nonmanifold_verts",
    "make_manifold",
    # Generate
    "build_cylinder_capped",
    "build_cube",
    "build_plane",
    "generate_icosahedron",
    "pointcloud_to_splatted_mesh",
    # Intersections
    "boolean_slice",
]
