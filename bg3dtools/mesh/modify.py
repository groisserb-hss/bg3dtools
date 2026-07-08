"""
Mesh modification operations.

This module provides functions for modifying mesh topology including
edge splitting, vertex count adjustment, and edge-face adjacency queries.
"""

from typing import Optional, Tuple, Union
import numpy as np
import igl

from bg3dtools.mesh.utils import as_igl_faces

__all__ = [
    "edge_neighbors",
    "split_edge",
    "split_to_num_verts",
    "resize_to_num_verts",
]


def edge_neighbors(
    faces: np.ndarray,
    face_adjacency: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the neighboring faces for each edge.

    Parameters
    ----------
    faces : (F, 3) ndarray
        Triangle face indices.
    face_adjacency : (F, 3) ndarray, optional
        Pre-computed face adjacency. Computed if None.

    Returns
    -------
    edges : (E, 2) ndarray
        Vertex indices defining each edge.
    edge_neighbors : (E, 2) ndarray
        Indices of the two neighboring faces for each edge.
        Value is -1 for boundary edges with only one neighbor.
    """
    # int64 chokepoint: is_edge_manifold / triangle_triangle_adjacency misbehave on the int32
    # faces that libigl's decimate/upsample/bfs_orient return on the Windows wheel.
    faces = as_igl_faces(faces)
    assert igl.is_edge_manifold(faces)

    if face_adjacency is None:
        face_adjacency = igl.triangle_triangle_adjacency(faces)[0]

    edges = igl.edges(faces)
    nE = len(edges)

    # Build all face-edge pairs: 3 edges per face, sorted
    # Edge corners: (0,1), (1,2), (2,0) — matching igl's triangle_triangle_adjacency order
    face_edges = np.stack([
        np.sort(faces[:, [0, 1]], axis=1),
        np.sort(faces[:, [1, 2]], axis=1),
        np.sort(faces[:, [2, 0]], axis=1),
    ], axis=1)  # (nF, 3, 2)

    sorted_edges = np.sort(edges, axis=1)
    # Map each sorted edge to its index via structured array lookup
    edge_keys = sorted_edges[:, 0].astype(np.int64) * (faces.max() + 1) + sorted_edges[:, 1]
    edge_key_to_idx = np.empty(edge_keys.max() + 1, dtype=np.int64)
    edge_key_to_idx[edge_keys] = np.arange(nE)

    nF = faces.shape[0]
    face_edge_keys = face_edges[:, :, 0].astype(np.int64) * (faces.max() + 1) + face_edges[:, :, 1]
    edge_idx_per_corner = edge_key_to_idx[face_edge_keys.ravel()]  # (3*nF,)
    neighbor_per_corner = face_adjacency.ravel()  # (3*nF,)

    # Fill edge_neighbors: use stable argsort to assign first/second occurrence
    edge_neighbors = np.full((nE, 2), -1, dtype=int)
    order = np.argsort(edge_idx_per_corner, kind='stable')
    sorted_eidx = edge_idx_per_corner[order]
    sorted_neigh = neighbor_per_corner[order]
    # First occurrence of each edge
    first_mask = np.concatenate([[True], sorted_eidx[1:] != sorted_eidx[:-1]])
    edge_neighbors[sorted_eidx[first_mask], 0] = sorted_neigh[first_mask]
    # Second occurrence
    second_mask = np.concatenate([[False], sorted_eidx[1:] == sorted_eidx[:-1]])
    # Filter to only second (not third+)
    if second_mask.any():
        second_indices = np.where(second_mask)[0]
        edge_neighbors[sorted_eidx[second_indices], 1] = sorted_neigh[second_indices]

    return edges, edge_neighbors


def split_edge(
    verts: np.ndarray,
    faces: np.ndarray,
    edge: Tuple[int, int],
    vtex: Optional[np.ndarray] = None,
    ftex: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Subdivide an edge by inserting a vertex at its midpoint.

    Parameters
    ----------
    verts : (N, 3) ndarray
        Mesh vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face indices.
    edge : tuple of int
        Vertex indices (v0, v1) defining the edge to split.
    vtex : (N, D) ndarray, optional
        Per-vertex texture coordinates. Interpolated for new vertex.
    ftex : (M, D) ndarray, optional
        Per-face texture coordinates. Duplicated for new faces.

    Returns
    -------
    verts : (N+1, 3) ndarray
        Updated vertices with new midpoint vertex.
    faces : (M+k, 3) ndarray
        Updated faces (k = number of faces sharing the edge).
    vtex : (N+1, D) ndarray, optional
        Updated vertex textures. Returned if vtex was provided.
    ftex : (M+k, D) ndarray, optional
        Updated face textures. Returned if ftex was provided.
    """
    # Calculate the midpoint of the selected edge
    midpoint = (verts[edge[0]] + verts[edge[1]]) / 2

    # Add the midpoint as a new vertex to the vertex list
    verts = np.vstack([verts, midpoint])
    new_vertex_index = len(verts) - 1
    vtex = None if vtex is None else np.vstack([vtex, (vtex[edge[0]] + vtex[edge[1]]) / 2])

    # Find all faces that need to be updated (those that contain the edge)
    faces_to_update = np.where(  np.sum((faces == edge[0]) + (faces == edge[1]), axis=1) == 2)[0]

    # For each face containing the edge, replace it with new faces incorporating the new vertex
    faces = np.array(faces, copy=True)
    new_face_list = []
    new_ftex_list = []
    for face_index in faces_to_update:
        face = faces[face_index]
        # Determine the third vertex of the face
        third_vertex = next(v for v in face if v not in edge)

        # Create new faces
        new_faces = np.array([
            [edge[0], new_vertex_index, third_vertex],
            [new_vertex_index, edge[1], third_vertex]
        ])
        # Replace the original face in-place and collect the new face
        faces[face_index] = new_faces[0]
        new_face_list.append(new_faces[1])
        if ftex is not None:
            new_ftex_list.append(ftex[face_index])
    if new_face_list:
        faces = np.vstack([faces] + new_face_list)
        if ftex is not None:
            ftex = np.vstack([ftex] + new_ftex_list)
    faces = as_igl_faces(igl.bfs_orient(faces)[0])  # bfs_orient returns int32 on the Windows wheel

    if vtex is None and ftex is None:
        return verts, faces
    elif ftex is None:
        return verts, faces, vtex
    elif vtex is None:
        return verts, faces, ftex
    else:
        return verts, faces, vtex, ftex


def split_to_num_verts(
    verts: np.ndarray,
    faces: np.ndarray,
    target_N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add vertices by edge splitting until reaching target count.

    Iteratively splits the edge with highest score (area + length)
    until the mesh has the target number of vertices.

    Parameters
    ----------
    verts : (N, 3) ndarray
        Mesh vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face indices.
    target_N : int
        Target number of vertices.

    Returns
    -------
    verts : (target_N, 3) ndarray
        Updated vertex coordinates.
    faces : (M', 3) ndarray
        Updated face indices.

    Warnings
    --------
    This function is inefficient for large target_N >> len(verts).
    Each iteration recomputes edge neighbors and scores.
    """

    while len(verts) < target_N:
        edges, edge_fidx = edge_neighbors(faces)
        f_area = igl.doublearea(verts, faces) / 2
        f_area = f_area / f_area.mean()
        edge_area = np.sum(f_area[edge_fidx], axis=1)

        edge_lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        edge_lengths /= edge_lengths.mean()

        edge_scores = edge_area + edge_lengths
        edge_idx = np.argmax(edge_scores)
        verts, faces = split_edge(verts, faces, edges[edge_idx])

    return verts, faces


def resize_to_num_verts(
    verts: np.ndarray,
    faces: np.ndarray,
    target_N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize mesh to exact vertex count using upsampling and decimation.

    Combines Loop subdivision, edge-collapse decimation, and edge
    splitting to achieve exactly the target number of vertices.

    Parameters
    ----------
    verts : (N, 3) ndarray
        Mesh vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face indices. Must be edge-manifold.
    target_N : int
        Exact target number of vertices.

    Returns
    -------
    verts : (target_N, 3) ndarray
        Resized vertex coordinates.
    faces : (M', 3) ndarray
        Updated face indices.

    Raises
    ------
    AssertionError
        If mesh is not edge-manifold or decimation fails.
    """
    assert igl.is_edge_manifold(faces), "Mesh must be edge manifold"

    # igl.upsample/decimate return int32 faces on the Windows wheel; pin int64 after each so the
    # downstream is_edge_manifold checks (here and in edge_neighbors) never see int32.
    # if we have fewer faces than the target number of vertices, upsample by splitting faces
    while len(verts) < 0.98 * target_N:
        verts, faces = igl.upsample(verts, faces)
        faces = as_igl_faces(faces)

    # first pass on decimation, keep 3 extra faces to try not to overshoot
    target_M = int(target_N / len(verts) * len(faces)) + 3
    if target_M < len(faces):
        sucess, verts, faces, _, _ = igl.decimate(verts, faces, target_M)
        faces = as_igl_faces(faces)
        assert sucess, "libigl decimation failed"

    # subsequent decimations try to hit the target number of vertices exactly
    while len(verts) > target_N:
        extra_verts = len(verts) - target_N
        sucess, verts, faces, _, _ = igl.decimate(verts, faces, len(faces) - 2*extra_verts)
        faces = as_igl_faces(faces)
        assert sucess, "libigl decimation failed"

    # if we overshot, add vertices back in by edge splitting
    verts, faces = split_to_num_verts(verts, faces, target_N)

    assert len(verts) == target_N, "Failed to resize to target number of vertices"
    return verts, faces