"""
Mesh utility functions for triangle mesh processing.

This module provides utilities for manipulating triangle meshes including
submesh extraction, normal computation, smoothing, and surface sampling.
"""

import igl
import numpy as np
from typing import Tuple, Optional, Union, List
from scipy.sparse import coo_matrix, csr_matrix
from bg3dtools.utils.np_helpers import row_normalize
from bg3dtools.pointclouds.quantize import sparse_quantize

__all__ = [
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
    "sample_E2V",
    "sparse_edge_map",
    "sample_obj_vtex",
    "get_genus",
]


def extract_manifold_patches(
    faces: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    Extract manifold patches from a mesh.

    Wrapper for various igl functions to extract connected manifold patches
    from a triangle mesh.

    Parameters
    ----------
    faces : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    n_patches : int
        Number of manifold patches found.
    f_labels : (nF,) ndarray
        Patch label for each face.
    """
    if hasattr(igl, 'extract_manifold_patches'):
        n_patches, f_labels = igl.extract_manifold_patches(faces)
    elif hasattr(igl, 'facet_components'):
        n_patches, f_labels = igl.facet_components(faces)
    elif hasattr(igl, 'connected_components'):
        f_labels = igl.connected_components(faces)
        n_patches = int(f_labels.max()) + 1
    else:
        raise RuntimeError('unable to extract manifold patches')

    return n_patches, f_labels


def join_meshes(
    v1: np.ndarray,
    f1: np.ndarray,
    v2: np.ndarray,
    f2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate two meshes into a single mesh.

    Parameters
    ----------
    v1 : (nV1, 3) ndarray
        Vertices of first mesh.
    f1 : (nF1, 3) ndarray
        Faces of first mesh.
    v2 : (nV2, 3) ndarray
        Vertices of second mesh.
    f2 : (nF2, 3) ndarray
        Faces of second mesh.

    Returns
    -------
    verts : (nV1 + nV2, 3) ndarray
        Combined vertices.
    faces : (nF1 + nF2, 3) ndarray
        Combined faces with updated indices.
    """
    verts = np.row_stack([v1, v2])
    faces = np.row_stack([f1, f2 + v1.shape[0]])
    return verts, faces


def submesh(
    old_verts: np.ndarray,
    old_faces: np.ndarray,
    f_idx: np.ndarray,
    return_indices: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Extract a submesh from a triangle mesh.

    Convenience wrapper for ``igl.remove_unreferenced`` that returns
    mappings between old and new indices.

    Parameters
    ----------
    old_verts : (nV, D) ndarray
        Vertex coordinates of the original mesh.
    old_faces : (nF, M) ndarray
        Face indices of the original mesh (typically M=3 for triangles).
    f_idx : (K,) ndarray or (nF,) bool ndarray
        Face indices to keep, or boolean mask selecting faces.
    return_indices : bool, optional
        If True (default), return index mappings.

    Returns
    -------
    new_verts : (nV', D) ndarray
        Vertices of the submesh.
    new_faces : (K, M) ndarray
        Faces of the submesh with updated vertex indices.
    f_idx : (K,) ndarray
        Original face indices (only if return_indices=True).
    v_idx : (nV',) ndarray
        Original vertex indices (only if return_indices=True).

    Examples
    --------
    >>> new_v, new_f, fi, vi = submesh(verts, faces, [0, 1, 2])
    >>> new_v, new_f = submesh(verts, faces, face_mask, return_indices=False)
    """
    n = old_verts.shape[1]
    m = old_faces.shape[1]
    if f_idx.size == old_faces.shape[0]:
        f_idx = np.argwhere(f_idx).flatten()

    sub_faces = old_faces[f_idx]
    seder = np.argsort(sub_faces[:, 0])
    sub_faces = sub_faces[seder]
    f_idx = f_idx[seder]
    new_verts, new_faces, _, v_idx = igl.remove_unreferenced(old_verts, sub_faces)
    new_verts = new_verts.reshape([-1, n])
    new_faces = new_faces.reshape([-1, m])

    if return_indices:
        return new_verts, new_faces, f_idx, v_idx
    else:
        return new_verts, new_faces


def per_vertex_normals(
    verts: np.ndarray,
    faces: np.ndarray,
    mode: int = igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
) -> np.ndarray:
    """
    Compute per-vertex normals for a triangle mesh.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.
    mode : int, optional
        Weighting type for normal averaging. Default is area-weighted.

    Returns
    -------
    normals : (nV, 3) ndarray
        Unit normals at each vertex.
    """
    pt_normals = igl.per_vertex_normals(verts, faces, mode)
    pt_normals[np.isnan(pt_normals)] = 0
    pt_normals = row_normalize(pt_normals)
    return pt_normals


def per_face_normals(
    verts: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute per-face normals for a triangle mesh.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    normals : (nF, 3) ndarray
        Unit normal for each face.
    """
    n = np.array([0, 1, 0], dtype=verts.dtype)
    face_normals = igl.per_face_normals(verts, faces, n).reshape([-1, 3])
    bad = np.any(np.isnan(face_normals), axis=1)
    face_normals[bad] = n
    face_normals = row_normalize(face_normals)
    return face_normals


def adj_from_edges(E: np.ndarray, nV: int) -> csr_matrix:
    """
    Build adjacency matrix from edge list.

    Parameters
    ----------
    E : (nE, 2) ndarray
        Edge list where each row is (vertex_i, vertex_j).
    nV : int
        Number of vertices.

    Returns
    -------
    A : (nV, nV) csr_matrix
        Symmetric adjacency matrix.
    """
    E = np.asarray(E)
    if E.ndim != 2 or E.shape[1] != 2:
        raise ValueError("E must be (nE,2)")
    # remove self-loops just in case
    E = E[E[:,0] != E[:,1]]
    # symmetric adjacency
    rows = np.concatenate([E[:,0], E[:,1]])
    cols = np.concatenate([E[:,1], E[:,0]])
    data = np.ones(rows.size, dtype=np.float64)
    A = coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def row_normalize_csr(A: csr_matrix) -> csr_matrix:
    """
    Row-normalize a sparse CSR matrix.

    Parameters
    ----------
    A : csr_matrix
        Input sparse matrix.

    Returns
    -------
    A_norm : csr_matrix
        Row-normalized matrix where each row sums to 1.
    """
    s = np.asarray(A.sum(axis=1)).ravel()
    invs = np.zeros_like(s, dtype=np.float64)
    nz = s != 0
    invs[nz] = 1.0 / s[nz]
    return A.multiply(invs[:, None])


def per_vertex_smoothing(
        a: np.ndarray,               # (nV,C)  or (nV,)
        f: np.ndarray,               # (nF,3)  triangles
        n_iters: int = 1,
        alpha: float = 0.5,          # 0 → none, 1 → pure neighbour average
        A: Optional[csr_matrix] = None,
) -> np.ndarray:
    """
    Explicit uniform-weight Laplacian smoothing.

    new_a ← (1-α)·old_a  +  α·mean(neighbour values)

    * alpha controls how strongly each iteration pulls toward the neighbours.
    * n_iters controls how many rounds you apply.
    """

    if a.ndim == 1:
        a = a[:, None]

    if A is None:
        F = np.ascontiguousarray(f, dtype=np.int64)
        A = igl.adjacency_matrix(F)
        if np.max(A) == 0:
            E = igl.edges(F)
            A = adj_from_edges(E, np.max(F)+1)

        A = row_normalize_csr(A)

    a_sm = a.copy().astype(np.float64)               # works for any #channels
    for _ in range(n_iters):
        a_sm = (1.0 - alpha) * a_sm + alpha * (A @ a_sm)

    return a_sm.squeeze()


def laplace_vertex_smoothing(verts: np.ndarray, faces: np.ndarray, v_values: np.ndarray) -> np.ndarray:
    """
    Wrapper for laplace.laplacian_smoothing.

    :param verts: [nV x 3] vertex coordinates
    :param faces: [nF x 3] triangle indices
    :param v_values: [nV x C] vertex values
    :return: [nV x C] smoothed vertex values
    """
    assert verts.ndim == 2 and verts.shape[1] == 3, "verts must be [nV x 3]"
    assert faces.ndim == 2 and faces.shape[1] == 3, "faces must be [nF x 3]"
    assert v_values.ndim == 2 and v_values.shape[0] == verts.shape[0], \
        "v_values must be [nV x C]"
    from bg3dtools.mesh.laplace import laplacian_smoothing, cotangent_weights, fem_mass_matrix
    l = cotangent_weights(verts, faces)
    m = fem_mass_matrix(verts, faces)
    return laplacian_smoothing(l, m, v_values)


def average_onto_vertices(
        verts: np.ndarray,
        faces: np.ndarray,
        face_values: np.ndarray,
        weighting: str = "area"
) -> np.ndarray:
    """
    Average per‑face quantities onto vertices (libigl's `average_onto_vertices`
    clone, but unlimited channels and no libigl dependency).

    Parameters
    ----------
    verts : (nV, 3) float
        Vertex coordinates.
    faces : (nF, 3) int
        Triangle indices.
    face_values : (nF,) or (nF, C)
        Quantity defined per face (any number of channels).
    weighting : {"area", "uniform"}, optional
        * "area"    – weight each face by its area (default libigl behaviour).
        * "uniform" – each incident face contributes equally.

    Returns
    -------
    vertex_values : (nV,) or (nV, C) ndarray
        Averaged value(s) at each vertex.
    """
    face_values = np.asarray(face_values, dtype=np.float64)
    if face_values.ndim == 1:
        face_values = face_values[:, None]  # promote to (nF, 1)

    nV = verts.shape[0]
    nF, C = face_values.shape

    # ------------------------------------------------------------
    # Determine weights per face
    # ------------------------------------------------------------
    if weighting == "area":
        v1, v2, v3 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
        w = areas.astype(face_values.dtype)
    else:  # uniform
        w = np.ones(nF, dtype=face_values.dtype)

    # ------------------------------------------------------------
    # Accumulate weighted sums onto vertices using bincount
    # ------------------------------------------------------------
    idx = faces.ravel()  # (3*nF,)
    w_rep = np.repeat(w, 3)  # weight per corner

    v_wgt = np.bincount(idx, weights=w_rep, minlength=nV)
    v_wgt[v_wgt == 0] = 1.0

    wfv = (face_values * w[:, None])  # (nF, C) weighted face values
    wfv_rep = np.repeat(wfv, 3, axis=0)  # (3*nF, C)

    v_sum = np.zeros((nV, C), dtype=face_values.dtype)
    for c in range(C):
        v_sum[:, c] = np.bincount(idx, weights=wfv_rep[:, c], minlength=nV)

    vertex_values = v_sum / v_wgt[:, None]

    return vertex_values.squeeze()


def face_2_vertex_map(
        verts: np.ndarray,
        faces: np.ndarray,
    ) -> coo_matrix:
    """
    Linear mapping to average a function defined on mesh faces onto the vertices

    Parameters
    ------------------------
    verts : (nV, 3) float
    faces: (nF, 3) int

    Return values
    -------------------------
    F2V : (nV, nF) coo_matrix

    """
    nF = len(faces)
    nV = len(verts)

    face_areas = igl.doublearea(verts, faces)

    # Build sparse matrix directly: each face corner contributes to its vertex
    fidx = np.repeat(np.arange(nF), 3)  # face index for each corner
    vidx = faces.ravel()                 # vertex index for each corner
    a = face_areas[fidx]                 # area of each corner's face

    # Normalize per-vertex: divide each entry by sum of areas at that vertex
    vertex_area_sum = np.bincount(vidx, weights=a, minlength=nV)
    vertex_area_sum[vertex_area_sum == 0] = 1.0
    val = a / vertex_area_sum[vidx]

    F2V = coo_matrix((val, (vidx, fidx)), shape=(nV, nF))
    return F2V


def edge_triangle_adjacency(f: np.ndarray, nV: Optional[int] = None) -> List[np.ndarray]:
    """Return per-edge lists of adjacent triangle indices."""
    edges = igl.edges(f)
    nE = edges.shape[0]

    if nV is None:
        nV = np.max(f) + 1

    v2f, ni = igl.vertex_triangle_adjacency(f, nV)
    v2f_list = [v2f[ni[vv]:ni[vv + 1]] for vv in range(nV)]

    adjacency_list = [[]] * nE
    for ee, edge in enumerate(edges):
        e0_f = v2f_list[edge[0]]
        e1_f = v2f_list[edge[1]]
        adjacency_list[ee] = np.intersect1d(e0_f, e1_f)

    return adjacency_list


def surface_sample(
    v: np.ndarray, f: np.ndarray, d: float = 40, N: Optional[int] = None, res: Optional[float] = None
) -> Tuple[coo_matrix, np.ndarray, np.ndarray]:
    """wrapper including workaround for bug in igl.random_points_on_mesh that returns invalid face indices
    This function works in two steps: first, a high-density sampling on the surface using igl.random_points_on_mesh
    Then downsample using sparse_quantize.
    parameters (d, N) determine the initial sampling density, while res determines the final downsample density
    (final sampling probably more important, as long as initial sample is sufficiently dense)

    :param v: [nV x 3] vertices of mesh
    :param f: [nF x 3] facets of mesh
    :param d: (default=40) scalar density of points per unit area (nP = 2000 * surface_area * d)
    :param N: (default=None) scalar number of points to sample (overrides d if specified)
    :param res: (default=None) 1 / sample "resolution"; higher values return points closer together

    :return point_map: [nP x nV] sparse map from mesh vertices to sampled points
    fidx_pt: [nP] index into f for each sampled point
    bc_pt: [nP x 3] barycentric coordinate of each point
    """
    from bg3dtools.mesh.barycentric import bc2sparse

    n = int(1000 * np.sum(igl.doublearea(v, f)) * d) if N is None else N
    res = 500 / np.linalg.norm(np.max(v, axis=0) - np.min(v, axis=0)) if res is None else res

    bc_pt, fidx_pt = igl.random_points_on_mesh(n, v, f)[:2]  # backwards compatibility
    good = fidx_pt < f.shape[0]
    bc_pt, fidx_pt = bc_pt[good], fidx_pt[good]
    point_map = bc2sparse(f, fidx_pt, bc_pt)

    # downsample; pseudo-uniform voxel-based sampling
    point_map = point_map.tocsr()  # for row-based indexing
    dense_pts = point_map @ v
    _, idx = sparse_quantize(dense_pts * res, return_index=True)
    if len(idx) < len(bc_pt):
        point_map, bc_pt, fidx_pt = point_map[idx], bc_pt[idx], fidx_pt[idx]

    return point_map, fidx_pt, bc_pt


def mesh_volume(V: np.ndarray, F: np.ndarray) -> float:
    """
    Compute the signed volume of a closed triangle mesh.

    Parameters
    ----------
    V : (nV, 3) ndarray
        Vertex coordinates.
    F : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    volume : float
        Signed volume of the mesh (positive if normals point outward).
    """
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.einsum('ij,ij->', v0, np.cross(v1, v2)) / 6.0


def ordered_edges(faces: np.ndarray) -> np.ndarray:
    """
    Get ordered edge list from triangle mesh.

    Parameters
    ----------
    faces : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    edges : (nE, 2) ndarray
        Sorted edge list for consistency with MATLAB ordering.
    """
    edges = igl.edges(faces.astype(np.int32))
    idx = np.lexsort(edges[:, [1, 0]].T)  # for consistency with matlab edges
    return edges[idx, :]


def sample_E2V(edges, verts=None, nV=None):
    """Build a sparse (nV, nE) matrix mapping edges to incident vertices.

    Each row sums to 1. If *verts* is provided, edges are weighted by
    inverse length (shorter edges contribute more); otherwise uniform.

    Parameters
    ----------
    edges : (nE, 2) ndarray
        Edge index array.
    verts : (nV, D) ndarray, optional
        Vertex positions used for inverse-edge-length weighting.
    nV : int, optional
        Number of vertices (inferred from *edges* if omitted).

    Returns
    -------
    E2V : (nV, nE) csr_matrix
        Row-normalised edge-to-vertex sampling matrix.
    edges : (nE, 2) ndarray
        Same edge array passed in (kept for API compatibility).
    """
    nV = int(np.max(edges)) + 1 if nV is None else nV
    nE = edges.shape[0]

    # Two entries per edge: (v0, e) and (v1, e)
    row = edges.ravel()                         # [v0_0, v1_0, v0_1, v1_1, ...]
    col = np.repeat(np.arange(nE), 2)           # [0, 0, 1, 1, ...]

    if verts is not None:
        inv_len = 1.0 / np.maximum(np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1), 1e-12)
        data = np.repeat(inv_len, 2)
    else:
        data = np.ones(2 * nE)

    E2V = csr_matrix((data, (row, col)), shape=(nV, nE))

    # Row-normalise so each row sums to 1
    row_sums = np.asarray(E2V.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    E2V = E2V.multiply(1.0 / row_sums[:, None]).tocsr()

    return E2V, edges


def sparse_edge_map(edges: np.ndarray, nV: int) -> coo_matrix:
    """Build sparse (nE, nV) matrix mapping vertex positions to edge vectors."""
    nE = edges.shape[0]
    ii = np.tile(np.arange(nE)[None, :], [2, 1]).T.flatten()
    jj = edges.flatten()
    val = (np.ones(edges.shape) * np.array([1, -1])).flatten()

    return coo_matrix((val, (ii, jj)), shape=(nE, nV))


def sample_obj_vtex(
    verts: np.ndarray, faces: np.ndarray, ftex: np.ndarray, tex: np.ndarray, img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    average sample texture data from image to verts & faces of mesh
    :param faces:     index into verts
    :param ftex:      index into mesh tex
    :param tex:       U,V coordinates in image space (normalized)
    :param img:       texture image

    : return vtex: [nV x 3] per vertex colors
    : return ftex: [nF x 3] per face colors
     Benjamin Groisser 2022
    """
    nF = faces.shape[0]
    nV = verts.shape[0]
    nRows, nCols, C = img.shape

    # convert from normalized coordinates (0-1) to row/column indices
    rows = np.minimum((1-tex[:, 1]) * nRows, nRows-1)  # indexing from top left
    cols = np.minimum(tex[:, 0] * nCols, nCols-1)
    texRC = (np.column_stack((rows, cols))+0.5).astype(np.int64)
    texRC[texRC < 0] = 0

    # restack image to [nRows*nCols x nChannels]
    img_flat = img.reshape(-1, C)

    # compute internal angles for triangles
    angles = igl.internal_angles(verts, faces)

    # Vectorized: gather all texture coordinates per face corner
    tex_rc_per_face = texRC[ftex]  # (nF, 3, 2) — row/col for each corner
    tex_iidx = np.ravel_multi_index(
        (tex_rc_per_face[:, :, 0], tex_rc_per_face[:, :, 1]), img.shape[:2]
    )  # (nF, 3)
    corner_colors = img_flat[tex_iidx]  # (nF, 3, C)

    # Per-face color from face center texel
    face_center_r = np.round(tex_rc_per_face[:, :, 0].mean(axis=1)).astype(np.int64)
    face_center_c = np.round(tex_rc_per_face[:, :, 1].mean(axis=1)).astype(np.int64)
    f_color = img[face_center_r, face_center_c]  # (nF, C)

    # Accumulate angle-weighted corner colors onto vertices
    weighted_colors = corner_colors * angles[:, :, None]  # (nF, 3, C)
    vidx = faces.ravel()  # (3*nF,)
    v_color = np.zeros((nV, C), dtype=np.float64)
    vcount = np.zeros(nV, dtype=np.float64)
    for c_ch in range(C):
        v_color[:, c_ch] = np.bincount(vidx, weights=weighted_colors[:, :, c_ch].ravel(), minlength=nV)
    vcount = np.bincount(vidx, weights=angles.ravel(), minlength=nV)
    vcount[vcount == 0] = 1.0

    return (v_color / vcount[:, None]).astype(img.dtype), f_color.astype(img.dtype)


def get_genus(verts, faces):
    """
    Compute the genus of a closed mesh.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    genus : int
        Genus of the mesh (0 for sphere-like, 1 for torus-like, etc.).
    """
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    return int(1 - m.euler_number / 2)
