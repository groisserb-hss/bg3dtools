"""
Mesh utility functions for triangle mesh processing.

This module provides utilities for manipulating triangle meshes including
submesh extraction, normal computation, smoothing, and surface sampling.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
from scipy.sparse import coo_matrix, csr_matrix
from bg3dtools.igl_compat import (
    AVAILABLE as IGL_AVAILABLE,
    PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,
    adjacency_matrix,
    doublearea,
    facet_components,
    internal_angles,
    random_points_on_mesh,
    remove_unreferenced,
    vertex_triangle_adjacency,
    edges as igl_edges,
    extract_manifold_patches as igl_extract_manifold_patches,
    per_face_normals as igl_per_face_normals,
    per_vertex_normals as igl_per_vertex_normals,
)
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
    "geodesic_submesh",
    "match_index_dtype",
]


def match_index_dtype(faces: np.ndarray, *arrays: np.ndarray):
    """Coerce integer index *arrays* to ``faces``'s integer dtype.

    libigl's pybind bindings require every *array* integer argument of a call
    (the faces ``f`` plus any vertex-index arrays such as ``exact_geodesic``'s
    source/target sets ``vs``/``vt``) to share ONE integer dtype, else they
    reject the call with::

        ValueError: Invalid type (int64, Row Major) for argument 'vs'.
        Expected it to match argument 'f' which is of type (int32, Row Major).

    NumPy's default integer differs by platform (int64 on Linux/macOS, int32 on
    Windows) and mesh I/O can mix the two, so the mismatch only ever surfaces on
    some hosts. Routing index arrays through here guarantees consistency
    regardless of platform or provenance. (Scalar index arguments — e.g.
    ``point_simplex_squared_distance``'s face index — are *not* affected by this
    matching and need no coercion.)

    Returns a single array when one is given, else a list in argument order.
    """
    dt = faces.dtype if faces.dtype.kind in 'iu' else np.dtype(np.int64)
    out = [np.ascontiguousarray(a, dtype=dt) for a in arrays]
    return out[0] if len(out) == 1 else out


def as_igl_faces(faces: np.ndarray) -> np.ndarray:
    """Canonicalize a face array to C-contiguous int64 for libigl.

    The class-2 companion to :func:`match_index_dtype`. The libigl bindings propagate
    the input face dtype to their outputs -- ``remove_unreferenced`` /
    ``collapse_small_triangles`` / ``decimate`` / ``upsample`` / ``bfs_orient`` hand back
    int32 when handed int32 -- and feeding int32 faces to predicates like
    ``is_edge_manifold`` / ``all_boundary_loop`` makes them report spurious
    non-manifoldness. Since NumPy's default integer is int32 on Windows, that only ever
    bit on some hosts.

    :mod:`bg3dtools.igl_compat` now pins int64 on **both** sides of every libigl call, so
    the igl boundary is covered there. This helper remains the chokepoint for keeping the
    *repo-wide* face dtype canonical -- faces that reach non-igl consumers (indexing,
    trimesh, PLY writers) or that were built by hand.

    Unlike ``match_index_dtype`` (which coerces to *faces' own* dtype, possibly int32),
    this always forces int64 -- the dtype the rest of the repo assumes.
    """
    return np.ascontiguousarray(faces, dtype=np.int64)


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
    if 'extract_manifold_patches' in IGL_AVAILABLE:
        n_patches, f_labels = igl_extract_manifold_patches(faces)
    else:
        # igl 2.6 removed extract_manifold_patches. facet_components is the
        # nearest equivalent, but it joins across the non-manifold edges that
        # extract_manifold_patches splits at, so patches come out coarser.
        f_labels = facet_components(faces)
        n_patches = int(f_labels.max()) + 1 if f_labels.size else 0

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
    new_verts, new_faces, _, v_idx = remove_unreferenced(old_verts, sub_faces)
    new_verts = new_verts.reshape([-1, n])
    # igl_compat.remove_unreferenced already returns int64; keep the pin so the dtype is
    # guaranteed for consumers even if a caller monkeypatches or reshapes around it.
    new_faces = as_igl_faces(new_faces.reshape([-1, m]))

    if return_indices:
        return new_verts, new_faces, f_idx, v_idx
    else:
        return new_verts, new_faces


def per_vertex_normals(
    verts: np.ndarray,
    faces: np.ndarray,
    mode: int = PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
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
    pt_normals = igl_per_vertex_normals(verts, faces, mode)
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
    face_normals = igl_per_face_normals(verts, faces, n).reshape([-1, 3])
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
        A = adjacency_matrix(F)
        if np.max(A) == 0:
            E = igl_edges(F)
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

    face_areas = doublearea(verts, faces)

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
    edges = igl_edges(f)
    nE = edges.shape[0]

    if nV is None:
        nV = np.max(f) + 1

    v2f, ni = vertex_triangle_adjacency(f, nV)
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

    n = int(1000 * np.sum(doublearea(v, f)) * d) if N is None else N
    res = 500 / np.linalg.norm(np.max(v, axis=0) - np.min(v, axis=0)) if res is None else res

    bc_pt, fidx_pt = random_points_on_mesh(n, v, f)[:2]  # backwards compatibility
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
    edges = igl_edges(faces)              # igl_compat returns int64 edges regardless of
    idx = np.lexsort(edges[:, [1, 0]].T)  # the input dtype, so no mismatch can be seeded
    return edges[idx, :]


def internal_edges(verts: np.ndarray, faces: np.ndarray, vweights: np.ndarray) -> np.ndarray:
    """Build segment-restricted internal "strut" edges for shape regularization.

    For each vertex, adds **one** edge to the partner vertex that maximizes a
    cross-body, same-segment, far, spread-out criterion::

        optimality = ang * segweight * d * exp(-hitcount)

    so the chosen partner is:

    - **across the body interior** -- ``ang = 1 - cos(dir_to_partner, vertex_normal)``
      is largest when the partner lies opposite the outward normal (a strut
      that spans the cavity rather than hugging the surface);
    - **in the same body segment** -- ``segweight = vweights @ vweights[v]`` (the
      blend-weight dot product) is high only when both vertices are driven by
      the same joints, so e.g. CHEST-CHEST struts form but never CHEST-HAND;
    - **far** -- ``d`` is the distance to the partner normalized by the mean
      vertex distance (long struts span a segment);
    - **spread out** -- ``exp(-hitcount)`` discourages many struts landing on
      the same partner.

    Stacked onto the face edges and fed to :func:`nonrigid_ICP` as extra
    regularization edges, these struts resist *segment-level shape change*
    (cross-section collapse / a limb being yanked toward stray points) while
    leaving the parametric body's articulation free.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Template (rest-pose) vertex positions.
    faces : (nF, 3) ndarray
        Triangle indices (used for per-vertex normals).
    vweights : (nV, nJ) ndarray
        Per-vertex linear-blend-skinning weights; their pairwise dot product
        defines body-segment co-membership.

    Returns
    -------
    edges : (nV, 2) int32 ndarray
        One internal edge ``[v, partner]`` per vertex.

    Notes
    -----
    Ported from the body-model coregistration toolbox; O(nV^2) and computed
    once on the rest pose (the resulting edge *list* is reused every frame).
    """
    vnormals = per_vertex_normals(verts, faces)
    inner_edges = np.empty([len(verts), 2], dtype=np.int32)
    hitcount = np.zeros(len(verts))

    for vv in range(len(verts)):
        # angle between direction-to-vertex and this vertex's normal
        vecs = verts - verts[vv]
        vecs = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-6)
        ang = 1 - np.sum(vecs * vnormals[vv], axis=-1)  # large => across the body
        ang[vv] = -1  # don't connect to self

        # distance from vertex (normalized by mean), prefers far partners
        d = np.linalg.norm(verts - verts[vv:vv + 1], axis=-1)
        d = d / np.mean(d)

        # restrict search to vertices in the same body segment
        segweight = np.dot(vweights, vweights[vv])

        optimality = ang * segweight * d * np.exp(-hitcount)

        paired_v = int(np.argmax(optimality))
        inner_edges[vv] = [vv, paired_v]
        hitcount[paired_v] += 1
        hitcount[vv] += 1

    return inner_edges


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
    angles = internal_angles(verts, faces)

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


def geodesic_submesh(
    verts: np.ndarray,
    faces: np.ndarray,
    center_idx: int,
    radius: float,
    method: str = "exact",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the submesh within geodesic distance of a source vertex.

    Computes geodesic distances from ``center_idx``, selects faces whose
    vertices all lie within ``radius``, and extracts the submesh with
    remapped indices.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle face indices.
    center_idx : int
        Source vertex index.
    radius : float
        Geodesic radius in mesh units (metres for metric meshes).
    method : {"exact", "heat"}, optional
        Geodesic algorithm. Default is "exact".

    Returns
    -------
    sub_verts : (nV', 3) ndarray
        Vertices of the extracted submesh.
    sub_faces : (nF', 3) ndarray
        Faces of the extracted submesh (indices into ``sub_verts``).
    orig_face_idx : (nF',) ndarray
        Original face indices corresponding to ``sub_faces``.
    orig_vert_idx : (nV',) ndarray
        Original vertex indices corresponding to ``sub_verts``.
    distances : (nV',) ndarray
        Geodesic distances of the submesh vertices from ``center_idx``.

    Examples
    --------
    >>> sv, sf, fi, vi, d = geodesic_submesh(verts, faces, 1000, 0.08)
    >>> sv.shape
    (2500, 3)
    """
    from bg3dtools.mesh.metrics import calc_geodesic

    # Geodesic distances from center vertex to all vertices
    D = calc_geodesic(
        verts, faces, np.array([center_idx]), exact=(method == "exact")
    )
    dist = D[:, 0]  # (nV,)

    # Select faces where all three vertices are within radius
    face_dists = dist[faces]  # (nF, 3)
    face_mask = np.all(face_dists <= radius, axis=1)
    f_idx = np.where(face_mask)[0]

    if f_idx.size == 0:
        empty_v = np.empty((0, 3), dtype=verts.dtype)
        empty_f = np.empty((0, 3), dtype=faces.dtype)
        return empty_v, empty_f, f_idx, np.array([], dtype=np.intp), np.array([])

    sub_verts, sub_faces, orig_face_idx, orig_vert_idx = submesh(
        verts, faces, f_idx, return_indices=True
    )

    distances = dist[orig_vert_idx]

    return sub_verts, sub_faces, orig_face_idx, orig_vert_idx, distances
