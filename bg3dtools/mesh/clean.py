"""
Mesh cleaning and repair utilities.

This module provides functions for cleaning and repairing triangle meshes,
including making meshes manifold, filling holes, and removing degenerate faces.
"""

import logging

from scipy.stats import mode
import numpy as np
from typing import Tuple, Optional, Union
import scipy.sparse as sparse

from bg3dtools.igl_compat import (
    AVAILABLE as IGL_AVAILABLE,
    all_boundary_loop,
    barycenter,
    bfs_orient,
    collapse_small_triangles,
    doublearea,
    ears as igl_ears,
    facet_components,
    is_edge_manifold,
    is_vertex_manifold,
    remove_unreferenced,
    resolve_duplicated_faces,
    triangle_triangle_adjacency,
)
from bg3dtools.mesh.utils import submesh, sample_E2V, mesh_volume, extract_manifold_patches, as_igl_faces

log = logging.getLogger(__name__)


def _mds_flatten(points: np.ndarray) -> np.ndarray:
    """Classical (Torgerson) MDS to 2D.

    Closed-form: double-center the squared-distance matrix and take the top
    two eigenvectors scaled by sqrt of their eigenvalues. For roughly planar
    boundary loops this is essentially a PCA projection; for non-planar
    loops it preserves pairwise distances as best as possible.
    """
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    D2 = (diff * diff).sum(-1)
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)  # ascending
    idx = np.argsort(w)[::-1][:2]
    w2 = np.clip(w[idx], 0.0, None)
    return V[:, idx] * np.sqrt(w2)


def _points_in_polygon(query_pts: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test."""
    qx = query_pts[:, 0:1]
    qy = query_pts[:, 1:2]
    px = polygon[:, 0]
    py = polygon[:, 1]
    px2 = np.roll(px, -1)
    py2 = np.roll(py, -1)
    straddles = (py > qy) != (py2 > qy)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_int = (px2 - px) * (qy - py) / (py2 - py) + px
    crosses = straddles & (qx < x_int)
    return (crosses.sum(axis=1) % 2).astype(bool)

__all__ = [
    "bounding_box_diagonal",
    "largest_patch",
    "remove_ears",
    "repair_with_model",
    "remove_large_faces",
    "fill_hole",
    "fill_hole_fan",
    "fill_hole_safe",
    "smooth_face_mask",
    "largest_component_mask",
    "close_end_caps",
    "nonmanifold_edges",
    "nonmanifold_verts",
    "split_nonmanifold_verts",
    "make_manifold",
]


def bounding_box_diagonal(verts: np.ndarray) -> float:
    """
    Compute the diagonal length of the axis-aligned bounding box.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.

    Returns
    -------
    diagonal : float
        Length of the bounding box diagonal.
    """
    bb_min = np.min(verts, axis=0)
    bb_max = np.max(verts, axis=0)
    return np.linalg.norm(bb_max - bb_min)


def largest_patch(
    verts: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the largest connected manifold patch from a mesh.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    verts : (nV', 3) ndarray
        Vertices of the largest patch.
    faces : (nF', 3) ndarray
        Faces of the largest patch.
    """
    p = extract_manifold_patches(faces)
    if p[0] > 1:
        verts, faces, f_idx, v_idx = submesh(verts, faces, p[1] == mode(p[1])[0])
    return verts, faces


def remove_ears(
    verts: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove ear triangles from a mesh.

    Ear triangles are faces with two boundary edges. This function
    iteratively removes them until none remain.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    verts : (nV', 3) ndarray
        Cleaned vertices.
    faces : (nF', 3) ndarray
        Cleaned faces without ear triangles.
    """
    ears = np.atleast_1d(igl_ears(faces)[0])
    while len(ears) > 0:
        mask = np.ones(len(faces), dtype=bool)
        mask[ears] = False
        verts, faces = submesh(verts, faces, mask, return_indices=False)
        ears = np.atleast_1d(igl_ears(faces)[0])

    return verts, faces


def repair_with_model(
    initV: np.ndarray, modelV: np.ndarray, edges: np.ndarray,
    E2V: Optional[sparse.spmatrix] = None, w: float = 0.25,
) -> np.ndarray:
    nV = initV.shape[-2]
    nE = edges.shape[0]
    assert modelV.shape[-2] == nV, 'modelV and regV must have same number of vertices'

    if E2V is None:
        E2V = sample_E2V(edges, modelV)[0]

    # edges are weighted higher for larger distortion
    reg_vecs = initV[edges[:, 0]] - initV[edges[:, 1]]
    model_vecs = modelV[edges[:, 0]] - modelV[edges[:, 1]]
    model_len = np.linalg.norm(model_vecs, axis=-1)
    vec_diff = np.linalg.norm(reg_vecs - model_vecs, axis=-1)
    edge_w = (vec_diff - np.mean(vec_diff)) / model_len
    edge_w = edge_w.clip(0)
    vert_w = np.exp(-3 * (E2V @ edge_w))

    # # vertices are weighted low for larger distortion
    # vert_diff = np.linalg.norm(initV - modelV, axis=-1)
    # vert_w = (vert_diff - np.mean(vert_diff)) / np.std(vert_diff)
    # vert_w = vert_w.clip(0)
    # vert_w = 1 - vert_w**2 / (4 + vert_w**2)
    # vert_w *= np.exp(-(E2V @ edge_w) / 5)

    # cartesian targets are weighted scan points
    weighted_targets = initV * vert_w[:, None]
    vertex_I = sparse.diags(vert_w, format='csr')

    # construct sparse matrix mapping edge vertices to edge vectors
    edge_w = np.ones([nE, 1])
    weighted_edges = w * model_vecs * edge_w

    eidx = np.tile(np.arange(nE), (2, 1)).T.flatten()
    vw = w * np.column_stack((edge_w, -edge_w)).flatten()
    vert_2_edge = sparse.csr_matrix((vw, (eidx, edges.flatten())), (nE, nV))

    # solve sparse linear equations
    A = sparse.vstack((vertex_I, vert_2_edge)).tocsc()  # strangely, csc seems fastest
    b = np.row_stack((weighted_targets, weighted_edges))

    # solve for new vertices
    x = sparse.linalg.lsqr(A, b[:, 0], x0=initV[:, 0])[0]
    y = sparse.linalg.lsqr(A, b[:, 1], x0=initV[:, 1])[0]
    z = sparse.linalg.lsqr(A, b[:, 2], x0=initV[:, 2])[0]
    fittedV = np.column_stack((x, y, z))

    return fittedV


def remove_large_faces(
    verts: np.ndarray, faces: np.ndarray, vtex: np.ndarray, ftex: np.ndarray,
    edge_thresh: float, return_idx: bool = False,
) -> Union[Tuple[np.ndarray, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    # remove large triangles
    e1 = np.sqrt(np.sum((verts[faces[:, 1], :] - verts[faces[:, 0], :]) ** 2, axis=1))
    e2 = np.sqrt(np.sum((verts[faces[:, 2], :] - verts[faces[:, 1], :]) ** 2, axis=1))
    e3 = np.sqrt(np.sum((verts[faces[:, 0], :] - verts[faces[:, 2], :]) ** 2, axis=1))
    fidx = np.max(np.column_stack((e1, e2, e3)), axis=1) < edge_thresh

    verts, faces, f_idx, v_idx = submesh(verts, faces, fidx)
    vtex, ftex = ftex[fidx], vtex[v_idx]

    if return_idx:
        return verts, faces, vtex, ftex, v_idx, f_idx
    else:
        return verts, faces, vtex, ftex


def fill_hole(verts: np.ndarray, faces: np.ndarray, boundary_vidx: np.ndarray) -> np.ndarray:
    len_boundary = len(boundary_vidx)

    if len_boundary < 3:
        pass

    elif len(boundary_vidx) == 3:
        faces = np.row_stack([faces, boundary_vidx])

    elif len(boundary_vidx) > 3:
        nV = boundary_vidx.shape[0]
        seg = np.column_stack((np.arange(nV), np.arange(1, nV + 1) % nV))
        b_verts = verts[boundary_vidx]
        flatV = _mds_flatten(b_verts)

        params = dict(vertices=flatV, segments=seg)
        import triangle
        patch = triangle.triangulate(params)

        # remove faces that are outside of polygon
        flatF = patch['triangles']
        bc = barycenter(flatV, flatF).reshape(-1, 2)
        flatF = flatF[_points_in_polygon(bc, flatV)]
        patch['triangles'] = flatF
        new_faces = boundary_vidx[flatF]
        faces = np.concatenate((faces, new_faces), axis=0)

    return faces


def fill_hole_fan(
    verts: np.ndarray,
    faces: np.ndarray,
    boundary_vidx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Close a boundary loop with a fan triangulation around a new centroid.

    Adds one new vertex (the boundary centroid) and `len(boundary_vidx)` new
    triangles, each spanning the centroid and a consecutive pair of
    boundary vertices.

    Topology guarantees: every new edge is either a boundary edge of the
    input (now incident to exactly 1 pre-existing face + 1 new fan
    triangle = manifold) or a centroid spoke (incident to exactly 2 new
    fan triangles = manifold). Unlike `fill_hole`, which 2D-flattens the
    boundary and Delaunay-triangulates, this never introduces a
    non-manifold edge — making it safe to use inside an iterative repair
    loop that would otherwise cycle on bad fills.

    Triangle quality is poor for non-convex / non-planar boundaries (long
    thin spokes), but for downstream boolean / inflation workflows where
    only topology matters, this is the right primitive.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.
    boundary_vidx : (nB,) ndarray
        Ordered boundary loop vertex indices (as returned by
        `igl.boundary_loop` or `igl.all_boundary_loop`).

    Returns
    -------
    verts : (nV + 1, 3) ndarray
        Vertices with the centroid appended.
    faces : (nF + nB, 3) ndarray
        Faces with the fan triangles appended. Winding follows the order
        of `boundary_vidx`, which preserves consistency with the existing
        mesh orientation when the boundary came from libigl.
    """
    boundary_vidx = np.asarray(boundary_vidx).ravel()
    n = boundary_vidx.shape[0]
    if n < 3:
        return verts, faces

    # Append the boundary centroid; its index is c_idx in the new array.
    centroid = verts[boundary_vidx].mean(axis=0, keepdims=True)
    new_verts = np.vstack([verts, centroid])
    c_idx = new_verts.shape[0] - 1

    # n triangles: (c, v_{i+1}, v_i) wrapped circularly. The reversed
    # order (v_{i+1}, v_i) is intentional: libigl's boundary loops are
    # returned in the same direction the adjacent faces traverse the
    # boundary edges, so a fan triangle wound (c, v_i, v_{i+1}) would
    # traverse the shared edge in the same direction as the existing
    # face — algebraically non-manifold. Reversing makes the fan
    # triangle traverse the edge in the opposite direction, restoring
    # manifold consistency.
    v_a = boundary_vidx
    v_b = np.roll(boundary_vidx, -1)
    fan = np.column_stack([
        np.full(n, c_idx, dtype=faces.dtype),
        v_b.astype(faces.dtype),
        v_a.astype(faces.dtype),
    ])

    new_faces = np.vstack([faces, fan])
    return new_verts, new_faces


def fill_hole_safe(
    verts: np.ndarray,
    faces: np.ndarray,
    boundary_vidx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill a hole with Delaunay if it stays edge-manifold, else fan.

    Delaunay (`fill_hole`) gives a higher-quality triangulation but on
    noisy / non-planar boundaries can produce triangles that, when
    stitched to the existing mesh, leave some edge incident to >2
    faces — algebraically non-manifold. This wrapper validates the
    Delaunay candidate by calling `igl.is_edge_manifold` and falls back
    to `fill_hole_fan` whenever that check fails or whenever the input
    itself was already non-edge-manifold (the latter case being one
    where Delaunay's output can't be meaningfully validated).

    `is_edge_manifold` is the correct gate here (counts faces per edge,
    independent of winding direction). Winding inconsistencies that
    `fill_hole` is prone to producing on libigl-ordered boundary loops
    will be repaired by downstream `bfs_orient` and aren't a reason to
    reject the Delaunay result.

    Fan triangulation is manifold-preserving by construction (each new
    edge is either a centroid spoke or an original boundary edge), so
    the fallback is guaranteed safe — at the cost of one centroid
    vertex and lower-quality triangles.

    Always returns `(verts, faces)`. The Delaunay path returns `verts`
    unchanged (Delaunay doesn't add vertices); the fan path appends
    one centroid.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.
    boundary_vidx : (nB,) ndarray
        Ordered boundary loop vertex indices.

    Returns
    -------
    verts : (nV, 3) or (nV + 1, 3) ndarray
        Original verts on the Delaunay path; centroid appended on the
        fan path.
    faces : (nF', 3) ndarray
        Faces with the hole closed.
    """
    faces = as_igl_faces(faces)  # int64: keep the returned candidate faces canonical
    if is_edge_manifold(faces):
        try:
            candidate = fill_hole(verts, faces, boundary_vidx)
            if (candidate.shape[0] > faces.shape[0]
                    and is_edge_manifold(candidate)):
                return verts, candidate
        except Exception:
            # Delaunay path raised (e.g., triangle library rejected the
            # boundary polygon); fall through to fan.
            pass
    return fill_hole_fan(verts, faces, boundary_vidx)


def smooth_face_mask(
    faces: np.ndarray,
    mask: np.ndarray,
    n_iters: int = 5,
    weight: float = 0.5,
) -> np.ndarray:
    """Diffusion-smooth a per-face mask and re-binarize.

    Useful for cleaning up noisy per-face classifications (e.g. from a
    ray-visibility test). Each iteration mixes each face's current value
    with the mean of its edge-neighbors; after `n_iters` iterations the
    result is thresholded at 0.5.

    Missing neighbors (boundary edges, encoded as -1 by libigl) are
    substituted with the face's own value, so boundary faces aren't biased
    by the choice of mesh boundary.

    Parameters
    ----------
    faces : (nF, 3) ndarray
        Triangle indices.
    mask : (nF,) bool or float ndarray
        Initial per-face values; converted to float for smoothing.
    n_iters : int
        Number of diffusion iterations. Default 5.
    weight : float
        Per-iteration mix ratio: x_new = (1 - w) * x_old + w * mean(neighbors).
        Default 0.5.

    Returns
    -------
    smoothed : (nF,) bool ndarray
        Cleaned mask, thresholded at 0.5.
    """
    TT, _ = triangle_triangle_adjacency(faces)
    self_idx = np.arange(len(faces))[:, None]
    neigh = np.where(TT >= 0, TT, self_idx)

    x = np.asarray(mask, dtype=np.float32)
    for _ in range(n_iters):
        neighbor_mean = x[neigh].mean(axis=1)
        x = (1 - weight) * x + weight * neighbor_mean
    return x > 0.5


def largest_component_mask(faces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restrict a per-face mask to its largest edge-connected component.

    Connectivity is defined among masked faces only (two masked faces are
    connected iff they share an edge in the original mesh). Faces outside
    `mask` are returned False.

    Useful for removing stray classified faces that are spatially
    disconnected from the dominant region.

    Parameters
    ----------
    faces : (nF, 3) ndarray
        Triangle indices of the full mesh.
    mask : (nF,) bool ndarray
        Selection mask.

    Returns
    -------
    keep : (nF,) bool ndarray
        Subset of `mask` containing only its largest connected component.
        All-False if the input mask was all-False.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()

    sub_idx = np.where(mask)[0]
    labels = facet_components(faces[sub_idx])
    largest = int(np.bincount(labels).argmax())

    out = np.zeros(len(faces), dtype=bool)
    out[sub_idx[labels == largest]] = True
    return out


def close_end_caps(
    verts: np.ndarray,
    faces: np.ndarray,
    n_expected: int = 2,
    rel_size_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fan-fill the N longest boundary loops, treating them as expected open ends.

    For surfaces with a known number of intentional open boundaries (e.g. 2
    for an open cylinder), this closes exactly those loops by descending
    length and leaves any remaining smaller boundaries untouched — those
    are typically noise or small holes and are better handled by
    downstream repair (e.g. `make_manifold`).

    Uses `fill_hole_safe` for closure: each cap first tries the higher-
    quality Delaunay triangulation, falling back to fan only if Delaunay
    would introduce non-manifold edges. Either way, the closure is
    manifold-consistent with the rest of the mesh.

    A boundary loop is considered "large" if its length is at least
    `rel_size_threshold` times the longest loop's length. The function
    returns the total count of large loops so the caller can flag
    unexpected topology (e.g. a fragmented surface that yielded more large
    loops than expected).

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.
    n_expected : int
        Number of large loops to fill. Default 2 (cylinder end caps).
    rel_size_threshold : float
        Minimum loop length, relative to the longest loop, to count as
        "large". Default 0.05.

    Returns
    -------
    verts : (nV', 3) ndarray
        Vertices, with one new centroid appended per filled loop.
    faces : (nF', 3) ndarray
        Faces with the `n_expected` longest large loops closed.
    n_large_loops : int
        Total number of large loops detected. Equals `n_expected` in the
        happy case; differs when topology is unexpected.
    """
    faces = as_igl_faces(faces)  # int64: keep the returned loop/face arrays canonical
    loops = all_boundary_loop(faces)
    if not loops:
        return verts, faces, 0

    loops = sorted(loops, key=len, reverse=True)
    longest_len = len(loops[0])
    large_loops = [L for L in loops if len(L) >= rel_size_threshold * longest_len]

    for loop in large_loops[:n_expected]:
        verts, faces = fill_hole_safe(verts, faces, np.asarray(loop))

    return verts, faces, len(large_loops)


def nonmanifold_edges(faces: np.ndarray, return_counts: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
      Inputs:
        F  #F by dim=3 list of facet indices
      Outputs:
        E  #E by 2 list of nonmanifold edges
        C  #E by 1 list of unsigned Counts (>2)
    """
    all_edges = np.row_stack((faces[:,[1,2]], faces[:,[2,0]], faces[:,[0,1]]))
    sorted_edges = np.sort(all_edges)

    # entries are summed together when converting to CSR format
    C = sparse.coo_matrix((np.ones_like(sorted_edges[:,0]), (sorted_edges[:,0], sorted_edges[:,1]))).tocsr()
    direction = 2*(all_edges[:,0] < all_edges[:,1]) - 1
    sC = sparse.coo_matrix((direction, (sorted_edges[:,0], sorted_edges[:,1]))).tocsr()

    # nonmanifold edges
    idx = C.nonzero()
    a = np.array(C[idx]).squeeze()  # number of bordering faces
    b = np.array(sC[idx]).squeeze()  # signed neighbors
    cc = np.logical_or(a > 2, np.abs(b) > 1)
    nme = np.column_stack(idx)[cc]

    if return_counts:
        return nme, a[cc]
    else:
        return nme


def nonmanifold_verts(faces: np.ndarray, nV: Optional[int] = None) -> Tuple[np.ndarray, sparse.spmatrix]:
    if nV is None:
        nV = np.max(faces) + 1

    # construct sparse matrix to navigate neighborhood
    nF = faces.shape[0]
    ff = np.column_stack(3*[np.arange(nF)]).flatten()
    V2F = sparse.csr_matrix((np.ones(3*nF), (faces.flatten(), ff)), shape=(nV, nF))

    # Fast path: libigl's own predicate, present only in igl >= 2.6 (see igl_compat)
    if 'is_vertex_manifold' in IGL_AVAILABLE:
        # The mask is only as long as the highest referenced index, which falls
        # short of nV whenever trailing vertices are unreferenced; those are
        # never non-manifold, so pad with True.
        manifold = is_vertex_manifold(faces)
        if manifold.size < nV:
            manifold = np.concatenate([manifold, np.ones(nV - manifold.size, dtype=bool)])
        referenced = np.zeros(nV, dtype=bool)
        referenced[np.unique(faces)] = True
        nmv = np.where(referenced & ~manifold[:nV])[0]
    else:
        # Fallback: per-vertex patch check
        nmv = []
        for vv in range(nV):
            fidx = V2F[vv].nonzero()[1]
            if fidx.size == 0:
                continue
            neighborhood = faces[fidx]
            if extract_manifold_patches(neighborhood)[0] > 1:
                nmv.append(vv)
        nmv = np.array(nmv)

    return nmv, V2F[nmv]


def split_nonmanifold_verts(
    verts: np.ndarray, faces: np.ndarray,
    suspect_v: Optional[np.ndarray] = None, vtex: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    orig_nV = verts.shape[0]
    nF = faces.shape[0]
    faces = np.copy(faces)

    if suspect_v is None:
        suspect_v = np.arange(orig_nV)

    # construct initial sparse matrix to navigate neighborhoods
    ff = np.column_stack(3*[np.arange(nF)]).flatten()
    V2F = sparse.coo_matrix((np.ones(3*nF), (faces.flatten(), ff)), shape=(orig_nV, nF)).tocsr()

    # edge_manifold patches neighboring vv
    for vv in suspect_v:
        fidx = V2F[vv].nonzero()[1]  # nonzero column indices are connected faces
        if fidx.size < 2:
            # no need to split if fewer than 2 faces attached
            continue
        neighborhood = faces[fidx]
        patches = extract_manifold_patches(neighborhood)

        if patches[0] == 1:
            # patch around vv is already manifold
            continue

        # extend verts to accommodate new vertices
        old_nV = verts.shape[0]
        verts = np.row_stack((verts, np.tile(verts[vv], [patches[0]-1, 1])))
        if vtex is not None:
            vtex = np.row_stack((vtex, np.tile(vtex[vv], [patches[0]-1, 1])))
        new_vidx = np.arange(old_nV, verts.shape[0]+1)
        new_vidx[-1] = vv

        # # extend sparse matrix-- not actually necessary since we never call face neighbors
        # V2F._shape = (verts.shape[0], nF)
        # V2F.indptr = np.pad(V2F.indptr, [0, patches[0]-1], constant_values=V2F.indptr[-1])
        # V2F = V2F.reshape(verts.shape[0], nF)

        for pp, nn in enumerate(new_vidx):
            # update faces to new vertex id
            pidx = fidx[patches[1] == pp]  # faces in this sub-patch
            patch = faces[pidx]
            patch[patch == vv] = nn  # replace with new vertex id
            faces[pidx] = patch

            # slightly move vertex towards patch center
            pverts = np.unique(patch)
            offset = np.mean(verts[pverts], axis=0) - verts[vv]
            eps = np.std(verts[pverts], axis=0) / 100
            verts[nn] = verts[vv] + offset * eps

            # # update sparse matrix -- not necessary?
            # V2F[vv, pidx] = 0
            # V2F[nn, pidx] = 1

    if vtex is None:
        return verts, faces
    else:
        return verts, faces, vtex


def make_manifold(
    verts: np.ndarray,
    faces: np.ndarray,
    vtex: Optional[np.ndarray] = None,
    ftex: Optional[np.ndarray] = None,
    suspect_v: Optional[np.ndarray] = None,
    double_check: bool = False,
    area_thresh: float = 0.000002,
    max_iters: int = 10,
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Convert to a manifold watertight mesh via fan-triangulated hole filling.

    Repairs the input by iteratively (a) restricting to the largest
    manifold patch, (b) splitting non-manifold vertices, (c) removing
    faces incident to non-manifold edges, and (d) fan-filling every
    boundary loop. Fan triangulation is provably manifold-preserving
    (each new edge is incident to exactly two of the new fan triangles,
    or one fan triangle + one pre-existing boundary face), so step (d)
    cannot feed step (c) on the next iteration — breaking the fill /
    remove cycle that a Delaunay-based hole fill would cause on noisy
    inputs.

    Once the loop converges to a closed, edge-manifold, vertex-manifold
    state, `collapse_small_triangles` runs *once* to clean up thin
    slivers introduced by fan triangulation. Any new boundaries created
    by the collapse are fan-filled in a single follow-up pass and the
    result re-verified; if that cosmetic collapse cannot be kept manifold
    + closed it is discarded in favour of the (already manifold + closed)
    pre-collapse mesh. The function only raises `RuntimeError` if the main
    repair loop itself fails to reach a manifold + closed state — it never
    returns a non-manifold mesh.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.
    vtex : (nV, C) ndarray, optional
        Per-vertex attributes; propagated through repairs. For each
        added fan centroid, the new vertex's attribute is the mean of
        the boundary vertices' attributes.
    ftex : (nF, C) ndarray, optional
        Per-face attributes; propagated through repairs. For each added
        fan triangle, the new row is NaN (synthetic face).
    suspect_v : (nL,) ndarray, optional
        Indices of vertices suspected to be non-manifold; used to scope
        the first `split_nonmanifold_verts` pass. Subsequent iterations
        re-scan all vertices.
    double_check : bool, optional
        If True, assert manifoldness after repair. Default False.
    area_thresh : float, optional
        Area threshold (relative to bbox-diagonal²) below which the
        post-loop `collapse_small_triangles` pass collapses thin
        slivers. Default 2e-6.
    max_iters : int, optional
        Convergence iteration cap. Default 10. Typical inputs converge
        in 1–2; the cap exists to fail fast on truly pathological
        inputs.

    Returns
    -------
    verts : (nV', 3) ndarray
    faces : (nF', 3) ndarray
    vtex : (nV', C) ndarray, only if `vtex` was provided.
    ftex : (nF', C) ndarray, only if `ftex` was provided.

    Raises
    ------
    RuntimeError
        If the main repair loop cannot bring the mesh to a manifold +
        closed state within `max_iters` iterations. A post-loop collapse
        that breaks manifoldness is non-fatal: the pre-collapse mesh is
        kept instead.
    """

    def _largest_patch(verts, faces, vtex, ftex):
        n_patches, labels = extract_manifold_patches(faces)
        if n_patches > 1:
            keep = labels == mode(labels)[0]
            verts, faces, f_idx, v_idx = submesh(verts, faces, keep)
            if vtex is not None:
                vtex = vtex[v_idx]
            if ftex is not None:
                ftex = ftex[f_idx]
        return verts, faces, vtex, ftex

    def _is_manifold_and_closed(faces):
        # all_boundary_loop / is_edge_manifold report spurious non-manifoldness on int32
        # faces; igl_compat pins int64 at the igl boundary, this keeps the local copy so too.
        faces = as_igl_faces(faces)
        return (not all_boundary_loop(faces)
                and is_edge_manifold(faces)
                and nonmanifold_verts(faces)[0].size == 0)

    def _fill_all_boundaries(verts, faces, vtex, ftex):
        """Close every boundary loop via fill_hole_safe; propagate vtex/ftex.

        fill_hole_safe may take the Delaunay path (no new vertex, adds
        ~len(loop)-2 faces) or the fan path (adds 1 centroid + len(loop)
        faces). vtex/ftex are extended by inspecting the actual change in
        verts.shape[0] and faces.shape[0] after the call, so the same
        helper handles both paths.
        """
        for loop in all_boundary_loop(faces):
            loop = np.asarray(loop)
            # Cache the centroid attribute now (the fan path would consume it
            # before vtex grows). Delaunay path simply ignores it.
            staged_centroid_attr = (
                vtex[loop].mean(axis=0, keepdims=True).astype(vtex.dtype)
                if vtex is not None else None
            )
            nV_before, nF_before = verts.shape[0], faces.shape[0]
            verts, faces = fill_hole_safe(verts, faces, loop)
            added_verts = verts.shape[0] - nV_before
            added_faces = faces.shape[0] - nF_before
            if vtex is not None and added_verts > 0:
                vtex = np.vstack([vtex, staged_centroid_attr])
            if ftex is not None and added_faces > 0:
                ftex = np.vstack([
                    ftex,
                    np.full((added_faces, ftex.shape[1]), np.nan, dtype=ftex.dtype),
                ])
        return verts, faces, vtex, ftex

    # The manifold predicates below misbehave on int32 faces; canonicalize once up front
    # so the whole routine stays int64 whatever the caller passed in.
    faces = as_igl_faces(faces)

    # Initial: keep only the largest manifold patch.
    verts, faces, vtex, ftex = _largest_patch(verts, faces, vtex, ftex)

    # Main repair loop: split / remove-bad / fan-fill / orient / dedup.
    for _it in range(max_iters):
        # Split non-manifold vertices. `suspect_v` is honored on the first
        # iteration (per caller hint); subsequent iters re-scan all verts.
        split = split_nonmanifold_verts(verts, faces, suspect_v, vtex=vtex)
        verts, faces = split[0], split[1]
        if vtex is not None:
            vtex = split[2]
        suspect_v = None

        # Fix winding BEFORE testing for non-manifold edges. The
        # `nonmanifold_edges` helper flags both ">2-incident" edges
        # (true non-manifoldness) AND winding-inconsistent edges
        # (recoverable by reorientation). Without this step, a fresh
        # Delaunay fill — wound in the same direction as the existing
        # boundary traversal — would flag every boundary edge of the
        # fill, prompting catastrophic face removal.
        oriented, _ = bfs_orient(faces)
        faces = np.asarray(oriented, dtype=faces.dtype)

        # Now remove faces incident to truly non-manifold edges
        # (>2-incident — winding inconsistencies are gone after orient).
        bad = nonmanifold_edges(faces)
        if bad.size > 0:
            keep = np.logical_not(np.any(np.isin(faces, bad), axis=1))
            verts, faces, f_idx, v_idx = submesh(verts, faces, keep)
            if vtex is not None:
                vtex = vtex[v_idx]
            if ftex is not None:
                ftex = ftex[f_idx]

        # Close every boundary loop (Delaunay where it keeps things
        # edge-manifold, else fan). The fill may introduce
        # winding-inconsistent edges that the next iteration's
        # bfs_orient will clean up.
        verts, faces, vtex, ftex = _fill_all_boundaries(verts, faces, vtex, ftex)

        # Re-orient again so the convergence check below sees a
        # consistently-wound mesh.
        oriented, _ = bfs_orient(faces)
        faces = np.asarray(oriented, dtype=faces.dtype)

        # Drop duplicate faces.
        ff, fidx = resolve_duplicated_faces(faces)
        if len(faces) > len(ff):
            verts, faces, f_idx, v_idx = submesh(verts, faces, fidx)
            if vtex is not None:
                vtex = vtex[v_idx]
            if ftex is not None:
                ftex = ftex[f_idx]

        converged = _is_manifold_and_closed(faces)
        if not converged and _it >= 2:
            # Diagnostic for pathological inputs (most converge in 1-2 iters).
            log.info(
                'make_manifold slow to converge: iter %d/%d, nV=%d, nF=%d, '
                'nonmanifold_edges=%d',
                _it + 1, max_iters, len(verts), len(faces),
                nonmanifold_edges(faces).size,
            )
        if converged:
            break
    else:
        raise RuntimeError(
            f'make_manifold did not converge after {max_iters} iterations'
        )

    # Post-loop: a single pass of small-triangle collapse, then one
    # follow-up fan-fill in case collapse created new boundaries.
    a = (np.min(doublearea(verts, faces)) / bounding_box_diagonal(verts) ** 2) / 2
    if a < area_thresh:
        # The main loop above already produced a manifold + closed mesh;
        # this collapse is only cosmetic thin-sliver cleanup. Pin int64
        # face indices across it: some libigl wheels (notably the Windows
        # build) return int32 faces from collapse_small_triangles /
        # remove_unreferenced, and those int32 arrays then make the
        # downstream manifold checks report spurious non-manifoldness --
        # the same dtype sensitivity the is_vertex_manifold call in
        # nonmanifold_verts() already guards against. Snapshot the
        # known-good pre-collapse mesh first so a still-failing collapse
        # can fall back to it rather than aborting the whole pipeline.
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        safe = (verts, faces, vtex, ftex)
        f_map = {tuple(f): idx for idx, f in enumerate(faces)}
        collapsed = np.ascontiguousarray(
            collapse_small_triangles(verts, faces, area_thresh), dtype=np.int64)
        if ftex is not None:
            ftex = ftex[[f_map[tuple(fc)] for fc in collapsed]]
        verts, faces, i, j = remove_unreferenced(verts, collapsed)
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        if vtex is not None:
            vtex = vtex[j]
        # Collapse may have opened new tiny boundaries; close them once.
        verts, faces, vtex, ftex = _fill_all_boundaries(verts, faces, vtex, ftex)
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        if not _is_manifold_and_closed(faces):
            # The cosmetic collapse could not be kept manifold + closed.
            # Discard it and keep the pre-collapse mesh (still manifold +
            # closed, only with the thin slivers left uncollapsed) rather
            # than failing the whole run.
            log.warning(
                'make_manifold: small-triangle collapse could not be kept '
                'manifold + closed; retaining the pre-collapse mesh '
                '(%d verts, %d faces, thin slivers uncollapsed).',
                len(safe[0]), len(safe[1]),
            )
            verts, faces, vtex, ftex = safe

    if double_check:
        assert is_edge_manifold(faces)
        assert nonmanifold_verts(faces)[0].size == 0

    # Drop any disjoint debris introduced by repairs.
    verts, faces, vtex, ftex = _largest_patch(verts, faces, vtex, ftex)

    # Flip if inside-out.
    if mesh_volume(verts, faces) < 0:
        faces = faces[:, [0, 2, 1]]

    if vtex is None and ftex is None:
        return verts, faces
    elif ftex is None:
        return verts, faces, vtex
    elif vtex is None:
        return verts, faces, ftex
    else:
        return verts, faces, vtex, ftex



