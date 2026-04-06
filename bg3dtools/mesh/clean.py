"""
Mesh cleaning and repair utilities.

This module provides functions for cleaning and repairing triangle meshes,
including making meshes manifold, filling holes, and removing degenerate faces.
"""

import igl
from scipy.stats import mode
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.manifold import MDS
import matplotlib.path as matpath
import scipy.sparse as sparse

from bg3dtools.mesh.utils import submesh, sample_E2V, mesh_volume, extract_manifold_patches

__all__ = [
    "bounding_box_diagonal",
    "largest_patch",
    "remove_ears",
    "repair_with_model",
    "remove_large_faces",
    "fill_hole",
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
    ears = np.atleast_1d(igl.ears(faces)[0])
    while len(ears) > 0:
        mask = np.ones(len(faces), dtype=bool)
        mask[ears] = False
        verts, faces = submesh(verts, faces, mask, return_indices=False)
        ears = np.atleast_1d(igl.ears(faces)[0])

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
        mds = MDS(n_components=2, normalized_stress='auto')

        nV = boundary_vidx.shape[0]
        seg = np.column_stack((np.arange(nV), np.arange(1, nV + 1) % nV))
        b_verts = verts[boundary_vidx]
        flatV = mds.fit_transform(b_verts)

        params = dict(vertices=flatV, segments=seg)
        import triangle
        patch = triangle.triangulate(params)

        # remove faces that are outside of polygon
        flatF = patch['triangles']
        bc = igl.barycenter(flatV, flatF).reshape(-1, 2)
        path = matpath.Path(flatV)
        flatF = flatF[path.contains_points(bc)]
        patch['triangles'] = flatF
        new_faces = boundary_vidx[flatF]
        faces = np.concatenate((faces, new_faces), axis=0)

    return faces


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

    # Fast path: use igl.is_vertex_manifold if available (igl >= 2.5)
    if hasattr(igl, 'is_vertex_manifold'):
        manifold = igl.is_vertex_manifold(faces.astype(np.int64))
        referenced = np.zeros(nV, dtype=bool)
        referenced[np.unique(faces)] = True
        nmv = np.where(referenced & ~manifold)[0]
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
    area_thresh: float = 0.000002
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Edit mesh to create a manifold watertight volume.

    Iteratively repairs the mesh by splitting non-manifold vertices,
    removing faces attached to non-manifold edges, filling holes,
    and collapsing small triangles.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.
    vtex : (nV, C) ndarray, optional
        Per-vertex texture/attribute data to propagate through repairs.
    ftex : (nF, C) ndarray, optional
        Per-face texture/attribute data to propagate through repairs.
    suspect_v : (nL,) ndarray, optional
        Indices of vertices suspected to be non-manifold.
    double_check : bool, optional
        If True, verify manifoldness after repair. Default is False.
    area_thresh : float, optional
        Threshold for collapsing small triangles. Default is 0.000002.

    Returns
    -------
    verts : (nV', 3) ndarray
        Repaired vertices.
    faces : (nF', 3) ndarray
        Repaired faces.
    vtex : (nV', C) ndarray
        Updated vertex attributes (if vtex was provided).
    ftex : (nF', C) ndarray
        Updated face attributes (if ftex was provided).

    Raises
    ------
    RuntimeError
        If repair fails to converge after 100 iterations.
    """
    # print('Edit mesh to manifold watertight volume')
    # extract largest contiguous edge-manifold patch
    if hasattr(igl, 'extract_manifold_patches'):
        p = igl.extract_manifold_patches(faces)
    elif hasattr(igl, 'facet_components'):
        p = igl.facet_components(faces)
    elif hasattr(igl, 'connected_components'):
        f_labels = igl.connected_components(faces)
        n_patches = int(f_labels.max()) + 1
        p = (n_patches, f_labels)
    else:
        raise RuntimeError('unable to extract manifold patches')

    if p[0] > 1:

        verts, faces, f_idx, v_idx = submesh(verts, faces, p[1] == mode(p[1])[0])
        ftex = None if ftex is None else ftex[f_idx]
        vtex = None if vtex is None else vtex[v_idx]
        # print('  largest manifold patch: %conv_channels v, %conv_channels f' % (verts.shape[0], faces.shape[0]))

    ii = 0
    # loop until faces don't change
    altered = True
    while altered:
        altered = False
        # print('  iteration %conv_channels' % ii)

        # split nonmanifold vertices
        nV = verts.shape[0]
        split = split_nonmanifold_verts(verts, faces, suspect_v, vtex=vtex)
        verts, faces = split[0], split[1]
        vtex = None if vtex is None else split[2]
        if verts.shape[0] > nV:
            altered = True
            # print('    split %conv_channels non-manifold vertices' % (verts.shape[0] - nV))

        # remove faces attached to non-manifold edges
        bad_edges = nonmanifold_edges(faces)
        if bad_edges.size > 0:
            altered = True
            # print('    removing faces attached to %conv_channels bad edges' % bad_edges.shape[0])
            fidx = np.logical_not(np.any(np.isin(faces, bad_edges), axis=1))
            verts, faces, f_idx, v_idx = submesh(verts, faces, fidx)
            ftex = None if ftex is None else ftex[f_idx]
            vtex = None if vtex is None else vtex[v_idx]

        # fill a single hole
        boundary_vidx = igl.boundary_loop(faces)
        if boundary_vidx.size >= 3:
            altered = True
            suspect_v = boundary_vidx
            faces = fill_hole(verts, faces, boundary_vidx)
            # print('    filled boundary loop of %conv_channels vertices' % len(boundary_vidx))
        else:
            suspect_v = []

        # fix winding/normals
        f, c = igl.bfs_orient(faces)
        if np.any(faces != f):
            altered = True
            faces = f
            # print('    corrected winding')

        # remove duplicate faces
        ff, fidx = igl.resolve_duplicated_faces(faces)
        num_removed = len(faces) - len(ff)
        if num_removed > 0:
            altered = True
            # print('    removed %conv_channels duplicate faces' % num_removed)
            verts, faces, f_idx, v_idx = submesh(verts, faces, fidx)
            ftex = None if ftex is None else ftex[f_idx]
            vtex = None if vtex is None else vtex[v_idx]
            suspect_v = np.where(np.isin(v_idx, suspect_v))[0]

        # collapse small triangles
        a = (np.min(igl.doublearea(verts, faces)) / bounding_box_diagonal(verts)**2) / 2
        if a < area_thresh:
            altered = True
            f_map = {tuple(f): idx for idx, f in enumerate(faces)}
            ff = igl.collapse_small_triangles(verts, faces, area_thresh)
            ftex = None if ftex is None else ftex[[f_map[tuple(fidx)] for fidx in ff]]

            # print('    collapsed %conv_channels small triangles' % (faces.shape[0] - ff.shape[0]))
            verts, faces, i, j = igl.remove_unreferenced(verts, ff)
            ftex = None if ftex is None else i[ftex]
            vtex = None if vtex is None else vtex[j]
            suspect_v = np.where(np.isin(j, suspect_v))[0]

        ii += 1
        if ii > 100:
            raise RuntimeError('Failed to converge after 100 iterations')

    # # for debugging
    if double_check:
        assert igl.is_edge_manifold(faces)
        assert nonmanifold_verts(faces)[0].size == 0

    p = extract_manifold_patches(faces)
    if p[0] > 1:

        verts, faces, f_idx, v_idx = submesh(verts, faces, p[1] == mode(p[1])[0])
        ftex = None if ftex is None else ftex[f_idx]
        vtex = None if vtex is None else vtex[v_idx]

    if mesh_volume(verts, faces) < 0:
        faces = faces[:, [0, 2, 1]]

    # print('  return mesh with %conv_channels v, %conv_channels f' % (verts.shape[0], faces.shape[0]))
    if vtex is None and ftex is None:
        return verts, faces
    elif ftex is None:
        return verts, faces, vtex
    elif vtex is None:
        return verts, faces, ftex
    else:
        return verts, faces, vtex, ftex

