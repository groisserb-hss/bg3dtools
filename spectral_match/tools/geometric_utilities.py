"""
Geometric utility functions for spectral matching.

Provides mesh normalization, geodesic distance computation, orthogonal
Procrustes alignment, boundary extraction, and sampling utilities.
"""

# Derived from pyFM by Robin Magnet (MIT License) — see /THIRD_PARTY_NOTICES.txt
import math

import igl
import numpy as np
from joblib import Memory, Parallel, delayed
from scipy.spatial.distance import cdist

from bg3dtools.mesh.laplace import biharmonic_embedding
from bg3dtools.mesh.utils import match_index_dtype

# To have a cache for computations which are taking time to complete
memory = Memory(location=".joblib_cache", verbose=0)


""" ================================================================================= """
"""                          Point Cloud Utilities                                    """
""" ================================================================================= """


def normalize_mesh(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre a mesh at the origin and scale so total surface area is 1."""
    v = v - np.mean(v, axis=0)
    v = v / math.sqrt(area(v, f))
    return v, f


def difference_matrix(
    array1: np.ndarray, array2: np.ndarray | None = None, norm: bool = True
) -> np.ndarray:

    """
    Pairwise distances between points in two arrays. Scales like N*M
    where N and M are the sizes of the point clouds in question.

    Args:
    array1: An array of shape [N1,D], where D is the dimension.
    array2: An array of shape [N2,D].

    Returns
    An array of shape [N2,N1].
    """

    if array2 is None:
        array2 = array1

    if len(array1.shape) == 1:
        array1 = np.expand_dims(array1, 1)

    if len(array2.shape) == 1:
        array2 = np.expand_dims(array2, 1)

    if norm:
        return cdist(array2, array1)

    # Non-norm case: return full difference tensor (N2, N1, D)
    return array2[:, np.newaxis, :] - array1[np.newaxis, :, :]


""" ================================================================================= """
"""                            Alignment Operations                                   """
""" ================================================================================= """


def centre_and_norm(X: np.ndarray) -> None:
    """Centre *X* at the origin and scale to unit RMS radius (in-place)."""
    X -= np.mean(X, axis=0)
    X /= np.sqrt(np.mean(np.sum(X ** 2, axis=-1)))
    return


def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Find rotation R that best aligns Y @ R to X (Procrustes)."""
    U, _, VT = np.linalg.svd(X.T @ Y)
    R = (VT.T).dot(U.T)
    return R


""" ================================================================================= """
"""                               Mesh Operations                                     """
""" ================================================================================= """


def reorder_mesh(
    v: np.ndarray, f: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reindex vertices according to *idx* and update face references."""
    # Build inverse permutation: inv_perm[old_idx] = new_idx
    inv_perm = np.empty(v.shape[0], dtype=f.dtype)
    inv_perm[idx] = np.arange(len(idx), dtype=f.dtype)
    nf = inv_perm[f]
    nv = v[idx]
    return nv, nf


def extract_edges(faces: np.ndarray) -> np.ndarray:
    """Return unique sorted edges (nE, 2) from a triangle mesh."""
    if not isinstance(faces, np.ndarray):
        faces = np.array(faces)

    edges = np.concatenate(
        [faces[:, i] for i in ([0, 1], [1, 2], [2, 0])], axis=0
    )
    edges = np.sort(edges, axis=1)
    _, ind = np.unique(edges, return_index=True, axis=0)
    return edges[ind]


def edge_lengths(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute the Euclidean length of every unique edge."""
    i, j = extract_edges(f).T
    return np.linalg.norm(v[i] - v[j], axis=-1)


def boundary_vertices(faces: np.ndarray) -> np.ndarray:
    """Return unique vertex indices that lie on the mesh boundary."""
    if not isinstance(faces, np.ndarray):
        faces = np.array(faces)
    e = np.concatenate([faces[:, i] for i in ([0, 1], [1, 2], [2, 0])], axis=0)
    e = np.sort(e, axis=1)
    e, c = np.unique(e, return_counts=True, axis=0)
    return np.unique(e[c == 1])


def mesh_neighbours(
    faces: np.ndarray,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """Group vertices by valence and return their 1-ring neighbours."""
    from scipy.sparse import csr_matrix

    edges = extract_edges(faces)
    p, q = edges.T
    nV = int(faces.max()) + 1

    # Build sparse adjacency for fast neighbour lookup
    data = np.ones(2 * len(edges), dtype=np.int32)
    rows = np.concatenate([p, q])
    cols = np.concatenate([q, p])
    adj = csr_matrix((data, (rows, cols)), shape=(nV, nV))

    nodes = np.unique(faces)
    valences = np.diff(adj.indptr)  # per-vertex degree
    node_valences = valences[nodes]

    # Group by valence
    unique_vals = np.unique(node_valences)
    numbers_list = unique_vals.tolist()
    nodes_list = []
    neighbours_list = []
    for val in unique_vals:
        mask = node_valences == val
        group_nodes = nodes[mask]
        nodes_list.append(group_nodes)
        # Gather neighbours for each node in this group
        nhbrs = np.stack([adj.indices[adj.indptr[n]:adj.indptr[n+1]] for n in group_nodes])
        neighbours_list.append(nhbrs)

    return numbers_list, nodes_list, neighbours_list


@memory.cache
def geodesic_matrix(
    v: np.ndarray,
    f: np.ndarray,
    i1: np.ndarray = np.array([]),
    i2: np.ndarray = np.array([]),
) -> np.ndarray:
    """
    Find the pairwise geodesic distances between a set of vertices
    within a mesh. Make sure that i1.size < i2.size
    """
    if i1.size == 0:
        i1 = np.arange(v.shape[0])

    if i2.size == 0:
        i2 = i1

    # libigl requires the source/target index arrays to share f's integer dtype
    # (else it raises on hosts where NumPy's default int differs from f's, e.g.
    # int32 faces vs int64 indices on Windows). Harmonize both to f.dtype.
    i2 = match_index_dtype(f, i2)
    func = lambda i: igl.exact_geodesic(v, f, match_index_dtype(f, np.array([i])), i2)
    d = Parallel(n_jobs=-1)(delayed(func)(j) for j in i1)
    return np.stack(d)


@memory.cache
def biharmonic_matrix(
    v: np.ndarray,
    f: np.ndarray,
    i1: np.ndarray = np.array([]),
    i2: np.ndarray = np.array([]),
    dim: int | None = None,
) -> np.ndarray:
    """
    Find the pairwise biharmonic distances between a set of vertices
    within a mesh.
    """
    if i1.size == 0:
        i1 = np.arange(v.shape[0])

    if i2.size == 0:
        i2 = i1

    if dim is None:
        dim = len(v) // 10

    B = biharmonic_embedding(v, f, dim=dim, p=2)
    d = cdist(B[i1], B[i2])
    return d


def area(v: np.ndarray, f: np.ndarray) -> np.floating:
    """Total surface area of a triangle mesh."""
    return face_areas(v, f).sum()


def face_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Per-face areas of a triangle mesh."""
    return igl.doublearea(v, f) / 2


def face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Unit face normals of a triangle mesh."""
    a, b, c = f.T
    normals = np.cross(v[b] - v[a], v[c] - v[a])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
    return normals


""" ================================================================================= """
"""                               Propagation Ops                                     """
""" ================================================================================= """


def metric_sampling(g: np.ndarray, target: int) -> np.ndarray:
    """Greedy farthest-point sampling using a precomputed distance matrix."""
    idx = [np.argmax(g.mean(axis=0))]
    while len(idx) < target:
        idx.append(np.argmax(g[idx].min(axis=0)))
    return np.asarray(idx)


def propogate_points(
    src: np.ndarray, dst: np.ndarray, points: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """Transfer points from *src* to *dst* mesh preserving barycentric + normal offset."""
    n_src = face_normals(src, f)
    n_dst = face_normals(dst, f)
    _, i, q = igl.point_mesh_squared_distance(points, src, f)
    l = np.linalg.norm((points - q) * n_src[i], axis=-1, keepdims=True)
    j = f[i]
    u, v, w = np.asarray(src[j.T], order="C", dtype=q.dtype)
    bc = igl.barycentric_coordinates_tri(q, u, v, w)
    bc[np.isnan(bc).any(axis=-1)] = 1 / 3
    pts = np.sum(bc[..., np.newaxis] * dst[j], axis=1)
    pts += l * n_dst[i]
    return pts


def extrapolate_geodesic_matrix(
    src: np.ndarray, dst: np.ndarray, g: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """Approximate geodesic distances on *dst* from those computed on *src*."""
    _, i, q = igl.point_mesh_squared_distance(dst, src, f)
    j = f[i]
    l = np.linalg.norm(src[j] - np.expand_dims(q, 1), axis=-1)
    h = (g[j] + np.expand_dims(l, -1)).min(axis=1)
    h = (h[:, j] + np.expand_dims(l, 0)).min(axis=-1)
    np.fill_diagonal(h, 0)
    return 0.5 * (h + h.T)


def extrapolate_scalars(
    src: np.ndarray, dst: np.ndarray, scalars: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """Interpolate per-vertex *scalars* from *src* mesh onto *dst* points."""
    if len(scalars.shape) == 1:
        scalars = scalars[..., np.newaxis]
    _, i, q = igl.point_mesh_squared_distance(dst, src, f)
    j = f[i]
    u, v, w = np.asarray(src[j.T], order="C", dtype=np.double)
    bc = igl.barycentric_coordinates_tri(q, u, v, w)
    return np.sum(bc[..., np.newaxis] * scalars[j], axis=1)


""" ================================================================================= """
"""                                        Misc                                       """
""" ================================================================================= """


def sign(x: np.ndarray) -> np.ndarray:
    """Element-wise sign: +1 for non-negative, -1 for negative."""
    return 1 - 2 * np.asarray(x < 0).astype(np.int32)


def safe_divide(x: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Divide *x* / *y* preserving sign and clamping near-zero denominators."""
    return sign(y) * x / (np.abs(y) + eps)


def safe_inverse(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Element-wise 1/x with sign-preserving near-zero clamping."""
    return safe_divide(1, x, eps)


def safe_sqrt(x: np.ndarray) -> np.ndarray:
    """Element-wise sqrt preserving sign: sign(x) * sqrt(|x|)."""
    return sign(x) * np.sqrt(np.abs(x))


def first_n(x: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """Boolean mask marking the *n* smallest values along *axis*."""
    i = np.argsort(x, axis=axis).take(indices=range(0, n), axis=axis)
    y = np.zeros(x.shape, dtype=bool)
    np.put_along_axis(y, indices=i, values=True, axis=axis)
    return y


""" ================================================================================= """
"""                                 Path Operations                                   """
""" ================================================================================= """


def arc_length(p: np.ndarray) -> np.ndarray:
    """Normalised cumulative arc-length parameter for a closed polygon."""
    p = np.append(p, np.expand_dims(p[0], 0), axis=0)
    dp = np.diff(p, axis=0)
    delta = np.linalg.norm(dp, axis=-1)
    s = np.concatenate(([0], np.cumsum(delta)))
    s = s / s[-1]
    return s[:-1]
