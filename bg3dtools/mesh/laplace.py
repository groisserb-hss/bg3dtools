"""
Laplacian operators and spectral mesh analysis.

This module provides functions for computing Laplace-Beltrami operators,
cotangent weight matrices, mass matrices, eigendecompositions, curvature,
and biharmonic embeddings on triangle meshes.
"""

from typing import List, Optional, Tuple
import numpy as np

from bg3dtools.igl_compat import (
    boundary_loop,
    gaussian_curvature as igl_gaussian_curvature,  # this module defines its own wrapper
)
from scipy import sparse
from scipy.sparse import diags, spmatrix
from scipy.sparse.linalg import eigsh, lobpcg, spsolve

from bg3dtools.mesh.utils import adj_from_edges

__all__ = [
    "cotangent_weights",
    "lumped_vertex_areas",
    "fem_mass_matrix",
    "laplace_beltrami_operator",
    "laplace_eigen_decomposition",
    "taubin_smoothing",
    "laplacian_smoothing",
    "laplacian_smoothing_batch",
    "gaussian_curvature",
    "biharmonic_embedding",
    "laplacian_spectrum",
]


# ---------------------------------------------------------------------------
# Laplacian matrix construction
# ---------------------------------------------------------------------------

def cotangent_weights(v: np.ndarray, f: np.ndarray, eps: float = 1e-8) -> sparse.coo_matrix:
    """
    Compute symmetric cotangent-weight matrix for mesh Laplacian.

    Parameters
    ----------
    v : (nV, 3) ndarray
        Vertex coordinates.
    f : (nF, 3) ndarray
        Triangle indices.
    eps : float, optional
        Small value to avoid numerical issues. Default is 1e-8.

    Returns
    -------
    W : (nV, nV) coo_matrix
        Symmetric cotangent-weight matrix.
    """
    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    u1, u2, u3 = v3 - v2, v1 - v3, v2 - v1

    # cot(angle) = dot / |cross|  (avoids trig and sqrt(1-cos²))
    cross1 = np.linalg.norm(np.cross(-u2, -u3), axis=1)
    cross2 = np.linalg.norm(np.cross(u1, -u3), axis=1)
    cross3 = np.linalg.norm(np.cross(-u1, u2), axis=1)

    cot1 = (-u2 * u3).sum(1) / np.maximum(cross1, eps)
    cot2 = (u1 * -u3).sum(1) / np.maximum(cross2, eps)
    cot3 = (-u1 * u2).sum(1) / np.maximum(cross3, eps)

    I = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    J = np.concatenate([f[:, 2], f[:, 0], f[:, 1]])
    S = 0.5 * np.concatenate([cot1, cot2, cot3])

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S, -S,  S,  S])

    return sparse.coo_matrix((Sn, (In, Jn)), shape=(v.shape[0], v.shape[0]))


def lumped_vertex_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute lumped per-vertex areas.

    Each vertex area is one-third of the sum of adjacent face areas.

    Parameters
    ----------
    v : (nV, 3) ndarray
        Vertex coordinates.
    f : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    areas : (nV,) ndarray
        Lumped area at each vertex.
    """
    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    return np.bincount(f.reshape(-1),
                       np.repeat(face_areas, 3) / 3,
                       minlength=v.shape[0])


def fem_mass_matrix(v: np.ndarray, f: np.ndarray) -> sparse.coo_matrix:
    """
    Compute consistent FEM mass matrix.

    Parameters
    ----------
    v : (nV, 3) ndarray
        Vertex coordinates.
    f : (nF, 3) ndarray
        Triangle indices.

    Returns
    -------
    M : (nV, nV) coo_matrix
        FEM mass matrix.
    """
    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

    I = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    J = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    S = np.concatenate([A, A, A])

    In = np.concatenate([I, J, I])
    Jn = np.concatenate([J, I, I])
    Sn = (1 / 12) * np.concatenate([S, S, 2 * S])

    return sparse.coo_matrix((Sn, (In, Jn)), shape=(v.shape[0], v.shape[0]))


# ---------------------------------------------------------------------------
# Laplace-Beltrami operator and spectral decomposition
# ---------------------------------------------------------------------------

def laplace_beltrami_operator(
    l: spmatrix,
    m: spmatrix
) -> spmatrix:
    """
    Compute the Laplace-Beltrami operator from stiffness and mass matrices.

    Parameters
    ----------
    l : (n, n) sparse matrix
        Cotangent Laplacian (stiffness matrix).
    m : (n, n) sparse matrix
        Mass matrix (diagonal).

    Returns
    -------
    lb : (n, n) sparse matrix
        Laplace-Beltrami operator M^{-1} @ L.
    """
    minv = diags(1 / (m.diagonal() + 1e-12))
    return minv.dot(l)


def laplace_eigen_decomposition(
    l: spmatrix,
    m: spmatrix,
    k: int,
    normalize: bool = True
) -> List[np.ndarray]:
    """
    Compute eigendecomposition of the generalized Laplacian problem.

    Solves -L @ v = lambda * M @ v for smallest eigenvalues.

    Parameters
    ----------
    l : (n, n) sparse matrix
        Cotangent Laplacian (stiffness matrix).
    m : (n, n) sparse matrix
        Mass matrix.
    k : int
        Number of eigenvalues/eigenvectors to compute.
    normalize : bool, optional
        If True, normalize eigenvectors w.r.t. mass matrix. Default is True.

    Returns
    -------
    eigenvalues : (k,) ndarray
        Smallest eigenvalues.
    eigenvectors : (n, k) ndarray
        Corresponding eigenvectors as columns.
    """
    evals, evecs = eigsh(A=-l, k=k, M=m, which="SM")
    if normalize:
        evecs /= np.sqrt(np.sum(m.dot(evecs ** 2), axis=0, keepdims=True))
    return [evals, evecs]


def laplacian_spectrum(
    W: spmatrix,
    A: spmatrix,
    spectrum_size: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of the mesh Laplacian.

    Solves the generalized eigenvalue problem W @ v = lambda * A @ v
    to obtain the Laplace-Beltrami spectrum.

    Parameters
    ----------
    W : (n, n) sparse matrix
        Cotangent weight matrix (Laplacian stiffness matrix).
    A : (n, n) sparse matrix
        Area weight matrix (mass matrix).
    spectrum_size : int, optional
        Number of eigenvalues/eigenvectors to compute. Default is 200.

    Returns
    -------
    eigenvalues : (spectrum_size,) ndarray
        Eigenvalues in ascending order.
    eigenvectors : (n, spectrum_size) ndarray
        Corresponding eigenvectors as columns.

    Notes
    -----
    Falls back to LOBPCG solver if the default eigsh solver fails.
    """
    try:
        eigenvalues, eigenvectors = eigsh(W, k=spectrum_size, M=A,
                                          sigma=-0.01)

    except RuntimeError:
        # raise ValueError('Matrices are not positive semidefinite')
        # Initial eigenvector values:
        print('Problem during LBO decomposition ! Please check')
        init_eigenvecs = np.random.random((A.shape[0], spectrum_size))
        eigenvalues, eigenvectors = lobpcg(W, init_eigenvecs,
                                           B=A, largest=False, maxiter=40)

        eigenvalues = np.real(eigenvalues)
        sorting_arr = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorting_arr]
        eigenvectors = eigenvectors[:, sorting_arr]

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Smoothing, curvature, and embeddings
# ---------------------------------------------------------------------------

def _laplacian_smooth_invariants(
    l: spmatrix,
    m: spmatrix,
) -> Tuple[spmatrix, spmatrix, spmatrix]:
    """
    Precompute the parts of Laplacian smoothing that depend only on l and m.

    Returns (m_csc, ql, m_csc) where ql = l^T @ (M^{-1} @ L).
    Reuse across multiple mu values to avoid redundant sparse solves.
    """
    m_csc = m.tocsc() if sparse.issparse(m) else m
    l_csc = l.tocsc() if sparse.issparse(l) else l
    is_diag = sparse.issparse(m_csc) and (m_csc.nnz <= m_csc.shape[0])
    if is_diag:
        m_inv_l = diags(1.0 / (m_csc.diagonal() + 1e-12)) @ l_csc
    else:
        m_inv_l = spsolve(m_csc, l_csc)
    ql = l_csc.T @ m_inv_l
    return m_csc, ql


def taubin_smoothing(
    verts: np.ndarray,
    faces: np.ndarray,
    n_iters: int = 10,
    lam: float = 0.9,
    mu: float = -0.905,
    pinned: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Taubin lambda|mu smoothing — low-pass filtering without shrinkage.

    Each iteration is a shrink step at `lam` followed by an inflate step at `mu`,
    on the row-normalised (uniform) Laplacian. With `lam > 0 > mu` and
    `|mu| > lam` the transfer function passes low frequencies at ~1 and attenuates
    high ones, so features are removed while overall scale is preserved.

    Use this rather than repeated plain Laplacian steps whenever the result is
    measured against the input. Plain Laplacian smoothing is mean-curvature flow:
    it contracts the surface without bound, so a "how much did the surface move"
    statistic reports contraction rather than the feature removal it is meant to
    measure, and enough iterations collapse the mesh to a point.

    Explicit, so there is no linear system to be near-singular, and stable for
    `lam < 1` because the row-normalised Laplacian has spectrum in [0, 2].

    Parameters
    ----------
    verts : (nV, 3) ndarray
    faces : (nF, 3) ndarray
    n_iters : int
        Number of lambda|mu pairs. Zero returns a copy.
    lam, mu : float
        Shrink and inflate factors. The defaults are the standard
        near-cancelling pair; `mu` must be negative and `|mu| > lam`.
    pinned : (nV,) bool ndarray, optional
        Vertices held fixed — typically a boundary ring, so a patch stays
        attached to the surface it was cut from. Note that pinning a ring while
        the interior contracts leaves a step at the ring; feather the result if
        that matters.

    Returns
    -------
    smoothed : (nV, 3) ndarray, float64

    See Also
    --------
    laplacian_smoothing : implicit, cotangent-weighted, single large step.
    """
    v = np.asarray(verts, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    if n_iters <= 0 or not len(f):
        return v
    if not (mu < 0 < lam and abs(mu) > lam):
        raise ValueError(f"need lam > 0 > mu and |mu| > lam; got lam={lam}, mu={mu}")

    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = adj_from_edges(e, len(v))
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)[:, None]

    free = slice(None)
    if pinned is not None:
        free = ~np.asarray(pinned, dtype=bool)

    for _ in range(int(n_iters)):
        for step in (lam, mu):
            delta = (A @ v) / deg - v
            v[free] += step * delta[free]
    return v


def laplacian_smoothing(
    l: spmatrix,
    m: spmatrix,
    s: np.ndarray,
    mu: float = 1e-3
) -> np.ndarray:
    """
    Perform Laplacian smoothing on mesh vertex positions.

    Parameters
    ----------
    l : (n, n) sparse matrix
        Cotangent Laplacian (stiffness matrix).
    m : (n, n) sparse matrix
        Mass matrix.
    s : (n, 3) ndarray
        Vertex positions to smooth.
    mu : float, optional
        Smoothing parameter. Smaller = more smoothing. Default is 1e-3.

    Returns
    -------
    s_smooth : (n, 3) ndarray
        Smoothed vertex positions.
    """
    m_csc, ql = _laplacian_smooth_invariants(l, m)
    return spsolve((mu * ql + (1 - mu) * m_csc).tocsc(), m_csc.dot(s))


def laplacian_smoothing_batch(
    l: spmatrix,
    m: spmatrix,
    s: np.ndarray,
    mus: list,
) -> list:
    """
    Laplacian smoothing at multiple mu values, computing invariants once.

    Parameters
    ----------
    l : (n, n) sparse matrix
        Cotangent Laplacian (stiffness matrix).
    m : (n, n) sparse matrix
        Mass matrix.
    s : (n, d) ndarray
        Signal to smooth (vertex positions or scalar field).
    mus : list of float
        Smoothing parameters.

    Returns
    -------
    results : list of ndarray
        Smoothed signals, one per mu value.
    """
    m_csc, ql = _laplacian_smooth_invariants(l, m)
    ms = m_csc.dot(s)
    return [spsolve((mu * ql + (1 - mu) * m_csc).tocsc(), ms) for mu in mus]


def gaussian_curvature(
    v: np.ndarray,
    f: np.ndarray
) -> np.ndarray:
    """
    Compute Gaussian curvature at each vertex.

    Parameters
    ----------
    v : (n, 3) ndarray
        Vertex coordinates.
    f : (m, 3) ndarray
        Triangle face indices.

    Returns
    -------
    curvature : (n,) ndarray
        Gaussian curvature per vertex. Boundary vertices set to 0.
    """
    s = igl_gaussian_curvature(v, f)
    b = boundary_loop(f)
    s[b] = 0
    s[np.isnan(s)] = 0
    return s


def biharmonic_embedding(
    verts: np.ndarray,
    faces: np.ndarray,
    dim: int = 4,
    p: float = 2
) -> np.ndarray:
    """
    Compute biharmonic embedding of a mesh.

    Creates an embedding where Euclidean distance approximates
    biharmonic distance on the surface.

    Parameters
    ----------
    verts : (n, 3) ndarray
        Vertex coordinates.
    faces : (m, 3) ndarray
        Triangle face indices.
    dim : int, optional
        Embedding dimension. Default is 4.
    p : float, optional
        Eigenvalue exponent controlling distance type:
        - 0.5: semi-harmonic embedding
        - 1: commute time (harmonic) embedding
        - 2: biharmonic embedding (default)
        - 3: triharmonic embedding

    Returns
    -------
    B : (n, dim) ndarray
        Biharmonic embedding coordinates.
    """

    # Custom implementations rather than igl's: cotangent_weights needs negating
    # (opposite sign convention) and fem_mass_matrix uses a different normalisation
    # from igl.massmatrix -- its diagonal sums to half igl's Voronoi one. The old
    # claim that igl's versions return all zeros on 2.5.1 is not reproducible.
    L = -cotangent_weights(verts, faces)  # negate: custom has opposite sign convention
    M = fem_mass_matrix(verts, faces)

    # get dim+1 smallest magnitude eigenvalues and corresponding vectors
    eig_vals, eig_vecs = laplace_eigen_decomposition(L, M, dim+1, normalize=False)
    eig_vals = np.abs(eig_vals[1:])
    eig_vecs = eig_vecs[:, 1:]

    #  divide each eigenvector by corresponding eigenvalue
    #  divide the power by 2 first because it will appear in the denominator of
    #  distance computation *outside* the squared difference
    B = eig_vecs * (1 / eig_vals)**(p/2)
    return B
