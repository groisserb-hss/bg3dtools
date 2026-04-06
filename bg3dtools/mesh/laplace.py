"""
Laplacian operators and spectral mesh analysis.

This module provides functions for computing Laplace-Beltrami operators,
cotangent weight matrices, mass matrices, eigendecompositions, curvature,
and biharmonic embeddings on triangle meshes.
"""

from typing import List, Tuple
import igl
import numpy as np
from scipy import sparse
from scipy.sparse import diags, spmatrix
from scipy.sparse.linalg import eigsh, lobpcg, spsolve

__all__ = [
    "cotangent_weights",
    "lumped_vertex_areas",
    "fem_mass_matrix",
    "laplace_beltrami_operator",
    "laplace_eigen_decomposition",
    "laplacian_smoothing",
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
    # Fast path: if m is diagonal, avoid dense NxN intermediate from spsolve
    m_csr = m.tocsr() if sparse.issparse(m) else m
    is_diag = sparse.issparse(m_csr) and (m_csr.nnz <= m_csr.shape[0])
    if is_diag:
        m_inv_l = diags(1.0 / (m_csr.diagonal() + 1e-12)) @ l
    else:
        m_inv_l = spsolve(m, l)
    ql = l.T @ m_inv_l
    return spsolve(mu * ql + (1 - mu) * m, m.dot(s))


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
    s = igl.gaussian_curvature(v, f)
    b = igl.boundary_loop(f)
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

    # Use custom implementations (igl.cotmatrix/massmatrix are broken in 2.5.1)
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
