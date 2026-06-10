"""
Functional map computation and optimization.

Solves for functional correspondence matrices between meshes using
spectral descriptors and orientation-preserving constraints, with
JAX-accelerated optimization.
"""

# Derived from pyFM by Robin Magnet (MIT License) — see /THIRD_PARTY_NOTICES.txt
import math

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, lil_matrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial.distance import cdist

from spectral_match.tools.mesh_class import Mesh

""" ======================================================================================================= """
"""                                    Functional Matching                                                  """
""" ======================================================================================================= """


def spectral_array(mesh: Mesh, x, k):
    x = mesh.pointwise_2_vector(x, k)
    x = mesh.pointwise_2_vector(x.T, k).T
    return x


def zero_diagonal(x):
    x = x.copy()
    np.fill_diagonal(x, 0)
    return x


def norm(x):
    return x / np.linalg.norm(x)


def root(x, n):
    return np.sign(x) * (np.abs(x) ** (1.0 / float(n)))


def coupling_array(mesh: Mesh, k, sigma):
    x = np.exp(-0.5 * (mesh.g / mesh.g.mean() / sigma) ** 2)
    x = zero_diagonal(spectral_array(mesh, x, k=-1))
    J0 = x[:k][:, :k]
    J1 = x[:k][:, k:] @ x[k:][:, :k]
    J2 = x[:k][:, k:] @ x[k:][:, k:] @ x[k:][:, :k]
    func = lambda a, n: norm(zero_diagonal(a)) / math.factorial(n) / 1.67
    couplings = [func(c, n + 1) for n, c in enumerate([J0, J1, J2])]
    return couplings


def spectral_extraction(mesh: Mesh, k):
    # Laplacian
    l = mesh.eigen[0][:k] + 1
    l = 1.0 / np.sqrt(l)
    l /= np.linalg.norm(l)
    # Metrics
    J = []
    for sigma in [0.3, 0.45, 0.6]:
        J += coupling_array(mesh, k, sigma=sigma)
    return l, J


def commutator(x, y, operator=np.subtract):
    x, y = np.meshgrid(x, y)
    return operator(x, y)


def spectral_functional(src, dst, k):
    # jax is imported lazily here (its only use in this module) so that importing
    # spectral_match does not require jax unless the functional-map solver is run.
    import jax
    import jax.numpy as jnp

    # Extraction
    ls, J_s = spectral_extraction(src, k)
    ld, J_d = spectral_extraction(dst, k)
    dl = commutator(ls, ld)

    # Metrics
    def functor(C):
        y = 3 * jnp.linalg.norm(dl * C, ord="fro")
        for a, b in zip(J_s, J_d):
            y += jnp.linalg.norm(C @ a @ C.T - b, ord="fro")  # alpha *
        return y

    # Compilation
    f = jax.jit(functor)
    df = jax.jit(jax.grad(functor, argnums=0))
    return f, df


def super_operator_promotion(A, n):
    from scipy.sparse import eye, kron, issparse
    if not issparse(A):
        from scipy.sparse import csr_matrix
        A = csr_matrix(A)
    return kron(A, eye(n, dtype=np.float64, format='csr'), format='csr')


def initialisation(src: Mesh, dst: Mesh, k, euclidean_weight=0.0):
    s1 = src.pointwise_2_vector(src.s, k)
    s2 = dst.pointwise_2_vector(dst.s, k)

    if euclidean_weight > 2e-8:
        from bg3dtools.mesh.registration import affine_ICP
        aligned_v = affine_ICP(dst.v, src.f, src.v)[1]
        src_idx = np.linspace(0, len(src.v) - 1, s1.shape[1], dtype=int)
        D = cdist(src.v[src_idx], aligned_v)
        dst_idx = np.argmin(D, axis=1)
        g1 = src.pointwise_2_vector(src.g[:, src_idx], k)
        g2 = dst.pointwise_2_vector(dst.g[:, dst_idx], k)
        s1 = np.concatenate([s1, g1], axis=1)
        s2 = np.concatenate([s2, g2], axis=1)

    l1, l2 = np.meshgrid(src.eigen[0][:k], dst.eigen[0][:k])
    # LHS
    A1 = super_operator_promotion(s1.dot(s1.T), k)
    A2 = diags(np.abs(1 / np.sqrt(1 + l1) - 1 / np.sqrt(1 + l2)).flatten())
    A = vstack([A1, A2])
    # RHS
    b1 = (s1.dot(s2.T)).flatten()
    b2 = np.zeros(A2.shape[-1])
    b = np.concatenate([b1, b2], axis=0)
    # Result
    return lsqr(A, b)[0].reshape(k, k)


def correspondence_matrix_solver(src, dst, k, optimise=True, euclidean_weight=0.0):
    # Setting up
    f, df = spectral_functional(src, dst, k)
    fun = lambda x: np.asarray(f(x.reshape(k, k)), dtype=np.float64)
    jac = lambda x: np.asarray(df(x.reshape(k, k)), dtype=np.float64).flatten()
    # Initial guess
    C = initialisation(src, dst, k, euclidean_weight)
    if optimise:
        res = minimize(
            fun=fun,
            jac=jac,
            x0=C.flatten(),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        C = res.x.reshape(k, k)
    return C


def soft_correspondence(src, dst, C, euclidean_weight=0.0):
    kd, ks = C.shape
    Q = src.mass @ src.eigen[-1][:, :ks] @ C.T @ dst.eigen[-1][:, :kd].T
    P = Q ** 2
    P /= np.sum(P, axis=1, keepdims=True)

    if euclidean_weight > 2e-8:
        from bg3dtools.mesh.registration import affine_ICP
        aligned_v = affine_ICP(dst.v, src.f, src.v)[1]
        D = cdist(src.v, aligned_v)
        radius = 2 * np.percentile(D, 50)
        E = np.exp(-D / radius)
        E /= np.sum(E, axis=1, keepdims=True)
        P = (1 - euclidean_weight) * P + euclidean_weight * E

    return P
