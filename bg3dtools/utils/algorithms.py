"""
General-purpose algorithms.

This module provides common algorithmic utilities including sorting,
sampling, and PCA persistence.
"""

from typing import Optional, List, Any
import numpy as np


def argsort(seq: List[Any]) -> List[int]:
    """
    Return indices that would sort a sequence.

    Parameters
    ----------
    seq : list
        Input sequence supporting __getitem__.

    Returns
    -------
    indices : list of int
        Indices that sort the sequence.

    Examples
    --------
    >>> argsort([3, 1, 2])
    [1, 2, 0]
    """
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def farthest_point_sampling(
    points: np.ndarray,
    n: int,
    init_idx: Optional[int] = None
) -> np.ndarray:
    """
    Sample points using farthest point sampling (FPS).

    Iteratively selects the point farthest from all previously
    selected points, producing well-distributed samples.

    Parameters
    ----------
    points : (N, D) ndarray
        Input point cloud.
    n : int
        Number of points to sample.
    init_idx : int, optional
        Index of first point. Random if None.

    Returns
    -------
    indices : (n,) ndarray
        Indices of sampled points.

    Notes
    -----
    Uses squared Euclidean distance for efficiency.
    """
    assert n <= len(points)

    indices = np.zeros(n, dtype=int)
    if init_idx is None:
        init_idx = np.random.randint(0, len(points))
    indices[0] = init_idx

    distances = np.sum((points - points[indices[0]])**2, axis=1)
    for i in range(1, n):
        indices[i] = np.argmax(distances)
        new_distances = np.sum((points - points[indices[i]])**2, axis=1)
        distances = np.minimum(distances, new_distances)

    return indices


def save_pca(path: str, pca, idx: Optional[np.ndarray] = None) -> None:
    """
    Save a fitted sklearn PCA model to disk.

    Saves only the components needed for transform/inverse_transform,
    enabling lightweight persistence of PCA models.

    Parameters
    ----------
    path : str
        Output file path (.npz extension recommended).
    pca : sklearn.decomposition.PCA
        Fitted PCA model.
    idx : (k,) ndarray, optional
        Indices of components to save. All components if None.
    """
    if idx is None:
        idx = np.arange(pca.n_components)

    np.savez(
        path,
        mean=pca.mean_.astype(np.float32),                     # (D,)
        components=pca.components_.astype(np.float32),   # (5,D)
        explained_variance=pca.explained_variance_.astype(np.float32),
        # Store total training variance so we can recompute global EVR if desired
        total_variance=np.float32(pca.explained_variance_.sum()),
        whiten=np.bool_(getattr(pca, "whiten", False)),
        n_features=np.int32(pca.n_features_in_),
        idx=idx.astype(np.int32),
    )


def load_pca(path: str, k: int = 65535, apply_idx: bool = True):
    """
    Load a PCA model saved with save_pca.

    Parameters
    ----------
    path : str
        Path to saved .npz file.
    k : int, optional
        Maximum number of components to load. Default is 65535 (all).
    apply_idx : bool, optional
        If True, use saved component indices. Default is True.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Reconstructed PCA model ready for transform/inverse_transform.
    """
    from sklearn.decomposition import PCA

    d = np.load(path, allow_pickle=False)
    mean   = d["mean"].astype(np.float64)
    comps  = d["components"].astype(np.float64)      # (k,D) rows orthonormal
    evals  = d["explained_variance"].astype(np.float64)
    total  = float(d["total_variance"])
    whiten = bool(d["whiten"])
    idx = d["idx"] if apply_idx else np.arange(comps.shape[0])
    D      = int(d["n_features"])
    if k < len(idx):
        idx = idx[:k]

    p = PCA(n_components=k, whiten=whiten)  # params only; not fitted yet

    # Rehydrate minimal fitted state required by sklearn for transform/inverse_transform
    p.mean_ = mean                          # (D,)
    p.components_ = comps[idx]                   # (k,D)
    p.explained_variance_ = evals[idx]           # (k,)
    # EVR relative to the **original** total variance (nice to have)
    p.explained_variance_ratio_ = (evals[idx] / total) if total > 0 else np.zeros_like(evals[idx])
    p.n_features_in_ = D
    p.n_components_ = k
    p.noise_variance_ = 0.0

    return p

