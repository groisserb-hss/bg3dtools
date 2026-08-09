"""
Bridging two nearby open boundaries into one connected surface.

The use case: a region of a mesh has been deleted and a replacement patch sits
just above the cut. ``cut_edges`` recovers the exact deletion boundary (and only
it — pre-existing holes never qualify), and ``zipper_loop_to_edges`` triangulates
a bridge strip between an ordered boundary loop and that (possibly fragmented)
edge set. The bridge adds **faces only, never vertices**, so provenance claims of
the form "every output vertex comes bit-identical from input X" survive stitching.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["cut_edges", "zipper_loop_to_edges"]


def cut_edges(faces: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """Edges shared between a masked (deleted) face and an unmasked (kept) face.

    This is the exact boundary a face deletion exposes, and nothing else: an edge
    on a pre-existing hole borders at most one face, so it can never be shared
    between a kept and a deleted face and never qualifies. No manifold
    assumption — an edge bordering several faces qualifies as soon as at least
    one is kept and one is deleted.

    Parameters
    ----------
    faces : (nF, 3) ndarray
        Triangle indices.
    face_mask : (nF,) bool ndarray
        True = deleted face.

    Returns
    -------
    edges : (nE, 2) int64 ndarray
        Unique undirected edges, each row sorted ascending. Vertex indices are in
        the input mesh's indexing.
    """
    f = np.asarray(faces, dtype=np.int64)
    fm = np.asarray(face_mask, dtype=bool)
    if len(fm) != len(f):
        raise ValueError(f"face_mask has {len(fm)} entries for {len(f)} faces")
    nV = int(f.max()) + 1 if len(f) else 0

    def _keys(ff):
        e = np.sort(np.vstack([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]]), axis=1)
        return np.unique(e[:, 0] * nV + e[:, 1])

    shared = np.intersect1d(_keys(f[fm]), _keys(f[~fm]), assume_unique=True)
    return np.stack([shared // nV, shared % nV], axis=1)


def _zip_polylines(verts: np.ndarray, A: np.ndarray, B: np.ndarray) -> list:
    """Greedy shortest-diagonal triangulation between two same-direction polylines."""
    faces = []
    i, j = 0, 0
    while i < len(A) - 1 or j < len(B) - 1:
        adv_a = i < len(A) - 1
        if adv_a and j < len(B) - 1:
            da = np.linalg.norm(verts[A[i + 1]] - verts[B[j]])
            db = np.linalg.norm(verts[B[j + 1]] - verts[A[i]])
            adv_a = da <= db
        if adv_a:
            faces.append((A[i], B[j], A[i + 1]))
            i += 1
        else:
            faces.append((A[i], B[j], B[j + 1]))
            j += 1
    return faces


def zipper_loop_to_edges(
    verts: np.ndarray,
    loop: np.ndarray,
    cut_verts: np.ndarray,
    *,
    axis: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    gap_max: Optional[float] = None,
    span_split_rad: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Triangulate a bridge between an ordered boundary loop and a scattered edge set.

    Both sides are parametrised by angle about ``axis`` through ``center``; the
    cut vertices are split into contiguous angular runs (a gap wider than
    ``span_split_rad`` — or a vertex farther than ``gap_max`` from the loop —
    starts a new run), and each run is zipped to the loop sub-polyline covering
    its angular range by a greedy shortest-diagonal march. Angular spans with no
    run stay **open**: bridging to geometry that is not there is not attempted.

    Winding is set per run by a radial test: the seam region is treated as
    locally cylindrical about ``axis``, and a run whose mean bridge normal points
    inward is flipped. Adds no vertices.

    Parameters
    ----------
    verts : (nV, 3) ndarray
        One vertex array containing BOTH sides (typical after concatenation).
    loop : (nL,) int ndarray
        Ordered boundary loop of the patch side (cyclic; direction irrelevant).
    cut_verts : (nC,) int ndarray
        Vertex indices of the opposite boundary, unordered and possibly
        fragmented (e.g. from :func:`cut_edges`).
    axis : (3,) ndarray, optional
        Seam axis. Default: normal of the best-fit plane of ``cut_verts``.
    center : (3,) ndarray, optional
        Point on the axis. Default: centroid of ``cut_verts``.
    gap_max : float, optional
        A cut vertex farther than this from its nearest loop vertex is left
        unbridged. Default: no limit.
    span_split_rad : float
        Angular gap between consecutive cut vertices that starts a new run.

    Returns
    -------
    bridge_faces : (nB, 3) int64 ndarray
        Triangles referencing existing vertices only; every triangle uses at
        least one vertex from each side.
    info : dict
        ``n_spans`` bridged runs, ``n_open_spans`` gaps left open,
        ``bridged_fraction`` of the full circle covered by runs,
        ``n_skipped_far`` cut vertices dropped by ``gap_max``.
    """
    from scipy.spatial import cKDTree

    P = np.asarray(verts, dtype=np.float64)
    loop = np.asarray(loop, dtype=np.int64)
    cv = np.unique(np.asarray(cut_verts, dtype=np.int64))
    if len(loop) < 2 or len(cv) == 0:
        return np.zeros((0, 3), dtype=np.int64), {
            "n_spans": 0, "n_open_spans": 0, "bridged_fraction": 0.0,
            "n_skipped_far": 0}

    c = P[cv].mean(axis=0) if center is None else np.asarray(center, np.float64)
    if axis is None:
        _, _, Vt = np.linalg.svd(P[cv] - c, full_matrices=False)
        n = Vt[-1]
    else:
        n = np.asarray(axis, np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    u = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, 0]
    u -= n * (u @ n)
    u /= max(np.linalg.norm(u), 1e-12)
    w = np.cross(n, u)

    def theta(idx):
        d = P[idx] - c
        return np.arctan2(d @ w, d @ u)

    # loop, in angular order (robust to boundary_loop direction)
    th_l = theta(loop)
    L = loop[np.argsort(th_l)]
    th_L = np.sort(th_l)

    n_far = 0
    if gap_max is not None:
        d_near = cKDTree(P[loop]).query(P[cv], k=1)[0]
        n_far = int((d_near > gap_max).sum())
        cv = cv[d_near <= gap_max]
        if len(cv) == 0:
            return np.zeros((0, 3), dtype=np.int64), {
                "n_spans": 0, "n_open_spans": 1, "bridged_fraction": 0.0,
                "n_skipped_far": n_far}
    th_c = theta(cv)
    order = np.argsort(th_c)
    cv, th_c = cv[order], th_c[order]

    # split into contiguous angular runs, starting at the widest gap
    gaps = np.diff(np.concatenate([th_c, [th_c[0] + 2 * np.pi]]))
    start = int(np.argmax(gaps)) + 1
    cv = np.roll(cv, -start)
    th_c = np.roll(th_c, -start)
    th_c = np.unwrap(th_c)  # monotone within the rolled ordering
    breaks = np.flatnonzero(np.diff(th_c) > span_split_rad) + 1
    runs = np.split(np.arange(len(cv)), breaks)
    closed = gaps.max() <= span_split_rad and len(runs) == 1

    all_faces: list = []
    covered = 0.0
    pad = np.pi / max(len(L), 3)  # half a typical loop-vertex spacing
    for run in runs:
        if len(run) < 2:
            continue
        B = cv[run]
        t0, t1 = th_c[run[0]] - pad, th_c[run[-1]] + pad
        # loop sub-polyline covering [t0, t1]; the loop's theta is periodic
        th3 = np.concatenate([th_L - 2 * np.pi, th_L, th_L + 2 * np.pi])
        L3 = np.concatenate([L, L, L])
        sel = (th3 >= t0) & (th3 <= t1)
        if sel.sum() < 2:
            j = np.argsort(np.abs(th3 - 0.5 * (t0 + t1)))[:2]
            sel = np.zeros(len(th3), bool)
            sel[j] = True
        A = L3[sel]
        if closed:
            A = np.concatenate([A, A[:1]])
            B = np.concatenate([B, B[:1]])
        faces = _zip_polylines(P, A, B)
        if not faces:
            continue
        fa = np.asarray(faces, dtype=np.int64)
        # radial winding test: outward = away from the axis at the face centroid
        cen = P[fa].mean(axis=1)
        nor = np.cross(P[fa[:, 1]] - P[fa[:, 0]], P[fa[:, 2]] - P[fa[:, 0]])
        rad = cen - c
        rad -= np.outer(rad @ n, n)
        if (np.einsum("ij,ij->i", nor, rad).sum()) < 0:
            fa = fa[:, [0, 2, 1]]
        all_faces.append(fa)
        covered += th_c[run[-1]] - th_c[run[0]]

    n_spans = len(all_faces)
    info = {
        "n_spans": n_spans,
        "n_open_spans": 0 if closed else max(len(runs), 1),
        "bridged_fraction": float(min(covered / (2 * np.pi), 1.0)),
        "n_skipped_far": n_far,
    }
    if not all_faces:
        return np.zeros((0, 3), dtype=np.int64), info
    out = np.vstack(all_faces)
    log.debug("zipper: %d bridge faces in %d span(s), %.0f%% of the circle, "
              "%d cut vert(s) beyond gap_max",
              len(out), n_spans, 100 * info["bridged_fraction"], n_far)
    return out, info
