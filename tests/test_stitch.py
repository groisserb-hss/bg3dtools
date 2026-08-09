"""Tests for mesh/stitch.py: cut-edge recovery and loop-to-edges zippering."""

import numpy as np
import pytest

from bg3dtools.mesh.generate import build_plane
from bg3dtools.mesh.stitch import cut_edges, zipper_loop_to_edges


# ---------------------------------------------------------------------------
# cut_edges
# ---------------------------------------------------------------------------


def _plane(n=6):
    faces, verts = build_plane(n, n, return_vertices=True)
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def _edge_set(faces):
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                axis=1)
    return {tuple(r) for r in e}


def test_cut_edges_are_exactly_the_kept_deleted_interface():
    v, f = _plane()
    fm = v[f].mean(axis=1)[:, 1] > 0.5 * v[:, 1].max()      # delete the top half
    ce = cut_edges(f, fm)
    assert len(ce) > 0
    kept_e, del_e = _edge_set(f[~fm]), _edge_set(f[fm])
    for e in map(tuple, ce):
        assert e in kept_e and e in del_e


def test_preexisting_hole_edges_never_qualify():
    """An edge on a hole borders at most one face, so it cannot be shared between
    a kept and a deleted face -- 'do not bridge to holes' falls out for free."""
    v, f = _plane()
    fm_full = v[f].mean(axis=1)[:, 1] > 0.5 * v[:, 1].max()

    hole_face = np.flatnonzero(~fm_full)[0]                 # a kept face...
    keep = np.ones(len(f), bool)
    keep[hole_face] = False                                 # ...removed beforehand
    f2, fm2 = f[keep], fm_full[keep]

    hole_edges = _edge_set(f[[hole_face]])
    shared_before = hole_edges & {tuple(r) for r in cut_edges(f, fm_full)}
    ce2 = {tuple(r) for r in cut_edges(f2, fm2)}
    # edges the hole face contributed to the interface must be gone, and no other
    # hole-boundary edge may appear
    for e in hole_edges - shared_before:
        assert e not in ce2


def test_outer_mesh_border_never_qualifies():
    v, f = _plane()
    fm = v[f].mean(axis=1)[:, 1] > 0.5 * v[:, 1].max()
    border = {e for e, cnt in _count_edges(f).items() if cnt == 1}
    assert not (border & {tuple(r) for r in cut_edges(f, fm)})


def _count_edges(faces):
    from collections import Counter
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                axis=1)
    return Counter(map(tuple, e))


# ---------------------------------------------------------------------------
# zipper_loop_to_edges
# ---------------------------------------------------------------------------


def _rims(n_top=16, n_bot=24, z_gap=0.3, seed=0):
    """Two circular rims about +Z: bottom verts first, then the top loop."""
    rng = np.random.default_rng(seed)
    tb = np.linspace(0, 2 * np.pi, n_bot, endpoint=False)
    tt = np.linspace(0, 2 * np.pi, n_top, endpoint=False) + 0.05
    bot = np.stack([np.cos(tb), np.sin(tb), np.zeros(n_bot)], axis=1)
    top = np.stack([np.cos(tt), np.sin(tt), np.full(n_top, z_gap)], axis=1)
    verts = np.vstack([bot, top])
    loop = np.arange(n_bot, n_bot + n_top)                  # ordered (cyclic)
    cut = rng.permutation(n_bot)                            # unordered, as promised
    return verts, loop, cut


def _sides_touched(face, n_bot):
    a = np.asarray(face)
    return (a < n_bot).any() and (a >= n_bot).any()


def test_zipper_connects_full_rims_with_no_new_vertices():
    verts, loop, cut = _rims()
    bf, info = zipper_loop_to_edges(verts, loop, cut, axis=np.array([0.0, 0, 1]))
    assert len(bf) > 0
    assert bf.max() < len(verts) and bf.min() >= 0
    for face in bf:
        assert _sides_touched(face, 24), "a bridge face must touch both sides"
    assert info["bridged_fraction"] > 0.9
    # connectivity: union-find over bridge faces joins the two sides
    parent = list(range(len(verts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, c in bf:
        for x, y in ((a, b), (b, c)):
            parent[find(x)] = find(y)
    assert find(0) == find(24)


def test_zipper_winding_is_radially_outward():
    verts, loop, cut = _rims()
    bf, _ = zipper_loop_to_edges(verts, loop, cut, axis=np.array([0.0, 0, 1]))
    tri = verts[bf]
    nor = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(axis=1)
    rad = cen.copy()
    rad[:, 2] = 0.0
    assert np.einsum("ij,ij->i", nor, rad).sum() > 0


def test_zipper_leaves_missing_sector_open():
    verts, loop, cut = _rims()
    th = np.arctan2(verts[cut, 1], verts[cut, 0])
    cut_open = cut[~((th > 0) & (th < np.pi / 2))]          # remove a 90-deg wedge
    bf, info = zipper_loop_to_edges(verts, loop, cut_open, axis=np.array([0.0, 0, 1]))
    assert 0.5 < info["bridged_fraction"] < 0.85
    cen = verts[bf].mean(axis=1)
    th_f = np.arctan2(cen[:, 1], cen[:, 0])
    inside = (th_f > 0.15) & (th_f < np.pi / 2 - 0.15)
    assert not inside.any(), "no bridge face may span the missing sector"


def test_zipper_gap_max_drops_far_vertices():
    verts, loop, cut = _rims()
    verts = verts.copy()
    verts[cut[0], 2] = -5.0                                  # one vert far away
    bf, info = zipper_loop_to_edges(verts, loop, cut, axis=np.array([0.0, 0, 1]),
                                    gap_max=1.0)
    assert info["n_skipped_far"] == 1
    assert cut[0] not in set(bf.ravel())


def test_zipper_empty_inputs_return_no_faces():
    verts, loop, cut = _rims()
    bf, info = zipper_loop_to_edges(verts, loop, np.array([], dtype=int))
    assert len(bf) == 0 and info["n_spans"] == 0
