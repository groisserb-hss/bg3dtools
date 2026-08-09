"""Per-vertex mask dilation and nearest-vertex field transfer."""

import numpy as np
import pytest

from bg3dtools.mesh.clean import dilate_vertex_mask
from bg3dtools.mesh.generate import build_plane
from bg3dtools.mesh.registration import transfer_vertex_field


@pytest.fixture
def grid():
    """An 11x11 vertex plane; index (r, c) is vertex r * 11 + c."""
    faces, verts = build_plane(11, 11, return_vertices=True)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


# --------------------------------------------------------------------------
# dilate_vertex_mask
# --------------------------------------------------------------------------


def test_zero_rings_is_identity(grid):
    verts, faces = grid
    m = np.zeros(len(verts), bool)
    m[60] = True
    assert np.array_equal(dilate_vertex_mask(faces, m, 0), m)


def test_result_is_a_copy_not_a_view(grid):
    verts, faces = grid
    m = np.zeros(len(verts), bool)
    m[60] = True
    out = dilate_vertex_mask(faces, m, 0)
    out[0] = True
    assert not m[0]


def test_each_ring_adds_the_edge_neighbours(grid):
    verts, faces = grid
    m = np.zeros(len(verts), bool)
    m[5 * 11 + 5] = True

    one = dilate_vertex_mask(faces, m, 1)
    # Whatever the triangulation's diagonal, the four axis neighbours are edges.
    for nb in (4 * 11 + 5, 6 * 11 + 5, 5 * 11 + 4, 5 * 11 + 6):
        assert one[nb]
    assert one[5 * 11 + 5]
    # A vertex two rows away cannot be reached in one ring.
    assert not one[3 * 11 + 5]


def test_dilation_is_monotone_and_composes(grid):
    verts, faces = grid
    m = np.zeros(len(verts), bool)
    m[60] = True
    prev = m
    for n in range(1, 5):
        cur = dilate_vertex_mask(faces, m, n)
        assert (cur | prev).sum() == cur.sum(), "each ring must be a superset"
        prev = cur
    # n rings == n applications of one ring
    once = m
    for _ in range(3):
        once = dilate_vertex_mask(faces, once, 1)
    assert np.array_equal(once, dilate_vertex_mask(faces, m, 3))


def test_growth_never_crosses_to_a_detached_component(grid):
    """The property that separates topological dilation from a Euclidean ball."""
    verts, faces = grid
    # A second copy of the plane, offset by a hair -- spatially adjacent,
    # topologically disjoint.
    verts2 = verts + np.array([0.0, 0.0, 1e-6])
    both_v = np.vstack([verts, verts2])
    both_f = np.vstack([faces, faces + len(verts)])

    m = np.zeros(len(both_v), bool)
    m[60] = True
    grown = dilate_vertex_mask(both_f, m, 50)

    assert grown[:len(verts)].all(), "should fill its own component"
    assert not grown[len(verts):].any(), "must not jump the gap"


def test_empty_mask_stays_empty(grid):
    verts, faces = grid
    m = np.zeros(len(verts), bool)
    assert not dilate_vertex_mask(faces, m, 5).any()


def test_face_mask_idiom(grid):
    """The documented recipe for dilating a per-face mask."""
    verts, faces = grid
    fmask = np.zeros(len(faces), bool)
    fmask[0] = True

    vmask = np.zeros(len(verts), bool)
    vmask[faces[fmask]] = True
    grown_f = dilate_vertex_mask(faces, vmask, 1)[faces].any(axis=1)

    assert grown_f[0]
    assert grown_f.sum() > fmask.sum()


# --------------------------------------------------------------------------
# transfer_vertex_field
# --------------------------------------------------------------------------


def test_transfer_onto_itself_is_the_identity(grid):
    verts, _ = grid
    field = np.arange(len(verts), dtype=np.float64)
    assert np.array_equal(transfer_vertex_field(verts, field, verts), field)


def test_boolean_field_stays_boolean(grid):
    verts, _ = grid
    field = np.zeros(len(verts), bool)
    field[::3] = True
    out = transfer_vertex_field(verts, field, verts, max_dist=1e-9, fill=False)
    assert out.dtype == np.bool_
    assert np.array_equal(out, field)


def test_max_dist_fills_rather_than_painting_far_geometry(grid):
    verts, _ = grid
    field = np.ones(len(verts), bool)
    far = verts + np.array([0.0, 0.0, 10.0])

    assert transfer_vertex_field(verts, field, far, max_dist=1.0, fill=False).sum() == 0
    # Without the bound, every point takes a value however far its match is.
    assert transfer_vertex_field(verts, field, far).all()


def test_max_dist_boundary_is_inclusive(grid):
    verts, _ = grid
    field = np.ones(len(verts), bool)
    offset = verts + np.array([0.0, 0.0, 0.5])
    assert transfer_vertex_field(verts, field, offset, max_dist=0.5, fill=False).all()
    assert not transfer_vertex_field(verts, field, offset, max_dist=0.4999, fill=False).any()


def test_multi_column_field(grid):
    verts, _ = grid
    field = np.tile(np.arange(len(verts))[:, None], (1, 4)).astype(np.float64)
    out = transfer_vertex_field(verts, field, verts, max_dist=1e-9, fill=-1.0)
    assert out.shape == field.shape
    assert np.array_equal(out, field)


def test_nearest_vertex_quantises_to_source_vertices(grid):
    """Documented behaviour: values are taken, never interpolated."""
    verts, _ = grid
    field = np.arange(len(verts), dtype=np.float64)
    midpoint = (verts[0] + verts[1])[None] / 2.0
    assert transfer_vertex_field(verts, field, midpoint)[0] in (0.0, 1.0)


def test_length_mismatch_raises(grid):
    verts, _ = grid
    with pytest.raises(ValueError, match="rows"):
        transfer_vertex_field(verts, np.zeros(len(verts) - 1), verts)
