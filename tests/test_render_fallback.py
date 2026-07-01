"""Tests for the tier-3 matplotlib software render fallback in ``bg3dtools.render.scan``.

The renderer tries offscreen → legacy → matplotlib → RenderUnavailable. The matplotlib tier is a
CPU-only Agg path that needs no GPU or display, so a host where both Open3D backends are dead still
gets a crude diagnostic image; only if matplotlib is *also* missing is ``RenderUnavailable`` raised.

These tests exercise the fallback directly and pin the dispatch contract:
  * ``_spec_to_mpl`` builds the right primitive for every spec type using pure NumPy (no Open3D);
  * ``_exec_matplotlib`` returns correctly-sized, non-blank uint8 frames;
  * the dispatcher falls THROUGH to matplotlib when the Open3D backends raise, and only raises
    ``RenderUnavailable`` when every tier (matplotlib included) fails.

The pure-NumPy / dispatch tests need no optional deps; the ones that actually rasterize
``pytest.importorskip("matplotlib")`` (the ``viz`` extra).
"""

import numpy as np
import pytest

import bg3dtools.render.scan as scan
from bg3dtools.render.scan import (
    CameraParams, RenderOptions, RenderStyle, RenderUnavailable,
    PointCloudSpec, Mesh, Wireframe, Skeleton, Lines, Floor, CameraFrustum,
    _spec_to_mpl, _mpl_up_rotation, _exec_matplotlib, _render_to_images,
)

_TETRA_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_TETRA_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)


def _camera():
    return CameraParams(lookat=np.array([0.4, 0.2, 0.1], float),
                        eye=np.array([1.6, 1.2, -2.6], float),
                        up=np.array([0, 1, 0], float), fov=45.0)


def _full_scene():
    """One spec of every supported type, including a NaN-tipped skeleton bone (must be skipped)."""
    rng = np.random.default_rng(0)
    joints = np.array([[0, 0, 0], [0, 0.6, 0], [0.4, 1.1, 0], [np.nan, np.nan, np.nan]], float)
    return [
        Floor(y=-0.6, half_extent=1.0, color=(0.8, 0.8, 0.8)),
        PointCloudSpec(points=rng.normal(size=(1200, 3)) * 0.3, point_size=4.0),
        Mesh(vertices=_TETRA_V, faces=_TETRA_F, color=(0.2, 0.6, 0.9)),
        Wireframe(vertices=_TETRA_V + np.array([1.5, 0, 0]), faces=_TETRA_F,
                  color=(0.9, 0.2, 0.2), line_width=1.0),
        Skeleton(joints=joints, connections=[(0, 1), (1, 2), (2, 3)], color=(0.1, 0.7, 0.1)),
        Lines(points=np.array([[-1, -1, 0], [1, 1, 0]], float),
              edges=np.array([[0, 1]], int), color=(0, 0, 0)),
        CameraFrustum(position=np.array([0, 0, -1.5], float), rotation=np.eye(3), scale=0.3),
    ]


def _background_fraction(img, thresh=250):
    """Fraction of pixels that are (near-)white background — 1.0 means a blank frame."""
    return float(np.mean(np.all(img >= thresh, axis=-1)))


# ---------------------------------------------------------------------------
# Pure-NumPy spec → primitive conversion (no matplotlib needed)
# ---------------------------------------------------------------------------

def test_spec_to_mpl_primitive_kinds():
    style = RenderStyle()
    kind = lambda spec: [p['kind'] for p in _spec_to_mpl(spec, style)]  # noqa: E731
    assert kind(Wireframe(vertices=_TETRA_V, faces=_TETRA_F)) == ['lines']
    assert kind(PointCloudSpec(points=_TETRA_V)) == ['points']
    assert kind(Mesh(vertices=_TETRA_V, faces=_TETRA_F)) == ['mesh']
    assert kind(Lines(points=_TETRA_V, edges=np.array([[0, 1], [1, 2]]))) == ['lines']
    assert kind(Floor(y=0.0)) == ['mesh']
    assert kind(CameraFrustum(position=np.zeros(3), rotation=np.eye(3))) == ['lines']


def test_spec_to_mpl_skeleton_skips_nan_bones():
    """A bone touching a non-finite joint is dropped; finite joints still become markers."""
    joints = np.array([[0, 0, 0], [0, 1, 0], [np.nan, np.nan, np.nan]], float)
    prims = _spec_to_mpl(Skeleton(joints=joints, connections=[(0, 1), (1, 2)]), RenderStyle())
    lines = [p for p in prims if p['kind'] == 'lines']
    points = [p for p in prims if p['kind'] == 'points']
    assert len(lines) == 1 and lines[0]['segs'].shape == (1, 2, 3)   # only the (0,1) bone survives
    assert len(points) == 1 and len(points[0]['xyz']) == 2           # two finite joints


def test_spec_to_mpl_mesh_vertex_colors_averaged_per_face():
    vc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    prims = _spec_to_mpl(Mesh(vertices=_TETRA_V, faces=_TETRA_F, vertex_colors=vc), RenderStyle())
    rgb = prims[0]['rgb']
    assert rgb.shape == (len(_TETRA_F), 3)                           # one colour per face
    np.testing.assert_allclose(rgb[0], vc[[0, 1, 2]].mean(axis=0))   # face 0 = mean of its 3 verts


def test_mpl_up_rotation_is_proper_and_maps_up_to_z():
    for up in (np.array([0, 1, 0.]), np.array([0, 0, 1.]), np.array([0.3, 0.5, 0.8])):
        M = _mpl_up_rotation(up)
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-9)    # orthonormal
        assert np.linalg.det(M) == pytest.approx(1.0, abs=1e-9)      # right-handed (a rotation)
        np.testing.assert_allclose(M @ (up / np.linalg.norm(up)), [0, 0, 1], atol=1e-9)


# ---------------------------------------------------------------------------
# Actual rasterization via matplotlib Agg
# ---------------------------------------------------------------------------

def test_exec_matplotlib_shape_dtype_and_nonblank():
    pytest.importorskip("matplotlib")
    opts = RenderOptions(width=320, height=240, style=RenderStyle(bg_color=(1, 1, 1, 1)))
    imgs = _exec_matplotlib([_full_scene()], _camera(), opts)
    assert len(imgs) == 1
    img = imgs[0]
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    assert _background_fraction(img) < 0.98                          # something was actually drawn


def test_exec_matplotlib_multiframe_and_odd_size_contract():
    """Multiple frames each come back at the exact requested (odd) size — dpi rounding is snapped."""
    pytest.importorskip("matplotlib")
    opts = RenderOptions(width=101, height=97)
    scene = _full_scene()
    imgs = _exec_matplotlib([scene, scene[:2]], _camera(), opts)
    assert len(imgs) == 2
    assert all(im.shape == (97, 101, 3) and im.dtype == np.uint8 for im in imgs)


def test_exec_matplotlib_empty_scene_is_blank_but_valid():
    """No geometry → a valid, all-background frame (never a crash on the degenerate bbox)."""
    pytest.importorskip("matplotlib")
    imgs = _exec_matplotlib([[]], _camera(), RenderOptions(width=64, height=64))
    assert imgs[0].shape == (64, 64, 3)
    assert _background_fraction(imgs[0]) == pytest.approx(1.0, abs=1e-3)


def test_exec_matplotlib_honors_view_aabb():
    pytest.importorskip("matplotlib")
    cam = CameraParams(eye=np.array([2, 2, 2], float), up=np.array([0, 1, 0], float),
                       zoom=0.5, fov=50.0,
                       view_aabb=(np.array([-1, -1, -1]), np.array([1, 1, 1])))
    img = _exec_matplotlib([_full_scene()], cam, RenderOptions(width=128, height=128))[0]
    assert img.shape == (128, 128, 3)
    assert _background_fraction(img) < 0.98


# ---------------------------------------------------------------------------
# Dispatcher tiering: fall through to matplotlib, and RenderUnavailable at the end
# ---------------------------------------------------------------------------

def test_dispatch_falls_through_to_matplotlib(monkeypatch):
    """Both Open3D backends raising must NOT abort — the matplotlib tier renders instead."""
    pytest.importorskip("matplotlib")

    def dead(*a, **k):
        raise RuntimeError("no Open3D backend here")

    monkeypatch.setattr(scan, "_exec_offscreen", dead)
    monkeypatch.setattr(scan, "_exec_legacy", dead)
    monkeypatch.setattr(scan.sys, "platform", "linux")              # exercise the full 3-tier chain
    imgs = _render_to_images([_full_scene()], _camera(), RenderOptions(width=96, height=96))
    assert imgs[0].shape == (96, 96, 3)
    assert _background_fraction(imgs[0]) < 0.98


def test_render_unavailable_when_every_tier_fails(monkeypatch):
    """When offscreen, legacy AND matplotlib are all unusable, raise a clean RenderUnavailable."""
    def dead(*a, **k):
        raise RuntimeError("backend down")

    monkeypatch.setattr(scan, "_exec_offscreen", dead)
    monkeypatch.setattr(scan, "_exec_legacy", dead)
    monkeypatch.setattr(scan, "_exec_matplotlib", dead)             # simulate matplotlib absent too
    monkeypatch.setattr(scan.sys, "platform", "linux")
    with pytest.raises(RenderUnavailable):
        _render_to_images([_full_scene()], _camera(), RenderOptions())
