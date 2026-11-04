"""Render calls in a fresh child process (``bg3dtools.render._child``) and the blank-output guard.

Open3D's headless renderer has segfaulted, and has silently gone on returning black frames, partway
through long processes (SAR-PC, 2026-11-03/04, causes not found), so off macOS every ``render_scan`` /
``render_frame`` renders in a child that exits after its one call, and a tier that returns only flat
images has failed.

The protocol tests run real children on stdlib callables (a value, an exception, a real native crash,
a logged warning); the routing tests replace the child with an in-process stand-in; one test renders
through a real child and runs only off macOS, where the child path is live.
"""
import ctypes
import logging
import math
import operator
import pickle
import sys

import numpy as np
import pytest

import bg3dtools.render._child as child
import bg3dtools.render.scan as scan
from bg3dtools.render import RenderChildDied
from bg3dtools.render.scan import CameraParams, Mesh, PointCloudSpec, RenderOptions, Wireframe, _RawGeom

_TETRA_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_TETRA_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)


def _camera():
    return CameraParams(lookat=np.array([0.3, 0.3, 0.3], float), eye=np.array([1.6, 1.2, -2.6], float),
                        up=np.array([0, 1, 0], float), fov=45.0)


def _scene():
    return [Mesh(vertices=_TETRA_V, faces=_TETRA_F, color=(0.2, 0.6, 0.9)),
            Wireframe(vertices=_TETRA_V, faces=_TETRA_F, color=(0.9, 0.2, 0.2))]


def _drawn(size):
    img = np.full((size, size, 3), 255, np.uint8)
    img[0, 0] = 0
    return img


# --- the child protocol, on real children -----------------------------------------------------------

def test_the_child_returns_the_calls_value():
    assert child.run_in_child(math.hypot, 3.0, 4.0) == 5.0


def test_the_childs_exception_is_raised_here_with_its_type():
    with pytest.raises(ZeroDivisionError):
        child.run_in_child(operator.truediv, 1, 0)


def test_a_native_crash_in_the_child_costs_only_that_call():
    with pytest.raises(RenderChildDied, match=r'exited with code -11'):
        child.run_in_child(ctypes.string_at, 0)               # reads address 0: a real segfault
    assert child.run_in_child(math.hypot, 3.0, 4.0) == 5.0     # and the caller goes on


def test_the_childs_warnings_reach_the_callers_log(caplog):
    with caplog.at_level(logging.WARNING):
        child.run_in_child(logging.warning, 'logged in the child')
    assert [r.getMessage() for r in caplog.records] == ['logged in the child']


# --- routing: off macOS through the child, on macOS here --------------------------------------------

@pytest.fixture()
def recorded_child(monkeypatch):
    """Replace the child with an in-process stand-in that records each call; the backend is faked."""
    calls = []

    def stand_in(fn, *args):
        calls.append(fn.__name__)
        return fn(*args)

    monkeypatch.setattr(child, 'run_in_child', stand_in)
    monkeypatch.setattr(scan, '_offscreen_crash_probe_ok', lambda: True)
    monkeypatch.setattr(scan, '_exec_offscreen', lambda frames, *a: [_drawn(16) for _ in frames])
    return calls


def test_render_frame_renders_in_the_child_off_macos(recorded_child, monkeypatch):
    monkeypatch.setattr(scan.sys, 'platform', 'linux')
    img = scan.render_frame(_scene(), _camera(), width=16, height=16)
    assert recorded_child == ['_render_to_images'] and img.shape == (16, 16, 3)


def test_render_scan_renders_and_encodes_in_the_child_off_macos(recorded_child, monkeypatch, tmp_path):
    pytest.importorskip('imageio')
    monkeypatch.setattr(scan.sys, 'platform', 'linux')
    out = tmp_path / 'video.mp4'
    scan.render_scan(str(out), [_scene()] * 4, _camera(), width=16, height=16)
    assert recorded_child == ['_render_video'] and out.stat().st_size > 0


def test_macos_renders_in_process(recorded_child, monkeypatch):
    monkeypatch.setattr(scan.sys, 'platform', 'darwin')
    monkeypatch.setattr(scan, '_exec_legacy', lambda frames, *a: [_drawn(16) for _ in frames])
    scan.render_frame(_scene(), _camera(), width=16, height=16)
    assert recorded_child == []


# --- blank output ----------------------------------------------------------------------------------

def test_a_backend_that_returns_only_flat_images_has_failed(monkeypatch, caplog):
    pytest.importorskip('matplotlib')
    monkeypatch.setattr(scan, '_dead_backends', set())
    monkeypatch.setattr(scan, '_offscreen_crash_probe_ok', lambda: True)
    def _exec_offscreen(frames, *a):                            # a renderer that stopped drawing
        return [np.zeros((48, 48, 3), np.uint8)] * len(frames)

    def _exec_legacy(*a):
        raise RuntimeError('no display')

    monkeypatch.setattr(scan, '_exec_offscreen', _exec_offscreen)    # named: dead backends are keyed by name
    monkeypatch.setattr(scan, '_exec_legacy', _exec_legacy)
    monkeypatch.setattr(scan.sys, 'platform', 'linux')
    with caplog.at_level(logging.WARNING, logger='bg3dtools.render.scan'):
        imgs = scan._render_to_images([_scene()] * 2, _camera(), RenderOptions(width=48, height=48))
    assert all(np.ptp(im) > 0 for im in imgs)                   # matplotlib drew the scene instead
    assert any('_exec_offscreen: returned only blank images' in r.getMessage() for r in caplog.records)


def test_an_empty_scene_may_render_flat():
    flat = [np.full((8, 8, 3), 237, np.uint8)] * 3
    assert not scan._blank([[], [], []], flat)                 # nothing to draw: the background is right
    assert scan._blank([[], _scene(), []], flat)
    assert not scan._blank([[], _scene(), []], [flat[0], _drawn(8), flat[0]])


# --- a raw open3d geometry crosses the process boundary ----------------------------------------------

def test_a_raw_open3d_mesh_survives_pickling():
    o3d = pytest.importorskip('open3d')
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=6)
    mesh.compute_vertex_normals()
    back = pickle.loads(pickle.dumps(_RawGeom(mesh, hint='mesh')))
    assert back.hint == 'mesh'
    np.testing.assert_array_equal(np.asarray(back.geom.triangles), np.asarray(mesh.triangles))
    np.testing.assert_allclose(np.asarray(back.geom.vertices), np.asarray(mesh.vertices))
    np.testing.assert_allclose(np.asarray(back.geom.vertex_normals), np.asarray(mesh.vertex_normals))


def test_an_unsupported_raw_geometry_names_itself():
    with pytest.raises(TypeError, match='an open3d dict cannot be pickled'):
        pickle.dumps(_RawGeom({}, hint='mesh'))


# --- a real render in a real child (the live path off macOS) -----------------------------------------

@pytest.mark.skipif(sys.platform == 'darwin', reason='macOS renders in-process (legacy tier)')
def test_a_real_child_renders_a_frame():
    pts = PointCloudSpec(points=np.random.default_rng(0).normal(size=(2000, 3)) * 0.3)
    img = scan.render_frame(_scene() + [pts], _camera(), width=64, height=64)
    assert img.shape[:2] == (64, 64) and np.ptp(img[..., :3]) > 0
