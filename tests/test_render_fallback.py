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

import logging

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


@pytest.fixture(autouse=True)
def _clear_dead_backends(monkeypatch):
    """The dispatcher memoizes failed backends (and the offscreen crash-probe verdict)
    process-wide; reset both between tests so backend-monkeypatching tests don't leak
    state into one another. Probing is forced OFF by default so no test spawns a real
    subprocess probe against the host's Open3D — the probe tests opt back in."""
    scan._dead_backends.clear()
    scan._offscreen_probe_ok = None
    monkeypatch.setenv("BG3DTOOLS_RENDER_PROBE", "never")
    yield
    scan._dead_backends.clear()
    scan._offscreen_probe_ok = None


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


def test_dead_backend_probed_once_then_skipped_silently(monkeypatch, caplog):
    """A dead Open3D backend is probed once, then skipped for the rest of the process.

    This is what keeps a headless host (e.g. Windows without EGL) from reprinting Open3D's
    native "EGL Headless is not supported" error on every frame: offscreen + legacy are each
    attempted once, the fallback warning is logged once, and later frames go straight to
    matplotlib in silence."""
    pytest.importorskip("matplotlib")
    calls = {"offscreen": 0, "legacy": 0, "mpl": 0}

    def dead_offscreen(*a, **k):
        calls["offscreen"] += 1
        raise RuntimeError("EGL Headless is not supported")         # the Windows native failure
    def dead_legacy(*a, **k):
        calls["legacy"] += 1
        raise RuntimeError("no display available")
    real_mpl = scan._exec_matplotlib
    def counting_mpl(*a, **k):
        calls["mpl"] += 1
        return real_mpl(*a, **k)

    monkeypatch.setattr(scan, "_exec_offscreen", dead_offscreen)
    monkeypatch.setattr(scan, "_exec_legacy", dead_legacy)
    monkeypatch.setattr(scan, "_exec_matplotlib", counting_mpl)
    monkeypatch.setattr(scan.sys, "platform", "linux")             # exercise the full 3-tier chain

    opts = RenderOptions(width=64, height=64)
    with caplog.at_level(logging.WARNING, logger="bg3dtools.render.scan"):
        for _ in range(3):
            imgs = _render_to_images([_full_scene()], _camera(), opts)
            assert imgs[0].shape == (64, 64, 3)

    assert calls["offscreen"] == 1                                  # probed once despite 3 renders
    assert calls["legacy"] == 1
    assert calls["mpl"] == 3                                        # matplotlib runs every frame
    fell_back = [r for r in caplog.records if "render fell back" in r.getMessage()]
    assert len(fell_back) == 1                                      # warned once, silent thereafter


# ---------------------------------------------------------------------------
# Offscreen crash probe: the uncatchable-EGL-crash guard (subprocess probe)
# ---------------------------------------------------------------------------
# EGL init can SIGSEGV natively on hosts without a usable GPU/EGL stack — no
# try/except can catch that in-process, so the dispatcher probes the offscreen
# tier once in a throwaway child before its first in-process use. These tests
# fake run_isolated (the child spawn) and the on-disk verdict cache to pin the
# contract without ever touching the host's real Open3D.

import bg3dtools.render as render_pkg


@pytest.fixture()
def probe_env(monkeypatch, tmp_path):
    """Opt back into probing, hermetically: fresh cache dir, a fake open3d dist,
    no EGL_PLATFORM, and 'open3d' guaranteed absent from sys.modules."""
    monkeypatch.delenv("BG3DTOOLS_RENDER_PROBE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("EGL_PLATFORM", raising=False)
    monkeypatch.setattr(scan._importlib_metadata, "version", lambda dist: "0.19.0")
    monkeypatch.delitem(scan.sys.modules, "open3d", raising=False)
    monkeypatch.setattr(scan.sys, "platform", "linux")


def _fake_run_isolated(monkeypatch, verdicts):
    """Install a run_isolated stub returning the given verdicts in order; returns the call log."""
    calls = []
    def fake(fn, *a, **k):
        calls.append(fn.__name__)
        return verdicts[len(calls) - 1]
    monkeypatch.setattr(render_pkg, "run_isolated", fake)
    return calls


def test_probe_crash_disables_offscreen_and_falls_through(probe_env, monkeypatch):
    """A probe child that dies (native EGL crash) must disable the tier catchably:
    the render still succeeds via a later tier and the real backend is never entered."""
    pytest.importorskip("matplotlib")
    probe_calls = _fake_run_isolated(monkeypatch, [False, False])   # plain + surfaceless both die
    entered = []
    monkeypatch.setattr(scan, "_exec_offscreen", lambda *a, **k: entered.append(1))
    monkeypatch.setattr(scan, "_exec_legacy",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no display")))

    imgs = _render_to_images([_full_scene()], _camera(), RenderOptions(width=64, height=64))
    assert imgs[0].shape == (64, 64, 3)
    assert entered == []                                            # offscreen never constructed in-process
    assert probe_calls == ["_offscreen_probe_worker"] * 2
    assert "EGL_PLATFORM" not in scan.os.environ                    # failed retry must not leak the env var
    # dead verdict is memoized: further renders spawn no more probes
    _render_to_images([_full_scene()], _camera(), RenderOptions(width=64, height=64))
    assert len(probe_calls) == 2


def test_probe_surfaceless_retry_adopts_env_and_enables_offscreen(probe_env, monkeypatch):
    """Plain probe dies, surfaceless probe survives → EGL_PLATFORM is adopted for the
    process and the offscreen tier runs in-process."""
    probe_calls = _fake_run_isolated(monkeypatch, [False, True])
    frame = np.zeros((64, 64, 3), np.uint8)
    monkeypatch.setattr(scan, "_exec_offscreen", lambda *a, **k: [frame])

    imgs = _render_to_images([_full_scene()], _camera(), RenderOptions(width=64, height=64))
    assert imgs == [frame]
    assert probe_calls == ["_offscreen_probe_worker"] * 2
    assert scan.os.environ["EGL_PLATFORM"] == "surfaceless"


def test_probe_verdict_cached_for_future_processes(probe_env, monkeypatch):
    """The verdict lands in the on-disk cache; a 'fresh process' (reset memo) trusts it
    without spawning any probe child."""
    _fake_run_isolated(monkeypatch, [False, True])
    monkeypatch.setattr(scan, "_exec_offscreen",
                        lambda *a, **k: [np.zeros((8, 8, 3), np.uint8)])
    _render_to_images([_full_scene()], _camera(), RenderOptions(width=8, height=8))

    cache = scan.json.loads(scan._probe_cache_path().read_text())
    assert list(cache.values()) == ["ok-surfaceless"]

    scan._offscreen_probe_ok = None                                 # simulate a brand-new process
    scan._dead_backends.clear()
    monkeypatch.delenv("EGL_PLATFORM", raising=False)               # (fresh process env too)
    second_calls = _fake_run_isolated(monkeypatch, [])              # any spawn would IndexError
    _render_to_images([_full_scene()], _camera(), RenderOptions(width=8, height=8))
    assert second_calls == []                                       # cache hit — no probe spawned
    assert scan.os.environ["EGL_PLATFORM"] == "surfaceless"         # cached verdict re-applied


def test_probe_respects_pinned_egl_platform(probe_env, monkeypatch):
    """An explicit EGL_PLATFORM is the user's call: no surfaceless second-guessing."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("EGL_PLATFORM", "x11")
    probe_calls = _fake_run_isolated(monkeypatch, [False])
    monkeypatch.setattr(scan, "_exec_legacy",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no display")))

    _render_to_images([_full_scene()], _camera(), RenderOptions(width=8, height=8))
    assert probe_calls == ["_offscreen_probe_worker"]               # exactly one probe, no retry
    assert scan.os.environ["EGL_PLATFORM"] == "x11"                 # untouched


def test_probe_never_mode_skips_probing_entirely(probe_env, monkeypatch):
    """BG3DTOOLS_RENDER_PROBE=never restores pre-probe behavior (caller accepts crash risk)."""
    monkeypatch.setenv("BG3DTOOLS_RENDER_PROBE", "never")
    probe_calls = _fake_run_isolated(monkeypatch, [])
    frame = np.zeros((8, 8, 3), np.uint8)
    monkeypatch.setattr(scan, "_exec_offscreen", lambda *a, **k: [frame])
    imgs = _render_to_images([_full_scene()], _camera(), RenderOptions(width=8, height=8))
    assert imgs == [frame] and probe_calls == []
