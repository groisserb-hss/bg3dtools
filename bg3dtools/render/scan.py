"""Unified 3D diagnostic renderer (the single home for what used to live in humanfit.utils.render).

Geometry specs are plain data objects describing *what* to draw. ``render_scan()`` writes a video and
``render_frame()`` returns a single image; both share ONE backend core (``_render_to_images``), which
tries these tiers in order and returns the first that works:

 1. OffscreenRenderer (headless EGL/Filament) — Linux servers / Modal.
 2. legacy Visualizer — macOS (visible window) and interactive Linux/Windows.
 3. matplotlib Agg (CPU only, no GPU or display) — universal last-resort fallback. Crude and
    diagnostic-only, but always available; runs when both Open3D backends are dead (e.g. headless
    Windows/Linux with no EGL and no display). Needs the ``viz`` extra's matplotlib.

If every tier fails (matplotlib not installed either), ``RenderUnavailable`` is raised.

``render_mesh_to_image`` (bg3dtools.render.o3d) is a thin wrapper over ``render_frame``, so the two
render engines that used to exist in parallel are now one implementation. humanfit.utils.render is a
compatibility shim that re-exports everything here.

NB ``bg3dtools.mesh.render`` is a DIFFERENT renderer (pytorch3d/open3d intrinsic-based depth/RGB/range
synthesis) — not the diagnostic offscreen+legacy path and out of scope for this module.

open3d is imported lazily inside functions (it's the bg3dtools ``viz`` extra), so importing this module
costs nothing and does not hard-require the extra.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


class RenderUnavailable(RuntimeError):
    """No rendering backend (offscreen, legacy, or the matplotlib fallback) is usable on this host.

    Raised only when ALL tiers fail: the headless OffscreenRenderer, the legacy
    Visualizer, AND the CPU matplotlib Agg fallback — e.g. a box with neither
    EGL/Filament headless support nor a display AND without matplotlib (the
    ``viz`` extra) installed. Callers should treat this as non-fatal and skip
    image output: the pipeline's data products (CSVs, meshes, gate files) are
    written independently of figures.
    """


# ---------------------------------------------------------------------------
# Geometry spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Wireframe:
    """Mesh wireframe (edges only, no filled faces)."""
    vertices: np.ndarray          # (V, 3)
    faces: np.ndarray             # (F, 3) — edges extracted automatically
    color: Tuple[float, float, float] = (0.3, 0.3, 0.9)
    line_width: Optional[float] = None   # px; None → style default. Set explicitly for cross-backend
                                         # parity (legacy GL clamps to ~1px, so use ~1.0).


@dataclass
class PointCloudSpec:
    """Point cloud with optional per-point colour."""
    points: np.ndarray            # (P, 3)
    colors: Optional[np.ndarray] = None   # (P, 3) float64 or None
    max_points: int = 50_000
    point_size: Optional[float] = None    # px; None → style default. Set explicitly so BOTH backends
                                          # use the same size (parity).


@dataclass
class Skeleton:
    """Joint skeleton drawn as a wireframe (bones + icosahedron spheres)."""
    joints: np.ndarray            # (J, 3)
    connections: Sequence[Tuple[int, int]]
    color: Tuple[float, float, float] = (0.9, 0.1, 0.1)
    line_width: Optional[float] = None   # px; None → style default


@dataclass
class Floor:
    """Horizontal quad at a given Y height."""
    y: float
    half_extent: float = 1.0
    color: Tuple[float, float, float] = (0.3, 0.3, 0.3)


@dataclass
class CameraFrustum:
    """Camera frustum wireframe placed at a world-space pose."""
    position: np.ndarray          # (3,) world position
    rotation: np.ndarray          # (3, 3) world-to-camera rotation
    hfov: float = 60.0            # degrees
    vfov: float = 45.0
    scale: float = 0.05
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class Mesh:
    """Solid triangle mesh with per-vertex or uniform colour."""
    vertices: np.ndarray          # (V, 3)
    faces: np.ndarray             # (F, 3)
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    vertex_colors: Optional[np.ndarray] = None  # (V, 3) float64 or None


@dataclass
class Lines:
    """Explicit line segments."""
    points: np.ndarray            # (N, 3)
    edges: np.ndarray             # (E, 2) int
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)


@dataclass
class _RawGeom:
    """Passthrough for an ALREADY-built open3d geometry (used by render_mesh_to_image's compat wrapper)."""
    geom: object                  # an o3d TriangleMesh / PointCloud / LineSet
    hint: str = "mesh"            # material hint: "mesh" / "point" / "line"


# A frame is a list of geometry specs.
GeometrySpec = Union[Wireframe, PointCloudSpec, Skeleton, Floor, CameraFrustum, Mesh, Lines, _RawGeom]


# ---------------------------------------------------------------------------
# Render style + options: ONE source of truth, consumed by BOTH backends
# ---------------------------------------------------------------------------

@dataclass
class RenderStyle:
    """Per-look knobs both backends read as the SCENE DEFAULT. A per-spec ``point_size``/``line_width``
    still overrides these (precedence: per-spec → style default)."""
    point_size: float = 5.0
    line_width: float = 2.0
    bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    max_points: int = 50_000      # global cap; a per-spec PointCloudSpec.max_points still overrides


@dataclass
class RenderOptions:
    """Output + camera-independent render config, bundling a RenderStyle. Additive: the legacy
    ``width/height/fps/bg_color`` kwargs on render_scan map onto this so existing callers are untouched."""
    width: int = 1280
    height: int = 960
    fps: float = 10.0
    fov: float = 45.0             # only used if a CameraParams.fov is unset; the camera wins otherwise
    style: RenderStyle = field(default_factory=RenderStyle)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wireframe_edges(faces: np.ndarray) -> np.ndarray:
    """Extract unique edges from a face array.  Returns (E, 2) int64."""
    import igl
    return igl.edges(faces.astype(np.int64))


def _height_colormap(pts: np.ndarray, vertical_axis: int = 1) -> np.ndarray:
    """Map point height to a blue-to-red colour ramp.  Returns (N, 3) float64."""
    h = pts[:, vertical_axis]
    lo, hi = np.nanmin(h), np.nanmax(h)
    t = (h - lo) / max(hi - lo, 1e-6)
    r = np.clip(t * 2, 0, 1)
    b = np.clip(2 - t * 2, 0, 1)
    g = np.clip(1 - np.abs(t - 0.5) * 2, 0, 1) * 0.4
    return np.column_stack([r, g, b])


def _subsample(pts: np.ndarray, colors: Optional[np.ndarray],
               max_points: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(pts) <= max_points:
        return pts, colors
    idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
    return pts[idx], colors[idx] if colors is not None else None


def _build_skeleton_lineset(skel: Skeleton):
    """Build an Open3D LineSet from a Skeleton spec."""
    import open3d as o3d

    joints = skel.joints
    joint_list = [p if np.all(np.isfinite(p)) else None for p in joints]

    valid = [(a, b) for a, b in skel.connections
             if a < len(joint_list) and b < len(joint_list)
             and joint_list[a] is not None and joint_list[b] is not None]

    if not valid:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        return ls

    start_idx, end_idx = zip(*valid)
    start_pts = joints[list(start_idx)]
    end_pts = joints[list(end_idx)]
    scale = np.mean(np.linalg.norm(start_pts - end_pts, axis=-1)) / 10.

    n_bones = len(start_idx)
    bone_points = np.concatenate([start_pts, end_pts], axis=0)
    bone_lines = np.column_stack([np.arange(n_bones),
                                  np.arange(n_bones, 2 * n_bones)])

    # Joint spheres (icosahedron wireframe)
    from bg3dtools.mesh.generate import generate_icosahedron
    icos_v, icos_f = generate_icosahedron()
    edge_set = set()
    for f in icos_f:
        for i in range(3):
            edge_set.add(tuple(sorted([int(f[i]), int(f[(i + 1) % 3])])))
    icos_edges = np.array(list(edge_set), dtype=np.int64)
    icos_v_scaled = icos_v * scale

    known_joints = [j for j in joint_list if j is not None]
    all_points = [bone_points]
    all_lines = [bone_lines]
    offset = bone_points.shape[0]

    for j in known_joints:
        all_points.append(icos_v_scaled + j[None, :])
        all_lines.append(icos_edges + offset)
        offset += len(icos_v_scaled)

    points = np.concatenate(all_points, axis=0).astype(np.float64)
    lines = np.concatenate(all_lines, axis=0).astype(np.int64)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(skel.color)
    return ls


def _build_frustum_lineset(frust: CameraFrustum):
    """Build an Open3D LineSet for a camera frustum in world space."""
    import open3d as o3d
    from bg3dtools.mesh.generate import build_camera_frustum

    verts, edges = build_camera_frustum(frust.hfov, frust.vfov, frust.scale)

    # Transform from camera local to world:
    # camera rotation R is world-to-camera, so camera-to-world = R^T
    R_cw = frust.rotation.T  # camera-to-world
    verts_world = (R_cw @ verts.T).T + frust.position[None, :]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(verts_world.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(frust.color)
    return ls


# ---------------------------------------------------------------------------
# Convert geometry specs → Open3D objects
# ---------------------------------------------------------------------------

def _spec_to_o3d(spec: GeometrySpec):
    """Convert a geometry spec into a list of (name_prefix, o3d_geom, material_hint) tuples."""
    import open3d as o3d

    if isinstance(spec, _RawGeom):
        return [("raw", spec.geom, spec.hint)]

    if isinstance(spec, Wireframe):
        edges = _wireframe_edges(spec.faces)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(spec.vertices.astype(np.float64))
        ls.lines = o3d.utility.Vector2iVector(edges)
        ls.paint_uniform_color(spec.color)
        return [("wireframe", ls, "line")]

    if isinstance(spec, PointCloudSpec):
        pts, colors = _subsample(spec.points, spec.colors, spec.max_points)
        if colors is None:
            colors = _height_colormap(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return [("pointcloud", pcd, "point")]

    if isinstance(spec, Skeleton):
        ls = _build_skeleton_lineset(spec)
        return [("skeleton", ls, "line")]

    if isinstance(spec, Floor):
        y = spec.y
        h = spec.half_extent
        verts = np.array([
            [-h, y, -h], [h, y, -h], [h, y, h], [-h, y, h],
        ], dtype=np.float64)
        tris = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.paint_uniform_color(spec.color)
        mesh.compute_vertex_normals()
        return [("floor", mesh, "mesh")]

    if isinstance(spec, CameraFrustum):
        ls = _build_frustum_lineset(spec)
        return [("frustum", ls, "line")]

    if isinstance(spec, Mesh):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(spec.vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(spec.faces.astype(np.int64))
        if spec.vertex_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                spec.vertex_colors.astype(np.float64))
        else:
            mesh.paint_uniform_color(spec.color)
        mesh.compute_vertex_normals()
        return [("mesh", mesh, "mesh")]

    if isinstance(spec, Lines):
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(spec.points.astype(np.float64))
        ls.lines = o3d.utility.Vector2iVector(spec.edges.astype(np.int64))
        ls.paint_uniform_color(spec.color)
        return [("lines", ls, "line")]

    raise TypeError(f"Unknown geometry spec: {type(spec)}")


# ---------------------------------------------------------------------------
# Camera + style: ONE source of truth shared by BOTH backends
# ---------------------------------------------------------------------------

@dataclass
class CameraParams:
    """Viewpoint camera parameters."""
    lookat: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    eye: np.ndarray = field(default_factory=lambda: np.array([0, 0, -2], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    zoom: float = 0.3
    fov: float = 45.0
    view_aabb: tuple = None   # optional (min,max) world-space corners to fit the view to (overrides scene bbox)

    @property
    def front(self) -> np.ndarray:
        f = (self.lookat - self.eye).astype(np.float32)
        n = np.linalg.norm(f)
        return f / n if n > 0 else np.array([0, 0, 1], dtype=np.float32)


# Both the OffscreenRenderer (setup_camera) and the legacy Visualizer (an explicit pinhole camera) take an
# EXPLICIT eye and an fov, so we resolve the final eye + sizes HERE, once, and both backends consume the
# result. Changing any of these (view_aabb/zoom/fov, point_size, line_width) propagates to both identically.
_AABB_FIT = 1.0                # absolute-scale calibration for the view_aabb→distance fit
_DEFAULT_POINT_SIZE = 5.0      # fallback when a PointCloudSpec leaves point_size unset (both backends)
_DEFAULT_LINE_WIDTH = 2.0      # fallback when a line spec leaves line_width unset (both backends)


def _resolve_eye(camera: CameraParams) -> np.ndarray:
    """Final eye position (shared by both backends). With view_aabb → distance fits the box (zoom-scaled)
    along the requested view direction; else → camera.eye verbatim. Both backends call this, so the crop
    is identical regardless of which renderer runs."""
    if getattr(camera, 'view_aabb', None) is None:
        return camera.eye.astype(np.float64)
    mn, mx = np.asarray(camera.view_aabb[0], float), np.asarray(camera.view_aabb[1], float)
    radius = 0.5 * float(np.linalg.norm(mx - mn))            # half-diagonal of the view box
    half_fov = np.radians(float(camera.fov)) / 2.0
    dist = _AABB_FIT * float(camera.zoom) * radius / max(np.tan(half_fov), 1e-6)
    return camera.lookat.astype(np.float64) - camera.front.astype(np.float64) * dist


def _scene_point_size(frames, default: float = _DEFAULT_POINT_SIZE) -> float:
    """First explicit PointCloudSpec.point_size across the frames, else ``default``."""
    ps = next((getattr(s, 'point_size', None) for fr in frames for s in fr
               if getattr(s, 'point_size', None) is not None), None)
    return float(ps) if ps is not None else float(default)


def _scene_line_width(frames, default: float = _DEFAULT_LINE_WIDTH) -> float:
    """First explicit line_width across the frames, else ``default``."""
    lw = next((getattr(s, 'line_width', None) for fr in frames for s in fr
               if getattr(s, 'line_width', None) is not None), None)
    return float(lw) if lw is not None else float(default)


def _pinhole_intrinsic(width: int, height: int, fov_deg: float):
    """open3d PinholeCameraIntrinsic from a VERTICAL fov — matches OffscreenRenderer.setup_camera's fov,
    so the legacy backend frames at the same focal length as offscreen."""
    import open3d as o3d
    fy = (height / 2.0) / np.tan(np.radians(float(fov_deg)) / 2.0)
    cx, cy = width / 2.0 - 0.5, height / 2.0 - 0.5
    return o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fy, fy, cx, cy)


def _look_at_extrinsic(eye, lookat, up) -> np.ndarray:
    """World→camera 4x4 extrinsic (OpenCV/open3d convention: camera +X right, +Y down, +Z forward)."""
    eye = np.asarray(eye, float); lookat = np.asarray(lookat, float); up = np.asarray(up, float)
    f = lookat - eye; f = f / (np.linalg.norm(f) + 1e-12)     # +Z forward
    r = np.cross(f, up); rn = np.linalg.norm(r)
    if rn < 1e-8:                                             # up ∥ forward → any orthogonal right
        alt = np.array([1.0, 0, 0]) if abs(f[0]) < 0.9 else np.array([0, 1.0, 0])
        r = np.cross(f, alt); rn = np.linalg.norm(r)
    r = r / rn                                               # +X right
    d = np.cross(f, r)                                       # +Y down
    R = np.stack([r, d, f], axis=0)
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = -R @ eye
    return E


# ---------------------------------------------------------------------------
# Backend executors — ONE implementation each, shared by render_scan + render_frame
# ---------------------------------------------------------------------------

def _exec_offscreen(frames, camera: CameraParams, options: RenderOptions) -> List[np.ndarray]:
    """Render every frame with OffscreenRenderer (EGL/Filament). Returns a list of (H,W,3/4) uint8 images."""
    import open3d.visualization.rendering as rendering
    style = options.style

    renderer = rendering.OffscreenRenderer(options.width, options.height)
    renderer.scene.set_background(np.array(style.bg_color, dtype=np.float32))

    mat_point = rendering.MaterialRecord(); mat_point.shader = "defaultUnlit"; mat_point.point_size = 2.0
    mat_line = rendering.MaterialRecord(); mat_line.shader = "unlitLine"; mat_line.line_width = 2.0
    mat_mesh = rendering.MaterialRecord(); mat_mesh.shader = "defaultLit"
    mat_map = {"point": mat_point, "line": mat_line, "mesh": mat_mesh}

    lookat = camera.lookat.astype(np.float32)
    eye = _resolve_eye(camera).astype(np.float32)            # shared eye (honors view_aabb+zoom)
    up = camera.up.astype(np.float32)

    images = []
    for spec_list in frames:
        renderer.scene.clear_geometry()
        geom_idx = 0
        for spec in spec_list:
            for prefix, geom, hint in _spec_to_o3d(spec):
                mat = rendering.MaterialRecord()
                mat.shader = mat_map[hint].shader
                if hint == "point":
                    ps = getattr(spec, 'point_size', None)
                    mat.point_size = float(ps) if ps is not None else float(style.point_size)
                elif hint == "line":
                    lw = getattr(spec, 'line_width', None)
                    mat.line_width = float(lw) if lw is not None else float(style.line_width)
                    if hasattr(geom, 'colors') and len(geom.colors) > 0:
                        pass  # per-line colours handled by geometry
                    elif hasattr(spec, 'color'):
                        mat.base_color = [*spec.color, 1.0]
                renderer.scene.add_geometry(f"{prefix}_{geom_idx}", geom, mat)
                geom_idx += 1
        renderer.setup_camera(camera.fov, lookat, eye, up)
        images.append(np.asarray(renderer.render_to_image()))
    return images


def _exec_legacy(frames, camera: CameraParams, options: RenderOptions) -> List[np.ndarray]:
    """Render every frame with the legacy Visualizer (visible window on macOS). Returns uint8 images.

    Explicit camera SHARED with the offscreen backend: eye via _resolve_eye (honors view_aabb+zoom),
    looking at ``lookat`` with ``up``, through a pinhole extrinsic + fov intrinsic — NOT open3d's bbox
    auto-fit. The first geometry is added with reset_bounding_box=True to INITIALIZE the view control
    (else convert_from_pinhole is silently ignored → blank); the intrinsic is built from the ACTUAL
    framebuffer size (Retina-safe), then the explicit camera is re-applied every frame."""
    import open3d as o3d
    style = options.style

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    visible = sys.platform == 'darwin'
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=options.width, height=options.height, visible=visible)

    opt = vis.get_render_option()
    opt.background_color = np.array(style.bg_color[:3])
    # Point/line sizes from the SAME shared resolvers the offscreen backend uses (open3d's legacy
    # Visualizer applies them globally, not per-geometry), defaulted from the style.
    opt.point_size = _scene_point_size(frames, style.point_size)
    opt.line_width = _scene_line_width(frames, style.line_width)

    ctr = vis.get_view_control()
    cam_params = o3d.camera.PinholeCameraParameters()
    extrinsic = _look_at_extrinsic(_resolve_eye(camera), camera.lookat, camera.up)

    images = []
    prev_geoms: list = []
    bbox_inited = False
    cam_ready = False
    for spec_list in frames:
        for g in prev_geoms:                                 # clear previous frame
            vis.remove_geometry(g, reset_bounding_box=False)
        prev_geoms.clear()
        for spec in spec_list:
            for _, geom, _ in _spec_to_o3d(spec):
                vis.add_geometry(geom, reset_bounding_box=(not bbox_inited))
                bbox_inited = True
                prev_geoms.append(geom)
        if not cam_ready:                                    # build intrinsic at the real framebuffer size
            intr = ctr.convert_to_pinhole_camera_parameters().intrinsic
            cam_params.intrinsic = _pinhole_intrinsic(intr.width, intr.height, float(camera.fov))
            cam_params.extrinsic = extrinsic
            cam_ready = True
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=False))
        images.append((np.clip(img, 0, 1) * 255).astype(np.uint8))

    vis.destroy_window()
    return images


# ---------------------------------------------------------------------------
# Matplotlib software fallback (tier 3): CPU-only, no GPU/display, always usable
# ---------------------------------------------------------------------------
# This runs entirely on matplotlib's Agg canvas, so it works on a headless host
# with neither a GPU/EGL (offscreen) nor a display (legacy). Quality is
# deliberately crude — it exists so a diagnostic figure still appears when both
# Open3D backends are dead, instead of none at all. Everything here is pure
# NumPy (no Open3D), which is the whole point: it must run when Open3D cannot.
# matplotlib is the ``viz`` extra; if it is absent, _exec_matplotlib raises
# ImportError and the dispatcher falls through to RenderUnavailable.

_MPL_MESH_FACE_CAP = 40_000    # above this a Mesh is drawn as its vertices (points), else Agg hangs


def _finite_points(arr) -> np.ndarray:
    """Rows of ``arr`` (reshaped to (-1, 3)) with all-finite coords — for bbox/aspect."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
    return a[np.isfinite(a).all(axis=1)]


def _mpl_up_rotation(up) -> np.ndarray:
    """3x3 rotation ``M`` with ``M @ up == +Z``, mapping the scene's up onto matplotlib's vertical
    axis. The remaining in-plane (azimuth) freedom is irrelevant: the view direction is recomputed in
    the SAME rotated frame, so the rendered orientation is invariant to which such ``M`` is chosen."""
    u = np.asarray(up, dtype=np.float64)
    n = np.linalg.norm(u)
    u = u / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - u * float(u @ ref)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(u, e1)
    return np.stack([e1, e2, u], axis=0)


def _rawgeom_to_mpl(spec: _RawGeom, style: RenderStyle) -> list:
    """Best-effort extraction of NumPy arrays from an already-built Open3D geometry (the compat path).
    Duck-typed so it never imports Open3D; returns [] (and logs) if the geometry can't be read."""
    g = spec.geom
    try:
        if hasattr(g, 'triangles') and hasattr(g, 'vertices'):
            verts = np.asarray(g.vertices, dtype=np.float64)
            faces = np.asarray(g.triangles, dtype=np.int64)
            if hasattr(g, 'vertex_colors') and len(g.vertex_colors) > 0:
                rgb = np.clip(np.asarray(g.vertex_colors, float), 0, 1)[faces].mean(axis=1)
            else:
                rgb = np.array([0.6, 0.6, 0.6])
            return [{'kind': 'mesh', 'verts': verts, 'faces': faces, 'rgb': rgb}]
        if hasattr(g, 'lines') and hasattr(g, 'points'):
            pts = np.asarray(g.points, dtype=np.float64)
            edges = np.asarray(g.lines, dtype=np.int64)
            return [{'kind': 'lines', 'segs': pts[edges], 'rgb': np.array([0.5, 0.5, 0.5]),
                     'width': float(style.line_width)}]
        if hasattr(g, 'points'):
            pts = np.asarray(g.points, dtype=np.float64)
            cols = (np.asarray(g.colors, float) if hasattr(g, 'colors') and len(g.colors) > 0
                    else _height_colormap(pts))
            return [{'kind': 'points', 'xyz': pts, 'rgb': np.clip(cols, 0, 1),
                     'size': float(style.point_size)}]
    except Exception as exc:
        log.debug("matplotlib fallback could not read _RawGeom (%s); skipping", exc)
    return []


def _spec_to_mpl(spec: GeometrySpec, style: RenderStyle) -> list:
    """Convert a geometry spec into backend-neutral matplotlib primitives (dicts). The NumPy sibling of
    ``_spec_to_o3d`` — uses NO Open3D so it runs when the Open3D backends are unavailable.

    Primitive kinds:
      {'kind':'points', 'xyz':(N,3), 'rgb':(N,3),      'size':float}
      {'kind':'lines',  'segs':(M,2,3), 'rgb':(3,)|(M,3), 'width':float}
      {'kind':'mesh',   'verts':(V,3), 'faces':(F,3),  'rgb':(3,)|(F,3)}
    """
    def line_w(s) -> float:
        v = getattr(s, 'line_width', None)
        return float(v) if v is not None else float(style.line_width)

    if isinstance(spec, _RawGeom):
        return _rawgeom_to_mpl(spec, style)

    if isinstance(spec, Wireframe):
        edges = _wireframe_edges(spec.faces)
        segs = np.asarray(spec.vertices, dtype=np.float64)[edges]         # (E, 2, 3)
        return [{'kind': 'lines', 'segs': segs, 'rgb': np.array(spec.color, float), 'width': line_w(spec)}]

    if isinstance(spec, PointCloudSpec):
        pts, colors = _subsample(spec.points, spec.colors, spec.max_points)
        if colors is None:
            colors = _height_colormap(pts)
        ps = spec.point_size if spec.point_size is not None else style.point_size
        return [{'kind': 'points', 'xyz': np.asarray(pts, float),
                 'rgb': np.clip(np.asarray(colors, float), 0, 1), 'size': float(ps)}]

    if isinstance(spec, Skeleton):
        joints = np.asarray(spec.joints, dtype=np.float64)
        finite = np.isfinite(joints).all(axis=1)
        valid = [(a, b) for a, b in spec.connections
                 if 0 <= a < len(joints) and 0 <= b < len(joints) and finite[a] and finite[b]]
        prims = []
        if valid:
            segs = np.stack([joints[[a, b]] for a, b in valid], axis=0)   # (M, 2, 3)
            prims.append({'kind': 'lines', 'segs': segs, 'rgb': np.array(spec.color, float),
                          'width': line_w(spec)})
        if finite.any():                                                  # joints as markers (cheaper than icosahedra)
            k = int(finite.sum())
            prims.append({'kind': 'points', 'xyz': joints[finite],
                          'rgb': np.tile(np.array(spec.color, float), (k, 1)),
                          'size': max(line_w(spec) * 3.0, 6.0)})
        return prims

    if isinstance(spec, Floor):
        y, h = spec.y, spec.half_extent
        verts = np.array([[-h, y, -h], [h, y, -h], [h, y, h], [-h, y, h]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return [{'kind': 'mesh', 'verts': verts, 'faces': faces, 'rgb': np.array(spec.color, float)}]

    if isinstance(spec, CameraFrustum):
        from bg3dtools.mesh.generate import build_camera_frustum
        verts, edges = build_camera_frustum(spec.hfov, spec.vfov, spec.scale)
        verts_world = (np.asarray(spec.rotation, float).T @ verts.T).T + np.asarray(spec.position, float)[None, :]
        return [{'kind': 'lines', 'segs': verts_world[edges], 'rgb': np.array(spec.color, float),
                 'width': line_w(spec)}]

    if isinstance(spec, Mesh):
        verts = np.asarray(spec.vertices, dtype=np.float64)
        faces = np.asarray(spec.faces, dtype=np.int64)
        if spec.vertex_colors is not None:
            rgb = np.clip(np.asarray(spec.vertex_colors, float), 0, 1)[faces].mean(axis=1)  # per-face avg
        else:
            rgb = np.array(spec.color, float)
        return [{'kind': 'mesh', 'verts': verts, 'faces': faces, 'rgb': rgb}]

    if isinstance(spec, Lines):
        segs = np.asarray(spec.points, float)[np.asarray(spec.edges, dtype=np.int64)]
        return [{'kind': 'lines', 'segs': segs, 'rgb': np.array(spec.color, float), 'width': line_w(spec)}]

    raise TypeError(f"Unknown geometry spec: {type(spec)}")


def _draw_prims_mpl(ax, prims: list, M: np.ndarray) -> None:
    """Draw backend-neutral primitives onto a matplotlib 3D axes, rotating every coordinate by ``M``
    (scene up → +Z). Non-finite rows are dropped per-primitive so a stray NaN never blanks the frame."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    for p in prims:
        if p['kind'] == 'points':
            raw = np.asarray(p['xyz'], float).reshape(-1, 3)
            mask = np.isfinite(raw).all(axis=1)
            if not mask.any():
                continue
            r = raw[mask] @ M.T
            s = float(np.clip(float(p['size']) ** 2, 1.0, 60.0))
            rgb = np.asarray(p['rgb'], float)
            if rgb.ndim == 2:
                ax.scatter(r[:, 0], r[:, 1], r[:, 2], c=np.clip(rgb[mask], 0, 1),
                           s=s, depthshade=False, edgecolors='none')
            else:
                ax.scatter(r[:, 0], r[:, 1], r[:, 2], color=tuple(np.clip(rgb, 0, 1)),
                           s=s, depthshade=False, edgecolors='none')

        elif p['kind'] == 'lines':
            segs = np.asarray(p['segs'], float)                          # (M, 2, 3)
            good = np.isfinite(segs).all(axis=(1, 2))
            if not good.any():
                continue
            segs = segs[good]
            rs = (segs.reshape(-1, 3) @ M.T).reshape(segs.shape)
            rgb = np.asarray(p['rgb'], float)
            cols = np.clip(rgb[good], 0, 1) if rgb.ndim == 2 else np.clip(rgb, 0, 1)
            ax.add_collection3d(Line3DCollection(list(rs), colors=cols,
                                                 linewidths=float(p.get('width', 1.0))))

        elif p['kind'] == 'mesh':
            verts = np.asarray(p['verts'], float)
            faces = np.asarray(p['faces'], dtype=np.int64)
            rgb = np.asarray(p['rgb'], float)
            if len(faces) > _MPL_MESH_FACE_CAP:                          # too heavy for the CPU rasterizer
                vf = _finite_points(verts) @ M.T
                base = np.clip(rgb if rgb.ndim == 1 else rgb.mean(axis=0), 0, 1)
                ax.scatter(vf[:, 0], vf[:, 1], vf[:, 2], color=tuple(base),
                           s=1.0, depthshade=False, edgecolors='none')
                log.warning("matplotlib fallback: mesh has %d faces (> %d cap); drawing vertices as points",
                            len(faces), _MPL_MESH_FACE_CAP)
                continue
            tris = (verts[faces].reshape(-1, 3) @ M.T).reshape(-1, 3, 3)
            fc = np.clip(rgb, 0, 1)
            ax.add_collection3d(Poly3DCollection(list(tris), facecolors=fc, edgecolors='none'))


def _exec_matplotlib(frames, camera: CameraParams, options: RenderOptions) -> List[np.ndarray]:
    """Tier-3 software fallback: render every frame on matplotlib's Agg 3D canvas (CPU only, no GPU or
    display). Crude but always available — used only when both Open3D backends fail. Returns a list of
    (H, W, 3) uint8 images at ``options.width`` x ``options.height``.

    Camera: the scene is framed to its overall finite bounding box (or ``camera.view_aabb``), centered
    in view, seen from ``_resolve_eye(camera)`` with ``camera.up`` mapped to the vertical axis and a
    perspective fov of ``camera.fov``. This approximates the Open3D pinhole framing — good enough for a
    diagnostic last resort, not pixel-identical to the GPU backends."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d import art3d  # noqa: F401  (imports the package → registers the '3d' projection)
    from PIL import Image

    style = options.style
    W, H = int(options.width), int(options.height)
    bg = tuple(float(c) for c in style.bg_color[:3])

    # Build primitives once per frame and collect a GLOBAL finite bbox → a stable camera across frames.
    frame_prims, allpts = [], []
    for spec_list in frames:
        prims = []
        for spec in spec_list:
            try:
                prims.extend(_spec_to_mpl(spec, style))
            except Exception as exc:
                log.debug("matplotlib fallback skipped a %s spec: %s", type(spec).__name__, exc)
        frame_prims.append(prims)
        for p in prims:
            key = {'points': 'xyz', 'lines': 'segs', 'mesh': 'verts'}[p['kind']]
            allpts.append(_finite_points(p[key]))

    pts_all = np.concatenate(allpts, axis=0) if allpts else np.zeros((0, 3))
    if getattr(camera, 'view_aabb', None) is not None:
        mn = np.asarray(camera.view_aabb[0], float)
        mx = np.asarray(camera.view_aabb[1], float)
    elif len(pts_all):
        mn, mx = pts_all.min(axis=0), pts_all.max(axis=0)
    else:
        mn, mx = -np.ones(3), np.ones(3)
    center = (mn + mx) / 2.0

    M = _mpl_up_rotation(camera.up)
    # View direction in the rotated frame (scene up → +Z), toward the bbox center that matplotlib centers on.
    vdir = M @ (_resolve_eye(camera) - center)
    nvd = np.linalg.norm(vdir)
    vdir = vdir / nvd if nvd > 1e-12 else np.array([0.0, 0.0, 1.0])
    elev = float(np.degrees(np.arcsin(np.clip(vdir[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(vdir[1], vdir[0])))

    # Rotated bbox corners → axis limits + true box aspect (so the fallback preserves proportions).
    corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    rc = corners @ M.T
    rmn, rmx = rc.min(axis=0), rc.max(axis=0)
    rspan = np.maximum(rmx - rmn, 1e-6)

    dpi = 100.0
    fov = float(camera.fov)
    focal = 1.0 / max(np.tan(np.radians(fov) / 2.0), 1e-3)

    images = []
    for prims in frame_prims:
        fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        fig.patch.set_facecolor(bg)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.set_facecolor(bg)

        _draw_prims_mpl(ax, prims, M)

        ax.set_xlim(rmn[0], rmx[0]); ax.set_ylim(rmn[1], rmx[1]); ax.set_zlim(rmn[2], rmx[2])
        try:
            ax.set_box_aspect(tuple(rspan))
        except Exception:
            pass
        try:
            ax.view_init(elev=elev, azim=azim, roll=0)
        except TypeError:                                                # matplotlib < 3.5: no roll kwarg
            ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type('persp', focal_length=focal)
        except (TypeError, ValueError):                                  # matplotlib < 3.6: no focal_length
            try:
                ax.set_proj_type('persp')
            except Exception:
                pass
        ax.set_axis_off()

        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[..., :3]
        if buf.shape[1] != W or buf.shape[0] != H:                       # dpi rounding → snap to contract size
            buf = np.asarray(Image.fromarray(buf).resize((W, H), Image.BILINEAR))
        images.append(np.ascontiguousarray(buf, dtype=np.uint8))
    return images


# Backends that have already failed in this process. Open3D's offscreen/legacy
# engines print a native (C++) error to stderr *before* raising when a
# GPU/EGL/display context can't be created (e.g. "EGL Headless is not supported"
# on headless Windows); re-probing a dead backend on every frame reprints that
# noise. We remember failures here and skip known-dead backends for the rest of
# the process, so the native error surfaces at most once and the fallback is
# silent thereafter. Process-local by design: a spawned render subprocess
# (run_isolated) starts with a fresh set and does its own single probe.
_dead_backends: set = set()


def _render_to_images(frames, camera: CameraParams, options: RenderOptions) -> List[np.ndarray]:
    """Backend dispatch (the single chokepoint). Tiers, first that works wins:
    offscreen → legacy → matplotlib → RenderUnavailable (macOS starts at legacy).

    The matplotlib tier is a CPU software renderer that needs no GPU or display, so a host where both
    Open3D backends are dead (e.g. headless Windows with no EGL and no display) still gets a crude
    diagnostic image. Only if matplotlib is ALSO unavailable do we raise RenderUnavailable, letting
    callers skip images and still produce their data outputs instead of aborting the run.

    A backend that fails once is recorded in ``_dead_backends`` and skipped for the rest of the
    process, so a dead Open3D backend prints its native error — and this function logs its fallback
    warning — at most once per process rather than on every frame."""
    backends = [_exec_legacy] if sys.platform == 'darwin' else [_exec_offscreen, _exec_legacy]
    backends = backends + [_exec_matplotlib]
    errors = []
    for backend in backends:
        if backend.__name__ in _dead_backends:                           # known-dead this process → skip silently, no re-probe
            continue
        try:
            images = backend(frames, camera, options)
        except Exception as exc:
            log.debug("render backend %s unavailable: %s", backend.__name__, exc)
            errors.append("%s: %s" % (backend.__name__, exc))
            _dead_backends.add(backend.__name__)                         # never re-probe → no repeated native stderr noise
            continue
        if errors:                                                       # a preferred backend failed *this call* → name the tier that ran
            log.warning("render fell back to %s after: %s", backend.__name__, "; ".join(errors))
        return images
    raise RenderUnavailable(
        "no usable render backend (" + ("; ".join(errors) or "all backends already failed this process") + ")")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_scan(
    vis_file: str,
    frames: List[List[GeometrySpec]],
    camera: CameraParams,
    *,
    width: int = 1280,
    height: int = 960,
    fps: float = 10.0,
    bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    options: Optional[RenderOptions] = None,
) -> None:
    """Render a sequence of geometry spec lists to an mp4 video.

    Backward compatible: the ``width/height/fps/bg_color`` kwargs work exactly as before. Pass a single
    ``options=RenderOptions(...)`` instead to control style (point_size/line_width/bg) in one object.
    """
    import imageio
    if options is None:
        options = RenderOptions(width=width, height=height, fps=fps,
                                style=RenderStyle(bg_color=bg_color))
    images = _render_to_images(frames, camera, options)
    writer = imageio.get_writer(str(vis_file), fps=float(options.fps))
    for im in images:
        writer.append_data(im)
    writer.close()


def render_frame(
    specs: List[GeometrySpec],
    camera: CameraParams,
    *,
    width: int = 400,
    height: int = 400,
    bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    style: Optional[RenderStyle] = None,
) -> np.ndarray:
    """Render ONE frame (a list of geometry specs) to an (H,W,3/4) uint8 image — same backend core as
    render_scan. ``render_mesh_to_image`` wraps this."""
    options = RenderOptions(width=width, height=height,
                            style=style or RenderStyle(bg_color=bg_color))
    return _render_to_images([specs], camera, options)[0]
