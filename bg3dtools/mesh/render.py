"""
Mesh rendering and ray tracing utilities.

This module provides GPU-accelerated and CPU mesh rendering using PyTorch3D
and Open3D backends. Includes depth and RGB rendering with camera intrinsics
and extrinsics support.

Functions
---------
render_point_cloud
    Ray trace mesh to compute surface points, depth, and normals.
render_RGB
    Render textured mesh to RGB image.
render_range
    Render mesh to depth image.
extrinsic_from_lookat
    Compute camera extrinsic matrix from look-at parameters.
intrinsic_from_fov
    Compute camera intrinsic matrix from field of view.
"""

from collections import namedtuple
import logging
import open3d as o3d
import numpy as np
import warnings
from typing import Tuple, Union
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


__all__ = [
    "CamParam",
    "CamMats",
    "iphone_intrinsics",
    "extrinsic_from_lookat",
    "intrinsic_from_fov",
    "render_point_cloud",
    "scale_intrinsics",
    "render_RGB",
    "render_range",
    "range_to_z",
    "add_depth_noise",
]

device = o3d.core.Device("CPU:0")
dtype_f = o3d.core.float32
dtype_i = o3d.core.int32

CamParam = namedtuple('CamParam', ['width_fov', 'target', 'position',
                                  'up_axis', 'width', 'height'])
CamMats = namedtuple('CamMats', ['intrinsic', 'extrinsic', 'width', 'height'])

iphone_intrinsics = np.array([[1358.5, 0, 960],  [0, 1358.5, 720],  [0, 0, 1]])


def extrinsic_from_lookat(
    camera_position: np.ndarray,   # (3,) world-space eye position
    camera_up:       np.ndarray,   # (3,) “up” vector (need not be normalised)
    target:          np.ndarray,   # (3,) world-space point the camera looks at
) -> np.ndarray:
    """
    Return the 4×4 *world-to-camera* extrinsic matrix that Open3D, OpenGL,
    etc. expect.

    The camera coordinate frame is:
        +X  → right
        +Y  → up
        +Z  → backwards (camera looks along −Z)

    Notes
    -----
    * FOV and image size are *intrinsic* parameters; they do **not** influence
      the extrinsic matrix.  They are accepted only so that the signature
      matches `create_rays_pinhole` and similar helpers.
    * All inputs can be Python sequences; they are cast to `np.float64`.
    """
    eye = np.asarray(camera_position, dtype=np.float64)
    up  = np.asarray(camera_up,       dtype=np.float64)
    cen = np.asarray(target,          dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Build an orthonormal basis (right, up', -forward)
    # ------------------------------------------------------------------
    forward = cen - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)          # already unit-length

    # Camera looks along −Z, so we store (right, true_up, −forward) as rows
    R = np.stack([right, true_up, -forward])    # shape (3,3)

    # ------------------------------------------------------------------
    # 2. Translation:  t = −R · eye
    # ------------------------------------------------------------------
    t = -R @ eye

    # ------------------------------------------------------------------
    # 3. Assemble homogeneous 4×4 matrix
    # ------------------------------------------------------------------
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3,  3] = t
    return extrinsic


def intrinsic_from_fov(fov: float, width: int, height: int) -> np.ndarray:
    """
    Create a pinhole camera intrinsic matrix from field-of-view (FOV) and image
    dimensions.

    Parameters
    ----------
    fov : float
        Horizontal field of view in degrees.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    intrinsic : np.ndarray
        3x3 intrinsic matrix.
    """
    fov_rad = np.deg2rad(fov)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx * (height / width)  # aspect ratio

    cx = width / 2.0
    cy = height / 2.0

    intrinsic = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)

    return intrinsic



# ---- Optional GPU backend (PyTorch3D) ----
def _render_pytorch3d(
    verts: np.ndarray,
    faces: np.ndarray,
    camera: Union[CamParam, CamMats],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize mesh via PyTorch3D (GPU-accelerated)."""
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        MeshRasterizer, RasterizationSettings, PerspectiveCameras
    )
    from pytorch3d.renderer.cameras import look_at_view_transform

    # detect if inputs are torch tensors or numpy arrays
    xp = 'torch' if torch.is_tensor(verts) else 'numpy'

    H, W = camera.height, camera.width
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    V = torch.as_tensor(verts, dtype=torch.float32, device=device)[None]    # (1, V, 3)
    F = torch.as_tensor(faces, dtype=torch.int64,   device=device)[None]    # (1, F, 3)
    mesh = Meshes(verts=V, faces=F)

    # ---- Build a PerspectiveCameras object ----
    if hasattr(camera, "intrinsic") and hasattr(camera, "extrinsic"):
        K = torch.as_tensor(camera.intrinsic, dtype=torch.float32, device=device)   # 3x3
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        # Assume extrinsic maps world->camera (OpenCV style). If yours is camera->world, invert it.
        E = torch.as_tensor(camera.extrinsic, dtype=torch.float32, device=device)   # 4x4
        R = E[:3,:3][None]                                  # (1,3,3)
        T = E[:3, 3][None]                                  # (1,3)

        cams = PerspectiveCameras(
            focal_length=torch.stack([fx, fy])[None],
            principal_point=torch.stack([cx, cy])[None],
            image_size=torch.tensor([[H, W]], device=device),
            R=R, T=T, in_ndc=False, device=device
        )
    else:
        # Look-at param set (eye/target/up) + horizontal FOV (deg).
        eye    = torch.tensor(camera.position, dtype=torch.float32, device=device)[None]
        target = torch.tensor(camera.target,   dtype=torch.float32, device=device)[None]
        up     = torch.tensor(camera.up_axis,  dtype=torch.float32, device=device)[None]
        R, T = look_at_view_transform(eye=eye, at=target, up=up)   # world->view
        # Derive fx/fy from horizontal FOV (assume square pixels)
        fx = fy = (W * 0.5) / np.tan(np.deg2rad(camera.width_fov) * 0.5)
        cams = PerspectiveCameras(
            focal_length=torch.tensor([[fx, fy]], device=device),
            principal_point=torch.tensor([[W/2.0, H/2.0]], device=device),
            image_size=torch.tensor([[H, W]], device=device),
            R=R, T=T, in_ndc=False, device=device
        )

    # ---- Rasterize on GPU (faces_per_pixel=1 → closest face per pixel) ----
    rast = MeshRasterizer(
        cameras=cams,
        raster_settings=RasterizationSettings(
            image_size=(H, W),
            faces_per_pixel=1,
            blur_radius=0.0,
            cull_backfaces=False
        )
    )
    frags = rast(mesh)  # fragments with .pix_to_face, .bary_coords, .zbuf
    cam_center = cams.get_camera_center()[0]  # (3,)

    # Extract per-pixel face index and barycentrics
    fidx = frags.pix_to_face[0, ..., 0]                  # (H,W), -1 for background
    bccoords = frags.bary_coords[0, ..., 0, :]           # (H,W,3), sums to 1
    hitmask = fidx >= 0

    # Compute world-space surface points from barycentrics
    faces = mesh.faces_packed()  # (F,3)
    verts = mesh.verts_packed()  # (V,3)
    tri = faces[torch.flatten(fidx).clamp_min(0)]  # (HW,3)
    c0, c1, c2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    surf_pts = bccoords[:, :, 0:1] * c0.view(H, W, 3) + \
               bccoords[:, :, 1:2] * c1.view(H, W, 3) + \
               bccoords[:, :, 2:3] * c2.view(H, W, 3)

    # Distance to surface along the ray
    depth = torch.linalg.norm(surf_pts - cam_center, dim=-1)   # Euclidean distance

    # Face normals (flat shading to match your Open3D primitive_normals)
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    fn = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)  # (F,3)
    # Gather normals per pixel; default to 0 where background
    normal_map = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    normal_map[hitmask] = fn[fidx[hitmask]]

    # Cosine of angle between ray and surface normal
    ray_vecs = surf_pts - cam_center.view(1, 1, 3)                    # (H,W,3)
    ray_vecs = torch.nn.functional.normalize(ray_vecs, dim=-1)
    nrm = torch.nn.functional.normalize(normal_map, dim=-1)
    hit_angle = (ray_vecs * nrm).sum(dim=-1)            # (H,W)

    # Background cleanup
    surf_pts = torch.where(hitmask[..., None], surf_pts, torch.inf * torch.ones_like(surf_pts))
    depth = depth.masked_fill(~hitmask, torch.inf)
    fidx = fidx.masked_fill(~hitmask, -1)
    bccoords = torch.where(hitmask[..., None], bccoords, torch.zeros_like(bccoords))
    hit_angle = hit_angle.masked_fill(~hitmask, 0.0)

    # rotate 180° to match Open3D's coordinate system / orientation
    surf_pts = torch.rot90(surf_pts, 2, dims=(0,1))
    depth = torch.rot90(depth, 2, dims=(0,1))
    fidx = torch.rot90(fidx, 2, dims=(0,1))
    bccoords = torch.rot90(bccoords, 2, dims=(0,1))
    hit_angle = torch.rot90(hit_angle, 2, dims=(0,1))

    if xp == 'numpy':
        surf_pts = surf_pts.detach().cpu().numpy()
        depth = depth.cpu().detach().numpy()
        fidx = fidx.cpu().detach().numpy()
        bccoords = bccoords.detach().cpu().numpy()
        hit_angle = hit_angle.detach().cpu().numpy()

    return surf_pts, depth, fidx, bccoords, hit_angle


# ---- Fallback: Open3D CPU ray-casting path ----
def _render_open3d_cpu(
    verts: np.ndarray,
    faces: np.ndarray,
    camera: Union[CamParam, CamMats],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize mesh via Open3D ray casting (CPU)."""
    import open3d as o3d
    dtype_f = o3d.core.Dtype.Float32
    dtype_i = o3d.core.Dtype.Int32
    device  = o3d.core.Device("CPU:0")

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(verts, dtype_f, device)
    mesh.triangle.indices = o3d.core.Tensor(faces, dtype_i, device)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    if hasattr(camera, "width_fov"):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=camera.width_fov,
            center=camera.target, eye=camera.position, up=camera.up_axis,
            width_px=camera.width, height_px=camera.height
        )
    else:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=camera.intrinsic,
            extrinsic_matrix=camera.extrinsic,
            width_px=camera.width, height_px=camera.height
        )

    cam_pos = rays.numpy()[..., :3]
    ray_vecs = rays.numpy()[..., 3:]
    ray_mags = np.linalg.norm(ray_vecs, axis=-1)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    ans = scene.cast_rays(rays)

    t_hit = ans['t_hit'].numpy()
    fidx = ans['primitive_ids'].numpy().astype(np.int32)
    bccoords = ans['primitive_uvs'].numpy()
    bc_sum = np.sum(bccoords, axis=-1, keepdims=True)
    bccoords = np.concatenate((bccoords, 1 - bc_sum), axis=-1).clip(0, 1)
    bccoords = bccoords[:, :, [2, 0, 1]]
    normal_map = ans['primitive_normals'].numpy()

    depth = t_hit * ray_mags
    surf_pts = cam_pos + ray_vecs * t_hit[..., None]

    normal_map /= np.maximum(1e-5, np.linalg.norm(normal_map, axis=-1, keepdims=True))
    hit_angle = np.sum(ray_vecs * normal_map, axis=-1)

    # Set background nicely
    miss = ~np.isfinite(t_hit)
    depth[miss] = np.inf
    fidx[miss] = -1
    bccoords[miss, :] = 0.0
    hit_angle[miss] = 0.0

    return surf_pts, depth, fidx, bccoords, hit_angle

# ---- Public wrapper ----
def render_point_cloud(
    verts: np.ndarray,
    faces: np.ndarray,
    camera: Union[CamParam, CamMats],
    prefer: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ray-trace a mesh to compute surface points, depth, and normals.

    Parameters
    ----------
    verts : (V, 3) ndarray
        Vertex positions.
    faces : (F, 3) ndarray
        Triangle indices.
    camera : CamParam or CamMats
        Camera parameters.
    prefer : {'auto', 'pytorch3d', 'open3d'}
        Backend preference.  ``'auto'`` uses GPU if available.

    Returns
    -------
    surf_pts : (H, W, 3) ndarray
        World-space surface hit points.
    depth : (H, W) ndarray
        Euclidean distance to each hit.
    fidx : (H, W) int ndarray
        Face index per pixel (-1 for background).
    bccoords : (H, W, 3) ndarray
        Barycentric coordinates of each hit.
    hit_angle : (H, W) ndarray
        Cosine of angle between ray and surface normal.
    """
    log = logging.getLogger(__name__)
    if prefer == "open3d":
        return _render_open3d_cpu(verts, faces, camera)

    # Try GPU path
    try:
        import torch
        import pytorch3d
        if torch.cuda.is_available():
            return _render_pytorch3d(verts, faces, camera)
    except Exception:
        log.warning("Failed to render point cloud with Pytorch, falling back to Open3D", category=UserWarning)
        pass  # fall back

    return _render_open3d_cpu(verts, faces, camera)

def scale_intrinsics(camera: Union[CamParam, CamMats], scale: float) -> Union[CamParam, CamMats]:
    if isinstance(camera, CamParam):
        return CamParam(
            width_fov=camera.width_fov,
            target=camera.target,
            position=camera.position,
            up_axis=camera.up_axis,
            width=round(camera.width * scale),
            height=round(camera.height * scale)
        )
    else:
        assert isinstance(camera, CamMats)
        intrinsic = camera.intrinsic.copy() * scale
        intrinsic[2, 2] = 1
        return CamMats(
            intrinsic=intrinsic,
            extrinsic=camera.extrinsic,
            width=round(camera.width * scale),
            height=round(camera.height * scale)
        )


def render_RGB(
    verts: np.ndarray,
    faces: np.ndarray,
    tex: np.ndarray,
    ftex: np.ndarray,
    img: np.ndarray,
    rgb_camera: Union[CamParam, CamMats],
    background_color: np.ndarray = (20, 220, 40),
    backend: str = "auto",
) -> np.ndarray:
    """Render a textured mesh to an RGB image via ray tracing.

    Parameters
    ----------
    verts : (V, 3) ndarray
        Vertex positions.
    faces : (F, 3) ndarray
        Triangle indices.
    tex : (T, 2) ndarray
        Texture coordinates normalised to [0, 1].
    ftex : (F, 3) ndarray
        Per-face texture-vertex indices into *tex*.
    img : (H_tex, W_tex, 3) ndarray
        Source texture image.
    rgb_camera : CamParam or CamMats
        Camera parameters.
    background_color : array-like
        RGB fill for pixels with no hit.
    backend : str
        Rendering backend (see :func:`render_point_cloud`).

    Returns
    -------
    rendered_rgb : (H, W, 3) uint8 ndarray
        Rendered colour image.
    """
    nRows, nCols, C = img.shape

    # convert from normalized coordinates (0-1) to row/column indices
    rows = np.minimum((1-tex[:, 1]) * nRows, nRows-1)  # indexing from top left
    cols = np.minimum(tex[:, 0] * nCols, nCols-1)
    texRC = np.column_stack((rows, cols))

    # ray trace to find contact points on the mesh
    _, dist, fi, bc, _ = render_point_cloud(verts, faces, rgb_camera, prefer=backend)
    hit_mask = np.isfinite(dist)  # [H, W] mask of valid hits

    # compute texture coordinates of hit points
    fi = fi[hit_mask]  # [nH] filter out invalid hits
    ftex_hits = ftex[fi, :]  # [nH x 3] face texture vertex indices of hit points

    # ([nH x 2], [nH x 2], [nH x 2]) pixel coordinates of vertices of hit faces:
    tex0, tex1, tex2 = texRC[ftex_hits[:, 0]], texRC[ftex_hits[:, 1]], texRC[ftex_hits[:, 2]]
    bc_hits = bc[hit_mask, :]  # [nH x 3] barycentric coordinates of hit points
    tex_rc = (tex0 * bc_hits[:, 0:1] + tex1 * bc_hits[:, 1:2] + tex2 * bc_hits[:, 2:3])  # [nH x 2] pixel coordinates
    tex_rc = np.round(tex_rc).astype(np.int64)

    # # create RGB image
    outR, outC = rgb_camera.height, rgb_camera.width
    background_color = np.asarray(background_color, dtype=np.uint8)
    if len(background_color) == 1:
        rendered_rgb = np.full((outR, outC, 3), background_color, dtype=np.uint8)
    elif background_color.ndim == 1:
        rendered_rgb = np.tile(background_color[None, None, :], (outR, outC, 1))
    else:
        rendered_rgb = background_color.copy()
    rendered_rgb[hit_mask, :] = img[tex_rc[:, 0], tex_rc[:, 1], :]

    return rendered_rgb


def render_range(
    verts: np.ndarray,
    faces: np.ndarray,
    depth_camera: Union[CamParam, CamMats],
    ang_thresh: Union[float, None] = -0.2,
    background_depth: float = np.inf,
    backend: str = "auto",
) -> np.ndarray:
    """Render a mesh to a depth image, with optional edge-angle masking."""
    _, dist, _, _, angle = render_point_cloud(verts, faces, depth_camera, prefer=backend)
    rendered_depth = np.minimum(dist, background_depth)

    if ang_thresh is not None:
        # find edges based on angle threshold
        mask = np.isfinite(dist)
        edges = (angle > ang_thresh) & mask

        # interpolate to fill in gaps
        d_h, d_w = rendered_depth.shape
        X, Y = np.meshgrid(np.arange(d_w), np.arange(d_h))
        interpolated = griddata((X[~edges], Y[~edges]), rendered_depth[~edges],
                                (X[edges], Y[edges]), method='linear', fill_value=0.0)
        rendered_depth[edges] = interpolated

    return rendered_depth


def range_to_z(range_img: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Convert a range (Euclidean distance) image to z-depth using camera intrinsics.

    Parameters
    ----------
    range_img : (H, W) ndarray
        Range image in metres.
    intrinsic : (3, 3) ndarray
        Camera intrinsic matrix.

    Returns
    -------
    z_depth : (H, W) ndarray
        Z-depth image in metres.
    """

    # Extract intrinsic parameters
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # Create meshgrid for pixel coordinates
    h, w = range_img.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    # Compute z-depth from range image using intrinsic parameters
    z = range_img / np.sqrt(1 + ((xx - cx) / fx) ** 2 + ((yy - cy) / fy) ** 2)
    return z


def add_depth_noise(
        depth_m: np.ndarray,
        *,
        radial_strength: float = 0.00,   # 0 → none, 0.03 ≈ +3 % at corners
        gauss_std: float = 0.000,         # σ in metres of white noise
        speckle_prob: float = 0.000,      # 0.02 → 2 % pixels dropped
        quant_step: float = 0.00,        # 0.004 → 4 mm LSB
        kernel_blur: float = 0.0,        # optional PSF blur σ (px)
        rng: Union[np.random.Generator, None] = None,
        intrinsics: Union[dict, None] = None   # {'fx','fy','cx','cy'}  OR None
) -> np.ndarray:
    """
    Returns a *copy* of `depth_m` with simulated LiDAR / ToF artefacts.
    All parameters are per-frame scalars; set any to 0 to disable that error.
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = depth_m.copy()

    # 1) radial bias ----------------------------------------------------------
    if radial_strength != 0:
        h, w = noisy.shape
        if intrinsics is not None:
            cx, cy = intrinsics["cx"], intrinsics["cy"]
        else:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2)
        r2 /= r2.max()                  # 0 at centre, 1 at far corner
        radial_bias = 1.0 + radial_strength * r2
        noisy *= radial_bias.astype(noisy.dtype)

    # 2) per-pixel Gaussian noise --------------------------------------------
    if gauss_std > 0:
        noisy += rng.normal(0.0, gauss_std, size=noisy.shape).astype(noisy.dtype)

    # 3) speckle drop-outs (set to zero) -------------------------------------
    if speckle_prob > 0:
        mask = rng.random(size=noisy.shape) < speckle_prob
        noisy[mask] = 0

    # 4) quantisation ---------------------------------------------------------
    if quant_step > 0:
        noisy = np.round(noisy / quant_step) * quant_step

    # 5) optional optical blur (simulates projector PSF) ---------------------
    if kernel_blur > 0:
        valid = noisy > 0
        noisy_blur = gaussian_filter(np.nan_to_num(noisy, nan=0.0),
                                     sigma=kernel_blur, mode='nearest')
        norm = gaussian_filter(valid.astype(float),
                               sigma=kernel_blur, mode='nearest')
        norm[norm == 0] = 1.0
        noisy = noisy_blur / norm
        noisy[~valid] = 0          # preserve existing NaNs

    return noisy
