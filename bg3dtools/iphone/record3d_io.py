"""
Record3D (TrueDepth camera) data I/O utilities.

Reads recordings exported from the Record3D iOS app in its native format:
  - ``metadata`` JSON with intrinsics, poses, timestamps
  - ``rgbd/*.depth`` LZFSE-compressed float32 depth frames
  - ``rgbd/*.jpg`` per-frame RGB images

The public helpers mirror the Stray Scanner API in ``data_io.py`` so that
downstream code can consume either format transparently.
"""

import os
import json
from glob import glob

import numpy as np
from scipy.spatial.transform import Rotation


def is_record3d(folder: str) -> bool:
    """Return True if *folder* looks like a Record3D recording."""
    return os.path.isfile(os.path.join(folder, 'metadata'))


def read_record3d(folder: str) -> dict:
    """Read a Record3D recording into the same dict shape as ``read_data()``.

    Returns
    -------
    dict
        Keys: ``intrinsics``, ``poses``, ``depth_frames``, ``rgb_frames``,
        ``mp4_file`` (None), ``fps``, ``source``, ``depth_size``.
    """
    with open(os.path.join(folder, 'metadata')) as f:
        meta = json.load(f)

    # --- Intrinsics (stored transposed: [[fx,0,0],[0,fy,0],[cx,cy,1]]) ---
    K_flat = np.array(meta['K'], dtype=np.float64).reshape(3, 3)
    intrinsics = K_flat.T  # -> OpenCV [[fx,0,cx],[0,fy,cy],[0,0,1]]

    # --- Depth / RGB resolution ---
    dh, dw = int(meta['dh']), int(meta['dw'])
    depth_size = (dh, dw)

    # --- Poses: [tx, ty, tz, qw, qx, qy, qz] → (T, 4, 4) ---
    raw_poses = np.asarray(meta['poses'], dtype=np.float64)
    translations = raw_poses[:, :3]                          # (T, 3): tx, ty, tz
    quats_wxyz = raw_poses[:, 3:]                            # (T, 4): qw, qx, qy, qz
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]                # scipy expects [x, y, z, w]
    rotations = Rotation.from_quat(quats_xyzw).as_matrix()   # (T, 3, 3)

    N = len(raw_poses)
    poses = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations

    # --- FPS ---
    fps = float(meta.get('fps', 30))

    # --- Frame lists (sort numerically, not lexicographically) ---
    rgbd_dir = os.path.join(folder, 'rgbd')
    depth_frames = sorted(
        glob(os.path.join(rgbd_dir, '*.depth')),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    rgb_frames = sorted(
        glob(os.path.join(rgbd_dir, '*.jpg')),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )

    assert len(depth_frames) > 0, f'No .depth files in {rgbd_dir}'
    assert len(rgb_frames) > 0, f'No .jpg files in {rgbd_dir}'

    return {
        'intrinsics': intrinsics,
        'poses': poses,
        'depth_frames': depth_frames,
        'rgb_frames': rgb_frames,
        'mp4_file': None,
        'fps': fps,
        'source': 'record3d',
        'depth_size': depth_size,
    }


def load_record3d_depth_img(path: str, depth_size: tuple) -> np.ndarray:
    """Decompress a single ``.depth`` LZFSE file to a float32 depth map (meters).

    Invalid pixels are NaN in the raw data; this function replaces them with 0
    to match the Stray Scanner convention.
    """
    import liblzfse

    with open(path, 'rb') as f:
        raw = liblzfse.decompress(f.read())

    dh, dw = depth_size
    expected_f32 = dh * dw * 4
    expected_f16 = dh * dw * 2

    if len(raw) == expected_f32:
        depth = np.frombuffer(raw, dtype=np.float32).reshape(dh, dw)
    elif len(raw) == expected_f16:
        depth = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(dh, dw)
    else:
        raise ValueError(
            f'Unexpected decompressed size {len(raw)} for depth_size={depth_size} '
            f'(expected {expected_f32} for float32 or {expected_f16} for float16)'
        )

    # Replace NaN with 0 (consistent with Stray Scanner filtered-out pixels)
    depth = np.nan_to_num(depth, nan=0.0)
    return depth


def load_record3d_depth(depth_frames: list, depth_size: tuple) -> np.ndarray:
    """Load all Record3D depth frames into a ``(T, H, W)`` float32 array."""
    dh, dw = depth_size
    depth_array = np.zeros((len(depth_frames), dh, dw), dtype=np.float32)
    for i, f in enumerate(depth_frames):
        depth_array[i] = load_record3d_depth_img(f, depth_size)
    return depth_array
