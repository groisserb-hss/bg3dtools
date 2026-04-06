#!/usr/bin/env python3
"""Compile Record3D rgbd/ frames into RGB and depth-heatmap MP4 videos.

Usage
-----
    python compile_video.py <path_to_rgbd>

*path_to_rgbd* may point to either the ``rgbd/`` folder itself or its parent
(the folder containing ``metadata``).  Both ``rgb.mp4`` and ``depth.mp4`` are
written to the parent of ``rgbd/``.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resolve imports from the toolbox
# ---------------------------------------------------------------------------
_toolbox = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.insert(0, os.path.normpath(_toolbox))

from bg3dtools.iphone.record3d_io import (
    load_record3d_depth_img,
    read_record3d,
)
from bg3dtools.render.colors import get_heatmap_color


def _resolve_folder(path: str) -> str:
    """Return the parent-of-rgbd folder, no matter what the user passed."""
    path = os.path.abspath(path)
    if os.path.basename(path) == "rgbd" and os.path.isdir(path):
        return os.path.dirname(path)
    if os.path.isdir(os.path.join(path, "rgbd")):
        return path
    sys.exit(f"Error: cannot find an rgbd/ directory from '{path}'")


def _compute_depth_caxis(
    depth_frames: list,
    depth_size: tuple,
    n_samples: int = 10,
) -> tuple:
    """Compute (vmin, vmax) from the center crop of sampled frames.

    
    p05 = 5th percentile of valid center-crop depth values
    vmin = p05 - 0.02 m
    vmax = p05 + 0.20 m
    """
    dh, dw = depth_size
    r0, r1 = 3 * dh // 8, 5 * dh // 8
    c0, c1 = 3 * dw // 8, 5 * dw // 8

    indices = np.linspace(0, len(depth_frames) - 1, n_samples, dtype=int)
    center_vals = []
    for idx in indices:
        depth = load_record3d_depth_img(depth_frames[idx], depth_size)
        crop = depth[r0:r1, c0:c1]
        valid = crop[(crop > 0) & np.isfinite(crop)]
        if len(valid):
            center_vals.append(valid)

    if not center_vals:
        return (0.25, 0.75)

    all_vals = np.concatenate(center_vals)
    p05 = float(np.percentile(all_vals, 5))
    return (p05 - 0.02, p05 + 0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to rgbd/ folder or its parent")
    args = parser.parse_args()

    folder = _resolve_folder(args.path)
    rec = read_record3d(folder)

    fps = rec["fps"]
    dh, dw = rec["depth_size"]
    rgb_frames = rec["rgb_frames"]
    depth_frames = rec["depth_frames"]

    rgb_out = os.path.join(folder, "rgb.mp4")
    depth_out = os.path.join(folder, "depth.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # --- RGB video ---
    # Read first frame to get dimensions (may differ from depth size)
    sample = cv2.imread(rgb_frames[0])
    h, w = sample.shape[:2]

    print(f"Writing RGB video: {len(rgb_frames)} frames, {w}x{h} @ {fps} fps")
    rgb_writer = cv2.VideoWriter(rgb_out, fourcc, fps, (w, h))
    for i, path in enumerate(rgb_frames):
        frame = cv2.imread(path)
        rgb_writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  rgb: {i + 1}/{len(rgb_frames)}")
    rgb_writer.release()
    print(f"  -> {rgb_out}  ({os.path.getsize(rgb_out) / 1024 / 1024:.1f} MB)")

    # --- Depth heatmap video ---
    print(f"Computing depth color range from center crop ...")
    vmin, vmax = _compute_depth_caxis(depth_frames, (dh, dw))
    print(f"  caxis: [{vmin:.3f}, {vmax:.3f}] m")

    print(f"Writing depth video: {len(depth_frames)} frames, {dw}x{dh} @ {fps} fps")
    depth_writer = cv2.VideoWriter(depth_out, fourcc, fps, (dw, dh))
    for i, path in enumerate(depth_frames):
        depth = load_record3d_depth_img(path, (dh, dw))
        rgb_img = get_heatmap_color(depth, caxis=(vmin, vmax), mapname="jet", out="uint8")
        # get_heatmap_color returns RGB; OpenCV expects BGR
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        depth_writer.write(bgr_img)
        if (i + 1) % 100 == 0:
            print(f"  depth: {i + 1}/{len(depth_frames)}")
    depth_writer.release()
    print(f"  -> {depth_out}  ({os.path.getsize(depth_out) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
