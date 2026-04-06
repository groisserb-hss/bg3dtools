"""
Human body joint landmarks and annotations.

This module defines joint landmark indices for BlazePose and SMPL body
models, along with utilities for visualizing and transforming pose data.

Classes
-------
BlazePoseLandmark
    MediaPipe BlazePose landmark indices.
MPIJoints
    SMPL body model joint indices.
"""

from enum import IntEnum
from typing import List, Tuple

import numpy as np


class BlazePoseLandmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    @classmethod
    def face_landmarks(cls) -> List['BlazePoseLandmark']:
        return [BlazePoseLandmark.NOSE,
                BlazePoseLandmark.LEFT_EYE_INNER,
                BlazePoseLandmark.LEFT_EYE,
                BlazePoseLandmark.LEFT_EYE_OUTER,
                BlazePoseLandmark.RIGHT_EYE_INNER,
                BlazePoseLandmark.RIGHT_EYE,
                BlazePoseLandmark.RIGHT_EYE_OUTER,
                BlazePoseLandmark.LEFT_EAR,
                BlazePoseLandmark.RIGHT_EAR,
                BlazePoseLandmark.MOUTH_LEFT,
                BlazePoseLandmark.MOUTH_RIGHT]

    @classmethod
    def torso_landmarks(cls) -> List['BlazePoseLandmark']:
        return [BlazePoseLandmark.LEFT_SHOULDER,
                BlazePoseLandmark.RIGHT_SHOULDER,
                BlazePoseLandmark.RIGHT_HIP,
                BlazePoseLandmark.LEFT_HIP]

    @classmethod
    def hand_landmarks(cls) -> List['BlazePoseLandmark']:
        return [BlazePoseLandmark.LEFT_WRIST,
                BlazePoseLandmark.RIGHT_WRIST,
                BlazePoseLandmark.LEFT_PINKY,
                BlazePoseLandmark.RIGHT_PINKY,
                BlazePoseLandmark.LEFT_INDEX,
                BlazePoseLandmark.RIGHT_INDEX,
                BlazePoseLandmark.LEFT_THUMB,
                BlazePoseLandmark.RIGHT_THUMB]

class MPIJoints(IntEnum):
    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    LUMBAR_SPINE = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    THORACOLUMBAR_SPINE = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    THORACIC_SPINE = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    LEFT_HAND = 22
    RIGHT_HAND = 23

BLAZE_POSE_CONNECTIONS = (
    (BlazePoseLandmark.NOSE, BlazePoseLandmark.LEFT_EYE_INNER),
    (BlazePoseLandmark.LEFT_EYE_INNER, BlazePoseLandmark.LEFT_EYE),
    (BlazePoseLandmark.LEFT_EYE, BlazePoseLandmark.LEFT_EYE_OUTER),
    (BlazePoseLandmark.LEFT_EYE_OUTER, BlazePoseLandmark.LEFT_EAR),

    (BlazePoseLandmark.NOSE, BlazePoseLandmark.RIGHT_EYE_INNER),
    (BlazePoseLandmark.RIGHT_EYE_INNER, BlazePoseLandmark.RIGHT_EYE),
    (BlazePoseLandmark.RIGHT_EYE, BlazePoseLandmark.RIGHT_EYE_OUTER),
    (BlazePoseLandmark.RIGHT_EYE_OUTER, BlazePoseLandmark.RIGHT_EAR),

    (BlazePoseLandmark.MOUTH_LEFT, BlazePoseLandmark.MOUTH_RIGHT),

    (BlazePoseLandmark.LEFT_SHOULDER, BlazePoseLandmark.RIGHT_SHOULDER),
    (BlazePoseLandmark.LEFT_SHOULDER, BlazePoseLandmark.LEFT_HIP),
    (BlazePoseLandmark.RIGHT_SHOULDER, BlazePoseLandmark.RIGHT_HIP),
    (BlazePoseLandmark.LEFT_HIP, BlazePoseLandmark.RIGHT_HIP),

    (BlazePoseLandmark.LEFT_SHOULDER, BlazePoseLandmark.LEFT_ELBOW),
    (BlazePoseLandmark.LEFT_ELBOW, BlazePoseLandmark.LEFT_WRIST),
    (BlazePoseLandmark.LEFT_WRIST, BlazePoseLandmark.LEFT_PINKY),
    (BlazePoseLandmark.LEFT_WRIST, BlazePoseLandmark.LEFT_INDEX),
    (BlazePoseLandmark.LEFT_WRIST, BlazePoseLandmark.LEFT_THUMB),
    (BlazePoseLandmark.LEFT_PINKY, BlazePoseLandmark.LEFT_INDEX),

    (BlazePoseLandmark.RIGHT_SHOULDER, BlazePoseLandmark.RIGHT_ELBOW),
    (BlazePoseLandmark.RIGHT_ELBOW, BlazePoseLandmark.RIGHT_WRIST),
    (BlazePoseLandmark.RIGHT_WRIST, BlazePoseLandmark.RIGHT_PINKY),
    (BlazePoseLandmark.RIGHT_WRIST, BlazePoseLandmark.RIGHT_INDEX),
    (BlazePoseLandmark.RIGHT_WRIST, BlazePoseLandmark.RIGHT_THUMB),
    (BlazePoseLandmark.RIGHT_PINKY, BlazePoseLandmark.RIGHT_INDEX),

    (BlazePoseLandmark.LEFT_HIP, BlazePoseLandmark.LEFT_KNEE),
    (BlazePoseLandmark.LEFT_KNEE, BlazePoseLandmark.LEFT_ANKLE),
    (BlazePoseLandmark.LEFT_ANKLE, BlazePoseLandmark.LEFT_HEEL),
    (BlazePoseLandmark.LEFT_HEEL, BlazePoseLandmark.LEFT_FOOT_INDEX),
    (BlazePoseLandmark.LEFT_ANKLE, BlazePoseLandmark.LEFT_FOOT_INDEX),

    (BlazePoseLandmark.RIGHT_HIP, BlazePoseLandmark.RIGHT_KNEE),
    (BlazePoseLandmark.RIGHT_KNEE, BlazePoseLandmark.RIGHT_ANKLE),
    (BlazePoseLandmark.RIGHT_ANKLE, BlazePoseLandmark.RIGHT_HEEL),
    (BlazePoseLandmark.RIGHT_HEEL, BlazePoseLandmark.RIGHT_FOOT_INDEX),
    (BlazePoseLandmark.RIGHT_ANKLE, BlazePoseLandmark.RIGHT_FOOT_INDEX),
)


MPI_KINTREE = (
    -1,                             # PELVIS
    MPIJoints.PELVIS,               # LEFT HIP
    MPIJoints.PELVIS,               # RIGHT HIP
    MPIJoints.PELVIS,               # LUMBAR SPINE
    MPIJoints.LEFT_HIP,             # LEFT KNEE
    MPIJoints.RIGHT_HIP,            # RIGHT KNEE
    MPIJoints.LUMBAR_SPINE,         # THORACOLUMBAR SPINE
    MPIJoints.LEFT_KNEE,            # LEFT ANKLE
    MPIJoints.RIGHT_KNEE,           # RIGHT ANKLE
    MPIJoints.THORACOLUMBAR_SPINE,  # THORACIC SPINE
    MPIJoints.LEFT_ANKLE,           # LEFT FOOT
    MPIJoints.RIGHT_ANKLE,          # RIGHT FOOT
    MPIJoints.THORACIC_SPINE,       # NECK
    MPIJoints.THORACIC_SPINE,       # LEFT COLLAR
    MPIJoints.THORACIC_SPINE,       # RIGHT COLLAR
    MPIJoints.NECK,                 # HEAD
    MPIJoints.LEFT_COLLAR,          # LEFT SHOULDER
    MPIJoints.RIGHT_COLLAR,         # RIGHT SHOULDER
    MPIJoints.LEFT_SHOULDER,        # LEFT ELBOW
    MPIJoints.RIGHT_SHOULDER,       # RIGHT ELBOW
    MPIJoints.LEFT_ELBOW,           # LEFT WRIST
    MPIJoints.RIGHT_ELBOW,          # RIGHT WRIST
    MPIJoints.LEFT_WRIST,           # LEFT HAND
    MPIJoints.RIGHT_WRIST,          # RIGHT HAND
)


MPI_POSE_CONNECTIONS = tuple([(i, j) for i, j in enumerate(MPI_KINTREE) if j >= 0])


def draw_landmarks_on_image(rgb_image: np.ndarray,
                            pose_landmarks: np.ndarray | list,
                            pose_connections: Tuple[Tuple[int, int]],
                            color: Tuple[int, int, int] = (0, 255, 0),
                            thickness: int = 3) -> np.ndarray:
    import cv2
    from ..image_tools.utils import normalized_to_pixel_coordinates

    annotated = rgb_image.copy()

    # OpenCV drawing
    h, w = annotated.shape[:2]

    pts = [None] * len(pose_landmarks)
    for i, lm in enumerate(pose_landmarks):
        if isinstance(lm, np.ndarray):
            p = normalized_to_pixel_coordinates(lm[0], lm[1], w, h)
        else:
            p = normalized_to_pixel_coordinates(lm.x, lm.y, w, h)

        pts[i] = p
        if p is not None:
            cv2.circle(annotated, p, radius=5*thickness, color=color)

    for a, b in pose_connections:
        if a >= len(pts) or b >= len(pts):
            raise ValueError('pose connections includes invalid connections')

        if pts[a] is not None and pts[b] is not None:
            cv2.line(annotated, pts[a], pts[b], color=color, thickness=thickness)

    return annotated


def draw_3d_skeleton(joints: np.ndarray,
                     pose_connections: Tuple[Tuple[int, int]],
                     color: Tuple[float, float, float] = (0, 1.0, 0),
                     render: bool = False,
                     ):
    """Draw a 3D skeleton as a LineSet with bones and joint spheres.

    Returns an ``open3d.geometry.LineSet`` that can be added to a
    Visualizer or OffscreenRenderer scene.
    """
    import open3d as o3d

    joint_list = [p if np.all(np.isfinite(p)) else None for p in joints]

    valid = [(a, b) for a, b in pose_connections
             if joint_list[a] is not None and joint_list[b] is not None]
    if not valid:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        return ls

    start_idx, end_idx = zip(*valid)
    start_pts = joints[list(start_idx)]
    end_pts = joints[list(end_idx)]
    scale = np.mean(np.linalg.norm(start_pts - end_pts, axis=-1)) / 10.

    # --- bone lines ---
    n_bones = len(start_idx)
    bone_points = np.concatenate([start_pts, end_pts], axis=0)  # (2B, 3)
    bone_lines = np.column_stack([np.arange(n_bones),
                                  np.arange(n_bones, 2 * n_bones)])

    # --- joint sphere wireframes ---
    from bg3dtools.mesh.generate import generate_icosahedron
    icos_v, icos_f = generate_icosahedron()
    # extract unique edges from icosahedron faces
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
        sphere_pts = icos_v_scaled + j[None, :]
        sphere_lines = icos_edges + offset
        all_points.append(sphere_pts)
        all_lines.append(sphere_lines)
        offset += len(icos_v_scaled)

    points = np.concatenate(all_points, axis=0).astype(np.float64)
    lines = np.concatenate(all_lines, axis=0).astype(np.int64)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)

    if render:
        o3d.visualization.draw_geometries([ls])
    return ls

