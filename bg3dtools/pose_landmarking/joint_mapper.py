"""Mapping between MediaPipe Pose landmarks and SMPL24 joint set."""

import numpy as np
from typing import List, Tuple, Optional
import logging
from enum import IntEnum

from .joint_annotations import BlazePoseLandmark as Blaze
from .joint_annotations import MPIJoints as MPI
from pathlib import Path


mapping = {
            MPI.PELVIS : {Blaze.LEFT_HIP: 0.47, Blaze.RIGHT_HIP: 0.47, Blaze.LEFT_SHOULDER: 0.03, Blaze.RIGHT_SHOULDER: 0.03},
            MPI.LEFT_HIP : {Blaze.LEFT_HIP: 0.75, Blaze.RIGHT_HIP: 0.15, Blaze.LEFT_KNEE: 0.1},
            MPI.RIGHT_HIP : {Blaze.RIGHT_HIP: 0.75, Blaze.LEFT_HIP: 0.15, Blaze.RIGHT_KNEE: 0.1},
            MPI.LUMBAR_SPINE : {Blaze.LEFT_HIP: 0.34, Blaze.RIGHT_HIP: 0.34, Blaze.LEFT_SHOULDER: 0.16, Blaze.RIGHT_SHOULDER: 0.16},
            MPI.LEFT_KNEE : {Blaze.LEFT_KNEE: 1.0},
            MPI.RIGHT_KNEE : {Blaze.RIGHT_KNEE: 1.0},
            MPI.THORACOLUMBAR_SPINE : {Blaze.LEFT_HIP: 0.2, Blaze.RIGHT_HIP: 0.2, Blaze.LEFT_SHOULDER: 0.3, Blaze.RIGHT_SHOULDER: 0.3},
            MPI.LEFT_ANKLE : {Blaze.LEFT_ANKLE: 1.0},
            MPI.RIGHT_ANKLE : {Blaze.RIGHT_ANKLE: 1.0},
            MPI.THORACIC_SPINE : {Blaze.LEFT_HIP: 0.15, Blaze.RIGHT_HIP: 0.15, Blaze.LEFT_SHOULDER: 0.35, Blaze.RIGHT_SHOULDER: 0.35},
            MPI.LEFT_FOOT : {Blaze.LEFT_FOOT_INDEX: 0.8, Blaze.LEFT_HEEL: 0.2},
            MPI.RIGHT_FOOT : {Blaze.RIGHT_FOOT_INDEX: 0.8, Blaze.RIGHT_HEEL: 0.2},
            MPI.NECK : {Blaze.LEFT_SHOULDER: 0.4, Blaze.RIGHT_SHOULDER: 0.4, Blaze.MOUTH_LEFT: 0.1, Blaze.MOUTH_RIGHT: 0.1},
            MPI.LEFT_COLLAR : {Blaze.LEFT_SHOULDER: 0.7, Blaze.RIGHT_SHOULDER: 0.18, Blaze.LEFT_HIP: 0.06, Blaze.RIGHT_HIP: 0.06},
            MPI.RIGHT_COLLAR : {Blaze.RIGHT_SHOULDER: 0.71, Blaze.LEFT_SHOULDER: 0.19, Blaze.RIGHT_HIP: 0.05, Blaze.LEFT_HIP: 0.05},
            MPI.HEAD : {Blaze.LEFT_SHOULDER: 0.15, Blaze.RIGHT_SHOULDER: 0.15, Blaze.LEFT_EAR: 0.35, Blaze.RIGHT_EAR: 0.35},
            MPI.LEFT_SHOULDER : {Blaze.LEFT_SHOULDER: 1.0},
            MPI.RIGHT_SHOULDER : {Blaze.RIGHT_SHOULDER: 1.0},
            MPI.LEFT_ELBOW : {Blaze.LEFT_ELBOW: 1.0},
            MPI.RIGHT_ELBOW : {Blaze.RIGHT_ELBOW: 1.0},
            MPI.LEFT_WRIST : {Blaze.LEFT_WRIST: 1.0},
            MPI.RIGHT_WRIST : {Blaze.RIGHT_WRIST: 1.0},
            MPI.LEFT_HAND : {Blaze.LEFT_PINKY: 0.5, Blaze.LEFT_INDEX: 0.5},
            MPI.RIGHT_HAND : {Blaze.RIGHT_PINKY: 0.5, Blaze.RIGHT_INDEX: 0.5}
        }

Blaze33toMPI24 = np.zeros([24, 33])
for dest_idx, map_dict in mapping.items():
    for src_idx, src_val in map_dict.items():
        Blaze33toMPI24[dest_idx, src_idx] = src_val
assert np.allclose(np.sum(Blaze33toMPI24, axis=-1), np.ones(24), atol=1e-6), 'Linear mappings must sum to 1'


# Load smpl_2_blaze regressor
_SMPL_2_BLAZE_PATH = Path(__file__).parent / 'smpl_2_blaze.npy'
_SMPL_2_BLAZE_CACHE = {}


def load_smpl_2_blaze(path: Optional[Path] = None) -> np.ndarray:
    """Load SMPL to BlazePose regressor matrix."""
    if path is None:
        path = _SMPL_2_BLAZE_PATH

    path_str = str(path)
    if path_str not in _SMPL_2_BLAZE_CACHE:
        _SMPL_2_BLAZE_CACHE[path_str] = np.load(path)

    return _SMPL_2_BLAZE_CACHE[path_str]
