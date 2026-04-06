# Optional: requires mediapipe
try:
    from .pose_detection import (get_mediapipe_detector,
                                 landmark_video,
                                 LandmarkVideoResult)
except ImportError:
    get_mediapipe_detector = None
    landmark_video = None
    LandmarkVideoResult = None

# Core: lightweight imports (enums, constants, mapper)
from .joint_annotations import (BlazePoseLandmark,
                                BLAZE_POSE_CONNECTIONS,
                                MPIJoints,
                                # MPIBones,
                                MPI_KINTREE,
                                MPI_POSE_CONNECTIONS,
                                draw_landmarks_on_image,
                                draw_3d_skeleton)

from .joint_mapper import Blaze33toMPI24, load_smpl_2_blaze

from .mask_cleanup import (
    morphological_closing_3d,
    gaussian_smooth_3d,
    cleanup_masks,
)


__all__ = ["get_mediapipe_detector",
           "landmark_video",
           "LandmarkVideoResult",
           "draw_landmarks_on_image",
           "BlazePoseLandmark",
           "BLAZE_POSE_CONNECTIONS",
           "MPIJoints",
           # "MPIBones",
           "MPI_KINTREE",
           "MPI_POSE_CONNECTIONS",
           "draw_3d_skeleton",
           "Blaze33toMPI24",
           "load_smpl_2_blaze",
           # Mask cleanup
           "morphological_closing_3d",
           "gaussian_smooth_3d",
           "cleanup_masks"]
