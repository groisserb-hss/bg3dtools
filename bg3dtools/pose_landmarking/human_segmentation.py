"""
Human segmentation using MediaPipe.

This module provides wrappers for MediaPipe's image segmentation for
isolating human subjects from backgrounds in video frames.
"""

import mediapipe as mp
from .resource_paths import segmenter_weights
from ..utils import SuppressCppStderr
from ._delegate import create_detector


def create_mp_segmenter(model_path: str = segmenter_weights, use_gpu=None):
    """Create a MediaPipe ImageSegmenter.

    `use_gpu`: None -> decide from MEDIAPIPE_DISABLE_GPU env; True/False -> force
    (falls back to CPU if the GPU delegate is unavailable).
    """
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    RunningMode = mp.tasks.vision.RunningMode

    def _build(delegate):
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
            running_mode=RunningMode.VIDEO,
            output_category_mask=True,
            output_confidence_masks=True,
        )
        with SuppressCppStderr():
            return ImageSegmenter.create_from_options(options)

    return create_detector(_build, use_gpu=use_gpu, what="ImageSegmenter")
