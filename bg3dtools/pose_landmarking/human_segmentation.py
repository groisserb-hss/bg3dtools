"""
Human segmentation using MediaPipe.

This module provides wrappers for MediaPipe's image segmentation for
isolating human subjects from backgrounds in video frames.
"""

import mediapipe as mp
from .resource_paths import segmenter_weights
from ..utils import SuppressCppStderr


def create_mp_segmenter(model_path: str = segmenter_weights):
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        output_category_mask=True,
        output_confidence_masks=True,
    )

    with SuppressCppStderr():
        segmenter = ImageSegmenter.create_from_options(options)
    return segmenter
