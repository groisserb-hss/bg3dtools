"""
File paths to pose landmarking model weights.

Provides default paths for MediaPipe Pose, BlazeFace, and selfie
segmentation model weight files.
"""

from os.path import join, dirname

_RESOURCES = join(dirname(__file__), 'resources')

landmarker_weights = join(_RESOURCES, 'pose_landmarker_heavy.task')
face_weights = join(_RESOURCES, 'blaze_face_short_range.tflite')
segmenter_weights = join(_RESOURCES, 'selfie_segmenter.tflite')
