"""
Face de-identification utilities.

This module provides functions for detecting and blurring faces in images
and video frames to protect subject privacy. Supports multiple detection
backends including MediaPipe and OpenCV Haar cascades.
"""

import os
from statistics import mean
from typing import Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ..image_tools import normalized_to_pixel_coordinates
from .resource_paths import face_weights
from ._delegate import create_detector

# Keep MediaPipe logging quieter (but do not swallow real errors).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "0")


# A small set of BlazePose landmark *indices* that roughly bound the face.
# These indices match the canonical BlazePose 33-landmark order used by MediaPipe Pose.
# 0:nose, 1:left_eye_inner, 2:left_eye, 3:left_eye_outer, 4:right_eye_inner, 5:right_eye,
# 6:right_eye_outer, 7:left_ear, 8:right_ear, 9:mouth_left, 10:mouth_right
_FACE_POSE_LMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def create_opencv_face_cascade(cascade_path: Optional[str] = None) -> cv2.CascadeClassifier:
    """Create an OpenCV Haar cascade for frontal face detection.

    If `cascade_path` is None, uses OpenCV's built-in haarcascades path.
    Raises if the cascade can't be loaded.
    """
    if cascade_path is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise FileNotFoundError(f"Failed to load OpenCV Haar cascade: {cascade_path}")

    return face_cascade


def create_tasks_face_detector(
    model_asset_path: Optional[str] = None,
    min_detection_confidence: float = 0.9,
    video: bool = True,
    use_gpu: Optional[bool] = None,
) -> mp_vision.FaceDetector:
    """Create a MediaPipe *Tasks* FaceDetector.

    `model_asset_path` must point to a face detector `.tflite` model compatible with the Tasks API.
    `use_gpu`: None -> CPU unless MEDIAPIPE_USE_GPU is set; True/False -> force.
    GPU is opt-in; it falls back to CPU if the GPU delegate cannot be created.
    """
    if model_asset_path is None:
        model_asset_path = face_weights

    running_mode = mp.tasks.vision.RunningMode.VIDEO if video else mp.tasks.vision.RunningMode.IMAGE

    def _build(delegate):
        options = mp_vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path, delegate=delegate),
            min_detection_confidence=min_detection_confidence,
            running_mode=running_mode,
        )
        return mp_vision.FaceDetector.create_from_options(options)

    return create_detector(_build, use_gpu=use_gpu, what="FaceDetector")


def _clamp_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def _bbox_from_pose_landmarks(pose_landmarks, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """Approximate a face bbox from BlazePose landmarks (nose/eyes/ears/mouth).

    Expects `pose_landmarks_list` in the MediaPipe Tasks PoseLandmarker format:
      - list of poses
      - each pose is a list of normalized landmarks

    Returns pixel bbox (x0, y0, x1, y1) or None.
    """
    if pose_landmarks is None:
        return None

    pts = []
    for lm_enum in _FACE_POSE_LMS:
        i = int(lm_enum)
        if i >= len(pose_landmarks):
            continue
        lm = pose_landmarks[i]
        p = normalized_to_pixel_coordinates(lm.x, lm.y, w, h)
        if p is not None:
            pts.append(p)

    if len(pts) < 3:
        return None

    xs = np.array([p[0] for p in pts], dtype=np.int32)
    ys = np.array([p[1] for p in pts], dtype=np.int32)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # Pad box (forehead/chin margin).
    pad_x = max(8, int(4 * ys.std()))
    pad_y = max(8, int(4 * xs.std()))

    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y

    return _clamp_bbox(x0, y0, x1, y1, w, h)


def _bbox_from_opencv(rgb_image: np.ndarray, face_cascade: cv2.CascadeClassifier) -> Optional[Tuple[int, int, int, int]]:
    """Detect face bbox via OpenCV Haar cascade (RGB input)."""
    # Lots of false positives, really can't use it
    if rgb_image.ndim != 3:
        return None
    h, w = rgb_image.shape[:2]
    s = min(h, w)

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(s // 20, s // 20),
    )

    if faces is None or len(faces) == 0:
        return None

    # Largest face.
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    x0, y0, x1, y1 = int(x), int(y), int(x + fw), int(y + fh)

    pad = max(6, int(0.12 * max(fw, fh)))
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad

    return _clamp_bbox(x0, y0, x1, y1, w, h)


def _bbox_from_mediapipe_face(rgb_image: np.ndarray,
                              mp_face_detector: mp_vision.FaceDetector,
                              timestamp: Optional[int] = None,
                              ) -> Optional[Tuple[int, int, int, int]]:
    """Detect face bbox via MediaPipe *Tasks* FaceDetector.

    - `mp_face_detector` must be a `mediapipe.tasks.python.vision.FaceDetector`.
    - `rgb_image` must be uint8 RGB.

    Returns pixel bbox (x0, y0, x1, y1) or None.
    """
    if rgb_image.ndim != 3:
        return None
    h, w = rgb_image.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    if timestamp is None:
        res = mp_face_detector.detect(mp_image)
    else:
        res = mp_face_detector.detect_for_video(mp_image, timestamp_ms=timestamp)

    if not res.detections:
        return None

    # Pick the highest-score detection.
    def _score(det) -> float:
        cats = getattr(det, "categories", None)
        if cats:
            return float(cats[0].score)
        return 0.0

    det = max(res.detections, key=_score)

    # Tasks bounding box is already in pixel coordinates.
    bb = det.bounding_box
    x0 = int(bb.origin_x)
    y0 = int(bb.origin_y)
    x1 = int(bb.origin_x + bb.width)
    y1 = int(bb.origin_y + bb.height)

    # Expand a bit.
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad_x = max(6, int(0.10 * bw))
    pad_y = max(6, int(0.18 * bh))

    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y

    return _clamp_bbox(x0, y0, x1, y1, w, h)


def _pixelate_roi(rgb_image: np.ndarray, bbox: Tuple[int, int, int, int], block: int = 12) -> np.ndarray:
    """Pixelate a rectangular ROI (in-place) and return the image."""
    x0, y0, x1, y1 = bbox
    roi = rgb_image[y0:y1, x0:x1]
    if roi.size == 0:
        return rgb_image

    rh, rw = roi.shape[:2]
    ds_w = max(1, rw // block)
    ds_h = max(1, rh // block)

    small = cv2.resize(roi, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    rgb_image[y0:y1, x0:x1] = pix
    return rgb_image


def deidentify_face_rgb(
    rgb_image: np.ndarray,
    pose_landmarks: Optional[list],
    face_cascade: cv2.CascadeClassifier,
    mp_face_detector: mp_vision.FaceDetector,
    block: int = 12,
    timestamp: Optional[int] = None,
) -> Tuple[np.ndarray, bool]:
    """Pixelate the face region using a union of:

      1) OpenCV Haar cascade bbox
      2) MediaPipe Tasks FaceDetector bbox
      3) BlazePose landmark-derived bbox

    Returns (output_image, found_face).
    """
    h, w = rgb_image.shape[:2]
    s = min(h, w)

    bboxes = []

    # b1 = _bbox_from_opencv(rgb_image, face_cascade)
    # if b1 is not None:
    #     bboxes.append(b1)

    b2 = _bbox_from_mediapipe_face(rgb_image, mp_face_detector, timestamp=timestamp)
    if b2 is not None:
        bboxes.append(b2)

    b3 = _bbox_from_pose_landmarks(pose_landmarks, w, h)
    if b3 is not None:
        bboxes.append(b3)

    if not bboxes:
        return rgb_image, False

    if b2 is not None and b3 is not None:
        # sanity check that landmarks are close together
        x2 = (b2[0] + b2[2]) / 2
        y2 = (b2[1] + b2[3]) / 2
        x3 = (b3[0] + b3[2]) / 2
        y3 = (b3[1] + b3[3]) / 2
        assert abs(x3 - x2) < (s // 20), 'face detection and landmark detection do not match'
        assert abs(y3 - y2) < (s // 20), 'face detection and landmark detection do not match'

    # Union all boxes.
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    bbox = _clamp_bbox(int(x0), int(y0), int(x1), int(y1), w, h)

    out = rgb_image.copy()
    _pixelate_roi(out, bbox, block=block)
    return out, True
