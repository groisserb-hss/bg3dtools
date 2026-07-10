"""
wrapper for mediapipe pose detection
"""
import os
import sys
import logging
from dataclasses import dataclass, asdict
from typing import Optional

# Suppress MediaPipe/TensorFlow C++ logs (must be before import)
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "0")

from ..utils import SuppressCppStderr

# Suppress C++ logs during mediapipe import
with SuppressCppStderr():
    import mediapipe as mp

import numpy as np

# MediaPipe "Tasks" API (preferred / modern).
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from scipy.signal import savgol_filter

from .resource_paths import landmarker_weights
from ._delegate import create_detector
from .human_segmentation import create_mp_segmenter
from .deidentify import deidentify_face_rgb, create_tasks_face_detector, create_opencv_face_cascade
from .joint_annotations import BlazePoseLandmark


@dataclass
class LandmarkVideoResult:
    """
    Result from landmark_video() pose detection.

    Attributes:
        blaze_2d: (T, 33, 4) BlazePose 2D landmarks [x, y, z, visibility].
            x, y are normalized to [0, 1] image coordinates.
        blaze_3d: (T, 33, 4) BlazePose 3D world landmarks [x, y, z, visibility].
            Coordinates are in meters, hip-centered.
        deidentified: (T, H, W, 3) Face-blurred RGB frames, or None if not requested.
        body_masks: (T, H, W) Body segmentation masks, or None if not requested.
        confidence_masks: (T, H, W) float32 foreground confidence in [0,1], or None.

    Usage:
        # Save to file
        np.savez(path, **result.to_dict())

        # Load from file
        result = LandmarkVideoResult.from_file(path)
    """
    blaze_2d: np.ndarray
    blaze_3d: np.ndarray
    deidentified: Optional[np.ndarray] = None
    body_masks: Optional[np.ndarray] = None
    confidence_masks: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dict for saving with np.savez."""
        d = {}
        for k, v in asdict(self).items():
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_file(cls, path: str) -> 'LandmarkVideoResult':
        """Load from npz file."""
        data = dict(np.load(path, allow_pickle=True))
        # Handle missing optional fields
        return cls(
            blaze_2d=data['blaze_2d'],
            blaze_3d=data['blaze_3d'],
            deidentified=data.get('deidentified'),
            body_masks=data.get('body_masks'),
            confidence_masks=data.get('confidence_masks'),
        )


def smooth_keypoints(
        kp: np.ndarray,
        window: int = 11,
        poly: int = 2,
        max_gap: int = 5
) -> np.ndarray:
    """
    kp : T×K×4  float32
    Returns same shape, NaNs kept for
    gaps > max_gap consecutive frames.
    """
    T, K, D = kp.shape
    out = kp.copy()
    for k in range(K):
        for d in range(4):                  # x / y
            vec = out[:,k,d]
            if np.all(np.isnan(vec)):
                continue                     # skip if all NaNs
            keep_nan = np.zeros_like(vec, bool)

            if np.isnan(vec[0]):
                idx = np.nonzero(np.isfinite(vec))[0][0]
                vec[0] = vec[idx]  # fill first NaN with first valid value
                keep_nan[0] = idx > max_gap
            if np.isnan(vec[-1]):
                idx = np.nonzero(np.isfinite(vec))[0][-1]
                vec[-1] = vec[idx]  # fill last NaN with last valid value
                keep_nan[-1] = (len(vec) - idx) > max_gap

            # --- patch gaps --------------------------------
            gaps = np.flatnonzero(np.diff(np.r_[False, np.isnan(vec), False]))
            starts, stops = gaps[::2], gaps[1::2]

            for s,e in zip(starts, stops):
                vec[s:e] = np.linspace(vec[s-1], vec[e], (e-s)+2)[1:-1]
                keep_nan[s:e] = (e - s) > max_gap        # leave big gaps NaN

            # --- savgol smoothing over valid points --------------
            smoothed = savgol_filter(vec, window, poly)
            vec = smoothed

            vec[keep_nan] = np.nan             # restore big-gap NaNs
            out[:,k,d] = vec
    return out


def get_mediapipe_detector(video: bool = True, model_asset_path: Optional[str] = None,
                           use_gpu=None):
    """Create a MediaPipe PoseLandmarker using the modern Tasks API.

    use_gpu: None -> decide from MEDIAPIPE_DISABLE_GPU env (default GPU when
    available); True/False -> force. Falls back to CPU if the GPU (OpenGL)
    delegate cannot be created on this platform.
    """
    running_mode = mp_vision.RunningMode.VIDEO if video else mp_vision.RunningMode.IMAGE

    if model_asset_path is None:
        model_asset_path = landmarker_weights
    assert os.path.isfile(model_asset_path), 'Failed to load model weights from %s' % model_asset_path

    def _build(delegate):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path, delegate=delegate),
            running_mode=running_mode,
        )
        # Suppress C++ logs during model creation
        with SuppressCppStderr():
            return mp_vision.PoseLandmarker.create_from_options(options)

    return create_detector(_build, use_gpu=use_gpu, what="PoseLandmarker")


def landmark_video(video_generator,
                   fps: float = 15.,
                   smooth: bool = True,
                   deidentify: bool = False,
                   segment: bool = False,
                   target_frames: Optional[int] = None,
                   use_gpu=None) -> 'LandmarkVideoResult':
    """
    Process a video and extract BlazePose landmarks.

    Args:
        video_generator: Iterable yielding RGB frames (H, W, 3).
        fps: Frame rate for timestamp calculation.
        smooth: Apply Savitzky-Golay smoothing to landmarks.
        deidentify: Blur faces in output video.
        segment: Generate body segmentation masks.
        target_frames: Number of frames to process (None = all).
        use_gpu: MediaPipe delegate selection. None -> decide from the
            MEDIAPIPE_DISABLE_GPU env var (default: GPU when available);
            True/False -> force. Falls back to CPU if the GPU (OpenGL)
            delegate is unavailable on this platform.

    Returns:
        LandmarkVideoResult with blaze_2d, blaze_3d, and optionally
        deidentified video and body_masks.
    """
    log = logging.getLogger('PoseLandmarkDetector')
    detector = get_mediapipe_detector(video=True, use_gpu=use_gpu)
    num_kp = len(BlazePoseLandmark)

    # Optional face de-identification resources.
    face_cascade = mp_face_detector = mp_segmenter = None
    if deidentify:
        face_cascade = create_opencv_face_cascade()
        mp_face_detector = create_tasks_face_detector(video=True, use_gpu=use_gpu)
    if segment:
        mp_segmenter = create_mp_segmenter(use_gpu=use_gpu)

    warn_no_landmarks = 0
    warn_no_face = 0

    landmarks_2d = np.full([target_frames, num_kp, 4], np.nan, dtype=np.float32)
    landmarks_3d = np.full([target_frames, num_kp, 4], np.nan, dtype=np.float32)
    # video files; size unknown
    blurred_video, body_masks, confidence_masks = [], [], []
    frame_step = 1000 / fps  # Calculate the frame step (ms) based on the desired FPS

    # progress = tqdm(range(video_generator.count_frames()), desc="Processing video", unit="frame", postfix="")
    for idx, rgb_img in enumerate(video_generator):
        if idx == target_frames:
            break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        ## Landmark detection
        detection_result = detector.detect_for_video(mp_image, int(frame_step * idx))
        if detection_result.pose_landmarks:
            pose_landmarks = detection_result.pose_landmarks[0]
            # collect 2d landmarks into numpy array
            for kk, landmark in enumerate(pose_landmarks):
                landmarks_2d[idx, kk] = landmark.x, landmark.y, landmark.z, landmark.visibility

            # collect 3d landmarks into numpy array
            pose_world = detection_result.pose_world_landmarks[0]
            for i, landmark in enumerate(pose_world):
                landmarks_3d[idx, i] = landmark.x, landmark.y, landmark.z, landmark.visibility

        else:
            warn_no_landmarks += 1
            pose_landmarks = None

        ## face blurring
        if deidentify:
            blurred_img, found = deidentify_face_rgb(
                rgb_image=rgb_img, pose_landmarks=pose_landmarks,
                face_cascade=face_cascade, mp_face_detector=mp_face_detector,
                block=24, timestamp=int(frame_step * idx))
            blurred_video.append(blurred_img)

            if not found:
                warn_no_face += 1

        ## body segmentation
        if segment:
            result = mp_segmenter.segment_for_video(mp_image, timestamp_ms=int(frame_step * idx))
            cm = result.category_mask.numpy_view() < 1
            body_masks.append(np.squeeze(cm))
            conf = result.confidence_masks[0].numpy_view()
            confidence_masks.append(conf.copy())

    # for some reason rgb video is not always same length as depth video??
    while deidentify and (len(blurred_video) < target_frames):
        blurred_video.append(blurred_video[-1].copy())
    while segment and (len(body_masks) < target_frames):
        body_masks.append(body_masks[-1].copy())
        confidence_masks.append(confidence_masks[-1].copy())

    if smooth:
        landmarks_2d = smooth_keypoints(landmarks_2d)
        landmarks_3d = smooth_keypoints(landmarks_3d)

    ## clean up after loop
    if warn_no_landmarks > 0:
        log.warning(f'Failed to find landmarks in {warn_no_landmarks} frames')
    if warn_no_face > 0:
        log.warning(f"Failed to find face in {warn_no_face} frames")

    detector.close()
    if mp_face_detector is not None:
        mp_face_detector.close()
    if mp_segmenter is not None:
        mp_segmenter.close()

    return LandmarkVideoResult(
        blaze_2d=landmarks_2d,
        blaze_3d=landmarks_3d,
        deidentified=np.asarray(blurred_video) if deidentify else None,
        body_masks=np.asarray(body_masks) if segment else None,
        confidence_masks=np.asarray(confidence_masks, dtype=np.float32) if segment else None,
    )

