"""
Mask cleanup utilities for post-processing segmentation masks.

Provides morphological operations and smoothing filters to clean up
noisy segmentation masks from MediaPipe or other sources.
"""

from typing import Tuple, Union
import numpy as np
from scipy import ndimage
import logging

log = logging.getLogger(__name__)


def morphological_closing_3d(
    masks: np.ndarray,
    structure: np.ndarray = None,
) -> np.ndarray:
    """
    Apply 3D morphological closing to fill small holes in masks.

    Closing is dilation followed by erosion, which fills small gaps
    while preserving the overall shape.

    Args:
        masks: (T, H, W) boolean mask array across frames
        structure: Structuring element for closing. Defaults to 3D cross.

    Returns:
        (T, H, W) closed mask array
    """
    if structure is None:
        structure = ndimage.generate_binary_structure(3, 1)  # 3D plus/cross

    result = ndimage.binary_closing(masks, structure=structure)
    log.debug("Applied 3D morphological closing; %d -> %d points",
              masks.sum(), result.sum())
    return result


def gaussian_smooth_3d(
    masks: np.ndarray,
    sigma: Union[float, Tuple[float, float, float]] = (1.0, 2.0, 2.0),
    threshold: float = 0.4,
) -> np.ndarray:
    """
    Apply 3D Gaussian blur to smooth masks across space and time.

    Converts masks to float, applies Gaussian filter, then thresholds
    back to binary. This fills small holes and smooths jagged edges.

    Args:
        masks: (T, H, W) boolean mask array
        sigma: Sigma for Gaussian blur. Can be single value or
            tuple of (temporal, height, width) sigmas.
        threshold: Threshold for converting blurred mask back to
            binary (0-1). Higher values = more conservative (smaller masks).

    Returns:
        (T, H, W) smoothed binary mask array
    """
    masks_float = masks.astype(np.float32)
    masks_blurred = ndimage.gaussian_filter(masks_float, sigma=sigma)
    result = masks_blurred > threshold
    log.debug("Applied 3D Gaussian blur (sigma=%s, thresh=%.2f); %d -> %d points",
              sigma, threshold, masks.sum(), result.sum())
    return result


def cleanup_masks(
    masks: np.ndarray,
    apply_closing: bool = True,
    closing_structure: np.ndarray = None,
    gaussian_sigma: Union[float, Tuple[float, float, float]] = (1.0, 2.0, 2.0),
    gaussian_threshold: float = 0.4
) -> np.ndarray:
    """
    Clean up segmentation masks with morphological closing and Gaussian smoothing.

    This is the standard cleanup pipeline for MediaPipe body masks:
    1. Pad borders to avoid edge erosion
    2. Morphological closing to fill small holes
    3. Gaussian smoothing across space and time
    4. Remove padding

    Args:
        masks: (T, H, W) boolean mask array
        apply_closing: Whether to apply morphological closing
        closing_structure: Structuring element for closing. Defaults to 3D cross.
        gaussian_sigma: Sigma for Gaussian blur (temporal, height, width)
        gaussian_threshold: Threshold for binarizing blurred masks

    Returns:
        (T, H, W) cleaned boolean mask array
    """
    assert masks.ndim == 3, f"Expected (T, H, W) masks, got shape {masks.shape}"

    result = masks

    # Pad to avoid border erosion from 3D operations
    if apply_closing:
        # Morphological closing to fill small holes
        result = np.pad(result, ((1, 1), (1, 1), (1, 1)), mode='edge')
        result = morphological_closing_3d(result, closing_structure)

    # Gaussian blur to smooth masks across space and time
    if gaussian_sigma is not None:
        result = gaussian_smooth_3d(result, gaussian_sigma, gaussian_threshold)

    # Remove padding
    if apply_closing:
        result = result[1:-1, 1:-1, 1:-1]

    return result
