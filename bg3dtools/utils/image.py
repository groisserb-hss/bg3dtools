"""
Image processing utilities.

This module provides helper functions for common image processing operations.
"""

import numpy as np


def auto_canny(
    image: np.ndarray,
    sigma: float = 0.33
) -> np.ndarray:
    """
    Apply Canny edge detection with automatic threshold selection.

    Thresholds are computed from the median pixel intensity.

    Parameters
    ----------
    image : (H, W) ndarray, uint8
        Grayscale input image.
    sigma : float, optional
        Threshold spread factor. Default is 0.33.

    Returns
    -------
    edges : (H, W) ndarray, uint8
        Binary edge map.
    """
    import cv2
    assert image.dtype == np.uint8

    v = np.nanmedian(image[image > 0])
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged