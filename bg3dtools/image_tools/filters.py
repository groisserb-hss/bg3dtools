"""
Image filtering utilities.

This module provides functions for edge detection and gradient computation
on images and depth maps.
"""

from typing import Optional
from scipy.ndimage import sobel
import numpy as np


def normal_edges(
    Z: np.ndarray,
    z_thresh: Optional[float] = None
) -> np.ndarray:
    """
    Detect edges in a grayscale/depth map using Sobel gradients.

    Parameters
    ----------
    Z : (H, W) ndarray
        Grayscale image or depth map.
    z_thresh : float, optional
        Gradient magnitude threshold. If None, uses 95th percentile.

    Returns
    -------
    edges : (H, W) ndarray of bool
        Binary edge mask.
    """

    dzx = sobel(Z, axis=1, mode='nearest') / 8.0
    dzy = sobel(Z, axis=0, mode='nearest') / 8.0
    gradZ = np.hypot(dzx, dzy)

    if z_thresh is None:
        z_thresh = np.percentile(gradZ, 95)
    edges = gradZ > z_thresh

    return edges