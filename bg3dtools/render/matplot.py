"""
Matplotlib-based visualization utilities.

This module provides simple matplotlib functions for 3D point cloud visualization.
"""

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt


def scatt(
    pts: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    s: float = 0.01
) -> None:
    """
    Display a 3D scatter plot of points.

    Parameters
    ----------
    pts : (N, 3) ndarray
        Point coordinates.
    rgb : (N, 3) or (N,) ndarray, optional
        Point colors. If (N,), creates a red-blue colormap.
        If None, colors by position.
    s : float, optional
        Point size. Default is 0.01.
    """
    if rgb is None:
        rgb = pts - np.min(pts, axis=0, keepdims=True)
        rgb = rgb / np.max(rgb, axis=0, keepdims=True)
    if rgb.ndim == 1 and len(rgb) == len(pts):
        rgb = rgb - np.min(rgb)
        rgb = rgb / np.max(rgb)
        rgb = np.column_stack((rgb, np.zeros(len(rgb)), 1 - rgb))

    ax = plt.axes(projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb, s=s)
    plt.show()
