"""
Color utilities for rendering.

This module provides colormap functions and default color palettes for
mesh and point cloud visualization.

Functions
---------
get_heatmap_color
    Map scalar values to RGB using MATLAB-style colormaps.

Attributes
----------
default_colors : list
    Default color palette (MATLAB-style) for categorical coloring.
"""

import numpy as np
from typing import Tuple, Literal, Optional

# Default color palette (MATLAB-style)
default_colors = [
    [0, 0.45, 0.74],    # blue
    [0.85, 0.33, 0.1],  # red
    [0.93, 0.69, 0.13], # yellow
    [0.49, 0.18, 0.56], # purple
    [0.47, 0.67, 0.19], # green
    [0.3, 0.75, 0.93],  # cyan
    [0.64, 0.08, 0.18], # maroon
]


def get_heatmap_color(
    values,
    caxis: Tuple[Optional[float], Optional[float]] = (None, None),
    mapname: str = "parula",
    n: int = 256,
    *,
    nan_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    out: Literal["float", "uint8"] = "float"
) -> np.ndarray:
    """
    Map scalar values to RGB using MATLAB-style colormaps.

    Parameters
    ----------
    values : array_like
        Scalar values of any shape.
    caxis : tuple of (float or None, float or None), optional
        Color axis limits (vmin, vmax). Use None to auto-compute from
        finite data. Default is (None, None).
    mapname : str, optional
        Colormap name. Options: 'parula', 'jet', 'hot', 'cool', 'spring'.
        Case-insensitive. Default is 'parula'.
    n : int, optional
        Number of colormap entries. Default is 256.
    nan_color : tuple of float, optional
        RGB color for NaN values (0-1 range). Default is (0.5, 0.5, 0.5).
    out : {'float', 'uint8'}, optional
        Output format: 'float' (0-1) or 'uint8' (0-255). Default is 'float'.

    Returns
    -------
    rgb : ndarray
        Array of shape ``values.shape + (3,)`` with RGB colors.

    Examples
    --------
    >>> colors = get_heatmap_color([0, 0.5, 1.0], mapname='jet')
    >>> colors.shape
    (3, 3)
    """
    v = np.asarray(values, dtype=float)
    flat = v.reshape(-1)
    finite = np.isfinite(flat)

    vmin, vmax = caxis
    if vmin is None:
        vmin = np.nanmin(flat) if np.any(finite) else 0.0
    if vmax is None:
        vmax = np.nanmax(flat) if np.any(finite) else 1.0
    vmin, vmax = float(vmin), float(vmax)

    # Avoid divide-by-zero; MATLAB effectively collapses to a single color
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        t = np.zeros_like(flat)
    else:
        t = (flat - vmin) / (vmax - vmin)

    t = np.clip(t, 0.0, 1.0)

    lut = _matlab_like_colormap(mapname, n)  # (n,3) float in [0,1]

    # Linear interpolation in LUT (smooth like MATLAB)
    idx = t * (n - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0)[:, None]

    rgb = (1.0 - w) * lut[i0] + w * lut[i1]

    # NaNs -> nan_color
    if not np.all(finite):
        rgb[~finite] = np.asarray(nan_color, dtype=float)

    rgb = rgb.reshape(v.shape + (3,))

    if out == "uint8":
        return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    if out == "float":
        return rgb.astype(np.float64)
    raise ValueError("out must be 'float' or 'uint8'")


def _matlab_like_colormap(name: str, n: int) -> np.ndarray:
    """
    Generate MATLAB-style colormap lookup table.

    Parameters
    ----------
    name : str
        Colormap name: 'parula', 'jet', 'hot', 'cool', 'spring'.
    n : int
        Number of colormap entries.

    Returns
    -------
    lut : ndarray
        Shape (n, 3) array of RGB values in [0, 1].
    """
    name = name.lower()

    x = np.linspace(0.0, 1.0, int(n))

    if name == "parula":
        r = np.minimum(1, np.abs(0.8 - (2 * x) ** 2.2) / 2.5)
        g = 1.41 * x ** 3 - 3.03 * x ** 2 + 2.46 * x + 0.06
        b = 5.45 * x ** 3 - 9.29 * x ** 2 + 3.36 * x + 0.65

    elif name == "jet":
        # Classic MATLAB jet (piecewise-linear ramps)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    elif name == "hot":
        # MATLAB hot: black -> red -> yellow -> white
        r = np.clip(3.0 * x, 0.0, 1.0)
        g = np.clip(3.0 * x - 1.0, 0.0, 1.0)
        b = np.clip(3.0 * x - 2.0, 0.0, 1.0)

    elif name == "cool":
        # MATLAB cool: cyan -> magenta
        r = x
        g = 1.0 - x
        b = np.ones_like(x)

    elif name == "spring":
        # MATLAB spring: magenta -> yellow
        r = np.ones_like(x)
        g = x
        b = 1.0 - x
    else:
        raise ValueError(f"Unknown colormap '{name}'. Use: parula, jet, hot, cool, spring.")

    return np.stack([r, g, b], axis=1)
