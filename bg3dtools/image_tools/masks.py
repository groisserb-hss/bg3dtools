"""Cleaning a 2-D binary mask with ``scipy.ndimage``: closing, opening, small-component removal, capped hole filling.

Pixel distances are Chebyshev (a ``(2k+1)``-square structuring element iterated from the 3x3 square), so ``close_px=1``
joins pieces up to 2 px apart and ``open_px=1`` removes anything thinner than 3 px. The image is edge-padded for the
morphology, so a region touching the border is not eroded there.
"""

import numpy as np
from scipy import ndimage

__all__ = ["clean_binary_mask"]

_SQUARE = np.ones((3, 3), dtype=bool)


def clean_binary_mask(mask: np.ndarray, *, close_px: int = 0, open_px: int = 0,
                      min_area: int = 0, max_hole_area: int = 0) -> np.ndarray:
    """Clean a 2-D bool mask, in this order:

    1. closing by ``close_px`` iterations of the 3x3 square (joins pieces, swallows gaps narrower than ``2*close_px``);
    2. opening by ``open_px`` iterations (removes specks and protrusions thinner than ``2*open_px + 1``);
    3. drop every 8-connected component with fewer than ``min_area`` pixels;
    4. fill every enclosed hole (a background region not reaching the image border) with fewer than
       ``max_hole_area`` pixels — larger holes stay open.

    Every step at 0 is skipped. Returns a new array; the input is not modified.
    """
    m = np.array(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"clean_binary_mask expects a 2-D mask, got shape {m.shape}")
    close_px, open_px = int(close_px), int(open_px)
    pad = close_px + open_px
    if pad:
        m = np.pad(m, pad, mode="edge")
        if close_px:
            m = ndimage.binary_closing(m, structure=_SQUARE, iterations=close_px)
        if open_px:
            m = ndimage.binary_opening(m, structure=_SQUARE, iterations=open_px)
        m = np.ascontiguousarray(m[pad:-pad, pad:-pad])
    if min_area > 0:
        labels, n = ndimage.label(m, structure=_SQUARE)
        if n:
            small = np.bincount(labels.ravel()) < min_area
            small[0] = False
            m[small[labels]] = False
    if max_hole_area > 0:
        holes = ndimage.binary_fill_holes(m) & ~m
        labels, n = ndimage.label(holes)
        if n:
            fill = np.bincount(labels.ravel()) < max_hole_area
            fill[0] = False
            m[fill[labels]] = True
    return m
