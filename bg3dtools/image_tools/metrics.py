"""Volumetric similarity metrics for registration quality assessment.

All functions accept two numpy arrays of the same shape (2-D, 3-D, or
higher) and an optional boolean *mask* of the same shape.  When *mask*
is provided only elements where ``mask == True`` contribute.

Pure numpy — no nibabel or other imaging library dependency.
"""

import numpy as np


def dice_score(vol_a, vol_b, mask=None, threshold_a=300, threshold_b=300):
    """Dice coefficient between binarised volumes.

    Parameters
    ----------
    vol_a, vol_b : ndarray
        Intensity arrays of the same shape.
    mask : ndarray[bool], optional
        Restrict computation to these voxels.
    threshold_a, threshold_b : float
        HU thresholds applied to *vol_a* and *vol_b* respectively.

    Returns
    -------
    float
        Dice coefficient in [0, 1].
    """
    bin_a = vol_a > threshold_a
    bin_b = vol_b > threshold_b
    if mask is not None:
        bin_a = bin_a & mask
        bin_b = bin_b & mask
    card_a = bin_a.sum()
    card_b = bin_b.sum()
    if card_a + card_b == 0:
        return 0.0
    intersection = (bin_a & bin_b).sum()
    return float(2 * intersection / (card_a + card_b))


def normalized_cross_correlation(vol_a, vol_b, mask=None):
    """Normalized cross-correlation (NCC) between two volumes.

    Parameters
    ----------
    vol_a, vol_b : ndarray
        Intensity arrays of the same shape.
    mask : ndarray[bool], optional
        Restrict computation to these voxels.

    Returns
    -------
    float
        NCC in [-1, 1].
    """
    if mask is not None:
        a = vol_a[mask].astype(np.float64)
        b = vol_b[mask].astype(np.float64)
    else:
        a = vol_a.ravel().astype(np.float64)
        b = vol_b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def gradient_correlation(vol_a, vol_b, mask=None):
    """NCC of gradient magnitudes.

    Computes the spatial gradient magnitude of each input via
    ``np.gradient`` (works for any dimensionality) then returns
    :func:`normalized_cross_correlation` of the two magnitude fields.

    Parameters
    ----------
    vol_a, vol_b : ndarray
        Intensity arrays of the same shape.
    mask : ndarray[bool], optional
        Restrict computation to these voxels.

    Returns
    -------
    float
        Gradient NCC in [-1, 1].
    """
    grad_a = np.sqrt(sum(g ** 2 for g in np.gradient(vol_a.astype(np.float64))))
    grad_b = np.sqrt(sum(g ** 2 for g in np.gradient(vol_b.astype(np.float64))))
    return normalized_cross_correlation(grad_a, grad_b, mask=mask)
