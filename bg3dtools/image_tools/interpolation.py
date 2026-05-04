"""Volumetric interpolation via Laplace's equation."""

import logging

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)


def laplace_interpolation(data, mask, voxel_size=None, ftol=1., maxiter=1000):
    """Interpolate masked voxels using iterative Gaussian smoothing (Laplace fill).

    Replaces voxels where ``mask`` is True with values obtained by iteratively
    diffusing from the surrounding known voxels until convergence.

    Parameters
    ----------
    data : ndarray (I, J, K)
        Input volume (float).
    mask : ndarray (I, J, K), bool
        True for voxels to interpolate.
    voxel_size : array-like (3,), optional
        Voxel dimensions in mm.  Used to set the Gaussian sigma so that the
        smoothing kernel is ~1 mm in each axis.  If *None*, isotropic 1-voxel
        sigma is used.
    ftol : float
        Mean absolute change threshold for convergence.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    ndarray (I, J, K)
        Volume with masked voxels filled by Laplace interpolation.
    """
    mask = np.asarray(mask, dtype=bool)
    fill_idx = mask.nonzero()

    if voxel_size is not None:
        sigma = 1.0 / np.abs(np.asarray(voxel_size, dtype=np.float64))
    else:
        sigma = 1.0

    result = np.array(data, dtype=np.float32)

    # Initial guess: mean of the immediate neighbourhood ring
    ring = ndimage.binary_dilation(mask) | mask
    result[fill_idx] = np.nanmean(result[ring])

    for step in range(maxiter):
        prev = result[fill_idx].copy()
        smoothed = ndimage.gaussian_filter(result, sigma)
        result[fill_idx] = smoothed[fill_idx]

        if np.mean(np.abs(prev - result[fill_idx])) < ftol:
            log.info('Laplace interpolation converged after %d iterations', step)
            break
    else:
        log.warning('Laplace interpolation did not converge after %d iterations', maxiter)

    return result
