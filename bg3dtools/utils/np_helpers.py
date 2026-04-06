"""
NumPy helper utilities.

This module provides common array manipulation functions used throughout
the codebase.
"""

import numpy as np
from typing import Union


def row_normalize(
    m: np.ndarray,
    AXIS: int = -1,
    EPS: float = 1e-07
) -> np.ndarray:
    """
    Normalize rows (or along specified axis) to unit length.

    Parameters
    ----------
    m : ndarray
        Input array to normalize.
    AXIS : int, optional
        Axis along which to compute norms. Default is -1 (last axis).
    EPS : float, optional
        Small value to avoid division by zero. Default is 1e-07.

    Returns
    -------
    normalized : ndarray
        Array with rows normalized to unit length. Same dtype as input.

    Examples
    --------
    >>> vecs = np.array([[3, 4], [0, 0]])
    >>> row_normalize(vecs)
    array([[0.6, 0.8],
           [0. , 0. ]])
    """
    t = m.dtype
    m = m.astype(np.float64, copy=False)
    n = np.sqrt(np.sum(m**2, axis=AXIS, keepdims=True))
    n[n < EPS] = EPS
    n[np.isnan(n)] = EPS
    return (m / n).astype(t)


def pad_to_size(
    input_arr: np.ndarray,
    target_length: int,
    val: Union[int, float] = 0,
    dim: int = 1
) -> np.ndarray:
    """
    Pad an array to a target length along a specified dimension.

    Parameters
    ----------
    input_arr : ndarray
        Input array to pad.
    target_length : int
        Desired length along the padding dimension.
    val : int or float, optional
        Value to use for padding. Default is 0.
    dim : int, optional
        Dimension along which to pad. Default is 1.

    Returns
    -------
    padded : ndarray
        Padded array with shape[dim] == target_length.

    Examples
    --------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> pad_to_size(arr, 4, val=-1, dim=1)
    array([[ 1,  2, -1, -1],
           [ 3,  4, -1, -1]])
    """
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)


def truncated_normal(
    avg: np.ndarray = 0,
    var: float = 0.05,
    spread: float = 2
) -> np.ndarray:
    """
    Generate truncated normal random values.

    Samples from a normal distribution but rejects values beyond
    a specified number of standard deviations.

    Parameters
    ----------
    avg : ndarray
        Mean values (determines output shape).
    var : float, optional
        Standard deviation scale factor. Default is 0.05.
    spread : float, optional
        Maximum allowed deviation in standard deviations. Default is 2.

    Returns
    -------
    samples : ndarray
        Random samples with same shape as avg, values in [avg - spread*var, avg + spread*var].
    """
    from scipy.stats import truncnorm
    return truncnorm.rvs(-spread, spread, loc=avg, scale=var, size=avg.shape)


def is_valid_array(
    x,
    *,
    allow_empty: bool = False,
    numeric_only: bool = True
) -> bool:
    """
    Check if x is a valid NumPy array for numeric work.

    Parameters
    ----------
    x : any
        Value to check.
    allow_empty : bool, optional
        If True, accept arrays with size == 0. Default is False.
    numeric_only : bool, optional
        If True, only accept numeric dtypes (float/int/complex).
        Default is True.

    Returns
    -------
    valid : bool
        True if x is a valid numeric array.

    Notes
    -----
    Rejects None and the 'saved None' case (0-D object array holding None).
    Boolean arrays are rejected when numeric_only=True.

    Examples
    --------
    >>> is_valid_array(np.array([1, 2, 3]))
    True
    >>> is_valid_array(None)
    False
    >>> is_valid_array(np.array([]))
    False
    >>> is_valid_array(np.array([]), allow_empty=True)
    True
    """
    if x is None:
        return False
    if not isinstance(x, np.ndarray):
        return False

    if not allow_empty and x.size == 0:
        return False

    if numeric_only:
        # Reject any object dtype outright (covers the saved-None case)
        if x.dtype == object:
            return False
        return np.issubdtype(x.dtype, np.number)  # float/int/complex (bool will be False)
    else:
        # Accept any ndarray dtype, but still reject the 'saved None' shapes
        if x.dtype == object:
            if x.ndim == 0 and x[()] is None:
                return False
            if x.size == 1 and x.ravel()[0] is None:
                return False
        return True
