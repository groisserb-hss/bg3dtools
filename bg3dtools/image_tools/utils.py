"""
Image coordinate utilities.

This module provides functions for converting between different
coordinate systems used in image processing.
"""

from typing import Optional, Tuple
import math


def normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int
) -> Optional[Tuple[int, int]]:
    """
    Convert normalized [0, 1] coordinates to pixel coordinates.

    Parameters
    ----------
    normalized_x : float
        X coordinate in [0, 1] range.
    normalized_y : float
        Y coordinate in [0, 1] range.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    coords : tuple of int or None
        (x_pixel, y_pixel) coordinates, or None if input is out of range.
    """

    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px