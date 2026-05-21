"""
Image processing utilities.

Provides filtering, coordinate conversion, video I/O, and image packing.
"""

from .filters import normal_edges
from .utils import normalized_to_pixel_coordinates
from .video import vreader
from .packing import (
    PackingResult,
    pack_images,
    create_packed_image,
    create_packed_video,
)
from .interpolation import laplace_interpolation
from .metrics import dice_score, normalized_cross_correlation, gradient_correlation

__all__ = [
    "normal_edges",
    "normalized_to_pixel_coordinates",
    "vreader",
    "PackingResult",
    "pack_images",
    "create_packed_image",
    "create_packed_video",
    "laplace_interpolation",
    "dice_score",
    "normalized_cross_correlation",
    "gradient_correlation",
]
