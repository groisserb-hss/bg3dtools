"""Image resizing with fallback across multiple libraries."""

import numpy as np
from typing import Tuple, Union, Literal

InterpolationType = Literal["nearest", "bilinear", "bicubic", "lanczos"]


def resize_img(
    img: np.ndarray,
    size: Tuple[int, int],
    interpolation: InterpolationType = "bilinear",
) -> np.ndarray:
    """
    Resize an image using the best available library.

    Tries libraries in order: Pillow, OpenCV, scipy, numpy (nearest only).

    Args:
        img: Input image as numpy array (H, W) or (H, W, C)
        size: Target size as (height, width)
        interpolation: Interpolation method - "nearest", "bilinear", "bicubic", "lanczos"

    Returns:
        Resized image as numpy array with same dtype as input
    """
    target_h, target_w = size
    original_dtype = img.dtype
    is_float = np.issubdtype(original_dtype, np.floating)

    # For float images, prefer scipy (handles float natively, no uint8 quantization)
    if is_float:
        try:
            from scipy import ndimage

            h, w = img.shape[:2]
            zoom_factors = (target_h / h, target_w / w)
            if img.ndim == 3:
                zoom_factors = zoom_factors + (1,)

            scipy_order = {
                "nearest": 0,
                "bilinear": 1,
                "bicubic": 3,
                "lanczos": 3,
            }
            order = scipy_order.get(interpolation, 1)

            resized = ndimage.zoom(img, zoom_factors, order=order)
            return resized.astype(original_dtype)
        except ImportError:
            pass

    # Try Pillow
    try:
        from PIL import Image

        pil_interp = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        mode = pil_interp.get(interpolation, Image.BILINEAR)

        # PIL needs uint8, so convert float images
        if is_float:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_uint8 = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_uint8 = np.zeros_like(img, dtype=np.uint8)
            pil_img = Image.fromarray(img_uint8)
            resized = pil_img.resize((target_w, target_h), mode)
            result = np.array(resized).astype(np.float64) / 255 * (img_max - img_min) + img_min
            return result.astype(original_dtype)
        else:
            pil_img = Image.fromarray(img)
            resized = pil_img.resize((target_w, target_h), mode)
            return np.array(resized).astype(original_dtype)
    except ImportError:
        pass

    # Try OpenCV
    try:
        import cv2

        cv2_interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        mode = cv2_interp.get(interpolation, cv2.INTER_LINEAR)

        resized = cv2.resize(img, (target_w, target_h), interpolation=mode)
        return resized.astype(original_dtype)
    except ImportError:
        pass

    # Try scipy (non-float path)
    try:
        from scipy import ndimage

        h, w = img.shape[:2]
        zoom_factors = (target_h / h, target_w / w)
        if img.ndim == 3:
            zoom_factors = zoom_factors + (1,)

        scipy_order = {
            "nearest": 0,
            "bilinear": 1,
            "bicubic": 3,
            "lanczos": 3,  # scipy doesn't have lanczos, use cubic
        }
        order = scipy_order.get(interpolation, 1)

        resized = ndimage.zoom(img, zoom_factors, order=order)
        return resized.astype(original_dtype)
    except ImportError:
        pass

    # Numpy fallback (nearest neighbor only)
    h, w = img.shape[:2]
    row_indices = (np.arange(target_h) * h / target_h).astype(int)
    col_indices = (np.arange(target_w) * w / target_w).astype(int)
    row_indices = np.clip(row_indices, 0, h - 1)
    col_indices = np.clip(col_indices, 0, w - 1)

    if interpolation != "nearest":
        import warnings
        warnings.warn(
            f"Only 'nearest' interpolation available with numpy fallback, "
            f"ignoring '{interpolation}'"
        )

    if img.ndim == 2:
        return img[row_indices][:, col_indices].astype(original_dtype)
    else:
        return img[row_indices][:, col_indices, :].astype(original_dtype)
