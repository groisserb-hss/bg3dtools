"""
Image packing utilities for arranging multiple images into a single canvas.

Provides functions for computing optimal layouts when combining multiple
images of potentially different sizes into a single output image.
"""

from dataclasses import dataclass
from itertools import permutations
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class PackingResult:
    """
    Result of packing multiple images into a target resolution.

    Attributes:
        scale: Uniform scale factor applied to all images.
        positions: List of (x, y) top-left positions for each image in output.
        scaled_sizes: List of (width, height) for each scaled image.
        canvas_size: (width, height) of the output canvas.
    """
    scale: float
    positions: List[Tuple[int, int]]
    scaled_sizes: List[Tuple[int, int]]
    canvas_size: Tuple[int, int]


def pack_images(
    sizes: List[Tuple[int, int]],
    target_width: int = 1470,
    target_height: int = 956,
    padding: int = 4,
) -> PackingResult:
    """
    Compute optimal 2D packing of multiple images into a target resolution.

    Uses brute-force search over all placement permutations with maximal
    rectangles algorithm. Finds the largest uniform scale factor that allows
    all images to fit.

    Args:
        sizes: List of (height, width) tuples for each image.
        target_width: Target canvas width in pixels.
        target_height: Target canvas height in pixels.
        padding: Padding between images in pixels.

    Returns:
        PackingResult with scale, positions, scaled sizes, and canvas size.
        Positions are in the original input order (not permuted).

    Example:
        >>> sizes = [(480, 640), (480, 640), (720, 1280)]
        >>> result = pack_images(sizes, 1920, 1080)
        >>> result.scale
        0.42
        >>> len(result.positions)
        3
    """
    if not sizes:
        return PackingResult(
            scale=1.0,
            positions=[],
            scaled_sizes=[],
            canvas_size=(target_width, target_height),
        )

    n = len(sizes)
    # Convert (H, W) to (W, H) for internal use, keep original indices
    sizes_wh = [(w, h, i) for i, (h, w) in enumerate(sizes)]

    # Binary search for optimal scale
    lo, hi = 0.001, 10.0
    best_scale = lo
    best_positions_by_idx = None

    for _ in range(40):  # Binary search iterations
        mid = (lo + hi) / 2
        positions_by_idx = _try_pack_bruteforce(
            sizes_wh, mid, target_width, target_height, padding
        )
        if positions_by_idx is not None:
            best_scale = mid
            best_positions_by_idx = positions_by_idx
            lo = mid
        else:
            hi = mid

    # Build result in original order
    if best_positions_by_idx is None:
        # Fallback: couldn't fit, use tiny scale
        best_scale = 0.01
        best_positions_by_idx = {i: (0, 0) for i in range(n)}

    positions = [best_positions_by_idx[i] for i in range(n)]
    scaled_sizes = [(int(w * best_scale), int(h * best_scale)) for h, w in sizes]

    return PackingResult(
        scale=best_scale,
        positions=positions,
        scaled_sizes=scaled_sizes,
        canvas_size=(target_width, target_height),
    )


def _try_pack_bruteforce(
    sizes_wh: List[Tuple[int, int, int]],  # (w, h, original_idx)
    scale: float,
    target_width: int,
    target_height: int,
    padding: int,
) -> Optional[dict]:
    """
    Try all permutations to pack images at given scale.

    Returns dict mapping original_idx -> (x, y) position, or None if no fit.
    """
    n = len(sizes_wh)

    # Scale all sizes
    scaled = [(int(w * scale), int(h * scale), idx) for w, h, idx in sizes_wh]

    # For small n, try all permutations
    if n <= 8:
        perms = list(permutations(range(n)))
    else:
        # For larger n, try some heuristic orderings
        perms = _get_heuristic_orderings(scaled)

    for perm in perms:
        ordered = [scaled[i] for i in perm]
        result = _try_pack_maxrects(ordered, target_width, target_height, padding)
        if result is not None:
            return result

    return None


def _get_heuristic_orderings(scaled: List[Tuple[int, int, int]]) -> List[Tuple[int, ...]]:
    """Generate heuristic orderings for larger n."""
    n = len(scaled)
    orderings = []

    # By area descending
    by_area = sorted(range(n), key=lambda i: scaled[i][0] * scaled[i][1], reverse=True)
    orderings.append(tuple(by_area))

    # By height descending
    by_height = sorted(range(n), key=lambda i: scaled[i][1], reverse=True)
    orderings.append(tuple(by_height))

    # By width descending
    by_width = sorted(range(n), key=lambda i: scaled[i][0], reverse=True)
    orderings.append(tuple(by_width))

    # By max dimension descending
    by_max = sorted(range(n), key=lambda i: max(scaled[i][0], scaled[i][1]), reverse=True)
    orderings.append(tuple(by_max))

    # Original order
    orderings.append(tuple(range(n)))

    return orderings


def _try_pack_maxrects(
    ordered: List[Tuple[int, int, int]],  # (w, h, original_idx) in placement order
    target_width: int,
    target_height: int,
    padding: int,
) -> Optional[dict]:
    """
    Pack images using maximal rectangles algorithm.

    Maintains a list of maximal free rectangles. For each image, finds the
    best-fit rectangle, places the image, then splits ALL overlapping free
    rectangles.

    Returns dict mapping original_idx -> (x, y), or None if doesn't fit.
    """
    # Free rectangles: list of (x, y, w, h)
    free_rects = [(0, 0, target_width, target_height)]
    positions = {}

    for img_w, img_h, orig_idx in ordered:
        # Add padding to required size
        req_w = img_w + padding
        req_h = img_h + padding

        # Find best rectangle (best short side fit)
        best_rect = None
        best_score = float('inf')
        best_pos = None

        for rect in free_rects:
            rx, ry, rw, rh = rect
            if rw >= req_w and rh >= req_h:
                # Score by leftover short side (best-short-side-fit)
                leftover_w = rw - req_w
                leftover_h = rh - req_h
                score = min(leftover_w, leftover_h)
                if score < best_score:
                    best_score = score
                    best_rect = rect
                    best_pos = (rx, ry)

        if best_rect is None:
            return None  # Doesn't fit

        # Place image at best_pos
        px, py = best_pos
        positions[orig_idx] = (px, py)

        # The placed rectangle (including padding)
        placed = (px, py, req_w, req_h)

        # Split ALL free rectangles that overlap with the placed rectangle
        new_free_rects = []
        for rect in free_rects:
            # Get splits of this rect around the placed rect
            splits = _split_rect_around(rect, placed)
            new_free_rects.extend(splits)

        # Remove rectangles fully contained in others
        free_rects = _remove_contained_rects(new_free_rects)

    return positions


def _rects_overlap(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> bool:
    """Check if two rectangles overlap."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def _split_rect_around(
    rect: Tuple[int, int, int, int],
    placed: Tuple[int, int, int, int]
) -> List[Tuple[int, int, int, int]]:
    """
    Split a free rectangle around a placed rectangle.

    Returns up to 4 new rectangles (left, right, top, bottom of placed rect),
    or the original rect if no overlap.
    """
    rx, ry, rw, rh = rect
    px, py, pw, ph = placed

    # Check if they overlap
    if not _rects_overlap(rect, placed):
        return [rect]

    result = []

    # Left portion: from rect left edge to placed left edge
    if px > rx:
        left_w = px - rx
        if left_w > 0:
            result.append((rx, ry, left_w, rh))

    # Right portion: from placed right edge to rect right edge
    if px + pw < rx + rw:
        right_x = px + pw
        right_w = (rx + rw) - right_x
        if right_w > 0:
            result.append((right_x, ry, right_w, rh))

    # Top portion: from rect top edge to placed top edge
    if py > ry:
        top_h = py - ry
        if top_h > 0:
            result.append((rx, ry, rw, top_h))

    # Bottom portion: from placed bottom edge to rect bottom edge
    if py + ph < ry + rh:
        bottom_y = py + ph
        bottom_h = (ry + rh) - bottom_y
        if bottom_h > 0:
            result.append((rx, bottom_y, rw, bottom_h))

    return result


def _remove_contained_rects(
    rects: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """Remove rectangles fully contained in others."""
    if not rects:
        return []

    result = []
    for i, r1 in enumerate(rects):
        x1, y1, w1, h1 = r1
        contained = False
        for j, r2 in enumerate(rects):
            if i == j:
                continue
            x2, y2, w2, h2 = r2
            # Check if r1 is fully contained in r2
            if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                contained = True
                break
        if not contained:
            result.append(r1)
    return result


def create_packed_image(
    images: List[np.ndarray],
    packing: PackingResult,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Create a packed image from individual images and packing result.

    Args:
        images: List of images as numpy arrays (H, W, 3).
        packing: PackingResult from pack_images().
        background: Background color as (R, G, B) tuple.

    Returns:
        Packed image as numpy array (target_H, target_W, 3).
    """
    import cv2

    canvas_w, canvas_h = packing.canvas_size
    canvas = np.full((canvas_h, canvas_w, 3), background, dtype=np.uint8)

    for img, (x, y), (scaled_w, scaled_h) in zip(
        images, packing.positions, packing.scaled_sizes
    ):
        if img is None or scaled_w <= 0 or scaled_h <= 0:
            continue

        # Resize image
        resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # Handle boundary clipping
        x_end = min(x + scaled_w, canvas_w)
        y_end = min(y + scaled_h, canvas_h)
        img_w = x_end - x
        img_h = y_end - y

        if img_w > 0 and img_h > 0:
            canvas[y:y_end, x:x_end] = resized[:img_h, :img_w]

    return canvas


def create_packed_video(
    videos: List[np.ndarray],
    packing: PackingResult,
    labels: List[str] = None,
    background: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.5,
    font_thickness: int = 1,
    label_color: Tuple[int, int, int] = (255, 255, 255),
    label_bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Create a packed video from multiple videos and packing result.

    Args:
        videos: List of videos as numpy arrays (T, H, W, 3).
        packing: PackingResult from pack_images().
        labels: Optional list of labels for each video.
        background: Background color as (R, G, B) tuple.
        font_scale: Font scale for labels.
        font_thickness: Font thickness for labels.
        label_color: Label text color as (R, G, B).
        label_bg_color: Label background color as (R, G, B).

    Returns:
        Packed video as numpy array (T, target_H, target_W, 3).
    """
    import cv2

    # Get number of frames (use max across all videos)
    T = max(len(v) for v in videos if len(v) > 0)
    canvas_w, canvas_h = packing.canvas_size

    output = np.full((T, canvas_h, canvas_w, 3), background, dtype=np.uint8)

    for t in range(T):
        # Collect frames for this timestep
        frames = []
        for video in videos:
            if t < len(video):
                frames.append(video[t])
            elif len(video) > 0:
                frames.append(video[-1])
            else:
                frames.append(None)

        # Create packed frame
        canvas = create_packed_image(frames, packing, background)

        # Add labels
        if labels:
            for label, (x, y) in zip(labels, packing.positions):
                if label:
                    _draw_label(canvas, label, x, y + 5,
                               font_scale, font_thickness, label_color, label_bg_color)

        output[t] = canvas

    return output


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    font_thickness: int,
    text_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int],
):
    """Draw a label with background on an image."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Draw background rectangle
    pad = 4
    cv2.rectangle(img,
                  (x, y),
                  (x + text_w + 2 * pad, y + text_h + 2 * pad + baseline),
                  bg_color, -1)

    # Draw text
    cv2.putText(img, text,
                (x + pad, y + text_h + pad),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
