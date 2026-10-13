"""Region anonymisation: strong Gaussian blur, then greyscale."""

from __future__ import annotations

import cv2
import numpy as np

from safeframe.pipeline.types import Box

# The blurred region is reduced to roughly this many pixels along its longest
# side before blurring. Detail finer than ~1/BLUR_RESOLUTION of the region is
# destroyed regardless of resolution, so a face 900 px tall is just as
# unrecognisable as one 90 px tall.
BLUR_RESOLUTION = 24
# Fraction of the box width/height added on every side, so hair, hands and
# small tracker misalignments are still covered.
DEFAULT_PADDING = 0.08


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def gaussian_blur_strong(roi: np.ndarray) -> np.ndarray:
    """Gaussian-blur ``roi`` with a sigma proportional to its size.

    A direct Gaussian with sigma ~ size/12 needs kernels hundreds of pixels
    wide on HD video. Blurring a downscaled copy and upscaling is equivalent
    for this purpose (the result is a smooth, low-frequency field) and is
    orders of magnitude faster.
    """
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return roi.copy()
    scale = min(1.0, BLUR_RESOLUTION / max(h, w))
    small_w, small_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    k = _odd(max(3, round(max(small_w, small_h) / 4)))
    small = cv2.GaussianBlur(small, (k, k), 0)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # Second pass at full resolution removes any interpolation blockiness.
    k_full = _odd(max(3, round(max(w, h) / BLUR_RESOLUTION)))
    return cv2.GaussianBlur(up, (k_full, k_full), 0)


def to_greyscale(roi: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def pixel_rect(
    box: Box, width: int, height: int, padding: float = DEFAULT_PADDING
) -> tuple[int, int, int, int] | None:
    """Padded, clipped integer rectangle (x1, y1, x2, y2) or None if empty."""
    b = box.expand(padding).clip(width, height)
    x1, y1 = int(np.floor(b.x1)), int(np.floor(b.y1))
    x2, y2 = int(np.ceil(b.x2)), int(np.ceil(b.y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def anonymize_regions(
    frame: np.ndarray, boxes: list[Box], padding: float = DEFAULT_PADDING
) -> np.ndarray:
    """Blur then greyscale every box in ``frame`` (in place). Returns ``frame``."""
    height, width = frame.shape[:2]
    for box in boxes:
        rect = pixel_rect(box, width, height, padding)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = to_greyscale(gaussian_blur_strong(roi))
    return frame
