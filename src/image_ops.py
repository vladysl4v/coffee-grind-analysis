"""Image cropping utilities for segmented coffee grain images."""

from __future__ import annotations

import numpy as np
from PIL import Image


def image_active_fraction(image: Image.Image, *, threshold: int = 8) -> float:
    """Share of pixels with max channel above ``threshold``."""

    arr = np.asarray(image.convert("RGB"))
    return float((arr.max(axis=2) > threshold).mean())


def is_degenerate_image(
    image: Image.Image,
    *,
    threshold: int = 8,
    min_active_fraction: float = 0.005,
) -> bool:
    """True if the image is empty or has almost no foreground."""

    arr = np.asarray(image.convert("RGB"))
    if arr.max() == 0:
        return True
    return image_active_fraction(image, threshold=threshold) < min_active_fraction


def nonzero_bounding_box(
    image: Image.Image,
    *,
    threshold: int = 8,
) -> tuple[int, int, int, int]:
    """Return ``(left, upper, right, lower)`` for pixels above ``threshold``."""

    arr = np.asarray(image.convert("RGB"))
    active = arr.max(axis=2) > threshold
    ys, xs = np.where(active)
    if ys.size == 0:
        w, h = image.size
        return 0, 0, w, h
    left = int(xs.min())
    upper = int(ys.min())
    right = int(xs.max()) + 1
    lower = int(ys.max()) + 1
    return left, upper, right, lower


class MaskBoundingBoxCrop:
    """Crop to the grain-region bounding box, then resize to a square.

    Intended for background-masked (segmented) images where inactive pixels are
    near-black. Falls back to a centered square crop when no foreground is found.
    """

    def __init__(
        self,
        size: int = 224,
        padding_ratio: float = 0.05,
        threshold: int = 8,
    ):
        self.size = int(size)
        self.padding_ratio = float(padding_ratio)
        self.threshold = int(threshold)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        left, upper, right, lower = nonzero_bounding_box(image, threshold=self.threshold)
        width, height = image.size

        box_w = right - left
        box_h = lower - upper
        if box_w < 2 or box_h < 2:
            side = min(width, height)
            left = (width - side) // 2
            upper = (height - side) // 2
            right, lower = left + side, upper + side
        else:
            pad_x = int(box_w * self.padding_ratio)
            pad_y = int(box_h * self.padding_ratio)
            left = max(0, left - pad_x)
            upper = max(0, upper - pad_y)
            right = min(width, right + pad_x)
            lower = min(height, lower + pad_y)

        cropped = image.crop((left, upper, right, lower))
        return cropped.resize((self.size, self.size), resample=Image.Resampling.BICUBIC)
