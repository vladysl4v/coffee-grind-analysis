import numpy as np
from scipy.ndimage import uniform_filter

MAX_CROPPED_SIZE = [528, 892]

# Pixels darker than this (L2 norm) are treated as background / fill, not coffee.
_MIN_PIXEL_NORM = 1.0


def _find_first_nonzero(img, axis):
    first_pixel_columns = (img > 0).all(axis=-1).argmax(axis=axis)
    return first_pixel_columns[first_pixel_columns > 0].min()


def get_mask(img):
    """Colour-direction mask for brown coffee grains; works for any H×W."""
    h, w = img.shape[:2]
    reshaped = img.reshape(-1, 3).astype(np.float64)
    norms = np.linalg.norm(reshaped, axis=1, keepdims=True)

    unit = np.zeros_like(reshaped)
    np.divide(reshaped, norms, out=unit, where=norms > _MIN_PIXEL_NORM)

    color = np.array([160.0, 82.0, 45.0])
    color_norm = color / np.linalg.norm(color)
    closeness = (unit @ color_norm).reshape(-1)
    closeness[norms.reshape(-1) <= _MIN_PIXEL_NORM] = 0.0

    plane = closeness.reshape(h, w)
    active = plane > 0.0
    if not np.any(active):
        return np.zeros((h, w, 1), dtype=bool)

    thresh = float(plane[active].mean())
    closeness_blurred = (plane > thresh).astype(np.float64)
    k = max(3, min(62, h // 18, w // 18))
    closeness_blurred = uniform_filter(closeness_blurred, size=k)
    return closeness_blurred[..., None] > 0.95


def extract_region(img, cropping=False):
    mask = get_mask(img)
    masked = img * mask
    if not cropping:
        return masked
    first_pixel_columns = _find_first_nonzero(masked, 0)
    first_pixel_rows = _find_first_nonzero(masked, 1)
    last_pixel_columns = masked.shape[0] - _find_first_nonzero(masked[::-1, ...], 0)
    last_pixel_rows = masked.shape[1] - _find_first_nonzero(masked[::-1, ...], 1)
    return masked[first_pixel_columns:last_pixel_columns, first_pixel_rows:last_pixel_rows]
