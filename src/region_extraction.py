import numpy as np
from scipy.ndimage import uniform_filter

MAX_CROPPED_SIZE = [528, 892]


def _find_first_nonzero(img, axis):
    first_pixel_columns = (img > 0).all(axis=-1).argmax(axis=axis)
    return first_pixel_columns[first_pixel_columns > 0].min()


def get_mask(img):
    reshaped = img.reshape(-1, 3)
    color = np.array([160, 82, 45]).reshape(3, 1)  # setup for a brown color
    color_norm = (color / np.linalg.norm(color, axis=0, keepdims=True))
    closeness = (reshaped / np.linalg.norm(reshaped, axis=1, keepdims=True)) @ color_norm
    closeness_blurred = (closeness.reshape(1080, 1080) > closeness.mean()).astype("float")
    closeness_blurred = uniform_filter(closeness_blurred, 62)
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
