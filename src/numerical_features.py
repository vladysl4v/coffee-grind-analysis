from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
import pywt
from PIL import Image
from scipy.ndimage import laplace
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import closing, disk, opening


CROP_SIZE = 224

FEATURE_NAMES = (
    "morph_closing_added_r21",
    "morph_closing_added_r13",
    "morph_opening_removed_r8",
    "lacunarity_box10",
    "autocorr_radial_mean",
    "autocorr_drop_010",
    "autocorr_drop_020",
    "autocorr_drop_050",
    "glcm_correlation_mean",
    "fft_low_power",
    "fft_mid_low_ratio",
    "fft_spectral_entropy",
    "local_entropy_mean_r5",
    "local_entropy_p90_r5",
    "local_std_p90_w9",
    "grad_p99",
    "grad_std",
    "haar_detail_mean",
    "laplace_mean_abs",
)


def crop_center_square(image: Image.Image | np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Return a center-cropped grayscale uint8 image."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image)

    if rgb.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {rgb.shape!r}")

    height, width = rgb.shape[:2]
    if height < size or width < size:
        raise ValueError(f"Image too small for {size}x{size} crop: {width}x{height}")

    top = (height - size) // 2
    left = (width - size) // 2
    crop = rgb[top:top + size, left:left + size]

    if crop.ndim == 2:
        return crop.astype(np.uint8)

    return cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = gray_f.mean()
    std = gray_f.std() + 1e-6

    z = (gray_f - mean) / std
    z = np.clip(z, -3.0, 3.0)

    return ((z + 3.0) / 6.0 * 255.0).astype(np.uint8)


def get_normalized_gray_crop(image: Image.Image | np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    return normalize_gray(crop_center_square(image, size=size))


def _fft_features(gray: np.ndarray) -> dict[str, float]:
    img = gray.astype(np.float32)
    img -= img.mean()

    fft = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(fft) ** 2

    height, width = power.shape
    cy, cx = height // 2, width // 2

    yy, xx = np.indices((height, width))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / (r.max() + 1e-12)

    low = power[r_norm < 0.15].mean()
    mid = power[(r_norm >= 0.15) & (r_norm < 0.35)].mean()

    p = power.ravel()
    p = p / (p.sum() + 1e-12)
    spectral_entropy = -np.sum(p * np.log2(p + 1e-12))

    return {
        "fft_low_power": float(low),
        "fft_mid_low_ratio": float(mid / (low + 1e-12)),
        "fft_spectral_entropy": float(spectral_entropy),
    }


def _glcm_features(gray: np.ndarray) -> dict[str, float]:
    levels = 32
    q = np.floor(gray.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    q = np.clip(q, 0, levels - 1)

    glcm = graycomatrix(
        q,
        distances=[1, 2, 4, 8, 16],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    return {"glcm_correlation_mean": float(graycoprops(glcm, "correlation").mean())}


def _morphology_features(gray: np.ndarray) -> dict[str, float]:
    img = gray.astype(np.float32)
    opened_r8 = opening(gray, disk(8)).astype(np.float32)
    closed_r13 = closing(gray, disk(13)).astype(np.float32)
    closed_r21 = closing(gray, disk(21)).astype(np.float32)

    return {
        "morph_closing_added_r21": float(np.mean(np.abs(closed_r21 - img))),
        "morph_closing_added_r13": float(np.mean(np.abs(closed_r13 - img))),
        "morph_opening_removed_r8": float(np.mean(np.abs(img - opened_r8))),
    }


def _autocorrelation_features(gray: np.ndarray) -> dict[str, float]:
    img = gray.astype(np.float32)
    img -= img.mean()

    fft = np.fft.fft2(img)
    corr = np.fft.ifft2(np.abs(fft) ** 2).real
    corr = np.fft.fftshift(corr).astype(np.float32)

    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    center = corr[cy, cx]
    corr = corr / (center + 1e-12)

    height, width = corr.shape
    yy, xx = np.indices((height, width))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    max_r = min(height, width) // 2
    radial = []
    for radius in range(1, max_r):
        mask = (r >= radius - 0.5) & (r < radius + 0.5)
        radial.append(corr[mask].mean())

    radial = np.asarray(radial, dtype=np.float32)

    def first_below(threshold: float) -> int:
        idx = np.where(radial < threshold)[0]
        if len(idx) == 0:
            return max_r
        return int(idx[0] + 1)

    return {
        "autocorr_radial_mean": float(radial.mean()),
        "autocorr_drop_010": float(first_below(0.10)),
        "autocorr_drop_020": float(first_below(0.20)),
        "autocorr_drop_050": float(first_below(0.50)),
    }


def _lacunarity_feature(gray: np.ndarray) -> dict[str, float]:
    binary = gray > np.percentile(gray, 50)
    box = 10
    masses = []

    height, width = binary.shape
    for y in range(0, height - box + 1):
        for x in range(0, width - box + 1):
            masses.append(binary[y:y + box, x:x + box].sum())

    masses = np.asarray(masses, dtype=np.float32)
    lacunarity = masses.var() / ((masses.mean() ** 2) + 1e-12)
    return {"lacunarity_box10": float(lacunarity)}


def _local_statistics_features(gray: np.ndarray) -> dict[str, float]:
    img = gray.astype(np.float32)

    mean = cv2.blur(img, (9, 9))
    mean_sq = cv2.blur(img ** 2, (9, 9))
    var = np.maximum(mean_sq - mean ** 2, 0.0)
    std = np.sqrt(var)

    ent = rank_entropy(gray, disk(5))
    return {
        "local_entropy_mean_r5": float(ent.mean()),
        "local_entropy_p90_r5": float(np.percentile(ent, 90)),
        "local_std_p90_w9": float(np.percentile(std, 90)),
    }


def _gradient_features(gray: np.ndarray) -> dict[str, float]:
    gray_f = gray.astype(np.float32)
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return {
        "grad_p99": float(np.percentile(mag, 99)),
        "grad_std": float(mag.std()),
    }


def _haar_feature(gray: np.ndarray) -> dict[str, float]:
    gray_f = gray.astype(np.float32)
    _, (c_h, c_v, c_d) = pywt.dwt2(gray_f, "haar")
    detail_power = (c_h ** 2 + c_v ** 2 + c_d ** 2) / 3.0
    nonzero = detail_power[detail_power != 0]
    value = float(nonzero.mean()) if nonzero.size else 0.0
    return {"haar_detail_mean": value}


def _laplace_feature(gray: np.ndarray) -> dict[str, float]:
    lap = laplace(gray.astype(np.float32))
    return {"laplace_mean_abs": float(np.abs(lap).mean())}


def extract_numerical_features_from_gray(
    gray_raw: np.ndarray,
    return_dict: bool = False,
) -> dict[str, float] | np.ndarray:
    gray = normalize_gray(gray_raw)

    features: OrderedDict[str, float] = OrderedDict()
    features.update(_morphology_features(gray))
    features.update(_lacunarity_feature(gray))
    features.update(_autocorrelation_features(gray))
    features.update(_glcm_features(gray))
    features.update(_fft_features(gray))
    features.update(_local_statistics_features(gray))
    features.update(_gradient_features(gray))
    features.update(_haar_feature(gray))
    features.update(_laplace_feature(gray))

    if return_dict:
        return features

    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract_numerical_features(
    image: Image.Image | np.ndarray,
    return_dict: bool = False,
    crop_size: int = CROP_SIZE,
) -> dict[str, float] | np.ndarray:
    gray_raw = crop_center_square(image, size=crop_size)
    return extract_numerical_features_from_gray(gray_raw, return_dict=return_dict)
