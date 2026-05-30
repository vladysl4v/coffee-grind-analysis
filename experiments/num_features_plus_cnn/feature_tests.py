import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from PIL import Image

from scipy.ndimage import laplace
from scipy.stats import pearsonr, spearmanr
from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern,
)
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk, opening, closing
from skimage.util import img_as_ubyte

from data_loader import CoffeeDataset


CROP_SIZE = 150
OUTPUT_DIR = Path("outputs/eda_texture_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_center_square(img: Image.Image, size: int = 150) -> np.ndarray:
    """
    Crop center square without resizing.
    Returns grayscale uint8 image.
    """
    img = img.convert("RGB")
    w, h = img.size

    if w < size or h < size:
        raise ValueError(f"Image too small for {size}x{size} crop: {w}x{h}")

    left = (w - size) // 2
    top = (h - size) // 2

    crop = img.crop((left, top, left + size, top + size))
    arr = np.asarray(crop, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    """
    Local crop normalization.
    Keeps texture but reduces brightness/exposure differences.
    """
    gray_f = gray.astype(np.float32)

    mean = gray_f.mean()
    std = gray_f.std() + 1e-6

    z = (gray_f - mean) / std
    z = np.clip(z, -3, 3)

    norm = ((z + 3) / 6 * 255).astype(np.uint8)
    return norm


def fft_features(gray: np.ndarray) -> dict:
    img = gray.astype(np.float32)
    img -= img.mean()

    fft = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(fft) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / r.max()

    low = power[r_norm < 0.15].mean()
    mid = power[(r_norm >= 0.15) & (r_norm < 0.35)].mean()
    high = power[r_norm >= 0.35].mean()

    p = power.ravel()
    p = p / (p.sum() + 1e-12)
    spectral_entropy = -np.sum(p * np.log2(p + 1e-12))

    # radial spectrum slope
    bins = np.linspace(0.03, 1.0, 32)
    radial_vals = []
    radial_centers = []

    for a, b in zip(bins[:-1], bins[1:]):
        mask = (r_norm >= a) & (r_norm < b)
        if mask.sum() > 0:
            radial_vals.append(power[mask].mean())
            radial_centers.append((a + b) / 2)

    radial_vals = np.array(radial_vals) + 1e-12
    radial_centers = np.array(radial_centers) + 1e-12

    slope = np.polyfit(np.log(radial_centers), np.log(radial_vals), 1)[0]

    return {
        "fft_low_power": low,
        "fft_mid_power": mid,
        "fft_high_power": high,
        "fft_high_low_ratio": high / (low + 1e-12),
        "fft_mid_low_ratio": mid / (low + 1e-12),
        "fft_spectral_entropy": spectral_entropy,
        "fft_radial_slope": slope,
    }


def glcm_features(gray: np.ndarray) -> dict:
    # reduce gray levels to make GLCM stable
    levels = 32
    q = (gray.astype(np.float32) / 256 * levels).astype(np.uint8)

    distances = [1, 2, 4, 8, 16]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    out = {}

    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals = graycoprops(glcm, prop)
        out[f"glcm_{prop}_mean"] = vals.mean()
        out[f"glcm_{prop}_std"] = vals.std()

    return out


def lbp_features(gray: np.ndarray) -> dict:
    out = {}

    for radius in [1, 2, 4, 8]:
        points = radius * 8

        lbp = local_binary_pattern(
            gray,
            P=points,
            R=radius,
            method="uniform",
        )

        n_bins = points + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, n_bins + 1),
            density=True,
        )

        out[f"lbp_r{radius}_entropy"] = -np.sum(hist * np.log2(hist + 1e-12))
        out[f"lbp_r{radius}_uniformity"] = np.sum(hist ** 2)

        for i, val in enumerate(hist):
            out[f"lbp_r{radius}_bin_{i}"] = val

    return out


def morphology_granulometry_features(gray: np.ndarray) -> dict:
    img = gray.astype(np.float32)
    out = {}

    for radius in [1, 2, 3, 5, 8, 13, 21]:
        se = disk(radius)

        opened = opening(gray, se).astype(np.float32)
        closed = closing(gray, se).astype(np.float32)

        opening_removed = np.mean(np.abs(img - opened))
        closing_added = np.mean(np.abs(closed - img))

        out[f"morph_opening_removed_r{radius}"] = opening_removed
        out[f"morph_closing_added_r{radius}"] = closing_added

    return out


def autocorrelation_features(gray: np.ndarray) -> dict:
    img = gray.astype(np.float32)
    img -= img.mean()

    fft = np.fft.fft2(img)
    corr = np.fft.ifft2(np.abs(fft) ** 2).real
    corr = np.fft.fftshift(corr).astype(np.float32)

    h, w = corr.shape
    cy, cx = h // 2, w // 2
    corr = corr / (corr[cy, cx] + 1e-12)

    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    max_r = min(h, w) // 2
    radial = []

    for radius in range(1, max_r):
        mask = (r >= radius - 0.5) & (r < radius + 0.5)
        radial.append(corr[mask].mean())

    radial = np.array(radial)

    def first_below(thr):
        idx = np.where(radial < thr)[0]
        if len(idx) == 0:
            return max_r
        return int(idx[0] + 1)

    return {
        "autocorr_drop_050": first_below(0.50),
        "autocorr_drop_020": first_below(0.20),
        "autocorr_drop_010": first_below(0.10),
        "autocorr_radial_mean": radial.mean(),
        "autocorr_radial_std": radial.std(),
    }


def edge_gradient_features(gray: np.ndarray) -> dict:
    gray_f = gray.astype(np.float32)

    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    lap = cv2.Laplacian(gray_f, cv2.CV_32F)

    edges = cv2.Canny(gray, 50, 150)

    hist, _ = np.histogram(mag.ravel(), bins=64, density=True)
    grad_entropy = -np.sum(hist * np.log2(hist + 1e-12))

    return {
        "grad_mean": mag.mean(),
        "grad_std": mag.std(),
        "grad_p90": np.percentile(mag, 90),
        "grad_p95": np.percentile(mag, 95),
        "grad_p99": np.percentile(mag, 99),
        "grad_entropy": grad_entropy,
        "laplacian_variance": lap.var(),
        "canny_edge_density": (edges > 0).mean(),
    }


def local_statistics_features(gray: np.ndarray) -> dict:
    out = {}
    img = gray.astype(np.float32)

    for win in [3, 5, 9, 15, 31]:
        mean = cv2.blur(img, (win, win))
        mean_sq = cv2.blur(img ** 2, (win, win))
        var = np.maximum(mean_sq - mean ** 2, 0)
        std = np.sqrt(var)

        out[f"local_std_mean_w{win}"] = std.mean()
        out[f"local_std_std_w{win}"] = std.std()
        out[f"local_std_p90_w{win}"] = np.percentile(std, 90)

    ent = rank_entropy(gray, disk(5))
    out["local_entropy_mean_r5"] = ent.mean()
    out["local_entropy_std_r5"] = ent.std()
    out["local_entropy_p90_r5"] = np.percentile(ent, 90)

    return out


def haar_and_laplace_features(gray: np.ndarray) -> dict:
    gray_f = gray.astype(np.float32)

    _, (c_h, c_v, c_d) = pywt.dwt2(gray_f, "haar")
    detail_power = (c_h ** 2 + c_v ** 2 + c_d ** 2) / 3.0
    nonzero = detail_power[detail_power != 0]
    haar_mean = float(nonzero.mean()) if nonzero.size else 0.0

    lap = laplace(gray_f)

    return {
        "haar_detail_mean": haar_mean,
        "laplace_mean_abs": float(np.abs(lap).mean()),
    }


def fractal_lacunarity_features(gray: np.ndarray) -> dict:
    # Use thresholded normalized image as texture map
    binary = gray > np.percentile(gray, 50)

    sizes = [2, 3, 5, 10, 15, 25]
    counts = []

    h, w = binary.shape

    for size in sizes:
        count = 0

        for y in range(0, h - size + 1, size):
            for x in range(0, w - size + 1, size):
                block = binary[y:y + size, x:x + size]
                if block.any() and not block.all():
                    count += 1

        counts.append(count + 1e-12)

    coeff = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeff[0]

    masses = []

    box = 10
    for y in range(0, h - box + 1):
        for x in range(0, w - box + 1):
            masses.append(binary[y:y + box, x:x + box].sum())

    masses = np.array(masses, dtype=np.float32)
    lacunarity = masses.var() / ((masses.mean() ** 2) + 1e-12)

    return {
        "fractal_dimension_boxcount": fractal_dim,
        "lacunarity_box10": lacunarity,
    }


def extract_features(gray_raw: np.ndarray) -> dict:
    gray = normalize_gray(gray_raw)

    features = {}
    features.update(fft_features(gray))
    features.update(glcm_features(gray))
    features.update(lbp_features(gray))
    features.update(morphology_granulometry_features(gray))
    features.update(autocorrelation_features(gray))
    features.update(edge_gradient_features(gray))
    features.update(local_statistics_features(gray))
    features.update(fractal_lacunarity_features(gray))
    features.update(haar_and_laplace_features(gray))

    return features


def save_feature_correlation_plots(feature_table: dict, labels: np.ndarray):
    feature_names = list(feature_table.keys())

    stats = []

    for name in feature_names:
        x = np.array(feature_table[name], dtype=np.float32)
        y = labels.astype(np.float32)

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 3 or np.std(x[valid]) < 1e-12:
            continue

        pearson = pearsonr(x[valid], y[valid]).statistic
        spearman = spearmanr(x[valid], y[valid]).statistic

        stats.append((name, pearson, spearman))

        fig = plt.figure(figsize=(6, 5))
        plt.scatter(y[valid], x[valid], alpha=0.55)

        m, b = np.polyfit(y[valid], x[valid], 1)
        xs = np.linspace(y[valid].min(), y[valid].max(), 100)
        plt.plot(xs, m * xs + b)

        plt.xlabel("Ground-truth fineness")
        plt.ylabel(name)
        plt.title(
            f"{name}\n"
            f"Pearson r={pearson:.3f}, Spearman ρ={spearman:.3f}"
        )
        plt.tight_layout()

        safe_name = name.replace("/", "_").replace(" ", "_")
        fig.savefig(OUTPUT_DIR / f"{safe_name}.png", dpi=150)
        plt.close(fig)

    stats = sorted(stats, key=lambda t: abs(t[1]), reverse=True)

    with open(OUTPUT_DIR / "feature_correlations.csv", "w") as f:
        f.write("feature,pearson,spearman\n")
        for name, pearson, spearman in stats:
            f.write(f"{name},{pearson:.6f},{spearman:.6f}\n")

    top = stats[:30]

    names = [x[0] for x in top][::-1]
    vals = [x[1] for x in top][::-1]

    fig = plt.figure(figsize=(10, 10))
    plt.barh(names, vals)
    plt.xlabel("Pearson correlation with fineness")
    plt.title("Top 30 features by absolute Pearson correlation")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_30_feature_correlations.png", dpi=150)
    plt.close(fig)

    print("\nTop features:")
    for name, pearson, spearman in top:
        print(f"{name:40s}  Pearson={pearson:+.4f}  Spearman={spearman:+.4f}")


def main():
    ds = CoffeeDataset("train", transform=None)

    feature_table = {}
    labels = []

    for i in range(len(ds)):
        img, label = ds[i]

        gray_crop = crop_center_square(img, CROP_SIZE)
        features = extract_features(gray_crop)

        for k, v in features.items():
            feature_table.setdefault(k, []).append(float(v))

        labels.append(float(label.item()))

        if (i + 1) % 25 == 0 or (i + 1) == len(ds):
            print(f"Processed {i + 1}/{len(ds)}")

    labels = np.array(labels, dtype=np.float32)

    save_feature_correlation_plots(feature_table, labels)

    print(f"\nSaved plots to: {OUTPUT_DIR}")
    print(f"Saved correlation table: {OUTPUT_DIR / 'feature_correlations.csv'}")


if __name__ == "__main__":
    main()
