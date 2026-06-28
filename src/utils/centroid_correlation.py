"""
Compute per-image FFT centroid vs fineness correlation for real and synthetic images.

Usage:
    cd Q:/coffee-grind-analysis
    uv run python src/centroid_correlation.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy import stats

_ROOT      = Path(__file__).parent.parent
_REAL_DIR  = _ROOT / "data" / "images" / "raw"
_LABELS    = _ROOT / "data" / "labels"
_SYNTH_DIR = Path("Q:/temp/coffee_grind_new_dataset")
_SYNTH_CSV = _SYNTH_DIR / "labels.csv"
_OUT       = _ROOT / "data" / "centroid_correlation.png"

CROP = 420


def load_csv(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    return [(str(row.iloc[0]), float(row.iloc[1])) for _, row in df.iterrows()]


def centroid(img_path, images_dir):
    img = Image.open(images_dir / img_path).convert("L")
    w, h = img.size
    left = (w - CROP) // 2
    top  = (h - CROP) // 2
    img  = img.crop((left, top, left + CROP, top + CROP))
    arr  = np.array(img, dtype=np.float32)

    fft   = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(fft) ** 2

    cy, cx = CROP // 2, CROP // 2
    y, x   = np.indices((CROP, CROP))
    r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r  = CROP // 2
    radial = np.zeros(max_r)
    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            radial[ri] = power[mask].mean()
    radial[0] = 0
    total = radial.sum()
    if total == 0:
        return None
    radial /= total
    freqs = np.arange(max_r)
    return float((freqs * radial).sum() / radial.sum())


def compute_all(samples, images_dir, label=""):
    centroids, fineness = [], []
    n = len(samples)
    for i, (fname, fin) in enumerate(samples):
        if i % 100 == 0:
            print(f"  {label}: {i}/{n}")
        try:
            c = centroid(fname, images_dir)
            if c is not None:
                centroids.append(c)
                fineness.append(fin)
        except Exception:
            pass
    return np.array(fineness), np.array(centroids)


def main():
    real_samples = []
    for split in ("train", "val", "test"):
        p = _LABELS / f"{split}.csv"
        if p.exists():
            real_samples += load_csv(p)

    synth_samples = load_csv(_SYNTH_CSV)

    print(f"Real : {len(real_samples)} images")
    print(f"Synth: {len(synth_samples)} images\n")

    print("Computing real centroids...")
    r_fin, r_cen = compute_all(real_samples, _REAL_DIR, "real")

    print("Computing synth centroids...")
    s_fin, s_cen = compute_all(synth_samples, _SYNTH_DIR, "synth")

    r_pearson = stats.pearsonr(r_fin, r_cen)
    s_pearson = stats.pearsonr(s_fin, s_cen)
    r_spearman = stats.spearmanr(r_fin, r_cen)
    s_spearman = stats.spearmanr(s_fin, s_cen)

    print(f"\n{'':30} {'Real':>10} {'Synth':>10}")
    print("-" * 52)
    print(f"{'Pearson r':30} {r_pearson.statistic:>10.3f} {s_pearson.statistic:>10.3f}")
    print(f"{'Pearson p-value':30} {r_pearson.pvalue:>10.2e} {s_pearson.pvalue:>10.2e}")
    print(f"{'Spearman r':30} {r_spearman.statistic:>10.3f} {s_spearman.statistic:>10.3f}")
    print(f"{'Spearman p-value':30} {s_spearman.pvalue:>10.2e} {s_spearman.pvalue:>10.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("FFT Centroid vs Fineness", fontsize=13)

    for ax, fin, cen, label, color, pr, sp in [
        (axes[0], r_fin, r_cen, "Real",      "saddlebrown", r_pearson, r_spearman),
        (axes[1], s_fin, s_cen, "Synthetic", "steelblue",   s_pearson, s_spearman),
    ]:
        ax.scatter(fin, cen, alpha=0.35, s=12, color=color)
        m, b = np.polyfit(fin, cen, 1)
        x_line = np.linspace(fin.min(), fin.max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5)
        ax.set_xlabel("Fineness (%)")
        ax.set_ylabel("FFT centroid (px radius)")
        ax.set_title(f"{label}  (n={len(fin)})\nPearson r={pr.statistic:.3f}  Spearman r={sp.statistic:.3f}")

    fig.tight_layout()
    fig.savefig(_OUT, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved to {_OUT}")


if __name__ == "__main__":
    main()
