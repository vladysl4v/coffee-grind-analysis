"""
Compare spatial frequency content of real vs synthetic images per fineness bin.

A camera scale issue (wrong distance) shifts the power spectrum horizontally.
If real and synthetic curves have the same shape but different frequency centroids,
the camera is too close or too far. If the shapes are completely different, the
problem is deeper than scale (domain gap in texture/lighting).

Centroids are reported in normalised units (cycles/pixel) so they are comparable
even though real and synthetic images use different crop sizes.

Usage:
    cd Q:/coffee-grind-analysis
    uv run python src/compare_texture_freq.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

_ROOT      = Path(__file__).parent.parent
_REAL_DIR  = _ROOT / "data" / "images" / "raw"
_LABELS    = _ROOT / "data" / "labels"
_OUT       = _ROOT / "data" / "texture_freq_analysis.png"

_SYNTH_DIR = Path("Q:/temp/coffee_grind_new_dataset")
_SYNTH_CSV = _SYNTH_DIR / "labels.csv"

CROP_REAL  = 420   # real images centre crop (px)
CROP_SYNTH = 420   # synthetic images centre crop (px)
BINS  = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]


def load_csv(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    return [(str(row.iloc[0]), float(row.iloc[1])) for _, row in df.iterrows()]


def radial_power_spectrum(img_path, images_dir, crop):
    """Return normalised radial average power spectrum for one image."""
    img = Image.open(images_dir / img_path).convert("L")
    w, h = img.size
    left = (w - crop) // 2
    top  = (h - crop) // 2
    img  = img.crop((left, top, left + crop, top + crop))
    arr  = np.array(img, dtype=np.float32)

    fft    = np.fft.fft2(arr)
    fft    = np.fft.fftshift(fft)
    power  = np.abs(fft) ** 2

    cy, cx = crop // 2, crop // 2
    y, x   = np.indices((crop, crop))
    r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r  = crop // 2
    radial = np.zeros(max_r)
    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            radial[ri] = power[mask].mean()

    # skip DC (r=0), normalise so area = 1 for shape comparison
    radial[0] = 0
    total = radial.sum()
    if total > 0:
        radial /= total
    return radial


def collect_spectra(samples, images_dir, crop, label=""):
    """Return dict of fineness_bin -> list of spectra."""
    from collections import defaultdict
    binned = defaultdict(list)
    n = len(samples)
    for i, (fname, fineness) in enumerate(samples):
        if i % 50 == 0:
            print(f"  {label}: {i}/{n}")
        try:
            spec = radial_power_spectrum(fname, images_dir, crop)
            for lo, hi in BINS:
                if lo <= fineness < hi or (hi == 100 and fineness == 100):
                    binned[(lo, hi)].append(spec)
                    break
        except Exception:
            pass
    return binned


def freq_centroid(spectrum):
    freqs = np.arange(len(spectrum))
    total = spectrum.sum()
    if total == 0:
        return 0
    return (freqs * spectrum).sum() / total


def main():
    real_samples  = []
    for split in ("train", "val", "test"):
        p = _LABELS / f"{split}.csv"
        if p.exists():
            real_samples += load_csv(p)

    if not _SYNTH_CSV.exists():
        print(f"No labels.csv found at {_SYNTH_CSV}")
        return
    synth_samples = load_csv(_SYNTH_CSV)
    if not synth_samples:
        print("Synthetic labels.csv is empty.")
        return

    print(f"Real images : {len(real_samples)}  (crop={CROP_REAL}px)")
    print(f"Synth images: {len(synth_samples)}  (crop={CROP_SYNTH}px)")

    print("\nComputing real spectra...")
    real_binned  = collect_spectra(real_samples,  _REAL_DIR,  CROP_REAL,  "real")
    print("Computing synthetic spectra...")
    synth_binned = collect_spectra(synth_samples, _SYNTH_DIR, CROP_SYNTH, "synth")

    freqs = np.arange(CROP_REAL // 2)

    fig, axes = plt.subplots(1, len(BINS), figsize=(18, 4), sharey=False)
    fig.suptitle(
        f"Radial Power Spectrum: Real vs Synthetic (crop={CROP_REAL}px)",
        fontsize=13,
    )

    print(f"\n{'Bin':<12} {'Real centroid':>15} {'Synth centroid':>16} {'Ratio (S/R)':>13}")
    print("-" * 58)

    for ax, (lo, hi) in zip(axes, BINS):
        label = f"{lo}–{hi}%"
        r_specs = real_binned.get((lo, hi), [])
        s_specs = synth_binned.get((lo, hi), [])

        if r_specs:
            r_mean = np.stack(r_specs).mean(axis=0)
            r_c    = freq_centroid(r_mean)
            ax.plot(freqs, r_mean, color="saddlebrown", linewidth=1.5,
                    label=f"real (n={len(r_specs)})")
        else:
            r_mean = None
            r_c    = None

        if s_specs:
            s_mean = np.stack(s_specs).mean(axis=0)
            s_c    = freq_centroid(s_mean)
            ax.plot(freqs, s_mean, color="steelblue", linewidth=1.5,
                    linestyle="--", label=f"synth (n={len(s_specs)})")
        else:
            s_mean = None
            s_c    = None

        ratio_str = f"{s_c/r_c:.3f}" if (r_c and s_c and r_c > 0) else "n/a"
        print(f"{label:<12} {r_c if r_c else 'n/a':>15.2f} {s_c if s_c else 'n/a':>16.2f} {ratio_str:>13}")

        ax.set_title(label)
        ax.set_xlabel("Spatial frequency (px radius)")
        ax.set_xlim(0, CROP_REAL // 6)
        ax.legend(fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Normalised power")

    fig.tight_layout()
    fig.savefig(_OUT, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved to {_OUT}")
    print("\nInterpretation:")
    print("  Ratio ~1.0  → scale is roughly correct")
    print("  Ratio  >1.0 → synthetic has higher freq → camera too close (particles look smaller)")
    print("  Ratio  <1.0 → synthetic has lower freq  → camera too far  (particles look larger)")


if __name__ == "__main__":
    main()
