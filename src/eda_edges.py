import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from data_loader import CoffeeDataset


TARGET_SIZE = (224, 224) # chnge to 1080 maybe
EDGE_MAPS_FILE = "eda_edge_maps_center_circle_gray.png"
EDGE_DENSITY_FILE = "eda_edge_density_center_circle.png"


def compute_thresholds(ds):
    labels = []
    for i in range(len(ds)):
        _, label = ds[i]
        labels.append(label.item())
    labels = np.array(labels, dtype=np.float32)
    low_thr = np.percentile(labels, 33)
    high_thr = np.percentile(labels, 66)
    return low_thr, high_thr


def prepare_gray(img: Image.Image) -> np.ndarray:
    img = img.resize(TARGET_SIZE, Image.NEAREST)   ################# thriple causion. change to BILINEAR maybe
    arr = np.asarray(img, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray


def build_center_circle_mask(height: int, width: int) -> np.ndarray:
    cy = height // 2
    cx = width // 2
    radius = width / 8.0

    yy, xx = np.ogrid[:height, :width]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist2 <= radius ** 2
    return mask


def main():
    ds = CoffeeDataset("train", transform=None)

    low_thr, high_thr = compute_thresholds(ds)
    print(f"33% threshold: {low_thr:.3f}")
    print(f"66% threshold: {high_thr:.3f}")

    width, height = TARGET_SIZE
    mask = build_center_circle_mask(height, width)
    mask_float = mask.astype(np.float32)
    mask_area = int(mask.sum())

    sum_fine = np.zeros((height, width), dtype=np.float64)
    sum_medium = np.zeros((height, width), dtype=np.float64)
    sum_coarse = np.zeros((height, width), dtype=np.float64)

    count_fine = 0
    count_medium = 0
    count_coarse = 0

    densities = []
    fineness_vals = []

    for i in range(len(ds)):
        img, label = ds[i]
        gray = prepare_gray(img)

        edges = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
        masked_edges = edges * mask_float

        center_edge_density = masked_edges.sum() / mask_area
        densities.append(center_edge_density)
        fineness_vals.append(label.item())

        val = label.item()
        if val <= low_thr:
            sum_fine += masked_edges
            count_fine += 1
        elif val <= high_thr:
            sum_medium += masked_edges
            count_medium += 1
        else:
            sum_coarse += masked_edges
            count_coarse += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(ds):
            print(f"Processed {i + 1}/{len(ds)}")

    avg_fine = (sum_fine / max(count_fine, 1)).astype(np.float32)
    avg_medium = (sum_medium / max(count_medium, 1)).astype(np.float32)
    avg_coarse = (sum_coarse / max(count_coarse, 1)).astype(np.float32)

    print(f"Fine: {count_fine}, Medium: {count_medium}, Coarse: {count_coarse}")

    # same scale for all 3 images
    vmax = max(
        float(avg_fine.max()),
        float(avg_medium.max()),
        float(avg_coarse.max()),
        1e-8,
    )

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(avg_fine, cmap="gray", vmin=0.0, vmax=vmax)
    ax1.set_title("Fine edges\n(center circle)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(avg_medium, cmap="gray", vmin=0.0, vmax=vmax)
    ax2.set_title("Medium edges\n(center circle)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(avg_coarse, cmap="gray", vmin=0.0, vmax=vmax)
    ax3.set_title("Coarse edges\n(center circle)")
    ax3.axis("off")

    fig.tight_layout()
    fig.savefig(EDGE_MAPS_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(fineness_vals, densities, alpha=0.5)
    plt.xlabel("Fineness")
    plt.ylabel("Center-circle edge density")
    plt.title("Center-circle edge density vs fineness")
    plt.tight_layout()
    fig.savefig(EDGE_DENSITY_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {EDGE_MAPS_FILE}")
    print(f"Saved: {EDGE_DENSITY_FILE}")


if __name__ == "__main__":
    main()