import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from data_loader import CoffeeDataset


TARGET_SIZE = (224, 224)   # much lighter
OUTPUT_FILE = "eda_average_images.png"


def compute_thresholds(ds):
    labels = []
    for i in range(len(ds)):
        _, label = ds[i]
        labels.append(label.item())
    labels = np.array(labels, dtype=np.float32)
    low_thr = np.percentile(labels, 33)
    high_thr = np.percentile(labels, 66)
    return low_thr, high_thr


def prepare_image(img: Image.Image) -> np.ndarray:
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def main():
    ds = CoffeeDataset("train", transform=None)

    low_thr, high_thr = compute_thresholds(ds)
    print(f"33% threshold: {low_thr:.3f}")
    print(f"66% threshold: {high_thr:.3f}")

    sum_fine = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.float64)
    sum_medium = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.float64)
    sum_coarse = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.float64)

    count_fine = 0
    count_medium = 0
    count_coarse = 0

    for i in range(len(ds)):
        img, label = ds[i]
        arr = prepare_image(img)
        value = label.item()

        if value <= low_thr:
            sum_fine += arr
            count_fine += 1
        elif value <= high_thr:
            sum_medium += arr
            count_medium += 1
        else:
            sum_coarse += arr
            count_coarse += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(ds):
            print(f"Processed {i + 1}/{len(ds)} images")

    avg_fine = (sum_fine / max(count_fine, 1)).astype(np.float32)
    avg_medium = (sum_medium / max(count_medium, 1)).astype(np.float32)
    avg_coarse = (sum_coarse / max(count_coarse, 1)).astype(np.float32)

    print(f"Fine: {count_fine}, Medium: {count_medium}, Coarse: {count_coarse}")

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(avg_fine)
    ax1.set_title("Fine (avg)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(avg_medium)
    ax2.set_title("Medium (avg)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(avg_coarse)
    ax3.set_title("Coarse (avg)")
    ax3.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()