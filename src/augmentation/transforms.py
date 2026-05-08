from __future__ import annotations

import csv
import io
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, overload

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms import functional


__all__ = [
    "AugmentationConfig",
    "AugmentedResult",
    "CoffeeAugmentor",
    "ApplyAugmentation",
    "augment_dataset_from_csv",
    "augment_image",
    "augment_many_layered",
    "augment_many_unique",
    "build_online_train_transform",
    "load_image",
    "save_image",
    "seed_worker",
    "set_seed",
]


_IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

_MAX_SEED = 2**63 - 1


@dataclass(frozen=True)
class AugmentationConfig:
    """Probability and intensity settings for coffee image augmentation."""

    horizontal_flip_prob: float = 0.50
    vertical_flip_prob: float = 0.15

    zoom_prob: float = 0.60
    zoom_scale_range: tuple[float, float] = (0.85, 1.00)

    rotation_prob: float = 0.80
    rotation_degrees: float = 12.0  # <45

    affine_prob: float = 0.80
    affine_translate: float = 0.05
    affine_scale_range: tuple[float, float] = (0.92, 1.08)
    affine_shear_degrees: float = 8.0

    perspective_prob: float = 0.25
    perspective_scale: float = 0.18

    brightness_prob: float = 0.80
    brightness_range: tuple[float, float] = (0.85, 1.15)

    contrast_prob: float = 0.80
    contrast_range: tuple[float, float] = (0.85, 1.15)

    saturation_prob: float = 0.25
    saturation_range: tuple[float, float] = (0.95, 1.05)

    sharpness_prob: float = 0.35
    sharpness_range: tuple[float, float] = (0.70, 1.40)

    blur_prob: float = 0.25
    blur_radius_range: tuple[float, float] = (0.20, 1.20)

    noise_prob: float = 0.25
    noise_std_range: tuple[float, float] = (0.005, 0.020)

    jpeg_prob: float = 0.15
    jpeg_quality_range: tuple[int, int] = (55, 95)

    cutout_prob: float = 0.0  # not sure if we really need it, so zero for now
    cutout_fraction_range: tuple[float, float] = (0.03, 0.10)
    cutout_patches: int = 2

    preserve_size: bool = True


@dataclass(frozen=True)
class AugmentedResult:
    """Metadata for one generated sample."""

    source_index: int
    source_name: str
    output_name: str
    image: Image.Image
    operations: tuple[str, ...]


class ApplyAugmentation:
    """Pickle-friendly callable for online augmentation pipelines."""

    def __init__(self, config: AugmentationConfig | None = None):
        self.config = config or AugmentationConfig()

    def __call__(self, image: Image.Image | str | Path | torch.Tensor) -> Image.Image:
        """Augment one image and return a PIL image for the next pipeline step."""

        return augment_image(image, config=self.config)


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible runs."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    """Seed DataLoader workers for reproducible online augmentation."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _ensure_random(rng: random.Random | None) -> random.Random:
    """Return a usable RNG, creating a fresh one when needed."""

    return rng if rng is not None else random.Random()


def _sample_uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    """Sample a float uniformly from the given interval."""

    low, high = bounds
    if low == high:
        return low
    return rng.uniform(low, high)


def _sample_int(rng: random.Random, bounds: tuple[int, int]) -> int:
    """Sample an integer uniformly from the given interval."""

    low, high = bounds
    if low == high:
        return low
    return rng.randint(low, high)


def _maybe(rng: random.Random, probability: float) -> bool:
    """Return True with the requested probability."""

    return probability > 0.0 and rng.random() < probability


def load_image(image: Image.Image | str | Path | torch.Tensor) -> Image.Image:
    """Load an image-like object and return a fresh RGB PIL image."""

    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return handle.convert("RGB").copy()
    if torch.is_tensor(image):
        return functional.to_pil_image(image.detach().cpu()).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def save_image(
    image: Image.Image,
    path: str | Path,
    *,
    image_format: str | None = None,
) -> None:
    """Save a PIL image and create the target directory if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format=image_format)


def _mean_fill_color(image: Image.Image) -> tuple[int, int, int]:
    """Estimate a neutral fill color from the average image color."""

    color = image.convert("RGB").resize((1, 1), Image.Resampling.BICUBIC).getpixel((0, 0))
    red, green, blue = color
    return int(red), int(green), int(blue)


def _layered_sample_indices(
    total: int,
    n_items: int,
    rng: random.Random,
    *,
    allow_reuse: bool,
) -> list[int]:
    """Build source indices for layered or replacement-based sampling."""

    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    if total <= 0:
        raise ValueError("total must be positive")

    if allow_reuse:
        return [rng.randrange(total) for _ in range(n_items)]

    selected_indices: list[int] = []
    remaining = n_items
    while remaining > 0:
        layer = list(range(total))
        rng.shuffle(layer)
        take = min(remaining, total)
        selected_indices.extend(layer[:take])
        remaining -= take
    return selected_indices


def _random_zoom_crop(image: Image.Image, rng: random.Random, config: AugmentationConfig) -> Image.Image:
    """Randomly crop and optionally resize to emulate framing variation."""

    width, height = image.size
    if width < 2 or height < 2:
        return image

    scale = _sample_uniform(rng, config.zoom_scale_range)
    crop_w = max(1, min(width, int(width * scale)))
    crop_h = max(1, min(height, int(height * scale)))

    if crop_w == width and crop_h == height:
        return image

    left = rng.randint(0, width - crop_w)
    top = rng.randint(0, height - crop_h)
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    if config.preserve_size:
        return cropped.resize((width, height), resample=Image.Resampling.BICUBIC)
    return cropped


def _random_rotate(image: Image.Image, rng: random.Random, config: AugmentationConfig) -> Image.Image:
    """Rotate the image by a small random angle."""

    angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=_mean_fill_color(image),
    )


def _random_affine(image: Image.Image, rng: random.Random, config: AugmentationConfig) -> Image.Image:
    """Apply small translation, scale, and shear perturbations."""

    width, height = image.size
    max_dx = int(width * config.affine_translate)
    max_dy = int(height * config.affine_translate)
    translate = (
        rng.randint(-max_dx, max_dx) if max_dx else 0,
        rng.randint(-max_dy, max_dy) if max_dy else 0,
    )
    scale = _sample_uniform(rng, config.affine_scale_range)
    shear = rng.uniform(-config.affine_shear_degrees, config.affine_shear_degrees)
    shear_rad = math.radians(shear)
    shear_term = math.tan(shear_rad)
    cx = width / 2.0
    cy = height / 2.0
    tx, ty = translate

    # PIL expects inverse affine coefficients.
    a = 1.0 / scale
    b = -shear_term / scale
    d = 0.0
    e = 1.0 / scale
    c = cx - (cx + tx) / scale + (shear_term * (cy + ty)) / scale
    f = cy - (cy + ty) / scale

    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BICUBIC,
        fillcolor=_mean_fill_color(image),
    )


def _random_perspective(image: Image.Image, rng: random.Random, config: AugmentationConfig) -> Image.Image:
    """Skew the image corners to mimic viewpoint changes."""

    width, height = image.size
    dx = max(1, int(width * config.perspective_scale))
    dy = max(1, int(height * config.perspective_scale))

    startpoints = [
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]
    endpoints = [
        [rng.randint(0, dx), rng.randint(0, dy)],
        [width - 1 - rng.randint(0, dx), rng.randint(0, dy)],
        [width - 1 - rng.randint(0, dx), height - 1 - rng.randint(0, dy)],
        [rng.randint(0, dx), height - 1 - rng.randint(0, dy)],
    ]
    matrix_rows: list[list[float]] = []
    rhs: list[float] = []
    for (x_out, y_out), (x_in, y_in) in zip(endpoints, startpoints, strict=True):
        matrix_rows.append([x_out, y_out, 1.0, 0.0, 0.0, 0.0, -x_in * x_out, -x_in * y_out])
        matrix_rows.append([0.0, 0.0, 0.0, x_out, y_out, 1.0, -y_in * x_out, -y_in * y_out])
        rhs.extend([x_in, y_in])

    coeffs = torch.linalg.solve(
        torch.tensor(matrix_rows, dtype=torch.float64),
        torch.tensor(rhs, dtype=torch.float64),
    )

    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        tuple(float(value) for value in coeffs.tolist()),
        resample=Image.Resampling.BICUBIC,
        fillcolor=_mean_fill_color(image),
    )


def _random_photometric(
    image: Image.Image,
    rng: random.Random,
    config: AugmentationConfig,
) -> tuple[Image.Image, tuple[str, ...]]:
    """Apply brightness, contrast, saturation, and sharpness tweaks."""

    output = image
    operations: list[str] = []

    if _maybe(rng, config.brightness_prob):
        factor = _sample_uniform(rng, config.brightness_range)
        output = ImageEnhance.Brightness(output).enhance(factor)
        operations.append("brightness")

    if _maybe(rng, config.contrast_prob):
        factor = _sample_uniform(rng, config.contrast_range)
        output = ImageEnhance.Contrast(output).enhance(factor)
        operations.append("contrast")

    if _maybe(rng, config.saturation_prob):
        factor = _sample_uniform(rng, config.saturation_range)
        output = ImageEnhance.Color(output).enhance(factor)
        operations.append("saturation")

    if _maybe(rng, config.sharpness_prob):
        factor = _sample_uniform(rng, config.sharpness_range)
        output = ImageEnhance.Sharpness(output).enhance(factor)
        operations.append("sharpness")

    return output, tuple(operations)


def _random_blur(image: Image.Image, rng: random.Random, config: AugmentationConfig) -> Image.Image:
    """Apply a mild Gaussian blur."""

    radius = _sample_uniform(rng, config.blur_radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _random_noise(image: Image.Image, rng: random.Random, config: AugmentationConfig) -> Image.Image:
    """Add small Gaussian noise to mimic sensor noise."""

    tensor = functional.to_tensor(image)
    noise_std = _sample_uniform(rng, config.noise_std_range)
    noise_seed = rng.randrange(_MAX_SEED)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
    noisy = torch.clamp(tensor + noise * noise_std, 0.0, 1.0)
    return functional.to_pil_image(noisy)


def _random_jpeg(image: Image.Image, rng: random.Random, config: AugmentationConfig) -> Image.Image:
    """Re-encode the image with lower JPEG quality."""

    quality = _sample_int(rng, config.jpeg_quality_range)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def _random_cutout(image: Image.Image, rng: random.Random, config: AugmentationConfig) -> Image.Image:
    """Mask out a couple of small regions with a neutral fill color."""

    output = image.copy()
    draw = ImageDraw.Draw(output)
    fill = _mean_fill_color(output)
    width, height = output.size

    for _ in range(max(0, config.cutout_patches)):
        frac = _sample_uniform(rng, config.cutout_fraction_range)
        box_w = max(1, int(width * frac))
        box_h = max(1, int(height * frac))
        if box_w >= width or box_h >= height:
            continue
        left = rng.randint(0, width - box_w)
        top = rng.randint(0, height - box_h)
        draw.rectangle([left, top, left + box_w, top + box_h], fill=fill)
    return output


@overload
def augment_image(
    image: Image.Image | str | Path | torch.Tensor,
    config: AugmentationConfig | None = None,
    *,
    rng: random.Random | None = None,
    return_ops: Literal[False] = False,
) -> Image.Image:
    ...


@overload
def augment_image(
    image: Image.Image | str | Path | torch.Tensor,
    config: AugmentationConfig | None = None,
    *,
    rng: random.Random | None = None,
    return_ops: Literal[True],
) -> tuple[Image.Image, tuple[str, ...]]:
    ...


def augment_image(
    image: Image.Image | str | Path | torch.Tensor,
    config: AugmentationConfig | None = None,
    *,
    rng: random.Random | None = None,
    return_ops: bool = False,
) -> Image.Image | tuple[Image.Image, tuple[str, ...]]:
    """Apply one random augmentation pass to a single image.

    The returned image stays in PIL format so it can be used both for
    offline generation and as part of a torchvision online transform.
    """

    config = config or AugmentationConfig()
    rng = _ensure_random(rng)
    output = load_image(image)
    operations: list[str] = []

    if _maybe(rng, config.horizontal_flip_prob):
        output = ImageOps.mirror(output)
        operations.append("horizontal_flip")

    if _maybe(rng, config.vertical_flip_prob):
        output = ImageOps.flip(output)
        operations.append("vertical_flip")

    if _maybe(rng, config.zoom_prob):
        output = _random_zoom_crop(output, rng, config)
        operations.append("zoom_crop")

    if _maybe(rng, config.rotation_prob):
        output = _random_rotate(output, rng, config)
        operations.append("rotation")

    if _maybe(rng, config.affine_prob):
        output = _random_affine(output, rng, config)
        operations.append("affine")

    if _maybe(rng, config.perspective_prob):
        output = _random_perspective(output, rng, config)
        operations.append("perspective")

    output, photometric_ops = _random_photometric(output, rng, config)
    operations.extend(photometric_ops)

    if _maybe(rng, config.blur_prob):
        output = _random_blur(output, rng, config)
        operations.append("blur")

    if _maybe(rng, config.noise_prob):
        output = _random_noise(output, rng, config)
        operations.append("noise")

    if _maybe(rng, config.jpeg_prob):
        output = _random_jpeg(output, rng, config)
        operations.append("jpeg")

    if _maybe(rng, config.cutout_prob):
        output = _random_cutout(output, rng, config)
        operations.append("cutout")

    if return_ops:
        return output, tuple(operations)
    return output


def _coerce_sample_name(sample: Image.Image | str | Path | torch.Tensor, source_index: int) -> str:
    """Derive a readable filename for generated samples."""

    if isinstance(sample, (str, Path)):
        return Path(sample).name
    return f"sample_{source_index:06d}.png"


def _build_output_name(source_name: str, output_index: int) -> str:
    """Create a stable output filename for an augmented sample."""

    return f"{Path(source_name).stem}__aug_{output_index:06d}.png"


def augment_many_unique(
    samples: Sequence[Image.Image | str | Path | torch.Tensor],
    n_new_images: int,
    *,
    config: AugmentationConfig | None = None,
    seed: int | None = None,
    allow_reuse: bool = False,
) -> list[AugmentedResult]:
    """Generate augmented samples in shuffled layers.

    The default mode performs repeated full passes over the source samples.
    Each pass uses every source at most once. If the requested amount exceeds
    the source count, the next layer starts with a fresh shuffle.

    ``allow_reuse=True`` switches to plain sampling with replacement.
    """

    if n_new_images < 0:
        raise ValueError("n_new_images must be non-negative")
    if not samples:
        raise ValueError("samples cannot be empty")

    rng = random.Random(seed)
    selected_indices = _layered_sample_indices(
        len(samples),
        n_new_images,
        rng,
        allow_reuse=allow_reuse,
    )

    results: list[AugmentedResult] = []
    for output_index, source_index in enumerate(selected_indices):
        child_rng = random.Random(rng.randrange(_MAX_SEED))
        augmented, ops = augment_image(
            samples[source_index],
            config=config,
            rng=child_rng,
            return_ops=True,
        )
        source_name = _coerce_sample_name(samples[source_index], source_index)
        output_name = _build_output_name(source_name, output_index)
        results.append(
            AugmentedResult(
                source_index=source_index,
                source_name=source_name,
                output_name=output_name,
                image=augmented,
                operations=ops,
            )
        )
    return results


def augment_many_layered(
    samples: Sequence[Image.Image | str | Path | torch.Tensor],
    n_new_images: int,
    *,
    config: AugmentationConfig | None = None,
    seed: int | None = None,
) -> list[AugmentedResult]:
    """Generate samples using repeated shuffled passes without replacement."""

    return augment_many_unique(
        samples,
        n_new_images,
        config=config,
        seed=seed,
        allow_reuse=False,
    )


def augment_dataset_from_csv(
    csv_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    n_new_images: int,
    *,
    config: AugmentationConfig | None = None,
    seed: int | None = None,
    allow_reuse: bool = False,
    output_csv_path: str | Path | None = None,
) -> list[AugmentedResult]:
    """Create an offline augmented dataset from a semicolon-delimited CSV.

    The source CSV is expected to follow the project format:
    ``Sample;Fineness``.
    The default mode generates repeated shuffled passes over the rows, so each
    source row appears at most once per pass.
    """

    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        rows = [row for row in reader if row]

    if len(header) < 2:
        raise ValueError("CSV must contain at least two columns: Sample and Fineness")
    if not rows:
        raise ValueError("CSV does not contain any data rows")

    rng = random.Random(seed)
    selected_indices = _layered_sample_indices(
        len(rows),
        n_new_images,
        rng,
        allow_reuse=allow_reuse,
    )

    results: list[AugmentedResult] = []
    csv_rows: list[list[str]] = []

    for output_index, source_index in enumerate(selected_indices):
        row = rows[source_index]
        source_name = row[0]
        label = row[1]
        source_path = images_dir / source_name
        child_rng = random.Random(rng.randrange(_MAX_SEED))
        augmented, ops = augment_image(
            source_path,
            config=config,
            rng=child_rng,
            return_ops=True,
        )

        output_name = _build_output_name(source_name, output_index)
        output_path = output_dir / output_name
        save_image(augmented, output_path)
        results.append(
            AugmentedResult(
                source_index=source_index,
                source_name=source_name,
                output_name=output_name,
                image=augmented,
                operations=ops,
            )
        )
        csv_rows.append([output_name, label])

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(header)
            writer.writerows(csv_rows)

    return results


def build_online_train_transform(
    config: AugmentationConfig | None = None,
    *,
    resize_short_side: int = 256,
    crop_size: int = 224,
    normalize: bool = True,
) -> transforms.Compose:
    """Build a torchvision transform for on-the-fly training augmentation."""

    config = config or AugmentationConfig()
    steps: list[object] = [
        ApplyAugmentation(config),
    ]

    if resize_short_side is not None:
        steps.append(transforms.Resize(resize_short_side))
    if crop_size is not None:
        steps.append(transforms.CenterCrop(crop_size))

    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(_IMAGENET_NORMALIZE)

    return transforms.Compose(steps)


class CoffeeAugmentor:
    """Stateful helper for reusable augmentation workflows."""

    def __init__(self, config: AugmentationConfig | None = None, seed: int | None = None):
        self.config = config or AugmentationConfig()
        self.seed = seed
        self._rng = random.Random(seed)

    def augment(self, image: Image.Image | str | Path | torch.Tensor, *, return_ops: bool = False):
        """Augment one image using the instance config and RNG state."""

        child_rng = random.Random(self._rng.randrange(_MAX_SEED))
        if return_ops:
            return augment_image(
                image,
                config=self.config,
                rng=child_rng,
                return_ops=True,
            )
        return augment_image(
            image,
            config=self.config,
            rng=child_rng,
            return_ops=False,
        )

    def augment_many(
        self,
        samples: Sequence[Image.Image | str | Path | torch.Tensor],
        n_new_images: int,
        *,
        allow_reuse: bool = False,
    ) -> list[AugmentedResult]:
        """Augment multiple in-memory samples using layered sampling."""

        return augment_many_unique(
            samples,
            n_new_images,
            config=self.config,
            seed=self._rng.randrange(_MAX_SEED),
            allow_reuse=allow_reuse,
        )

    def build_online_transform(
        self,
        *,
        resize_short_side: int = 256,
        crop_size: int = 224,
        normalize: bool = True,
    ) -> transforms.Compose:
        """Build a callable transform suitable for DataLoader usage."""

        return build_online_train_transform(
            self.config,
            resize_short_side=resize_short_side,
            crop_size=crop_size,
            normalize=normalize,
        )

    def expand_csv_dataset(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        output_dir: str | Path,
        n_new_images: int,
        *,
        allow_reuse: bool = False,
        output_csv_path: str | Path | None = None,
    ) -> list[AugmentedResult]:
        """Create an offline augmented dataset from the project CSV format."""

        return augment_dataset_from_csv(
            csv_path,
            images_dir,
            output_dir,
            n_new_images,
            config=self.config,
            seed=self._rng.randrange(_MAX_SEED),
            allow_reuse=allow_reuse,
            output_csv_path=output_csv_path,
        )
