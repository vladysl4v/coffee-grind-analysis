# Data Augmentation

This module contains the first standalone augmentation pipeline for coffee grind images.

## What it does

- Augments a single image with controlled random transforms.
- Generates offline augmented datasets in shuffled layers, so each source image is used at most once per layer.
- Builds an online torchvision transform for training-time augmentation.
- Provides a small notebook for visual checks and quick experiments.

## Main entry points

- `augment_image(...)` - augment exactly one image and inspect the result.
- `augment_many_unique(...)` - generate a controlled batch from images already loaded in memory.
- `augment_many_layered(...)` - use the same layered sampling rule, but make it explicit at the call site.
- `augment_dataset_from_csv(...)` - create offline augmented files on disk from the project CSV format.
- `build_online_train_transform(...)` - build a transform for training-time augmentation on every epoch. *(not tested yet!)*
- `CoffeeAugmentor` - reuse one config and seed across multiple augmentation operations.

## Reproducibility

- Use `set_seed(...)` for global seeding.
- Use `seed_worker(...)` in `DataLoader(worker_init_fn=...)` for reproducible online augmentation. *(not tested yet!)*

## Offline generation

Use `augment_dataset_from_csv(...)` to generate new image files from the existing training CSV.
By default, the function performs repeated shuffled passes over the dataset:

- the first `N` generated images use each source row once, where `N` is the number of source rows;
- the next `N` generated images start a new shuffled pass;
- the pattern continues for `2N`, `3N`, and so on.

If you explicitly set `allow_reuse=True`, the function falls back to sampling with replacement.

## Online training *(not tested yet!)*

Use `build_online_train_transform(...)` and pass the result into the training dataset transform.
