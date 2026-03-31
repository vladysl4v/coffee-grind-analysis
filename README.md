# Coffee Grind Fineness Dataset

## Contributing

> [!CAUTION]
> **NO DIRECT PUSHES TO `main` — EVER.**
> **NO SELF-MERGES — ALWAYS GET A REVIEW FIRST.(@vladysl4v)**
> **NO MERGE WITHOUT TEAM DISCUSSION.**

- PRs must be focused and reasonably scoped — one feature or fix per PR.
- **Keep branches short-lived** — merge back to `main` (or your epic) as soon as the work is stable. Long-lived feature branches cause painful merges.

> **Note:** The first 3 commits to `main` were initial project scaffolding and are the only exception to the rules above.

### Branch naming

| Type | Pattern | Example |
|------|---------|---------|
| New feature | `feature/<description>` | `feature/resnet-backbone` |
| Chore / tooling | `chore/<description>` | `chore/update-dependencies` |
| Bug fix | `fix/<description>` | `fix/csv-decimal-parsing` |
| Experiment | `experiment/<description>` | `experiment/augmentation-tuning` |
| Long-running epic | `epic/<description>` | `epic/model-a-exploration` |

Keep branches short-lived — merge back to `main` (or your epic) as soon as the work is stable.

### Workflow

1. Branch off `main`, do your work, push your branch.
2. Open a Pull Request and assign **@vladysl4v** as reviewer *(temporary — review ownership may be revisited as the project matures, but kept this way for now to maintain a consistent process)*.
3. Wait for approval — **do not merge your own PR**.
4. Reviewer merges into `main` after approval.

### Epics — long-running work

If your work will diverge significantly for an extended period (e.g. a model exploration running in parallel with the rest of the team), create an `epic/<description>` branch and treat it as your local `main`:

- Working **with others**: branch off the epic with `feature/`, `fix/` etc. and open PRs into it — same review rules apply.
- Working **alone**: you may commit directly to the epic branch.

When the epic is complete, open a single PR from the epic branch into `main`.


## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv and install dependencies
uv sync

# run any script inside the venv
uv run python src/split_labels.py
```

To add a new dependency:

```bash
uv add torch torchvision
```

## Dataset

`data/labels/labels_train2.csv` — 648 labeled images, semicolon-delimited (`Sample;Fineness`).

Place the raw images in `data/images/` — the directory is tracked in git (via `.gitkeep`) but its contents are not.

## Train / Val / Test Split

**Script:** `src/split_labels.py`  
**Outputs:** `data/labels/labels_train.csv`, `labels_val.csv`, `labels_test.csv`

### Rationale

During data collection, photographers took **4 consecutive photos of the same physical grain sample** and assigned them the same fineness value. Splitting at the individual image level would leak images of the same grain across train/val/test, causing the model to appear better than it is on held-out data.

To prevent this, the split operates on **grain groups** (consecutive images sharing a fineness label) rather than individual rows:

| Split | Groups | Images |
|-------|--------|--------|
| Train | 130 (80%) | ~520 |
| Val   | 16 (10%)  | ~64  |
| Test  | 16 (10%)  | ~64  |

**Total:** 162 groups × 4 images = 648 images

Groups are shuffled with a fixed seed (`SEED = 42`) before splitting, so the split is reproducible. Each group is kept intact — all 4 photos of a grain go exclusively to one split.

## Data Loader

**Module:** `src/data_loader.py`  
**Requires:** `torch`, `torchvision`, `Pillow`, `pandas`

### Quickstart

```python
from data_loader import get_loaders

train_loader, val_loader, test_loader = get_loaders(batch_size=32)

for images, labels in train_loader:
    # images: FloatTensor [B, 3, 224, 224], ImageNet-normalised
    # labels: FloatTensor [B]  — fineness value (e.g. 47.2)
    ...
```

### `get_loaders()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `32` | Images per batch |
| `num_workers` | `4` | DataLoader worker processes |
| `train_transform` | see below | Override train augmentation pipeline |
| `eval_transform` | see below | Override val/test pipeline |

### Default transforms

**Train** — `RandomResizedCrop(224)` → random horizontal/vertical flip → `ColorJitter` → `ToTensor` → ImageNet normalisation

**Val / Test** — `Resize(256)` → `CenterCrop(224)` → `ToTensor` → ImageNet normalisation

### Using `CoffeeDataset` directly

```python
from data_loader import CoffeeDataset, DEFAULT_EVAL_TRANSFORM
from torch.utils.data import DataLoader

ds = CoffeeDataset("test", transform=DEFAULT_EVAL_TRANSFORM)
print(len(ds))          # number of images in the split
image, label = ds[0]    # PIL Image (no transform) or Tensor (with transform)

loader = DataLoader(ds, batch_size=16, shuffle=False)
```

### Generating the split CSVs

Before first use, run the split script once:

```bash
python src/split_labels.py
```

This reads `data/labels/labels_train2.csv` and writes `labels_train.csv`, `labels_val.csv`, `labels_test.csv` into the same directory.
