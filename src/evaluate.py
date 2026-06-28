"""
Evaluate a saved ConvNeXt-Small model on the real val set.

Usage:
    cd Q:/coffee-grind-analysis
    uv run python src/evaluate.py run_025
    uv run python src/evaluate.py run_025 run_043 run_045
    uv run python src/evaluate.py data/runs/convnext_small/run_025/best_model.pt
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models.convnext import get_convnext_small

_ROOT       = Path(__file__).parent.parent
_IMAGES_DIR = _ROOT / "data" / "images" / "raw"
_LABELS_DIR = _ROOT / "data" / "labels"

TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1557, 0.0899, 0.0404],
                         std =[0.0483, 0.0349, 0.0190]),
])


class ValDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(_IMAGES_DIR / fname).convert('RGB')
        return TRANSFORM(img), torch.tensor(label, dtype=torch.float32)


def load_csv(path):
    df = pd.read_csv(path, sep=';', decimal=',')
    return [(row.iloc[0], float(row.iloc[1]) / 100.0) for _, row in df.iterrows()]


def evaluate(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_convnext_small(freeze_backbone=False)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()

    samples = load_csv(_LABELS_DIR / 'val.csv')
    loader  = DataLoader(ValDataset(samples), batch_size=32, shuffle=False,
                         num_workers=0, pin_memory=True)

    preds_all, gt_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds_all.append(model(imgs).view(-1).cpu().numpy())
            gt_all.append(labels.numpy())

    preds = np.concatenate(preds_all) * 100
    gt    = np.concatenate(gt_all)    * 100
    mae   = np.abs(preds - gt).mean()
    rmse  = np.sqrt(((preds - gt) ** 2).mean())
    return mae, rmse


def resolve_path(arg):
    p = Path(arg)
    if p.suffix == '.pt':
        return p
    # treat as run name or partial path
    run_dir = _ROOT / "data" / "runs" / "convnext_small" / p.name
    if not run_dir.exists():
        run_dir = p  # maybe absolute
    best = run_dir / "best_model.pt"
    if not best.exists():
        raise FileNotFoundError(f"No best_model.pt found in {run_dir}")
    return best


def main():
    args = sys.argv[1:] or ["run_025"]
    print(f"{'Run':<30}  {'Val MAE':>8}  {'Val RMSE':>9}")
    print("-" * 52)
    for arg in args:
        path = resolve_path(arg)
        mae, rmse = evaluate(path)
        print(f"{path.parent.name:<30}  {mae:>7.2f}%  {rmse:>8.2f}%")


if __name__ == '__main__':
    main()
