import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import CoffeeDataset, DEFAULT_EVAL_TRANSFORM
from models.swin import get_swin_s

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt      = Path(__file__).parent.parent / "data/runs/swin_s/run_003/best_model.pt"

model = get_swin_s(freeze_backbone=False)
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.to(device).eval()

loader = torch.utils.data.DataLoader(
    CoffeeDataset("val", transform=DEFAULT_EVAL_TRANSFORM),
    batch_size=32, shuffle=False, num_workers=0
)

gts, preds = [], []
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        out = model(images).view(-1).float().cpu() * 100
        preds.append(out); gts.append(labels * 100)

gts   = torch.cat(gts).numpy()
preds = torch.cat(preds).numpy()
errors = preds - gts

print(f"Val MAE:  {np.abs(errors).mean():.2f}%")
print(f"Bias:     {errors.mean():+.2f}%")
print(f"RMSE:     {np.sqrt((errors**2).mean()):.2f}%")
print(f"\n{'─'*50}")
print(f"{'#':>3}  {'Ground Truth':>12}  {'Predicted':>10}  {'Error':>8}")
print(f"{'─'*50}")
for i, (gt, pred, err) in enumerate(zip(gts, preds, errors), 1):
    print(f"{i:>3}  {gt:>12.2f}  {pred:>10.2f}  {err:>+8.2f}")
print(f"{'─'*50}")
