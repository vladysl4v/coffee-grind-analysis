"""
Run a trained ConvNeXt-Small model on one or more images.

Usage:
    python inference.py path/to/model.pt path/to/image.jpg
    python inference.py path/to/model.pt path/to/folder/
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import get_model

CROP = 420
MEAN = [0.1557, 0.0899, 0.0404]
STD  = [0.0483, 0.0349, 0.0190]

_transform = transforms.Compose([
    transforms.CenterCrop(CROP),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_model(model_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = get_model(freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device).eval()
    return model


def predict(model: torch.nn.Module, image_path: str | Path, device: torch.device) -> float:
    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).item()
    return round(pred * 100, 2)


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model.pt> <image_or_folder>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    target     = Path(sys.argv[2])
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)

    if target.is_dir():
        images = [p for p in sorted(target.iterdir()) if p.suffix.lower() in IMG_EXTENSIONS]
    else:
        images = [target]

    for img_path in images:
        fineness = predict(model, img_path, device)
        print(f"{img_path.name}: {fineness:.2f}%")


if __name__ == "__main__":
    main()
