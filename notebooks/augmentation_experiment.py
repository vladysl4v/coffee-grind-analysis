import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from data_loader import get_loaders, DEFAULT_EVAL_TRANSFORM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def create_model():
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def val_loss(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

minimal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

default_transform = None

aggressive_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

configs = {
    "minimal": minimal_transform,
    "default": default_transform,
    "aggressive": aggressive_transform,
}

results = {}

for name, train_transform in configs.items():
    print(f"\n=== {name.upper()} ===")

    train_loader, val_loader, _ = get_loaders(
        batch_size=16,
        num_workers=0,
        train_transform=train_transform,
        eval_transform=DEFAULT_EVAL_TRANSFORM,
    )

    model = create_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val = float("inf")

    for epoch in range(20):
        train_l = train_epoch(model, train_loader, criterion, optimizer)
        val_l = val_loss(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/20 | Train: {train_l:.3f} | Val: {val_l:.3f}")
        best_val = min(best_val, val_l)

    results[name] = best_val

print("\n" + "=" * 50)
print("FINAL RESULTS (Lower = Better):")
for k, v in sorted(results.items(), key=lambda x: x[1]):
    print(f"{k:10}: {v:.3f}")
