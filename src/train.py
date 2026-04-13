import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_loaders, _NORMALIZE
from models.simple_cnn import SimpleCNN
from torchvision import transforms
from pathlib import Path


BASE_DIR = Path("data/statistics/train/cnn")

run_versions = sorted(BASE_DIR.glob("run_*"))
run_id = len(run_versions) + 1

RUN_DIR = BASE_DIR / f"run_{run_id:03d}"
MODELS_DIR = RUN_DIR / "models"
GRAPHS_DIR = RUN_DIR / "graphs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NO_AUG_TRAIN = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])

train_loader, val_loader, test_loader = get_loaders(
    train_transform=NO_AUG_TRAIN
)


model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch():
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)



def evaluate(loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


def save_loss_plot(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "loss_curve.png")
    plt.close()


def save_scatter(epoch):
    model.eval()

    preds_list = []
    gt_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            preds_list.append(preds.cpu().numpy())
            gt_list.append(labels.cpu().numpy())

    preds = np.concatenate(preds_list)
    gt = np.concatenate(gt_list)

    mse = np.mean((preds - gt) ** 2)

    plt.figure()
    plt.scatter(gt, preds, alpha=0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(f"GT vs Pred (epoch {epoch})")

    plt.text(
        0.05, 0.95,
        f"MSE: {mse:.4f}",
        transform=plt.gca().transAxes
    )

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f"scatter_epoch_{epoch}.png")
    plt.close()


train_losses = []
val_losses = []

for epoch in range(50):

    train_loss = train_one_epoch()
    val_loss = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    save_loss_plot(train_losses, val_losses)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), MODELS_DIR / f"cnn_epoch_{epoch}.pt")
        save_scatter(epoch)
