import torch.nn as nn
from torchvision import models


def get_resnext50(freeze_backbone: bool = True) -> nn.Module:
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
