import torch.nn as nn
from torchvision import models


def get_model(freeze_backbone: bool = True) -> nn.Module:
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)
    return model
