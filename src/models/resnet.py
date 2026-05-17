import torch.nn as nn
from torchvision import models


def get_resnet152(freeze_backbone: bool = True) -> nn.Module:
    """ResNet152 pretrained on ImageNet with a single regression output.

    Parameters
    ----------
    freeze_backbone: if True, only the final FC layer is trained initially.
                     Set to False to fine-tune the whole network.
    """
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)

    return model

def get_resnet18(freeze_backbone: bool = True) -> nn.Module:
    """ResNet18 pretrained on ImageNet with a single regression output.

    Parameters
    ----------
    freeze_backbone: if True, only the final FC layer is trained initially.
                     Set to False to fine-tune the whole network.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)

    return model