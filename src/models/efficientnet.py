import torch.nn as nn
from torchvision import models


def get_efficientnet_b0(freeze_backbone: bool = True) -> nn.Module:
    """EfficientNet-B0 pretrained on ImageNet with a single regression output.

    Parameters
    ----------
    freeze_backbone: if True, only the final classifier layer is trained initially.
                     Set to False to fine-tune the whole network.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # EfficientNet uses "classifier" instead of "fc"
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    return model