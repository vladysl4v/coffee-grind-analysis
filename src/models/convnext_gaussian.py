import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvNeXtSmallGaussian(nn.Module):
    """ConvNeXt-Small backbone with separate mu and log_var heads.

    Separate linear layers prevent gradients for uncertainty from corrupting
    the mean prediction, which stabilises MAE during training.

    forward() returns raw (mu, log_var).
    Variance is recovered via softplus(log_var) before passing to GaussianNLLLoss.
    predict() returns (mu, sigma) for inference.
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        base = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        if freeze_backbone:
            for param in base.parameters():
                param.requires_grad = False
        in_features = base.classifier[2].in_features
        base.classifier[2] = nn.Identity()
        self.backbone = base
        self.mu_head      = nn.Linear(in_features, 1)
        self.log_var_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        mu      = self.mu_head(features).squeeze(1)
        log_var = self.log_var_head(features).squeeze(1)
        return mu, log_var

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, sigma) in label space."""
        mu, log_var = self.forward(x)
        return mu, F.softplus(log_var)
