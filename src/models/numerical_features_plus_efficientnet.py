import torch
import torch.nn as nn
from torchvision import models


class NumericalFeaturesPlusEfficientNetB0(nn.Module):
    def __init__(self, num_features: int, freeze_backbone: bool = True):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        backbone_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.feature_branch = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(backbone_dim + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        image_embedding = self.backbone(image)
        feature_embedding = self.feature_branch(features)
        fused = torch.cat((image_embedding, feature_embedding), dim=1)
        return self.regression_head(fused).squeeze(1)
