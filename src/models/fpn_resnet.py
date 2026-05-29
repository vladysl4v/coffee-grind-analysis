from torch import nn
import torch
import timm
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class FPNResnet18(nn.Module):
    def __init__(self, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_small",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.backbone.feature_info.channels(),
            out_channels=256
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.w = nn.Parameter(torch.ones(1, len(self.backbone.feature_info.channels())))

    def forward(self, x):
        features = self.backbone(x)
        features = {str(i): f for i, f in enumerate(features)}
        fpn_feats = self.fpn(features)
        pooled = []
        for k in sorted(fpn_feats.keys()):
            pooled.append(self.pool(fpn_feats[k]).flatten(1))

        #x = torch.stack(pooled, dim=1) * torch.softmax(self.w, dim=1).unsqueeze(-1)
        return self.head(pooled[0])