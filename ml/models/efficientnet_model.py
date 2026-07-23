"""EfficientNet-B0 deepfake detector (via timm)."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from ml.models.base import BaseDetector


class EfficientNetDetector(BaseDetector):
    architecture = "efficientnet_b0"

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def target_layer(self) -> nn.Module:
        # Final conv block before pooling.
        return self.backbone.conv_head
