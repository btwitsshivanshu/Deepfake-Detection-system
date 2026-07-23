"""XceptionNet deepfake detector (via timm).

Xception is the de-facto strong baseline for FaceForensics++ style
deepfake detection thanks to its depthwise-separable convolutions.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from ml.models.base import BaseDetector


class XceptionDetector(BaseDetector):
    architecture = "xception"

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def target_layer(self) -> nn.Module:
        # Use the final batch-norm feature map (`bn4`) rather than the following
        # activation: timm's xception uses in-place ReLUs, and registering a
        # backward hook on an in-place op breaks autograd (view+inplace).
        return self.backbone.bn4
