"""EfficientNet-B4 (Noisy-Student weights) — backbone used by the Self-Blended
Images (SBI) face-swap detector (Shiohara & Yamasaki, CVPR 2022)."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from ml.models.base import BaseDetector


class EfficientNetB4Detector(BaseDetector):
    architecture = "efficientnet_b4"

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.backbone = timm.create_model(
            "tf_efficientnet_b4.ns_jft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def target_layer(self) -> nn.Module:
        return self.backbone.conv_head
