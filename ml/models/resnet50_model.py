"""ResNet50-based deepfake detector."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from ml.models.base import BaseDetector


class ResNet50Detector(BaseDetector):
    architecture = "resnet50"

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        # Replace the classifier head for binary real/fake with light regularization.
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def target_layer(self) -> nn.Module:
        # Last residual block's final conv feature map.
        return self.backbone.layer4[-1]
