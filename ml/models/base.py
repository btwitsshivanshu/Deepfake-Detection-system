"""Abstract base for all deepfake detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDetector(nn.Module, ABC):
    """
    Common interface for every detector so training/inference code is
    architecture-agnostic (Open/Closed & Liskov Substitution — SOLID).
    """

    architecture: str = "base"

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return raw logits of shape (B, num_classes)."""
        raise NotImplementedError

    @abstractmethod
    def target_layer(self) -> nn.Module:
        """Return the conv layer Grad-CAM should hook (last spatial feature map)."""
        raise NotImplementedError

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return torch.softmax(self.forward(x), dim=1)
