"""
Model factory — the single entry point for creating any detector.

Keeps the rest of the codebase decoupled from concrete classes
(Dependency Inversion). Add a new architecture in one place.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ml.models.base import BaseDetector

# Project root (…/Deepfake_finalyear) — used to resolve relative weights paths
# so inference works regardless of the process working directory (e.g. the
# backend launches from backend/, but weights_path values are root-relative).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
from ml.models.resnet50_model import ResNet50Detector
from ml.models.efficientnet_model import EfficientNetDetector
from ml.models.xception_model import XceptionDetector
from ml.models.efficientnet_b4_model import EfficientNetB4Detector

_REGISTRY: dict[str, type[BaseDetector]] = {
    "resnet50": ResNet50Detector,
    "efficientnet_b0": EfficientNetDetector,
    "xception": XceptionDetector,
    "efficientnet_b4": EfficientNetB4Detector,
}


def available_models() -> list[str]:
    return list(_REGISTRY.keys())


def build_model(
    architecture: str,
    num_classes: int = 2,
    pretrained: bool = True,
) -> BaseDetector:
    key = architecture.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {available_models()}"
        )
    return _REGISTRY[key](num_classes=num_classes, pretrained=pretrained)


def load_model(
    architecture: str,
    weights_path: str,
    num_classes: int = 2,
    device: str | torch.device = "cpu",
) -> BaseDetector:
    """Build a model and load trained weights for inference."""
    model = build_model(architecture, num_classes=num_classes, pretrained=False)
    path = Path(weights_path)
    if not path.is_absolute() and not path.exists():
        path = _PROJECT_ROOT / weights_path
    state = torch.load(path, map_location=device)
    # Support both raw state_dict and {"model_state": ...} checkpoints.
    state = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.to(device).eval()
    return model
