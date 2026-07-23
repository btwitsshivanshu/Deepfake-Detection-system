"""
Central configuration for the ML pipeline.

All paths, hyper-parameters and model settings live here so training,
evaluation and inference stay consistent and reproducible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


@dataclass
class Paths:
    data_dir: Path = PROJECT_ROOT / "datasets"
    checkpoints: Path = ROOT / "checkpoints"
    logs: Path = ROOT / "logs"
    reports: Path = ROOT / "reports"

    def ensure(self) -> None:
        for p in (self.data_dir, self.checkpoints, self.logs, self.reports):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    architecture: str = "xception"          # resnet50 | efficientnet_b0 | xception
    num_classes: int = 2                    # real / fake
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"                # adam | adamw | sgd
    scheduler: str = "cosine"               # cosine | step | none
    early_stopping_patience: int = 5
    label_smoothing: float = 0.05
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    seed: int = 42
    device: str = os.getenv("DEVICE", "cuda")

    # data splitting
    val_split: float = 0.15
    test_split: float = 0.15

    # video inference
    video_frames: int = 20                  # frames sampled per video
    frame_agg: str = "mean"                 # mean | max | vote


PATHS = Paths()
CONFIG = TrainConfig()

# Class index mapping (kept explicit and stable everywhere)
CLASS_NAMES = ["real", "fake"]
CLASS_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_CLASS = {0: "real", 1: "fake"}
