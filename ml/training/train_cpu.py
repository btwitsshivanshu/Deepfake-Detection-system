"""
CPU-friendly training launcher for the small HF sample.

Overrides a few defaults that matter on a CPU-only box with a small dataset:
  - num_workers = 0  (Windows + small data: spawning workers costs more than it saves)
  - mixed_precision off (CPU has no AMP benefit)
  - fewer epochs / smaller batch by default

Usage:
    python -m ml.training.train_cpu --arch xception --epochs 8
"""
from __future__ import annotations

import argparse

from ml.config import CONFIG
from ml.training.train import train


def main() -> None:
    p = argparse.ArgumentParser(description="CPU training launcher")
    p.add_argument("--arch", default="xception",
                   choices=["resnet50", "efficientnet_b0", "xception"])
    p.add_argument("--data", default="datasets/hemg")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    CONFIG.device = "cpu"
    CONFIG.num_workers = 0
    CONFIG.pin_memory = False
    CONFIG.mixed_precision = False
    CONFIG.epochs = args.epochs
    CONFIG.batch_size = args.batch_size
    CONFIG.learning_rate = args.lr
    CONFIG.early_stopping_patience = 3

    ckpt = train(args.arch, args.data, CONFIG)
    print(f"\nSaved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
