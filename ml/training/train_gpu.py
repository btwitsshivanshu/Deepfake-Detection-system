"""
GPU training launcher (CUDA).

Trains one architecture with GPU-appropriate settings: AMP on, a batch size
that fits ~6 GB VRAM, cosine schedule, early stopping. Saves the best
checkpoint to ml/checkpoints/{arch}_best.pth (the path the seeded DB expects).

Usage:
    python -m ml.training.train_gpu --arch xception --epochs 15
    python -m ml.training.train_gpu --arch resnet50 --epochs 15
    python -m ml.training.train_gpu --arch efficientnet_b0 --epochs 15
"""
from __future__ import annotations

import argparse

from ml.config import CONFIG
from ml.training.train import train


def main() -> None:
    p = argparse.ArgumentParser(description="GPU training launcher")
    p.add_argument("--arch", default="xception",
                   choices=["resnet50", "efficientnet_b0", "xception"])
    p.add_argument("--data", default="datasets/processed")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers (0 is safest on Windows)")
    args = p.parse_args()

    CONFIG.device = "cuda"
    CONFIG.mixed_precision = True
    CONFIG.num_workers = args.workers
    CONFIG.epochs = args.epochs
    CONFIG.batch_size = args.batch_size
    CONFIG.learning_rate = args.lr
    CONFIG.early_stopping_patience = 4

    ckpt = train(args.arch, args.data, CONFIG)
    print(f"\nSaved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
