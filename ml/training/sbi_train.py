"""
Train the face-swap detector with Self-Blended Images (SBI).

Training data = REAL faces only (Hemg train pool + 140k-faces train/real).
Each real face yields a pair: (real, SBI pseudo-fake). Both then get the same
kind of random "real-world" degradation (JPEG, downscale, blur, noise) so the
model survives WhatsApp / screenshot quality.

Held-out Hemg val/test images are excluded so later tests stay clean.
Face boxes + landmarks (YuNet) are cached on first run.

Usage:
    python -m ml.training.sbi_train --per-source 30000 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from ml.config import CONFIG, PATHS
from ml.dataset.dataset import _gather_samples, previous_split
from ml.dataset.face_extraction import detect_largest_face, load_image_rgb
from ml.dataset.sbi import self_blend
from ml.dataset.transforms import IMAGENET_MEAN, IMAGENET_STD
from ml.models.factory import build_model
from ml.utils.logger import get_logger
from ml.utils.metrics import compute_metrics
from ml.utils.seed import get_device, set_seed

CACHE = Path("datasets/sbi_cache.json")


def degrade(size: int) -> A.Compose:
    """Random real-world degradation, applied independently to real & fake."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-8, 8), p=0.4),
        A.ImageCompression(quality_range=(30, 95), p=0.6),
        A.Downscale(scale_range=(0.4, 0.9), p=0.25),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.1),
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def plain(size: int) -> A.Compose:
    return A.Compose([A.Resize(size, size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class SBIPairs(Dataset):
    def __init__(self, items, transform, fixed_seed: int | None = None):
        self.items, self.t, self.fixed_seed = items, transform, fixed_seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, box, lm = self.items[i]
        img = load_image_rgb(path)
        rng = (np.random.default_rng(self.fixed_seed + i) if self.fixed_seed is not None
               else np.random.default_rng())
        lm = np.array(lm, np.float32) if lm is not None else None
        fake = self_blend(img, box, lm, rng)
        return self.t(image=img)["image"], self.t(image=fake)["image"]


def build_cache(paths, logger):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [p for p in paths if str(p) not in cache]
    logger.info("Landmark cache: %d cached, %d to detect", len(cache), len(todo))
    t0 = time.time()
    for k, p in enumerate(todo, 1):
        try:
            box, lm = detect_largest_face(load_image_rgb(p))
        except Exception:
            box, lm = None, None
        cache[str(p)] = None if box is None else [list(box), lm.tolist() if lm is not None else None]
        if k % 5000 == 0 or k == len(todo):
            CACHE.write_text(json.dumps(cache))
            logger.info("  landmarks %d/%d (%.0fs)", k, len(todo), time.time() - t0)
    return cache


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, n, y_true, y_prob = 0.0, 0, [], []
    for real, fake in loader:
        x = torch.cat([real, fake]).to(device)
        y = torch.cat([torch.zeros(len(real)), torch.ones(len(fake))]).long().to(device)
        with autocast(enabled=device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        loss_sum += criterion(logits, y).item() * len(y)
        n += len(y)
        y_prob += torch.softmax(logits, 1)[:, 1].cpu().tolist()
        y_true += y.cpu().tolist()
    y_pred = [int(p > 0.5) for p in y_prob]
    return loss_sum / n, compute_metrics(y_true, y_pred, y_prob)


def main() -> None:
    p = argparse.ArgumentParser(description="SBI face-swap detector training")
    p.add_argument("--arch", default="efficientnet_b4")
    p.add_argument("--hemg", default="datasets/hemg")
    p.add_argument("--archive", default="D:/archive/real_vs_fake/real-vs-fake/train")
    p.add_argument("--per-source", type=int, default=30000, help="real faces per dataset")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--pairs-per-batch", type=int, default=12)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=10000, help="pairs between checks")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--ckpt-name", default="efficientnet_b4_sbi_best.pth")
    args = p.parse_args()

    PATHS.ensure()
    logger = get_logger("sbi", PATHS.logs / "sbi_train.log")
    set_seed(CONFIG.seed)
    device = get_device("cuda")
    rng = random.Random(CONFIG.seed)

    # --- real faces only (exclude Hemg held-out val/test)
    hemg_all = _gather_samples(args.hemg)
    h_val, h_test = previous_split(hemg_all, 15000, CONFIG.val_split, CONFIG.test_split, CONFIG.seed)
    held = {str(s[0]) for s in h_val} | {str(s[0]) for s in h_test}
    hemg_real = [s[0] for s in hemg_all if s[1] == 0 and str(s[0]) not in held]
    arch_real = sorted(Path(args.archive, "real").iterdir())
    rng.shuffle(hemg_real); rng.shuffle(arch_real)
    paths = hemg_real[: args.per_source] + arch_real[: args.per_source]

    cache = build_cache(paths, logger)
    items = [(str(p), *cache[str(p)]) for p in paths if cache.get(str(p))]
    logger.info("Faces with a detected face: %d / %d", len(items), len(paths))
    rng.shuffle(items)
    n_val = min(max(500, len(items) // 40), len(items) // 5)
    val_items, train_items = items[:n_val], items[n_val:]

    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(SBIPairs(train_items, degrade(args.size)),
                              batch_size=args.pairs_per_batch, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(SBIPairs(val_items, plain(args.size), fixed_seed=12345),
                            batch_size=args.pairs_per_batch * 2, shuffle=False, **common)
    logger.info("Pairs: train=%d val=%d | %d epochs | input %dpx | batch %d imgs",
                len(train_items), len(val_items), args.epochs, args.size, args.pairs_per_batch * 2)

    model = build_model(args.arch, num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                                                pct_start=0.05)
    scaler = GradScaler(enabled=True)
    ckpt = PATHS.checkpoints / args.ckpt_name
    best, seen, next_eval, t0, history = float("inf"), 0, args.eval_every, time.time(), []
    total_pairs = len(train_loader) * args.pairs_per_batch * args.epochs

    for epoch in range(1, args.epochs + 1):
        model.train()
        run, run_n = 0.0, 0
        for real, fake in train_loader:
            x = torch.cat([real, fake]).to(device, non_blocking=True)
            y = torch.cat([torch.zeros(len(real)), torch.ones(len(fake))]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            seen += len(real)
            run += loss.item() * len(y); run_n += len(y)
            if seen >= next_eval:
                next_eval += args.eval_every
                vl, m = evaluate(model, val_loader, criterion, device)
                tag = ""
                if vl < best:
                    best = vl
                    torch.save({"model_state": model.state_dict(), "arch": args.arch,
                                "image_size": args.size, "pairs_seen": seen,
                                "val_metrics": m.to_dict()}, ckpt)
                    tag = "  [best, saved]"
                history.append({"pairs_seen": seen, "train_loss": run / run_n, "val_loss": vl,
                                **m.to_dict()})
                logger.info("CHECK %d/%d pairs (epoch %d) | train_loss=%.4f val_loss=%.4f "
                            "acc=%.4f auc=%.4f (%.0fs)%s", seen, total_pairs, epoch,
                            run / run_n, vl, m.accuracy, m.roc_auc, time.time() - t0, tag)
                run, run_n = 0.0, 0
                model.train()

    (PATHS.reports / "sbi_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
