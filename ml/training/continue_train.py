"""
Continue training a checkpoint on the FULL dataset, validating only every N
training images and testing once at the end.

The val/test sets are reproduced exactly from the earlier capped run
(same seed + same max_per_class), so they stay unseen and results stay
comparable. Every other image goes into training.

Usage:
    python -m ml.training.continue_train --arch xception --data datasets/hemg \
        --resume ml/checkpoints/xception_hemg_best.pth \
        --prev-max-per-class 15000 --eval-every 10000 --passes 1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ml.dataset.dataset import previous_split as _previous_split

from ml.config import CONFIG, PATHS
from ml.dataset.dataset import DeepfakeImageDataset, _gather_samples
from ml.dataset.transforms import eval_transforms, train_transforms
from ml.models.factory import build_model
from ml.training.train import _evaluate
from ml.utils.logger import get_logger
from ml.utils.seed import get_device, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Continue training on the full dataset")
    p.add_argument("--arch", default="xception")
    p.add_argument("--data", default="datasets/hemg")
    p.add_argument("--resume", required=True)
    p.add_argument("--prev-max-per-class", type=int, default=15000,
                   help="max_per_class of the run whose val/test sets to reuse")
    p.add_argument("--ckpt-name", default=None)
    p.add_argument("--eval-every", type=int, default=10000, help="train images between val checks")
    p.add_argument("--passes", type=float, default=1.0, help="passes over the training set")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    args = p.parse_args()

    cfg = CONFIG
    PATHS.ensure()
    logger = get_logger("continue", PATHS.logs / f"continue_{args.arch}.log")
    set_seed(cfg.seed)
    device = get_device("cuda")

    all_samples = _gather_samples(args.data)
    val, test = _previous_split(all_samples, args.prev_max_per_class,
                                cfg.val_split, cfg.test_split, cfg.seed)
    held_out = {str(s[0]) for s in val} | {str(s[0]) for s in test}
    train = [s for s in all_samples if str(s[0]) not in held_out]
    logger.info("Data: train=%d val=%d test=%d (val/test reused from previous run)",
                len(train), len(val), len(test))

    common = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(DeepfakeImageDataset(train, train_transforms(cfg.image_size)),
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(DeepfakeImageDataset(val, eval_transforms(cfg.image_size)),
                            shuffle=False, **common)
    test_loader = DataLoader(DeepfakeImageDataset(test, eval_transforms(cfg.image_size)),
                             shuffle=False, **common)

    model = build_model(args.arch, num_classes=cfg.num_classes, pretrained=False)
    state = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(state.get("model_state", state))
    model = model.to(device)
    logger.info("Resumed weights from %s", args.resume)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=cfg.weight_decay)
    total_steps = math.ceil(len(train_loader) * args.passes)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = GradScaler(enabled=True)

    ckpt_path = PATHS.checkpoints / (args.ckpt_name or f"{args.arch}_hemg_full_best.pth")
    best_val, history = float("inf"), []
    seen, next_eval, step, t0 = 0, args.eval_every, 0, time.time()
    running, running_n = 0.0, 0

    model.train()
    while step < total_steps:
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            seen += images.size(0)
            running += loss.item() * images.size(0)
            running_n += images.size(0)

            if seen >= next_eval or step >= total_steps:
                next_eval += args.eval_every
                val_loss, m = _evaluate(model, val_loader, criterion, device)
                train_loss = running / max(running_n, 1)
                running, running_n = 0.0, 0
                history.append({"images_seen": seen, "train_loss": train_loss,
                                "val_loss": val_loss, **m.to_dict()})
                tag = ""
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model_state": model.state_dict(), "arch": args.arch,
                                "images_seen": seen, "val_metrics": m.to_dict()}, ckpt_path)
                    tag = "  [best, saved]"
                logger.info("CHECK %d/%d imgs | train_loss=%.4f val_loss=%.4f acc=%.4f "
                            "auc=%.4f f1=%.4f (%.0fs)%s", seen, total_steps * args.batch_size,
                            train_loss, val_loss, m.accuracy, m.roc_auc, m.f1,
                            time.time() - t0, tag)
                model.train()
            if step >= total_steps:
                break

    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    _, test_m = _evaluate(model, test_loader, criterion, device)
    logger.info("TEST metrics: %s", test_m.to_dict())
    (PATHS.reports / f"{args.arch}_continue_history.json").write_text(json.dumps(history, indent=2))
    (PATHS.reports / f"{args.arch}_continue_test_metrics.json").write_text(
        json.dumps(test_m.to_dict(), indent=2))
    print(f"\nSaved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
