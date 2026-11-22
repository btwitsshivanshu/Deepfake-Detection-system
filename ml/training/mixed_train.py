"""
Mixed-dataset fine-tuning: Hemg + 140k Real-and-Fake-Faces (Kaggle "archive").

- Hemg val/test are reproduced exactly from the earlier capped run (unseen).
- Archive uses its own train/valid/test folders.
- Validates every N training images on a val subset of EACH dataset (reported
  separately); best checkpoint = lowest mean val loss across the two.
- Tests once at the end on Hemg test + the full archive test split.

Usage:
    python -m ml.training.mixed_train --resume ml/checkpoints/xception_best.pth
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ml.config import CONFIG, PATHS
from ml.dataset.dataset import DeepfakeImageDataset, _gather_samples, previous_split
from ml.dataset.transforms import eval_transforms, train_transforms
from ml.models.factory import build_model
from ml.training.train import _evaluate
from ml.utils.logger import get_logger
from ml.utils.seed import get_device, set_seed


def _balanced_sample(samples, n, seed):
    """Take n samples, half per class (or all if fewer)."""
    rng = random.Random(seed)
    out = []
    for cls in sorted({l for _, l in samples}):
        group = [s for s in samples if s[1] == cls]
        rng.shuffle(group)
        out.extend(group[: n // 2])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Hemg + archive mixed fine-tuning")
    p.add_argument("--arch", default="xception")
    p.add_argument("--hemg", default="datasets/hemg")
    p.add_argument("--archive", default="D:/archive/real_vs_fake/real-vs-fake")
    p.add_argument("--resume", required=True)
    p.add_argument("--hemg-train", type=int, default=100000)
    p.add_argument("--archive-train", type=int, default=100000)
    p.add_argument("--val-per-set", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--ckpt-name", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    args = p.parse_args()

    cfg = CONFIG
    PATHS.ensure()
    logger = get_logger("mixed", PATHS.logs / f"mixed_{args.arch}.log")
    set_seed(cfg.seed)
    device = get_device("cuda")

    # --- Hemg: reuse earlier held-out val/test, train on the rest
    hemg_all = _gather_samples(args.hemg)
    h_val, h_test = previous_split(hemg_all, 15000, cfg.val_split, cfg.test_split, cfg.seed)
    held = {str(s[0]) for s in h_val} | {str(s[0]) for s in h_test}
    h_train = _balanced_sample([s for s in hemg_all if str(s[0]) not in held],
                               args.hemg_train, cfg.seed)
    # --- Archive: official splits
    a_train = _balanced_sample(_gather_samples(Path(args.archive, "train")),
                               args.archive_train, cfg.seed)
    a_val = _gather_samples(Path(args.archive, "valid"))
    a_test = _gather_samples(Path(args.archive, "test"))

    h_val_s = _balanced_sample(h_val, args.val_per_set, cfg.seed)
    a_val_s = _balanced_sample(a_val, args.val_per_set, cfg.seed)
    train = h_train + a_train
    logger.info("Data: train=%d (hemg %d + archive %d) | val hemg=%d archive=%d | "
                "test hemg=%d archive=%d", len(train), len(h_train), len(a_train),
                len(h_val_s), len(a_val_s), len(h_test), len(a_test))

    common = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    ev = eval_transforms(cfg.image_size)
    train_loader = DataLoader(DeepfakeImageDataset(train, train_transforms(cfg.image_size)),
                              shuffle=True, drop_last=True, **common)
    mk = lambda s: DataLoader(DeepfakeImageDataset(s, ev), shuffle=False, **common)
    val_loaders = {"hemg": mk(h_val_s), "archive": mk(a_val_s)}
    test_loaders = {"hemg": mk(h_test), "archive": mk(a_test)}

    model = build_model(args.arch, num_classes=cfg.num_classes, pretrained=False)
    state = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(state.get("model_state", state))
    model = model.to(device)
    logger.info("Resumed weights from %s", args.resume)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = GradScaler(enabled=True)

    ckpt_path = PATHS.checkpoints / (args.ckpt_name or f"{args.arch}_mixed_best.pth")
    best_val, history = float("inf"), []
    seen, next_eval, t0 = 0, args.eval_every, time.time()
    running, running_n = 0.0, 0
    total_imgs = total_steps * args.batch_size

    model.train()
    for step, (images, labels) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        seen += images.size(0)
        running += loss.item() * images.size(0)
        running_n += images.size(0)

        if seen >= next_eval or step == total_steps:
            next_eval += args.eval_every
            res = {k: _evaluate(model, dl, criterion, device) for k, dl in val_loaders.items()}
            val_loss = sum(r[0] for r in res.values()) / len(res)
            row = {"images_seen": seen, "train_loss": running / max(running_n, 1),
                   "val_loss": val_loss}
            for k, (l, m) in res.items():
                row[k] = {"val_loss": l, **m.to_dict()}
            history.append(row)
            running, running_n = 0.0, 0
            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state": model.state_dict(), "arch": args.arch,
                            "images_seen": seen,
                            "val_metrics": {k: m.to_dict() for k, (_, m) in res.items()}},
                           ckpt_path)
                tag = "  [best, saved]"
            h, a = res["hemg"][1], res["archive"][1]
            logger.info("CHECK %d/%d imgs | val_loss=%.4f | hemg acc=%.4f auc=%.4f | "
                        "archive acc=%.4f auc=%.4f (%.0fs)%s", seen, total_imgs, val_loss,
                        h.accuracy, h.roc_auc, a.accuracy, a.roc_auc, time.time() - t0, tag)
            model.train()

    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    out = {}
    for k, dl in test_loaders.items():
        _, m = _evaluate(model, dl, criterion, device)
        out[k] = m.to_dict()
        logger.info("TEST %s: %s", k, m.to_dict())
    (PATHS.reports / f"{args.arch}_mixed_history.json").write_text(json.dumps(history, indent=2))
    (PATHS.reports / f"{args.arch}_mixed_test_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
