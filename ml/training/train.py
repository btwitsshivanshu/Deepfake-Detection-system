"""
Training pipeline (Phase 5/6).

Architecture-agnostic trainer: works for ResNet50, EfficientNet-B0 and Xception
via the model factory. Features:
  - AdamW/SGD optimizers, cosine/step LR schedules
  - mixed-precision (AMP)
  - early stopping on val loss
  - best-checkpoint saving
  - per-epoch metric logging

Usage:
    python -m ml.training.train --arch xception --data datasets/hemg --epochs 25
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from ml.config import CONFIG, PATHS
from ml.dataset.dataset import build_dataloaders
from ml.models.factory import build_model
from ml.utils.logger import get_logger
from ml.utils.metrics import compute_metrics
from ml.utils.seed import set_seed, get_device


def _make_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                               momentum=0.9, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                             weight_decay=cfg.weight_decay)


def _make_scheduler(optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return None


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, y_true, y_pred, y_prob = 0.0, [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
        logits = logits.float()
        loss_sum += criterion(logits, labels).item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_prob.extend(probs.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.cpu().tolist())
    avg_loss = loss_sum / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return avg_loss, metrics


def train(arch: str, data_root: str, cfg=CONFIG, resume: str | None = None,
          ckpt_name: str | None = None, log_every: int = 200) -> Path:
    PATHS.ensure()
    logger = get_logger("train", PATHS.logs / f"train_{arch}.log")
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger.info("Device: %s | Architecture: %s", device, arch)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root, image_size=cfg.image_size, batch_size=cfg.batch_size,
        val_split=cfg.val_split, test_split=cfg.test_split,
        num_workers=cfg.num_workers, seed=cfg.seed,
        max_per_class=cfg.max_per_class,
    )

    logger.info("Data: %s | train=%d val=%d test=%d", data_root,
                len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    model = build_model(arch, num_classes=cfg.num_classes, pretrained=resume is None)
    if resume:
        state = torch.load(resume, map_location="cpu")
        state = state.get("model_state", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
        logger.info("Resumed weights from %s", resume)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

    best_val = float("inf")
    patience = 0
    ckpt_path = PATHS.checkpoints / (ckpt_name or f"{arch}_best.pth")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        n_batches = len(train_loader)
        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            if log_every and step % log_every == 0:
                logger.info("  epoch %02d step %d/%d loss=%.4f (%.0fs)",
                            epoch, step, n_batches, loss.item(), time.time() - t0)

        if scheduler:
            scheduler.step()

        train_loss = running / len(train_loader.dataset)
        val_loss, val_metrics = _evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **val_metrics.to_dict()})
        logger.info(
            "Epoch %02d/%d | train_loss=%.4f val_loss=%.4f acc=%.4f auc=%.4f f1=%.4f (%.1fs)",
            epoch, cfg.epochs, train_loss, val_loss,
            val_metrics.accuracy, val_metrics.roc_auc, val_metrics.f1, time.time() - t0,
        )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({"model_state": model.state_dict(), "arch": arch,
                        "epoch": epoch, "val_metrics": val_metrics.to_dict()}, ckpt_path)
            logger.info("  ↳ new best model saved to %s", ckpt_path)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Final test evaluation with best weights
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    _, test_metrics = _evaluate(model, test_loader, criterion, device)
    logger.info("TEST metrics: %s", test_metrics.to_dict())

    (PATHS.reports / f"{arch}_history.json").write_text(json.dumps(history, indent=2))
    (PATHS.reports / f"{arch}_test_metrics.json").write_text(
        json.dumps(test_metrics.to_dict(), indent=2))
    return ckpt_path


def main() -> None:
    p = argparse.ArgumentParser(description="Train a deepfake detector")
    p.add_argument("--arch", default=CONFIG.architecture,
                   choices=["resnet50", "efficientnet_b0", "xception"])
    p.add_argument("--data", default="datasets/hemg")
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    p.add_argument("--resume", default=None, help="checkpoint to start from")
    p.add_argument("--ckpt-name", default=None, help="output file in ml/checkpoints/")
    args = p.parse_args()

    CONFIG.epochs = args.epochs
    CONFIG.batch_size = args.batch_size
    CONFIG.learning_rate = args.lr
    train(args.arch, args.data, CONFIG, resume=args.resume, ckpt_name=args.ckpt_name)


if __name__ == "__main__":
    main()
