"""
Evaluation & Results module (Phase 10).

Loads a trained checkpoint, runs it over the test split and produces:
  - metrics JSON (accuracy, precision, recall, F1, ROC-AUC)
  - confusion-matrix plot
  - ROC curve plot

Usage:
    python -m ml.evaluation.evaluate --arch xception \
        --weights ml/checkpoints/xception_best.pth --data datasets/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from ml.config import CONFIG, PATHS, CLASS_NAMES
from ml.dataset.dataset import build_dataloaders
from ml.models.factory import load_model
from ml.utils.logger import get_logger
from ml.utils.metrics import compute_metrics
from ml.utils.seed import get_device

logger = get_logger("evaluate")


@torch.no_grad()
def _collect(model, loader, device):
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_prob.extend(probs.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_confusion(cm, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_roc(y_true, y_prob, auc, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def evaluate(arch: str, weights: str, data_root: str) -> dict:
    PATHS.ensure()
    device = get_device(CONFIG.device)
    model = load_model(arch, weights, num_classes=CONFIG.num_classes, device=device)

    _, _, test_loader = build_dataloaders(
        data_root, image_size=CONFIG.image_size, batch_size=CONFIG.batch_size,
        val_split=CONFIG.val_split, test_split=CONFIG.test_split,
        num_workers=CONFIG.num_workers, seed=CONFIG.seed,
    )

    y_true, y_pred, y_prob = _collect(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("%s metrics: %s", arch, metrics.to_dict())

    cm = np.array(metrics.confusion_matrix)
    plot_confusion(cm, PATHS.reports / f"{arch}_confusion.png")
    plot_roc(y_true, y_prob, metrics.roc_auc, PATHS.reports / f"{arch}_roc.png")
    out = PATHS.reports / f"{arch}_eval.json"
    out.write_text(json.dumps(metrics.to_dict(), indent=2))
    logger.info("Saved report artifacts to %s", PATHS.reports)
    return metrics.to_dict()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained detector")
    p.add_argument("--arch", required=True,
                   choices=["resnet50", "efficientnet_b0", "xception"])
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="datasets/processed")
    args = p.parse_args()
    evaluate(args.arch, args.weights, args.data)


if __name__ == "__main__":
    main()
