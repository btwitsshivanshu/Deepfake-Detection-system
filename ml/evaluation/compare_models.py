"""
Model comparison script (Phase 6 — comparison / Phase 10 — results).

Evaluates all three architectures on the same test split and writes a
combined comparison table (JSON + bar chart) for the results chapter.

Usage:
    python -m ml.evaluation.compare_models --data datasets/hemg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.config import PATHS
from ml.evaluation.evaluate import evaluate
from ml.utils.logger import get_logger

logger = get_logger("compare")

MODELS = {
    "resnet50": "ml/checkpoints/resnet50_best.pth",
    "efficientnet_b0": "ml/checkpoints/efficientnet_b0_best.pth",
    "xception": "ml/checkpoints/xception_best.pth",
}


def compare(data_root: str) -> dict:
    PATHS.ensure()
    results: dict[str, dict] = {}
    for arch, weights in MODELS.items():
        if not Path(weights).exists():
            logger.warning("Skipping %s — weights not found at %s", arch, weights)
            continue
        results[arch] = evaluate(arch, weights, data_root)

    if not results:
        logger.error("No trained models found to compare.")
        return {}

    # Bar chart of key metrics
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.15
    for i, arch in enumerate(results):
        vals = [results[arch][m] for m in metrics]
        xs = [x + i * width for x in range(len(metrics))]
        ax.bar(xs, vals, width, label=arch)
    ax.set_xticks([x + width for x in range(len(metrics))], labels=metrics)
    ax.set_ylim(0, 1); ax.set_title("Model Comparison"); ax.legend()
    fig.tight_layout(); fig.savefig(PATHS.reports / "model_comparison.png", dpi=150)

    (PATHS.reports / "model_comparison.json").write_text(json.dumps(results, indent=2))
    logger.info("Comparison saved to %s", PATHS.reports / "model_comparison.json")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Compare all detectors")
    p.add_argument("--data", default="datasets/hemg")
    args = p.parse_args()
    compare(args.data)


if __name__ == "__main__":
    main()
