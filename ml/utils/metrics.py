"""
Classification metrics used by the evaluation module (Phase 10).

Wraps scikit-learn so every model is scored identically:
accuracy, precision, recall, F1, ROC-AUC and confusion matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> MetricResult:
    """
    Args:
        y_true: ground-truth labels (0=real, 1=fake)
        y_pred: predicted labels
        y_prob: predicted probability of the positive (fake) class
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present

    return MetricResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(auc),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )
