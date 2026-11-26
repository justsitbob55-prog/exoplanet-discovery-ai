"""Classification metrics for baseline and future models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Return standard binary classification metrics."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_array, np.asarray(y_score)))
        except ValueError:
            metrics["roc_auc"] = 0.0
    matrix = confusion_matrix(y_true_array, y_pred_array)
    metrics["tn"] = float(matrix[0, 0]) if matrix.shape[0] > 0 else 0.0
    if matrix.shape == (2, 2):
        metrics.update({
            "fp": float(matrix[0, 1]),
            "fn": float(matrix[1, 0]),
            "tp": float(matrix[1, 1]),
        })
    return metrics
