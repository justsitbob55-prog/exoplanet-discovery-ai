"""Lightweight baseline model training and persistence."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ..common.exceptions import ModelError


@dataclass
class TrainingResult:
    model: RandomForestClassifier
    feature_names: list[str]
    train_size: int
    test_size: int
    metrics: dict[str, float]

    def predict_proba(self, features: Any) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return np.asarray(self.model.predict_proba(matrix), dtype=np.float64)


def train_baseline(
    features: Any,
    labels: Any,
    feature_names: list[str] | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> TrainingResult:
    """Train a reproducible Random Forest while preventing cross-split leakage."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y):
        raise ModelError("Features must be a 2D matrix aligned with labels.")
    if len(np.unique(y)) < 2:
        raise ModelError("Training requires at least two target classes.")
    counts = np.bincount(y.astype(int)) if np.issubdtype(y.dtype, np.integer) else None
    if counts is not None and np.min(counts[counts > 0]) < 2:
        raise ModelError("Each class needs at least two samples.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    predicted_proba = model.predict_proba(x_test)
    positive_index = int(np.where(model.classes_ == 1)[0][0]) if 1 in model.classes_ else 1
    from .evaluation import binary_metrics

    metrics = binary_metrics(y_test, predicted, predicted_proba[:, positive_index])
    names = feature_names or [f"feature_{index}" for index in range(x.shape[1])]
    return TrainingResult(model, names, len(x_train), len(x_test), metrics)


def save_model(result: TrainingResult, directory: str | Path) -> tuple[Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "baseline_random_forest.pkl"
    metadata_path = target / "baseline_random_forest.json"
    with model_path.open("wb") as handle:
        pickle.dump(result.model, handle)
    metadata = {
        "feature_names": result.feature_names,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "metrics": result.metrics,
        "model_type": "RandomForestClassifier",
        "training_status": "trained",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def train_model_comparison(
    features: Any,
    labels: Any,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compare Random Forest, SVM, and Gradient Boosting on one reproducible split."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y):
        raise ModelError("Features must be a 2D matrix aligned with labels.")
    if len(np.unique(y)) < 2:
        raise ModelError("Training requires at least two target classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y,
    )
    models: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "svm": SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_seed),
    }
    from .evaluation import binary_metrics

    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        y_score = None
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(x_test))
            if 1 in model.classes_:
                positive_index = int(np.where(model.classes_ == 1)[0][0])
                y_score = probabilities[:, positive_index]
        results[name] = binary_metrics(y_test, predicted, y_score)
    return results
