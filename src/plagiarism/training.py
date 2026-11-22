"""CPU XGBoost training and persistence for the passage-reuse experiment."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np

from .features import FEATURE_NAMES


def load_logged_classifier(model_uri: str) -> Any:
    from mlflow.xgboost import load_model

    # MLflow rejects model artifacts under sys.path when code loading is off.
    # A local SQLite tracker can store artifacts under the working directory.
    with TemporaryDirectory(prefix="passage-reuse-model-") as destination:
        return load_model(model_uri, dst_path=destination)


def train_classifier(
    features: np.ndarray,
    labels: Iterable[int],
    *,
    seed: int = 903,
    n_estimators: int = 80,
    max_depth: int = 4,
    learning_rate: float = 0.08,
) -> Any:
    """Fit a modest, deterministic XGBClassifier using CPU histogram trees."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is not installed in this environment; use the plagiarism runtime") from exc
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(list(labels), dtype=np.int32)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) != len(y) or len(y) == 0:
        raise ValueError("features must align with labels and the configured feature schema")
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("training requires finite features and both binary labels 0 and 1")
    if n_estimators < 1 or max_depth < 1 or learning_rate <= 0:
        raise ValueError("invalid XGBoost training settings")
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        n_jobs=1,
        random_state=seed,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=0.0,
        reg_lambda=1.0,
    )
    model.fit(x, y, verbose=False)
    return model


def positive_class_probability(model: Any, features: np.ndarray) -> np.ndarray:
    """Return the probability column for class 1 with output validation."""
    classes = np.asarray(model.classes_)
    indices = np.flatnonzero(classes == 1)
    if len(indices) != 1:
        raise ValueError("estimator must expose exactly one class labelled 1")
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)[:, int(indices[0])]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("model returned invalid class-1 probabilities")
    return probabilities
