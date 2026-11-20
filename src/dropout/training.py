"""Deterministic training and validation metrics for the dropout model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .validation import FEATURE_COLUMNS


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float | None
    precision: float | None
    recall: float | None
    candidates_considered: int


def build_pipeline(seed: int = 903) -> Pipeline:
    """Build the sole supported estimator; every learned transform fits on train."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=seed, max_iter=1000)),
        ]
    )


def positive_class_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return P(class=1), independent of sklearn's class-column ordering."""
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("estimator must expose exactly one positive class labelled 1")
    probabilities = np.asarray(model.predict_proba(features), dtype=float)[:, int(matches[0])]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("estimator returned invalid class-1 probabilities")
    return probabilities


def choose_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    minimum_precision: float = 0.30,
    lower: float = 0.05,
    upper: float = 0.95,
    step: float = 0.05,
) -> ThresholdChoice:
    """Choose max recall subject to precision; ties prefer precision then threshold."""
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not 0 <= lower <= upper <= 1 or step <= 0 or not 0 <= minimum_precision <= 1:
        raise ValueError("invalid threshold selection settings")

    # Integer arithmetic avoids floating point gaps at inclusive endpoints.
    n_steps = int(round((upper - lower) / step))
    candidates = [round(lower + index * step, 10) for index in range(n_steps + 1)]
    qualifying: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = p >= threshold
        precision = float(precision_score(y, predicted, zero_division=0))
        recall = float(recall_score(y, predicted, zero_division=0))
        if precision >= minimum_precision:
            qualifying.append((recall, precision, threshold))
    if not qualifying:
        return ThresholdChoice(None, None, None, len(candidates))
    recall, precision, threshold = max(qualifying, key=lambda item: (item[0], item[1], item[2]))
    return ThresholdChoice(threshold, precision, recall, len(candidates))


def reliability_bins(y_true: Iterable[int], probabilities: Iterable[float], bins: int = 10) -> list[dict[str, float | int]]:
    """Return fixed-width probability bins, retaining empty-bin counts."""
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) != len(p) or bins < 1:
        raise ValueError("labels and probabilities must align and bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    result: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        result.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": float(p[mask].mean()) if count else 0.0,
                "observed_rate": float(y[mask].mean()) if count else 0.0,
            }
        )
    return result


def evaluate_probabilities(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float | None,
    *,
    train_prevalence: float,
) -> dict[str, Any]:
    """Compute validation probability and threshold metrics, never test metrics."""
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be binary 0/1")
    baseline = np.full(len(y), float(train_prevalence), dtype=float)
    base = {
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, p)),
        "train_prevalence_brier_score": float(brier_score_loss(y, baseline)),
    }
    if threshold is None:
        decisions = np.zeros(len(y), dtype=bool)
        threshold_metrics = {"precision": None, "recall": None, "f1": None, "confusion_matrix": None}
    else:
        decisions = p >= threshold
        threshold_metrics = {
            "precision": float(precision_score(y, decisions, zero_division=0)),
            "recall": float(recall_score(y, decisions, zero_division=0)),
            "f1": float(f1_score(y, decisions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, decisions, labels=[0, 1]).tolist(),
        }
    return {
        "split": "validation",
        "support": int(len(y)),
        "positive_count": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": threshold,
        **base,
        **threshold_metrics,
        "reliability_bins": reliability_bins(y, p),
    }


def threshold_choice_dict(choice: ThresholdChoice) -> dict[str, Any]:
    return asdict(choice)


def assert_feature_schema(features: pd.DataFrame) -> None:
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise ValueError("training feature columns/order do not match the feature schema")
