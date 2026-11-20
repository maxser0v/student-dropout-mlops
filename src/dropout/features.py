"""Shared allowlisted day-35 features for training and scoring."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .validation import DataContractError, FEATURE_COLUMNS, _LEAKAGE_COLUMNS, validate_snapshot_frame

RAW_FEATURE_COLUMNS = (
    "active_days_14d", "active_days_previous_14d", "interactions_14d",
    "interactions_previous_14d", "resource_views_14d", "forum_posts_14d",
    "late_submissions_to_date", "assignments_due_to_date",
    "scores_available_count", "mean_score_pct_to_date",
)


def make_features(snapshots: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    """Return the 14 explicitly named model inputs in the documented order.

    No identifiers, dates, labels, outcomes, cohort fields, or other metadata
    are selected. All rows are retained. Blank grades remain ``NaN`` for the
    model pipeline's training-only imputer. ``days_since_last_activity`` uses
    fractional elapsed UTC days (seconds / 86,400), capped at 35.
    """
    if not isinstance(snapshots, pd.DataFrame):
        raise TypeError("snapshots must be a pandas DataFrame")
    leaked = sorted(_LEAKAGE_COLUMNS.intersection(snapshots.columns))
    if leaked:
        raise DataContractError(f"snapshot input contains target/outcome leakage columns: {', '.join(leaked)}")
    if validate:
        validate_snapshot_frame(snapshots)
    missing = sorted(set(RAW_FEATURE_COLUMNS + ("assignments_submitted_to_date", "last_activity_at", "snapshot_at")) - set(snapshots.columns))
    if missing:
        raise DataContractError(f"cannot derive features; missing columns: {', '.join(missing)}")

    features = snapshots.loc[:, RAW_FEATURE_COLUMNS].copy()
    for column in RAW_FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column].replace("", pd.NA), errors="raise")
    due = pd.to_numeric(snapshots["assignments_due_to_date"], errors="raise")
    submitted = pd.to_numeric(snapshots["assignments_submitted_to_date"], errors="raise")
    recent = pd.to_numeric(snapshots["active_days_14d"], errors="raise")
    previous = pd.to_numeric(snapshots["active_days_previous_14d"], errors="raise")
    features["missed_assignments"] = due - submitted
    features["activity_change"] = recent - previous

    ages: list[float] = []
    no_history: list[int] = []
    for cutoff_raw, activity_raw in zip(snapshots["snapshot_at"], snapshots["last_activity_at"]):
        cutoff = datetime.fromisoformat(str(cutoff_raw).removesuffix("Z") + "+00:00").astimezone(timezone.utc)
        if pd.isna(activity_raw) or str(activity_raw).strip() == "":
            ages.append(35.0)
            no_history.append(1)
        else:
            activity = datetime.fromisoformat(str(activity_raw).removesuffix("Z") + "+00:00").astimezone(timezone.utc)
            ages.append(min(35.0, max(0.0, (cutoff - activity).total_seconds() / 86400.0)))
            no_history.append(0)
    features["days_since_last_activity"] = ages
    features["no_activity_history"] = no_history
    features = features.loc[:, FEATURE_COLUMNS]
    if len(features) != len(snapshots):
        raise DataContractError("feature generation changed the scoring row count")
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise DataContractError("generated feature order does not match the explicit schema")
    return features
