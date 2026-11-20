"""Strict validators for the LMS snapshot and outcome export contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

KEY_COLUMNS = ("tenant_id", "student_key", "course_id", "cohort_id", "snapshot_at")
SNAPSHOT_COLUMNS = (
    *KEY_COLUMNS,
    "course_start_at", "course_end_at", "enrolled_at",
    "enrolment_status_at_snapshot", "activity_coverage_start_at",
    "activity_complete_through_at", "active_days_14d",
    "active_days_previous_14d", "interactions_14d",
    "interactions_previous_14d", "resource_views_14d", "forum_posts_14d",
    "last_activity_at", "assignments_due_to_date",
    "assignments_submitted_to_date", "late_submissions_to_date",
    "mean_score_pct_to_date", "scores_available_count", "source_system",
    "schema_version",
)
OUTCOME_COLUMNS = (*KEY_COLUMNS, "final_outcome", "withdrawn_at", "outcome_observed_at")
FEATURE_COLUMNS = (
    "active_days_14d", "active_days_previous_14d", "interactions_14d",
    "interactions_previous_14d", "resource_views_14d", "forum_posts_14d",
    "late_submissions_to_date", "assignments_due_to_date",
    "scores_available_count", "mean_score_pct_to_date", "missed_assignments",
    "activity_change", "days_since_last_activity", "no_activity_history",
)
ASSIGNMENT_OUTCOME_DELAY_DAYS = 14
TRAIN_COHORTS = ("2023_S1", "2023_S2", "2024_S1", "2024_S2")
VALIDATION_COHORTS = ("2025_S1",)
TEST_COHORTS = ("2025_S2",)

_TIMESTAMP_COLUMNS = (
    "snapshot_at", "course_start_at", "course_end_at", "enrolled_at",
    "activity_coverage_start_at", "activity_complete_through_at",
)
_INTEGER_RANGES = {
    "active_days_14d": (0, 14),
    "active_days_previous_14d": (0, 14),
    "interactions_14d": (0, None),
    "interactions_previous_14d": (0, None),
    "resource_views_14d": (0, None),
    "forum_posts_14d": (0, None),
    "assignments_due_to_date": (0, None),
    "assignments_submitted_to_date": (0, None),
    "late_submissions_to_date": (0, None),
    "scores_available_count": (0, None),
}
_LEAKAGE_COLUMNS = {
    "final_outcome", "withdrawn_at", "outcome_observed_at", "dropout_label",
    "predicted_dropout", "label", "target",
}


class DataContractError(ValueError):
    """Raised when an export cannot safely be used for training or scoring."""


@dataclass(frozen=True)
class TrainingData:
    """Validated, mature rows and cohort metadata for the training flow."""

    snapshots: pd.DataFrame
    labels: pd.Series
    exclusions: dict[str, int]
    split_counts: dict[str, int]
    student_overlap: dict[str, int]


def _required_string(value: object, field: str, row: object, errors: list[str]) -> str:
    if pd.isna(value) or str(value).strip() == "":
        errors.append(f"row {row}: {field} is required")
        return ""
    return str(value).strip()


def _timestamp(value: object, field: str, row: object, errors: list[str], *, blank_ok: bool = False) -> datetime | None:
    if pd.isna(value) or str(value).strip() == "":
        if not blank_ok:
            errors.append(f"row {row}: {field} is required")
        return None
    raw = str(value).strip()
    if not raw.endswith("Z"):
        errors.append(f"row {row}: {field} must be a UTC ISO 8601 timestamp ending in Z")
        return None
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError:
        errors.append(f"row {row}: {field} is not a valid ISO 8601 timestamp")
        return None
    if parsed.utcoffset() != timedelta(0):
        errors.append(f"row {row}: {field} must be UTC")
        return None
    return parsed.astimezone(timezone.utc)


def _integer(value: object, field: str, row: object, errors: list[str]) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        errors.append(f"row {row}: {field} is required; blank is not zero")
        return None
    try:
        number = float(value)
        if not number.is_integer():
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"row {row}: {field} must be an integer")
        return None


def _check_columns(frame: pd.DataFrame, expected: Iterable[str], name: str, errors: list[str]) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(frame.columns))
    extra = sorted(set(frame.columns) - expected_set)
    if missing:
        errors.append(f"{name}: missing required columns: {', '.join(missing)}")
    if extra:
        errors.append(f"{name}: unexpected columns: {', '.join(extra)}")


def validate_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a complete snapshot CSV or scoring batch without dropping rows.

    Returns the original frame. Any malformed row rejects the entire batch.
    Outcome columns or any unrecognized columns are rejected explicitly.
    """
    errors: list[str] = []
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise DataContractError("snapshot: duplicate column names are not allowed")
    _check_columns(frame, SNAPSHOT_COLUMNS, "snapshot", errors)
    if errors:
        raise DataContractError("\n".join(errors))
    if frame.empty:
        raise DataContractError("snapshot: file must contain at least one row")
    if frame.loc[:, KEY_COLUMNS].isna().any().any() or frame.loc[:, KEY_COLUMNS].astype(str).apply(lambda s: s.str.strip().eq("")).any().any():
        errors.append("snapshot: all join-key fields are required")
    duplicated = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicated.any():
        errors.append(f"snapshot: duplicate join keys in {int(duplicated.sum())} row(s)")

    for index, row in frame.iterrows():
        ts = {field: _timestamp(row[field], field, index, errors) for field in _TIMESTAMP_COLUMNS}
        last = _timestamp(row["last_activity_at"], "last_activity_at", index, errors, blank_ok=True)
        nums = {field: _integer(row[field], field, index, errors) for field in _INTEGER_RANGES}
        for field, (minimum, maximum) in _INTEGER_RANGES.items():
            value = nums[field]
            if value is not None and (value < minimum or maximum is not None and value > maximum):
                errors.append(f"row {index}: {field} must be between {minimum} and {maximum if maximum is not None else 'unbounded'}")

        score_count = nums["scores_available_count"]
        submitted = nums["assignments_submitted_to_date"]
        due = nums["assignments_due_to_date"]
        late = nums["late_submissions_to_date"]
        active = nums["active_days_14d"]
        prior = nums["active_days_previous_14d"]
        recent_events = nums["interactions_14d"]
        previous_events = nums["interactions_previous_14d"]
        resource = nums["resource_views_14d"]
        forum = nums["forum_posts_14d"]
        if score_count is not None and submitted is not None and score_count > submitted:
            errors.append(f"row {index}: scores_available_count cannot exceed assignments_submitted_to_date")
        if submitted is not None and due is not None and submitted > due:
            errors.append(f"row {index}: assignments_submitted_to_date cannot exceed assignments_due_to_date")
        if late is not None and submitted is not None and late > submitted:
            errors.append(f"row {index}: late_submissions_to_date cannot exceed assignments_submitted_to_date")
        if recent_events is not None and resource is not None and forum is not None and resource + forum > recent_events:
            errors.append(f"row {index}: resource_views_14d + forum_posts_14d cannot exceed interactions_14d")
        if recent_events is not None and active is not None and ((recent_events == 0) != (active == 0) or active > recent_events):
            errors.append(f"row {index}: active_days_14d and interactions_14d are inconsistent")
        if previous_events is not None and prior is not None and ((previous_events == 0) != (prior == 0) or prior > previous_events):
            errors.append(f"row {index}: active_days_previous_14d and interactions_previous_14d are inconsistent")

        mean = row["mean_score_pct_to_date"]
        mean_blank = pd.isna(mean) or str(mean).strip() == ""
        if score_count is not None and mean_blank != (score_count == 0):
            errors.append(f"row {index}: mean_score_pct_to_date must be blank iff scores_available_count is zero")
        if not mean_blank:
            try:
                score = float(mean)
                if not 0 <= score <= 100:
                    errors.append(f"row {index}: mean_score_pct_to_date must be between 0 and 100")
            except (TypeError, ValueError):
                errors.append(f"row {index}: mean_score_pct_to_date must be numeric or blank")

        if row["enrolment_status_at_snapshot"] != "active":
            errors.append(f"row {index}: enrolment_status_at_snapshot must be active")
        if pd.isna(row["source_system"]) or str(row["source_system"]).strip() == "":
            errors.append(f"row {index}: source_system is required")
        if row["schema_version"] != "1.0":
            errors.append(f"row {index}: unsupported schema_version {row['schema_version']!r}")
        if all(ts.get(f) is not None for f in ("snapshot_at", "course_start_at", "course_end_at", "enrolled_at", "activity_coverage_start_at", "activity_complete_through_at")):
            cutoff = ts["snapshot_at"]
            if cutoff - ts["course_start_at"] != timedelta(days=35):
                errors.append(f"row {index}: snapshot_at must equal course_start_at + 35 days")
            if not ts["enrolled_at"] < cutoff < ts["course_end_at"]:
                errors.append(f"row {index}: enrolment must precede the cutoff and course_end_at must follow it")
            if ts["activity_coverage_start_at"] > cutoff - timedelta(days=28):
                errors.append(f"row {index}: activity coverage does not cover the 28-day lookback")
            if ts["activity_complete_through_at"] < cutoff:
                errors.append(f"row {index}: activity coverage watermark is before snapshot_at")
            if last is not None:
                if not ts["enrolled_at"] <= last < cutoff:
                    errors.append(f"row {index}: last_activity_at must be on/after enrolment and before cutoff")
                if active is not None and ((last >= cutoff - timedelta(days=14)) != (active > 0)):
                    errors.append(f"row {index}: last_activity_at conflicts with active_days_14d")
                if active == 0 and prior is not None and (last >= cutoff - timedelta(days=28)) != (prior > 0):
                    errors.append(f"row {index}: last_activity_at conflicts with active_days_previous_14d")
            elif active not in (0, None) or prior not in (0, None):
                errors.append(f"row {index}: blank last_activity_at requires zero activity in both windows")

    if errors:
        raise DataContractError("\n".join(errors))
    return frame


def _validate_outcome_frame(frame: pd.DataFrame) -> None:
    errors: list[str] = []
    if frame.columns.duplicated().any():
        raise DataContractError("outcomes: duplicate column names are not allowed")
    _check_columns(frame, OUTCOME_COLUMNS, "outcomes", errors)
    if errors:
        raise DataContractError("\n".join(errors))
    if frame.empty:
        raise DataContractError("outcomes: file must contain at least one row")
    if frame.loc[:, KEY_COLUMNS].isna().any().any() or frame.loc[:, KEY_COLUMNS].astype(str).apply(lambda s: s.str.strip().eq("")).any().any():
        errors.append("outcomes: all join-key fields are required")
    if frame.duplicated(list(KEY_COLUMNS), keep=False).any():
        errors.append("outcomes: duplicate join keys")
    allowed = {"withdrawn", "completed", "failed", "unknown"}
    for index, row in frame.iterrows():
        outcome = str(row["final_outcome"])
        if outcome not in allowed:
            errors.append(f"row {index}: unsupported final_outcome {outcome!r}")
        withdrawn_at = _timestamp(row["withdrawn_at"], "withdrawn_at", index, errors, blank_ok=True)
        observed = _timestamp(row["outcome_observed_at"], "outcome_observed_at", index, errors)
        if outcome == "withdrawn" and withdrawn_at is None:
            errors.append(f"row {index}: withdrawn outcome requires withdrawn_at")
        if outcome == "withdrawn" and withdrawn_at is not None and observed is not None and observed < withdrawn_at:
            errors.append(f"row {index}: outcome_observed_at cannot precede withdrawn_at")
        if outcome != "withdrawn" and withdrawn_at is not None:
            errors.append(f"row {index}: withdrawn_at must be blank unless final_outcome is withdrawn")
    if errors:
        raise DataContractError("\n".join(errors))


def load_training_data(data_dir: str | Path) -> TrainingData:
    """Load, validate, join and cohort-split the two historical CSV exports.

    Unknown and immature completed/failed outcomes are excluded and counted.
    Invalid/out-of-window withdrawals, schema defects and unmatched joins fail.
    The returned rows include only mature supervised labels; features are built
    separately with :func:`dropout.features.make_features`.
    """
    root = Path(data_dir)
    snapshot_path = root / "raw" / "lms_snapshots.csv"
    outcome_path = root / "raw" / "student_outcomes.csv"
    try:
        snapshots = pd.read_csv(snapshot_path, dtype=str, keep_default_na=False)
        outcomes = pd.read_csv(outcome_path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataContractError(f"cannot read dataset: {exc}") from exc
    validate_snapshot_frame(snapshots)
    _validate_outcome_frame(outcomes)
    left = snapshots.set_index(list(KEY_COLUMNS), verify_integrity=True)
    right = outcomes.set_index(list(KEY_COLUMNS), verify_integrity=True)
    missing_outcome = len(left.index.difference(right.index))
    missing_snapshot = len(right.index.difference(left.index))
    if missing_outcome or missing_snapshot:
        raise DataContractError(f"snapshot/outcome join is not one-to-one: {missing_outcome} unmatched snapshots, {missing_snapshot} unmatched outcomes")
    joined = left.join(right, how="inner").reset_index()
    exclusions = {"unknown_outcome": 0, "immature_outcome": 0}
    labels: list[int | None] = []
    for index, row in joined.iterrows():
        outcome = row["final_outcome"]
        cutoff = _timestamp(row["snapshot_at"], "snapshot_at", index, [])
        end = _timestamp(row["course_end_at"], "course_end_at", index, [])
        observed = _timestamp(row["outcome_observed_at"], "outcome_observed_at", index, [])
        withdrawn = _timestamp(row["withdrawn_at"], "withdrawn_at", index, [], blank_ok=True)
        if outcome == "withdrawn":
            if withdrawn is None or not cutoff < withdrawn <= end:
                raise DataContractError(f"row {index}: withdrawn_at must be strictly after cutoff and no later than course_end_at")
            labels.append(1)
        elif outcome in {"completed", "failed"}:
            if observed < end:
                exclusions["immature_outcome"] += 1
                labels.append(None)
            elif observed < end + timedelta(days=ASSIGNMENT_OUTCOME_DELAY_DAYS):
                exclusions["immature_outcome"] += 1
                labels.append(None)
            else:
                labels.append(0)
        else:
            exclusions["unknown_outcome"] += 1
            labels.append(None)
    joined["dropout_label"] = labels
    mature = joined["dropout_label"].notna()
    supervised = joined.loc[mature].copy()
    supervised["dropout_label"] = supervised["dropout_label"].astype("int64")
    cohorts = supervised["cohort_id"]
    cohort_groups = {
        "train": cohorts.isin(TRAIN_COHORTS),
        "validation": cohorts.isin(VALIDATION_COHORTS),
        "test": cohorts.isin(TEST_COHORTS),
    }
    split_counts = {name: int(mask.sum()) for name, mask in cohort_groups.items()}
    if set(supervised["cohort_id"]) - set((*TRAIN_COHORTS, *VALIDATION_COHORTS, *TEST_COHORTS)):
        raise DataContractError("dataset contains cohort IDs absent from the documented temporal split")
    train_students = set(supervised.loc[cohort_groups["train"], "student_key"])
    validation_students = set(supervised.loc[cohort_groups["validation"], "student_key"])
    test_students = set(supervised.loc[cohort_groups["test"], "student_key"])
    student_overlap = {
        "train_validation": len(train_students & validation_students),
        "train_test": len(train_students & test_students),
        "validation_test": len(validation_students & test_students),
    }
    return TrainingData(
        snapshots=supervised.drop(columns=["dropout_label", "final_outcome", "withdrawn_at", "outcome_observed_at"]),
        labels=supervised["dropout_label"].rename("dropout_label"),
        exclusions=exclusions,
        split_counts=split_counts,
        student_overlap=student_overlap,
    )


def verify_manifest(data_dir: str | Path) -> dict[str, object]:
    """Verify the starter manifest's SHA-256 values and row counts."""
    root = Path(data_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"cannot read data manifest: {exc}") from exc
    if not manifest.get("synthetic"):
        raise DataContractError("expected a synthetic manifest for this local demonstration")
    errors = []
    files = manifest.get("files", {})
    required_training_inputs = {"data/raw/lms_snapshots.csv", "data/raw/student_outcomes.csv"}
    if not isinstance(files, dict):
        raise DataContractError("manifest files must be a mapping of relative paths to checksums")
    missing_training_inputs = sorted(required_training_inputs - set(files))
    if missing_training_inputs:
        errors.append(f"manifest missing required training input checksum(s): {', '.join(missing_training_inputs)}")
    invalid_training_hashes = sorted(
        relative
        for relative in required_training_inputs.intersection(files)
        if not isinstance(files[relative], dict)
        or not isinstance(files[relative].get("sha256"), str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", files[relative].get("sha256", ""))
    )
    if invalid_training_hashes:
        errors.append(f"manifest has invalid training input checksum(s): {', '.join(invalid_training_hashes)}")
    for relative, info in files.items():
        checksum = info.get("sha256") if isinstance(info, dict) else None
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
            errors.append(f"manifest checksum must be 64 hexadecimal characters: {relative}")
            continue
        path = root.parent / relative
        if not path.is_file():
            errors.append(f"manifest file missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != checksum.lower():
            errors.append(f"manifest checksum mismatch: {relative}")
    if errors:
        raise DataContractError("\n".join(errors))
    snapshots = pd.read_csv(root / "raw" / "lms_snapshots.csv", dtype=str, keep_default_na=False)
    outcomes = pd.read_csv(root / "raw" / "student_outcomes.csv", dtype=str, keep_default_na=False)
    if len(snapshots) != len(outcomes) or len(snapshots) != sum(c["rows"] for c in manifest["cohorts"].values()):
        raise DataContractError("manifest row counts do not match the snapshot/outcome files")
    actual = snapshots["cohort_id"].value_counts().to_dict()
    for cohort, details in manifest["cohorts"].items():
        if actual.get(cohort, 0) != details["rows"]:
            raise DataContractError(f"manifest cohort count mismatch for {cohort}")
    return manifest
