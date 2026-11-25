"""Lightweight, label-safe feature drift and outcome monitoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import brier_score_loss, precision_score, recall_score

from .features import make_features
from .model_io import load_verified_bundle
from .score import score_frame
from .validation import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    TRAIN_COHORTS,
    DataContractError,
    _timestamp,
    _validate_outcome_frame,
    validate_snapshot_frame,
)

PROFILE_TAG = "monitoring_profile_sha256"
PROFILE_VERSION = "1.0"
MONITORED_FEATURES = ("active_days_14d", "interactions_14d", "days_since_last_activity")
DEFAULT_PSI_ALERT = 0.20
DEFAULT_EPSILON = 1e-6
DEFAULT_MINIMUM_ROWS = 100
QUANTILE_CUTS = tuple(index / 10 for index in range(1, 10))


def _profile_artifact_path(model_version: str) -> str:
    return f"monitoring/reference-profile-v{model_version}.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _feature_bin_spec(values: np.ndarray, feature_name: str) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"method": "unavailable", "valid_reference_count": 0, "bin_counts": [], "bin_proportions": []}
    if feature_name == "active_days_14d":
        categories = list(range(15))
        if not np.equal(finite, np.floor(finite)).all() or ((finite < 0) | (finite > 14)).any():
            raise ValueError("active_days_14d reference values must be integers from 0 through 14")
        counts = np.bincount(finite.astype(int), minlength=len(categories)).astype(int)
        method = "integer_values"
        details: dict[str, Any] = {"categories": categories}
    else:
        cuts = np.unique(np.quantile(finite, QUANTILE_CUTS)).astype(float).tolist()
        assignments = np.searchsorted(np.asarray(cuts, dtype=float), finite, side="right")
        counts = np.bincount(assignments, minlength=len(cuts) + 1).astype(int)
        method = "train_quantiles"
        details = {"internal_edges": cuts, "interval_convention": "(-inf, edge), then [edge, next_edge), with unbounded outer bins"}
    proportions = counts.astype(float) / float(counts.sum())
    return {
        "method": method,
        **details,
        "valid_reference_count": len(finite),
        "missing_reference_count": int(len(values) - len(finite)),
        "missing_reference_rate": float((len(values) - len(finite)) / len(values)),
        "bin_counts": counts.tolist(),
        "bin_proportions": proportions.tolist(),
    }


def create_reference_profile(
    features: pd.DataFrame,
    *,
    model_name: str,
    model_version: str,
    mlflow_run_id: str,
    data_manifest_sha256: str,
    target_version: str,
    feature_schema_version: str,
    train_cohorts: tuple[str, ...] = TRAIN_COHORTS,
    platform_scope: dict[str, Any] | None = None,
    platform_export_id: str | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    """Build fixed bins/proportions from the TRAIN feature matrix only."""
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise ValueError("reference features do not match the explicit feature schema")
    if features.empty:
        raise ValueError("cannot create a reference profile from zero training rows")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("PSI smoothing epsilon must be finite and positive")
    profiles: dict[str, Any] = {}
    for name in MONITORED_FEATURES:
        values = pd.to_numeric(features[name], errors="raise").to_numpy(dtype=float)
        profiles[name] = _feature_bin_spec(values, name)
    return {
        "profile_version": PROFILE_VERSION,
        "model_name": model_name,
        "model_version": str(model_version),
        "mlflow_run_id": str(mlflow_run_id),
        "data_manifest_sha256": data_manifest_sha256,
        "target_version": target_version,
        "feature_schema_version": feature_schema_version,
        "feature_columns": list(FEATURE_COLUMNS),
        "train_cohorts": list(train_cohorts),
        "platform_scope": platform_scope,
        "platform_export_id": platform_export_id,
        "train_rows": len(features),
        "psi_epsilon": float(epsilon),
        "psi_alert_threshold": DEFAULT_PSI_ALERT,
        "minimum_batch_rows": DEFAULT_MINIMUM_ROWS,
        "features": profiles,
    }


def profile_from_snapshot_export(data_dir: str | Path, bundle: dict[str, Any]) -> dict[str, Any]:
    """Create a profile from the snapshot CSV without opening student outcomes."""
    root = Path(data_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    snapshot_path = root / "raw" / "lms_snapshots.csv"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        snapshot_bytes = snapshot_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"cannot read reference snapshot export or manifest: {exc}") from exc
    manifest_hash = _sha256_bytes(manifest_bytes)
    lineage = bundle.get("lineage") or {}
    if manifest_hash != lineage.get("data_manifest_sha256"):
        raise ValueError("reference data manifest does not match the model's recorded training manifest")
    if manifest.get("platform_scope") is not None and manifest.get("platform_scope") != lineage.get("platform_scope"):
        raise ValueError("reference manifest scope does not match the model's training scope")
    if manifest.get("platform_scope") is not None and not isinstance(lineage.get("platform_scope"), dict):
        raise ValueError("scoped monitoring profile requires a scoped model bundle")
    if manifest.get("export_id") is not None and manifest.get("export_id") != lineage.get("platform_export_id"):
        raise ValueError("reference manifest export ID does not match the model's training lineage")
    files = manifest.get("files") or {}
    relative_snapshot = next(
        (name for name in ("raw/lms_snapshots.csv", "data/raw/lms_snapshots.csv") if name in files),
        None,
    )
    if relative_snapshot is None:
        raise DataContractError("reference manifest does not declare the LMS snapshot export")
    expected_file = files[relative_snapshot].get("sha256")
    if _sha256_bytes(snapshot_bytes) != expected_file:
        raise DataContractError("reference snapshot checksum does not match its manifest")
    split = manifest.get("split") or {}
    declared_train = tuple(split.get("train", ()))
    declared_all = [*declared_train, *split.get("validation", ()), *split.get("test", ())]
    cohort_manifest = manifest.get("cohorts") or []
    if isinstance(cohort_manifest, dict):
        chronological_cohorts = list(cohort_manifest)
    elif isinstance(cohort_manifest, list):
        chronological_cohorts = cohort_manifest
    else:
        chronological_cohorts = []
    if (
        not declared_train
        or len(set(declared_all)) != len(declared_all)
        or set(declared_all) != set(chronological_cohorts)
    ):
        raise DataContractError("reference manifest must declare a complete, disjoint chronological split")
    positions = [chronological_cohorts.index(cohort) for cohort in declared_all]
    if positions != sorted(positions):
        raise DataContractError("reference manifest train cohorts are not chronological")
    snapshots = pd.read_csv(snapshot_path, dtype=str, keep_default_na=False)
    validate_snapshot_frame(snapshots)
    train_rows = snapshots.loc[snapshots["cohort_id"].astype(str).isin(declared_train)].reset_index(drop=True)
    if train_rows.empty:
        raise DataContractError("reference export contains no rows from the fixed TRAIN cohorts")
    if set(snapshots["cohort_id"].astype(str)) - set(declared_all):
        raise DataContractError("reference export contains a cohort outside the documented temporal split")
    features = make_features(train_rows, validate=False)
    return create_reference_profile(
        features,
        model_name=str(bundle["model_name"]),
        model_version=str(bundle["model_version"]),
        mlflow_run_id=str(bundle["mlflow_run_id"]),
        data_manifest_sha256=manifest_hash,
        target_version=str(bundle["target_version"]),
        feature_schema_version=str(bundle["feature_schema_version"]),
        train_cohorts=declared_train,
        platform_scope=(bundle.get("lineage") or {}).get("platform_scope"),
        platform_export_id=(bundle.get("lineage") or {}).get("platform_export_id"),
    )


def _validated_profile(profile: dict[str, Any], bundle: dict[str, Any], expected_sha256: str) -> None:
    if profile.get("profile_version") != PROFILE_VERSION:
        raise ValueError("monitoring reference profile version is unsupported")
    expected = {
        "model_name": bundle.get("model_name"),
        "model_version": str(bundle.get("model_version")),
        "mlflow_run_id": str(bundle.get("mlflow_run_id")),
        "data_manifest_sha256": (bundle.get("lineage") or {}).get("data_manifest_sha256"),
        "target_version": bundle.get("target_version"),
        "feature_schema_version": bundle.get("feature_schema_version"),
        "feature_columns": list(bundle.get("feature_columns", ())),
        "platform_scope": (bundle.get("lineage") or {}).get("platform_scope"),
        "platform_export_id": (bundle.get("lineage") or {}).get("platform_export_id"),
    }
    lineage_split = (bundle.get("lineage") or {}).get("split") or {}
    if lineage_split.get("train") is not None:
        expected["train_cohorts"] = list(lineage_split["train"])
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f"monitoring profile {key} does not match the verified model bundle")
    if not np.isfinite(float(profile.get("psi_epsilon", 0))) or float(profile["psi_epsilon"]) <= 0:
        raise ValueError("monitoring profile has an invalid PSI epsilon")
    if not isinstance(expected_sha256, str) or _sha256_bytes(_json_bytes(profile)) != expected_sha256:
        raise ValueError("monitoring profile payload does not match its expected SHA-256")


def load_verified_reference_profile(
    client: MlflowClient,
    model_name: str,
    model_version: str,
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Load the profile artifact from the concrete model run and verify its checksum/bindings."""
    registered = client.get_model_version(model_name, str(model_version))
    if str(registered.run_id) != str(bundle.get("mlflow_run_id")):
        raise ValueError("monitoring profile registry run does not match the verified bundle")
    tags = registered.tags or {}
    expected_hash = tags.get(PROFILE_TAG)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("model version has no registered monitoring profile checksum")
    local_path = mlflow.artifacts.download_artifacts(run_id=registered.run_id, artifact_path=_profile_artifact_path(str(model_version)))
    payload = Path(local_path).read_bytes()
    actual_hash = _sha256_bytes(payload)
    if actual_hash != expected_hash:
        raise ValueError("downloaded monitoring profile checksum does not match the model-version tag")
    try:
        profile = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("monitoring reference profile is not valid JSON") from exc
    _validated_profile(profile, bundle, expected_hash)
    return profile, actual_hash


def _assign_bins(values: np.ndarray, feature_name: str, feature_profile: dict[str, Any]) -> np.ndarray:
    method = feature_profile.get("method")
    if method == "integer_values":
        categories = np.asarray(feature_profile["categories"], dtype=float)
        if not np.equal(values, np.floor(values)).all() or ((values < categories[0]) | (values > categories[-1])).any():
            raise ValueError(f"monitor feature {feature_name} is outside the reference integer bins")
        return values.astype(int)
    if method == "train_quantiles":
        edges = np.asarray(feature_profile.get("internal_edges", []), dtype=float)
        if not np.isfinite(edges).all():
            raise ValueError(f"monitor feature {feature_name} has invalid train-derived bin edges")
        return np.searchsorted(edges, values, side="right")
    raise ValueError(f"monitor feature {feature_name} has no usable reference distribution")


def _smoothed_proportions(counts: np.ndarray, epsilon: float) -> np.ndarray:
    proportions = counts.astype(float) / max(float(counts.sum()), 1.0)
    smoothed = np.maximum(proportions, epsilon)
    return smoothed / smoothed.sum()


def calculate_psi(reference_proportions: list[float], current_counts: list[int], epsilon: float = DEFAULT_EPSILON) -> float:
    """Population stability index with epsilon smoothing and renormalization."""
    reference = np.asarray(reference_proportions, dtype=float)
    counts = np.asarray(current_counts, dtype=float)
    if len(reference) == 0 or len(reference) != len(counts):
        raise ValueError("reference proportions and current counts must have matching non-zero bins")
    if not np.isfinite(reference).all() or (reference < 0).any() or (counts < 0).any():
        raise ValueError("PSI inputs must be finite and non-negative")
    ref = np.maximum(reference, epsilon)
    ref = ref / ref.sum()
    cur = np.maximum(counts / max(counts.sum(), 1.0), epsilon)
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _feature_distribution(values: np.ndarray, feature_name: str, profile: dict[str, Any]) -> dict[str, Any]:
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    missing = int((~finite_mask).sum())
    result: dict[str, Any] = {
        "rows": len(values),
        "valid_count": len(finite),
        "missing_count": missing,
        "missing_rate": float(missing / len(values)) if len(values) else None,
        "mean": float(finite.mean()) if len(finite) else None,
        "minimum": float(finite.min()) if len(finite) else None,
        "maximum": float(finite.max()) if len(finite) else None,
    }
    if not len(finite):
        result.update({"psi": None, "status": "no_valid_values", "bin_counts": []})
        return result
    counts = np.bincount(
        _assign_bins(finite, feature_name, profile),
        minlength=len(profile["bin_proportions"]),
    ).astype(int)
    result["bin_counts"] = counts.tolist()
    result["psi"] = calculate_psi(profile["bin_proportions"], counts.tolist(), float(profile.get("psi_epsilon", DEFAULT_EPSILON)))
    result["status"] = "computed"
    return result


def _outcome_metrics(
    snapshots: pd.DataFrame,
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame | None,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    if outcomes is None:
        return {"status": "data_not_ready", "reason": "no matured outcomes were supplied", "support": 0, "pending_count": len(snapshots), "simulated": None}
    _validate_outcome_frame(outcomes)
    input_keys = set(map(tuple, snapshots.loc[:, KEY_COLUMNS].astype(str).to_numpy()))
    outcome_keys = set(map(tuple, outcomes.loc[:, KEY_COLUMNS].astype(str).to_numpy()))
    if outcome_keys - input_keys:
        raise DataContractError("outcome stream contains keys absent from the scored batch")
    prediction_by_key = {
        tuple(row[column] for column in KEY_COLUMNS): float(probability)
        for row, probability in zip(
            predictions.loc[:, KEY_COLUMNS].astype(str).to_dict(orient="records"),
            predictions["dropout_probability"].to_numpy(dtype=float),
        )
    }
    outcomes_by_key = {
        tuple(row[column] for column in KEY_COLUMNS): row
        for row in outcomes.loc[:, [*KEY_COLUMNS, "final_outcome", "withdrawn_at", "outcome_observed_at"]].astype(str).to_dict(orient="records")
    }
    labels: list[int] = []
    probabilities: list[float] = []
    pending = 0
    for _, snapshot in snapshots.iterrows():
        key = tuple(str(snapshot[column]) for column in KEY_COLUMNS)
        outcome = outcomes_by_key.get(key)
        if outcome is None or outcome["final_outcome"] == "unknown":
            pending += 1
            continue
        cutoff = _timestamp(snapshot["snapshot_at"], "snapshot_at", key, [])
        course_end = _timestamp(snapshot["course_end_at"], "course_end_at", key, [])
        observed = _timestamp(outcome["outcome_observed_at"], "outcome_observed_at", key, [])
        if outcome["final_outcome"] == "withdrawn":
            withdrawn = _timestamp(outcome["withdrawn_at"], "withdrawn_at", key, [])
            if withdrawn is None or not cutoff < withdrawn <= course_end:
                raise DataContractError("withdrawn outcome falls outside the prediction window")
            labels.append(1)
            probabilities.append(prediction_by_key[key])
        elif observed < course_end + timedelta(days=14):
            pending += 1
        else:
            labels.append(0)
            probabilities.append(prediction_by_key[key])
    simulated = None
    if "source_system" in snapshots.columns:
        simulated = bool(snapshots["source_system"].astype(str).str.startswith("synthetic").all())
    if not labels:
        return {"status": "data_not_ready", "reason": "no mature labels were available", "support": 0, "pending_count": pending, "simulated": simulated}
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = p >= float(bundle["threshold"])
    positive_support = int(y.sum())
    predicted_positive = int(predicted.sum())
    return {
        "status": "reported",
        "support": len(y),
        "positive_support": positive_support,
        "pending_count": pending,
        "precision": float(precision_score(y, predicted, zero_division=0)) if predicted_positive else 0.0,
        "recall": float(recall_score(y, predicted, zero_division=0)) if positive_support else None,
        "brier_score": float(brier_score_loss(y, p)),
        "threshold": float(bundle["threshold"]),
        "simulated": simulated,
    }


def monitor_frame(
    snapshots: pd.DataFrame,
    model: Any,
    bundle: dict[str, Any],
    model_version: str,
    profile: dict[str, Any],
    *,
    profile_sha256: str,
    outcomes: pd.DataFrame | None = None,
    batch_id: str | None = None,
    minimum_rows: int | None = None,
    psi_alert_threshold: float | None = None,
) -> dict[str, Any]:
    """Score a complete batch and return an input-drift/outcome status report."""
    validate_snapshot_frame(snapshots)
    _validated_profile(profile, bundle, profile_sha256)
    if str(model_version) != str(bundle.get("model_version")):
        raise ValueError("monitor model version does not match the verified bundle")
    features = make_features(snapshots, validate=False)
    predictions = score_frame(snapshots, model, bundle, str(model_version), batch_id=batch_id)
    min_rows = int(minimum_rows if minimum_rows is not None else profile.get("minimum_batch_rows", DEFAULT_MINIMUM_ROWS))
    alert_threshold = float(psi_alert_threshold if psi_alert_threshold is not None else profile.get("psi_alert_threshold", DEFAULT_PSI_ALERT))
    if min_rows < 1 or not np.isfinite(alert_threshold) or alert_threshold < 0:
        raise ValueError("minimum batch size and PSI alert threshold are invalid")
    feature_reports: dict[str, Any] = {}
    enough_rows = len(snapshots) >= min_rows
    for name in MONITORED_FEATURES:
        values = pd.to_numeric(features[name], errors="raise").to_numpy(dtype=float)
        if enough_rows:
            feature_reports[name] = _feature_distribution(values, name, profile["features"][name])
        else:
            finite = values[np.isfinite(values)]
            missing = int((~np.isfinite(values)).sum())
            feature_reports[name] = {
                "rows": len(values), "valid_count": len(finite), "missing_count": missing,
                "missing_rate": float(missing / len(values)) if len(values) else None,
                "mean": float(finite.mean()) if len(finite) else None,
                "minimum": float(finite.min()) if len(finite) else None,
                "maximum": float(finite.max()) if len(finite) else None,
                "psi": None, "status": "insufficient_data",
            }
    statuses = {entry["status"] for entry in feature_reports.values()}
    enough_reference = "no_valid_values" not in statuses
    if not enough_rows:
        drift_status = "insufficient_data"
        alert = None
    elif not enough_reference:
        drift_status = "insufficient_reference"
        alert = None
    else:
        alert = any(entry["psi"] is not None and entry["psi"] > alert_threshold for entry in feature_reports.values())
        drift_status = "alert" if alert else "clear"
    manifest = bundle["lineage"]["data_manifest_sha256"]
    outcome_report = _outcome_metrics(snapshots, predictions, outcomes, bundle)
    if outcome_report["status"] == "data_not_ready":
        retraining_state = "data_not_ready"
        recommendation_reason = (
            "input distribution alert; collect mature outcomes before supervised retraining"
            if alert
            else "mature outcome labels are unavailable"
        )
    elif alert:
        retraining_state = "recommended"
        recommendation_reason = "input distribution alert with mature outcomes available for supervised evaluation"
    else:
        retraining_state = "none"
        recommendation_reason = "no drift alert and no retraining trigger"
    return {
        "report_version": "1.0",
        "status": "completed",
        "batch_id": str(predictions["batch_id"].iloc[0]),
        "model_name": bundle["model_name"],
        "model_version": str(model_version),
        "model_run_id": bundle["mlflow_run_id"],
        "target_version": bundle["target_version"],
        "feature_schema_version": bundle["feature_schema_version"],
        "training_data_manifest_sha256": manifest,
        "row_count": len(snapshots),
        "valid_row_count": len(snapshots),
        "rejected_row_count": 0,
        "drift_status": drift_status,
        "drift_alert": alert,
        "psi_alert_threshold": alert_threshold,
        "minimum_rows": min_rows,
        "features": feature_reports,
        "mean_probability": float(predictions["dropout_probability"].mean()),
        "review_rate": float((predictions["risk_band"] == "review").mean()),
        "outcomes": outcome_report,
        "retraining_recommendation": {
            "state": retraining_state,
            "reason": recommendation_reason,
        },
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(report)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def create_and_register_reference_profile(
    data_dir: str | Path,
    model_version: str,
    *,
    tracking_uri: str,
    model_name: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create and attach the TRAIN-only profile to an existing concrete model run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    _, bundle = load_verified_bundle(client, model_name, str(model_version))
    profile = profile_from_snapshot_export(data_dir, bundle)
    payload = _json_bytes(profile)
    digest = _sha256_bytes(payload)
    local_path = Path(output_path).expanduser().resolve()
    expected_artifact = _profile_artifact_path(str(model_version))
    if local_path.name != Path(expected_artifact).name:
        raise ValueError("monitoring profile output filename must match the immutable versioned artifact path")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(payload)
    registered = client.get_model_version(model_name, str(model_version))
    tags = registered.tags or {}
    existing_digest = tags.get(PROFILE_TAG)
    if existing_digest:
        if existing_digest != digest:
            raise ValueError("reference profile is already registered with a different checksum; do not replace a released model profile")
        artifact_path = _profile_artifact_path(str(model_version))
        remote = mlflow.artifacts.download_artifacts(run_id=registered.run_id, artifact_path=artifact_path)
        if _sha256_bytes(Path(remote).read_bytes()) != digest:
            raise ValueError("existing reference profile artifact does not match its checksum tag")
        return {"status": "already_registered", "sha256": digest, "artifact": artifact_path, "model_version": str(model_version)}
    client.log_artifact(str(registered.run_id), str(local_path), artifact_path="monitoring")
    client.set_model_version_tag(model_name, str(model_version), PROFILE_TAG, digest)
    return {"status": "registered", "sha256": digest, "artifact": _profile_artifact_path(str(model_version)), "model_version": str(model_version), "mlflow_run_id": str(registered.run_id)}


def monitor_file(
    input_path: str | Path,
    model_version: str,
    *,
    tracking_uri: str,
    config_path: str | Path,
    report_path: str | Path,
    outcomes_path: str | Path | None = None,
) -> dict[str, Any]:
    """Monitor a complete, validated snapshot batch against a fixed model/profile."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("model configuration must be a YAML mapping")
    source = Path(input_path).expanduser().resolve()
    report_target = Path(report_path).expanduser().resolve()
    outcomes_target = Path(outcomes_path).expanduser().resolve() if outcomes_path is not None else None
    if report_target == source or (outcomes_target is not None and report_target == outcomes_target):
        raise ValueError("monitor report path must differ from both input and outcomes files")
    try:
        snapshots = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataContractError(f"cannot read monitoring input CSV: {exc}") from exc
    validate_snapshot_frame(snapshots)
    outcomes = None
    if outcomes_path is not None:
        try:
            outcomes = pd.read_csv(Path(outcomes_path).expanduser().resolve(), dtype=str, keep_default_na=False)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataContractError(f"cannot read outcome CSV: {exc}") from exc
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model, bundle = load_verified_bundle(
        client,
        str(config["model_name"]),
        str(model_version),
        expected_target_version=str(config["target"]["version"]),
        expected_feature_schema_version=str(config["feature_schema"]["version"]),
    )
    profile, profile_hash = load_verified_reference_profile(client, str(config["model_name"]), str(model_version), bundle)
    report = monitor_frame(snapshots, model, bundle, str(model_version), profile, profile_sha256=profile_hash, outcomes=outcomes)
    report["reference_profile_sha256"] = profile_hash
    report["report_path"] = str(report_target)
    write_report(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Complete UTF-8 LMS snapshot CSV batch")
    parser.add_argument("--model-version", required=True, help="Concrete gate-passed MLflow version")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--report", required=True, help="JSON report output path")
    parser.add_argument("--outcomes", help="Optional matured-outcomes CSV matching the scored batch")
    args = parser.parse_args(argv)
    try:
        report = monitor_file(args.input, args.model_version, tracking_uri=args.tracking_uri, config_path=args.config, report_path=args.report, outcomes_path=args.outcomes)
    except Exception as exc:
        print(f"monitoring failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
