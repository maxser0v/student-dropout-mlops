"""Evaluate the frozen model once on the held-out 2025_S2 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dropout.features import make_features
from dropout.model_io import load_verified_bundle
from dropout.training import positive_class_probability, reliability_bins
from dropout.validation import TEST_COHORTS, load_training_data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _record_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def final_test_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    train_prevalence: float,
    *,
    model_version: str,
    target_version: str,
    feature_schema_version: str,
    training_manifest_sha256: str,
    frozen_bundle_sha256: str,
) -> dict[str, Any]:
    """Compute final-test metrics once, at the already-frozen decision threshold."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("final-test labels and probabilities must have the same non-zero length")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("final-test labels must be binary")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("final-test probabilities must be finite and within [0, 1]")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("frozen threshold must be within [0, 1]")
    decision = p >= float(threshold)
    positive_count = int(y.sum())
    predicted_positive = int(decision.sum())
    return {
        "evaluation_split": "final_test",
        "synthetic": True,
        "cohorts": list(TEST_COHORTS),
        "support": int(len(y)),
        "positive_support": positive_count,
        "prevalence": float(y.mean()),
        "model_version": str(model_version),
        "target_version": target_version,
        "feature_schema_version": feature_schema_version,
        "training_manifest_sha256": training_manifest_sha256,
        "frozen_deployment_bundle_sha256": frozen_bundle_sha256,
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "precision": float(precision_score(y, decision, zero_division=0)) if predicted_positive else 0.0,
        "recall": float(recall_score(y, decision, zero_division=0)) if positive_count else None,
        "f1": float(f1_score(y, decision, zero_division=0)) if positive_count or predicted_positive else 0.0,
        "brier_score": float(brier_score_loss(y, p)),
        "train_prevalence_brier_score": float(brier_score_loss(y, np.full(len(y), train_prevalence))),
        "confusion_matrix": confusion_matrix(y, decision, labels=[0, 1]).tolist(),
        "reliability_bins": reliability_bins(y, p),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "simulated held-out results; not evidence of real-world student risk or performance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-record", default="reports/demo/frozen_model.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--output", default="reports/demo/final_test_metrics.json")
    parser.add_argument("--attempt-record", default="reports/demo/final_test_attempt.json")
    args = parser.parse_args()
    freeze_path = Path(args.freeze_record).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    attempt_path = Path(args.attempt_record).expanduser().resolve()
    if not freeze_path.is_file():
        raise SystemExit(f"required frozen-model record does not exist: {freeze_path}")
    if output == attempt_path or output == freeze_path:
        raise SystemExit("final-test metrics path must differ from the attempt and frozen-model records")
    data_dir = Path(args.data_dir).expanduser().resolve()
    if output == data_dir / "manifest.json" or output == data_dir / "raw" / "student_outcomes.csv" or output == data_dir / "raw" / "lms_snapshots.csv":
        raise SystemExit("final-test report path must not overwrite input data")
    if output.exists() or attempt_path.exists():
        raise SystemExit("final-test evaluation has already started or completed; refusing a second evaluation")
    frozen_bytes = freeze_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    if frozen.get("test_evaluation_status") != "not_started":
        raise SystemExit("frozen-model record does not permit a final-test evaluation")
    if tuple(frozen.get("cohorts", TEST_COHORTS)) != TEST_COHORTS:
        raise SystemExit("frozen-model record names a different test cohort")

    config_path = Path(args.config).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest_path = data_dir / "manifest.json"
    if hashlib.sha256(config_bytes).hexdigest() != frozen.get("config_sha256"):
        raise SystemExit("model config changed after the model was frozen")
    if _sha256(manifest_path) != frozen.get("training_data_manifest_sha256"):
        raise SystemExit("dataset manifest changed after the model was frozen")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    model_name = str(frozen["model_name"])
    version = str(frozen["model_version"])
    alias = client.get_model_version_by_alias(model_name, "champion")
    if str(alias.version) != version:
        raise SystemExit("champion alias changed after freeze; refusing final-test evaluation")
    model, bundle = load_verified_bundle(
        client,
        model_name,
        version,
        expected_target_version=str(frozen["target_version"]),
        expected_feature_schema_version=str(frozen["feature_schema_version"]),
    )
    bundle_path = mlflow.artifacts.download_artifacts(run_id=str(bundle["mlflow_run_id"]), artifact_path="deployment_bundle.json")
    bundle_hash = _sha256(Path(bundle_path))
    for field, value in (
        ("model_uri", bundle.get("model_uri")),
        ("mlflow_run_id", bundle.get("mlflow_run_id")),
        ("deployment_bundle_sha256", bundle_hash),
        ("threshold", float(bundle["threshold"])),
        ("target_version", bundle.get("target_version")),
        ("feature_schema_version", bundle.get("feature_schema_version")),
    ):
        if frozen.get(field) != value:
            raise SystemExit(f"frozen {field} no longer matches the exact registered model bundle")
    if not np.array_equal(np.asarray(bundle["feature_columns"], dtype=object), np.asarray(frozen["feature_columns"], dtype=object)):
        raise SystemExit("feature columns changed after freeze")

    experiment_name = "student-dropout-final-evaluation"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        existing = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.evaluation.split = 'final_test' and tags.model.version = '{version}'",
            max_results=1,
        )
        if existing:
            raise SystemExit("a final-test MLflow evaluation already exists for this model version")

    _record_once(attempt_path, {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "model_uri": frozen["model_uri"],
        "freeze_record_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "training_manifest_sha256": frozen["training_data_manifest_sha256"],
    })

    training_data = load_training_data(data_dir)
    test_mask = training_data.snapshots["cohort_id"].astype(str).isin(TEST_COHORTS).to_numpy()
    test_snapshots = training_data.snapshots.loc[test_mask].reset_index(drop=True)
    test_labels = training_data.labels.loc[test_mask].reset_index(drop=True).to_numpy(dtype=int)
    if len(test_snapshots) != 600 or len(test_labels) != len(test_snapshots):
        raise RuntimeError(f"expected 600 mature held-out rows, found {len(test_snapshots)}")
    features = make_features(test_snapshots, validate=False)
    probabilities = positive_class_probability(model, features)
    report = final_test_metrics(
        test_labels,
        probabilities,
        float(frozen["threshold"]),
        float(frozen["train_prevalence"]),
        model_version=version,
        target_version=str(frozen["target_version"]),
        feature_schema_version=str(frozen["feature_schema_version"]),
        training_manifest_sha256=str(frozen["training_data_manifest_sha256"]),
        frozen_bundle_sha256=str(frozen["deployment_bundle_sha256"]),
    )
    report.update({
        "mlflow_experiment": experiment_name,
        "config_sha256": frozen["config_sha256"],
        "mlflow_run_id": None,
    })
    _atomic_json(output, report)

    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
    run = client.create_run(
        experiment_id,
        run_name=f"final-test-{model_name}-v{version}",
        tags={
            "synthetic": "true",
            "evaluation.split": "final_test",
            "model.name": model_name,
            "model.version": version,
            "model.run_id": str(bundle["mlflow_run_id"]),
            "target.version": str(frozen["target_version"]),
            "feature_schema.version": str(frozen["feature_schema_version"]),
            "training_manifest.sha256": str(frozen["training_data_manifest_sha256"]),
            "deployment_bundle.sha256": str(frozen["deployment_bundle_sha256"]),
        },
    )
    run_id = run.info.run_id
    report["mlflow_run_id"] = run_id
    _atomic_json(output, report)
    for metric_name in ("average_precision", "roc_auc", "precision", "recall", "f1", "brier_score", "train_prevalence_brier_score", "prevalence", "threshold"):
        value = report.get(metric_name)
        if value is not None:
            client.log_metric(run_id, f"final_test_{metric_name}", float(value))
    client.log_metric(run_id, "final_test_support", int(report["support"]))
    client.log_metric(run_id, "final_test_positive_support", int(report["positive_support"]))
    for name, value in {
        "model_version": version,
        "model_uri": str(frozen["model_uri"]),
        "threshold": str(frozen["threshold"]),
        "target_version": str(frozen["target_version"]),
        "feature_schema_version": str(frozen["feature_schema_version"]),
        "training_manifest_sha256": str(frozen["training_data_manifest_sha256"]),
        "frozen_bundle_sha256": str(frozen["deployment_bundle_sha256"]),
        "support": str(report["support"]),
    }.items():
        client.log_param(run_id, name, value)
    client.log_artifact(run_id, str(output), artifact_path="final_test")
    client.set_terminated(run_id, status="FINISHED")

    frozen["test_evaluation_status"] = "completed"
    frozen["test_evaluated_at_utc"] = report["evaluated_at_utc"]
    frozen["final_test_mlflow_run_id"] = run_id
    _atomic_json(freeze_path, frozen)
    _atomic_json(attempt_path, {
        "status": "completed",
        "started_at_utc": json.loads(attempt_path.read_text(encoding="utf-8"))["started_at_utc"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "mlflow_run_id": run_id,
        "metrics_path": str(output),
    })
    print(json.dumps({"status": "final_test_evaluated_once", "model_version": version, "mlflow_run_id": run_id, "support": report["support"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
