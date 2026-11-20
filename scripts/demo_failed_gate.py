"""Demonstrate a genuine validation-only gate failure without registry release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dropout.gates import evaluate_release_gate
from dropout.training import evaluate_probabilities
from dropout.validation import VALIDATION_COHORTS, load_training_data


def _model_versions(client: MlflowClient, model_name: str) -> list[str]:
    versions = client.search_model_versions(f"name='{model_name}'")
    return sorted(str(version.version) for version in versions)


def _atomic_json(path: Path, payload: dict) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default="1", help="Current champion version used for the guard check")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--output", default="reports/demo/failed_baseline_gate.json")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"failed-gate demonstration already exists: {output}")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    model_name = str(config["model_name"])
    champion_before = str(client.get_model_version_by_alias(model_name, "champion").version)
    if champion_before != str(args.model_version):
        raise SystemExit(f"expected champion {args.model_version}, found {champion_before}")
    versions_before = _model_versions(client, model_name)

    training_data = load_training_data(args.data_dir)
    validation_mask = training_data.snapshots["cohort_id"].astype(str).isin(VALIDATION_COHORTS).to_numpy()
    validation_labels = training_data.labels.loc[validation_mask].to_numpy(dtype=int)
    if not len(validation_labels):
        raise SystemExit("validation cohort has no mature supervised labels")
    frozen_run = client.get_model_version(model_name, champion_before)
    bundle_path = mlflow.artifacts.download_artifacts(run_id=frozen_run.run_id, artifact_path="deployment_bundle.json")
    bundle_bytes = Path(bundle_path).read_bytes()
    bundle = json.loads(bundle_bytes)
    train_prevalence = float(bundle["train_prevalence"])
    threshold = float(bundle["threshold"])
    constant_probabilities = np.full(len(validation_labels), train_prevalence, dtype=float)
    metrics = evaluate_probabilities(
        validation_labels,
        constant_probabilities,
        threshold,
        train_prevalence=train_prevalence,
    )
    candidate = {
        **metrics,
        "candidate_type": "constant_train_prevalence_baseline",
        "target_version": bundle["target_version"],
        "feature_schema_version": bundle["feature_schema_version"],
        "model_version": None,
        "mlflow_run_id": None,
        "config_version": bundle["config_version"],
        "config_sha256": bundle["config_sha256"],
    }
    report = evaluate_release_gate(candidate, config, lineage_passed=bool(bundle.get("lineage", {}).get("complete_data_checksums")))
    report.update({
        "demonstration": "failed_validation_gate_for_constant_baseline",
        "synthetic": True,
        "champion_before": champion_before,
        "champion_version_under_comparison": str(bundle["model_version"]),
        "deployment_bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "registered_model_versions_before": versions_before,
        "candidate_registered": False,
    })
    if report["passed"]:
        raise RuntimeError("constant prevalence baseline unexpectedly passed the unchanged release gate")

    _atomic_json(output, report)
    experiment_name = "student-dropout-gate-demo"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
    run = client.create_run(
        experiment_id,
        run_name="constant-prevalence-validation-gate-failure",
        tags={
            "synthetic": "true",
            "demo": "failed-release-gate",
            "evaluation.split": "validation",
            "model.name": model_name,
            "champion.version": champion_before,
            "gate.decision": str(report["decision"]),
            "candidate.registered": "false",
        },
    )
    run_id = run.info.run_id
    for name in ("average_precision", "roc_auc", "precision", "recall", "f1", "brier_score", "train_prevalence_brier_score", "threshold", "prevalence"):
        value = candidate.get(name)
        if value is not None:
            client.log_metric(run_id, f"validation_{name}", float(value))
    client.log_metric(run_id, "release_gate_passed", 0.0)
    client.log_param(run_id, "baseline", "constant_train_prevalence")
    client.log_param(run_id, "validation_rows", len(validation_labels))
    report["mlflow_run_id"] = run_id
    _atomic_json(output, report)
    client.log_artifact(run_id, str(output))
    client.set_terminated(run_id, status="FINISHED")

    champion_after = str(client.get_model_version_by_alias(model_name, "champion").version)
    versions_after = _model_versions(client, model_name)
    if champion_after != champion_before or versions_after != versions_before:
        raise RuntimeError("failed-gate demonstration unexpectedly changed the model registry")
    report["champion_after"] = champion_after
    report["registered_model_versions_after"] = versions_after
    _atomic_json(output, report)
    print(json.dumps({"status": "gate_failed_as_expected", "mlflow_run_id": run_id, "champion": champion_after, "candidate_registered": False, "failed_checks": [check["name"] for check in report["checks"] if not check["passed"]], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
