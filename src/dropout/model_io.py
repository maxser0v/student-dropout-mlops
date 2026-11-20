"""Verified loading of a gated, concrete MLflow sklearn model version."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import skops.io
import yaml
from mlflow import MlflowClient

from .validation import FEATURE_COLUMNS

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load_verified_bundle(
    client: MlflowClient,
    model_name: str,
    model_version: str,
    *,
    expected_target_version: str | None = None,
    expected_feature_schema_version: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Verify registry tags, run artifacts, lineage and immutable model URI."""
    if not model_version.isdecimal() or int(model_version) < 1:
        raise ValueError("model version must be a positive concrete integer version")
    registered = client.get_model_version(model_name, model_version)
    tags = registered.tags or {}
    if tags.get("gate_passed", "").lower() != "true":
        raise ValueError(f"model version {model_name}/{model_version} is not gate-passed")

    bundle_path = mlflow.artifacts.download_artifacts(
        run_id=registered.run_id,
        artifact_path="deployment_bundle.json",
    )
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    expected_uri = f"models:/{model_name}/{model_version}"
    expected_fields = {
        "bundle_version": "1.0",
        "model_name": model_name,
        "model_version": model_version,
        "model_uri": expected_uri,
        "mlflow_run_id": str(registered.run_id),
        "gate_passed": True,
    }
    for field, expected in expected_fields.items():
        if bundle.get(field) != expected:
            raise ValueError(f"deployment bundle field {field!r} does not match registry version {model_name}/{model_version}")
    if tuple(bundle.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise ValueError("deployment bundle feature columns do not match the scorer allowlist")
    if not bundle.get("target_version") or not bundle.get("feature_schema_version"):
        raise ValueError("deployment bundle is missing target or feature schema version")
    if expected_target_version is not None and bundle["target_version"] != expected_target_version:
        raise ValueError("registered model target version does not match the configured scorer target")
    if expected_feature_schema_version is not None and bundle["feature_schema_version"] != expected_feature_schema_version:
        raise ValueError("registered model feature schema version does not match the configured scorer schema")
    threshold = bundle.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("deployment bundle threshold must be finite and within [0, 1]")
    lineage = bundle.get("lineage")
    training_hash = lineage.get("data_manifest_sha256") if isinstance(lineage, dict) else None
    if not isinstance(training_hash, str) or not _SHA256.fullmatch(training_hash):
        raise ValueError("deployment bundle has no valid training manifest SHA-256")
    for tag_name, bundle_name in (("target_version", "target_version"), ("feature_schema_version", "feature_schema_version")):
        if tags.get(tag_name) != str(bundle.get(bundle_name)):
            raise ValueError(f"registry tag {tag_name!r} does not match deployment bundle")

    gate_path = mlflow.artifacts.download_artifacts(
        run_id=registered.run_id,
        artifact_path="gate_report.json",
    )
    gate_report = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    candidate = gate_report.get("candidate") or {}
    if gate_report.get("passed") is not True or gate_report.get("decision") != "passed":
        raise ValueError("model version does not have a passing gate report")
    if str(gate_report.get("mlflow_run_id")) != str(registered.run_id):
        raise ValueError("gate report run ID does not match registry lineage")
    if candidate.get("target_version") != bundle["target_version"]:
        raise ValueError("gate report target version does not match deployment bundle")
    if candidate.get("feature_schema_version") != bundle["feature_schema_version"]:
        raise ValueError("gate report feature schema does not match deployment bundle")
    approved_threshold = candidate.get("threshold")
    if isinstance(approved_threshold, bool) or not isinstance(approved_threshold, (int, float)):
        raise ValueError("gate report has no numeric approved threshold")
    if float(approved_threshold) != float(threshold):
        raise ValueError("deployment bundle threshold does not match the gate-approved threshold")
    threshold_choice = candidate.get("threshold_choice")
    if isinstance(threshold_choice, dict) and threshold_choice.get("threshold") != approved_threshold:
        raise ValueError("gate report threshold choice does not match its approved threshold")

    model_directory = Path(mlflow.artifacts.download_artifacts(artifact_uri=expected_uri)).resolve()
    mlmodel_path = model_directory / "MLmodel"
    if not mlmodel_path.is_file():
        raise ValueError("registered model artifact is missing its MLmodel metadata")
    model_metadata = yaml.safe_load(mlmodel_path.read_text(encoding="utf-8"))
    if not isinstance(model_metadata, dict) or str(model_metadata.get("run_id")) != str(registered.run_id):
        raise ValueError("registered model MLmodel run ID does not match registry lineage")
    flavor = model_metadata.get("flavors", {}).get("sklearn", {})
    if flavor.get("serialization_format") != "skops":
        raise ValueError("registered model must use the locked skops serialization format")
    if flavor.get("skops_trusted_types") != ["numpy.dtype"]:
        raise ValueError("registered model skops trust list must contain only numpy.dtype")
    flavor_metadata = model_metadata.get("metadata") or {}
    if flavor_metadata.get("target_version") != bundle["target_version"]:
        raise ValueError("registered model target metadata does not match its deployment bundle")
    if flavor_metadata.get("feature_schema_version") != bundle["feature_schema_version"]:
        raise ValueError("registered model schema metadata does not match its deployment bundle")
    model_filename = flavor.get("pickled_model")
    if not isinstance(model_filename, str) or Path(model_filename).name != model_filename:
        raise ValueError("registered model metadata contains an invalid model artifact path")
    model_path = model_directory / model_filename
    if not model_path.is_file():
        raise ValueError("registered model artifact file is missing")
    untrusted_types = set(skops.io.get_untrusted_types(file=model_path))
    trusted_types = ["numpy.dtype"]
    if not untrusted_types.issubset(set(trusted_types)):
        raise ValueError("registered model requests a type outside the approved skops trust list")
    model = skops.io.load(model_path, trusted=trusted_types)
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("registered estimator does not provide predict_proba")
    if tuple(getattr(model, "feature_names_in_", ())) != FEATURE_COLUMNS:
        raise ValueError("registered estimator feature names do not match the deployment bundle")
    if 1 not in np.asarray(getattr(model, "classes_", [])).tolist():
        raise ValueError("registered estimator does not expose positive class 1")
    return model, bundle
