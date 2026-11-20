"""Strict, all-or-nothing batch scoring against a concrete MLflow model version."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient

from .features import make_features
from .model_io import load_verified_bundle
from .training import positive_class_probability
from .validation import FEATURE_COLUMNS, KEY_COLUMNS, DataContractError, validate_snapshot_frame


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the expected model name and local feature-schema contract."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("model configuration must be a YAML mapping")
    expected = tuple(config.get("feature_schema", {}).get("columns", ()))
    if expected != FEATURE_COLUMNS:
        raise ValueError("configured feature schema does not match the scorer allowlist")
    if not config.get("model_name"):
        raise ValueError("model configuration is missing model_name")
    return config


def score_frame(
    snapshots: pd.DataFrame,
    model: Any,
    bundle: dict[str, Any],
    model_version: str,
    *,
    predicted_at: str | None = None,
    batch_id: str | None = None,
) -> pd.DataFrame:
    """Validate every row and return the complete prediction contract."""
    validate_snapshot_frame(snapshots)
    feature_columns = tuple(bundle.get("feature_columns", ()))
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError("deployment bundle feature columns do not match the scorer allowlist")
    features = make_features(snapshots, validate=False)
    probabilities = positive_class_probability(model, features)
    if len(probabilities) != len(snapshots):
        raise ValueError("estimator returned a different number of predictions than input rows")
    threshold = float(bundle["threshold"])
    training_hash = bundle["lineage"]["data_manifest_sha256"]
    timestamp = predicted_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    identifier = batch_id or str(uuid.uuid4())

    output = snapshots.loc[:, KEY_COLUMNS].copy()
    output["predicted_at"] = timestamp
    output["dropout_probability"] = probabilities
    output["risk_band"] = np.where(probabilities >= threshold, "review", "routine")
    output["prediction_status"] = "scored"
    output["model_version"] = str(model_version)
    output["feature_schema_version"] = str(bundle["feature_schema_version"])
    output["training_data_hash"] = training_hash
    output["batch_id"] = identifier
    return output


def write_csv_atomically(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a complete CSV in the destination directory, then replace atomically."""
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".tmp", prefix=f".{target.name}.",
            dir=target.parent, delete=False,
        ) as handle:
            temporary = handle.name
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def score_file(
    input_path: str | Path,
    output_path: str | Path,
    model_version: str,
    *,
    tracking_uri: str,
    config_path: str | Path,
) -> dict[str, Any]:
    """Run one all-or-nothing batch and atomically persist its CSV result."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("output path must differ from input path; the source export cannot be overwritten")
    if not source.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {source}")
    try:
        snapshots = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataContractError(f"cannot read scoring input CSV: {exc}") from exc
    validate_snapshot_frame(snapshots)
    config = load_config(config_path)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model, bundle = load_verified_bundle(
        client,
        str(config["model_name"]),
        str(model_version),
        expected_target_version=str(config["target"]["version"]),
        expected_feature_schema_version=str(config["feature_schema"]["version"]),
    )
    result = score_frame(snapshots, model, bundle, str(model_version))
    destination = write_csv_atomically(result, output_path)
    return {
        "status": "scored",
        "rows": int(len(result)),
        "model_name": config["model_name"],
        "model_version": str(model_version),
        "batch_id": str(result["batch_id"].iloc[0]),
        "output_path": str(destination),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 CSV matching the LMS snapshot export contract")
    parser.add_argument("--output", required=True, help="Destination CSV; replaced only after the full batch succeeds")
    parser.add_argument("--model-version", required=True, help="Concrete gate-passed MLflow registered model version")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--config", default="config/model.yaml", help="Model name and feature-schema contract")
    args = parser.parse_args(argv)
    try:
        result = score_file(
            args.input,
            args.output,
            args.model_version,
            tracking_uri=args.tracking_uri,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"scoring failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
