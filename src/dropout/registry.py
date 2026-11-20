"""Safe local registry helpers for explicit promotion and rollback commands."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import mlflow
from mlflow import MlflowClient


@contextmanager
def release_lock(path: str | Path) -> Iterator[None]:
    """Serialize local alias changes and their release-log writes."""
    import fcntl

    lock_path = Path(path).with_suffix(Path(path).suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def read_release_log(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    parsed = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"release log must contain a JSON list: {target}")
    return parsed


def write_release_log(path: str | Path, events: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(events, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def version_bundle(client: MlflowClient, model_name: str, version: str) -> tuple[Any, dict[str, Any]]:
    model_version = client.get_model_version(model_name, str(version))
    tags = model_version.tags or {}
    if tags.get("gate_passed", "").lower() != "true":
        raise ValueError(f"model version {model_name}/{version} does not have gate_passed=true")
    artifact = mlflow.artifacts.download_artifacts(run_id=model_version.run_id, artifact_path="deployment_bundle.json")
    bundle = json.loads(Path(artifact).read_text(encoding="utf-8"))
    if bundle.get("gate_passed") is not True or str(bundle.get("model_version")) != str(version):
        raise ValueError(f"model version {model_name}/{version} has an invalid deployment bundle")
    expected_uri = f"models:/{model_name}/{version}"
    if bundle.get("model_name") != model_name:
        raise ValueError(f"model version {model_name}/{version} bundle names another registered model")
    if bundle.get("model_uri") != expected_uri:
        raise ValueError(f"model version {model_name}/{version} bundle does not point to its immutable model URI")
    if str(bundle.get("mlflow_run_id")) != str(model_version.run_id):
        raise ValueError(f"model version {model_name}/{version} bundle run ID does not match registry lineage")
    for tag_name, bundle_name in (("target_version", "target_version"), ("feature_schema_version", "feature_schema_version")):
        if tags.get(tag_name) != str(bundle.get(bundle_name)):
            raise ValueError(f"model version {model_name}/{version} {tag_name} tag does not match its bundle")
    gate_path = mlflow.artifacts.download_artifacts(run_id=model_version.run_id, artifact_path="gate_report.json")
    gate_report = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    if gate_report.get("passed") is not True or gate_report.get("decision") != "passed":
        raise ValueError(f"model version {model_name}/{version} has no passing gate report")
    gate_candidate = gate_report.get("candidate") or {}
    if str(gate_report.get("mlflow_run_id")) != str(model_version.run_id):
        raise ValueError(f"model version {model_name}/{version} gate report belongs to another run")
    if (
        gate_candidate.get("target_version") != bundle.get("target_version")
        or gate_candidate.get("feature_schema_version") != bundle.get("feature_schema_version")
    ):
        raise ValueError(f"model version {model_name}/{version} gate report schema or target does not match its bundle")
    return model_version, bundle


def current_champion(client: MlflowClient, model_name: str) -> tuple[str | None, dict[str, Any] | None]:
    try:
        model_version = client.get_model_version_by_alias(model_name, "champion")
    except Exception as exc:
        message = str(exc).lower()
        if "does not exist" in message or "not found" in message or "resource_does_not_exist" in message:
            return None, None
        raise
    version = str(model_version.version)
    _, bundle = version_bundle(client, model_name, version)
    return version, bundle


def release_entry(version: str | None, bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if version is None or bundle is None:
        return None
    return {
        "model_version": str(version),
        "model_uri": bundle.get("model_uri"),
        "config_version": bundle.get("config_version"),
        "config_sha256": bundle.get("config_sha256"),
        "feature_schema_version": bundle.get("feature_schema_version"),
        "target_version": bundle.get("target_version"),
        "threshold": bundle.get("threshold"),
        "mlflow_run_id": bundle.get("mlflow_run_id"),
        "bundle_artifact": "deployment_bundle.json",
    }
