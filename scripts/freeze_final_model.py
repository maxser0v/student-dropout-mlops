"""Freeze one verified registry version before the single final-test assessment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dropout.model_io import load_verified_bundle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _git_state() -> dict[str, object]:
    def call(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = call("status", "--porcelain")
    return {
        "commit": call("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_entry_count": len(status.splitlines()) if status else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--config", default=str(ROOT / "config/model.yaml"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "reports/demo/frozen_model.json"))
    args = parser.parse_args()
    model_version = str(args.model_version)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"freeze record already exists: {output}")
    config_path = Path(args.config).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest_path = Path(args.data_dir).expanduser().resolve() / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    model_name = str(config["model_name"])
    champion = client.get_model_version_by_alias(model_name, "champion")
    if str(champion.version) != model_version:
        raise SystemExit(f"refusing to freeze version {model_version}: current champion is {champion.version}")
    _, bundle = load_verified_bundle(
        client,
        model_name,
        model_version,
        expected_target_version=str(config["target"]["version"]),
        expected_feature_schema_version=str(config["feature_schema"]["version"]),
    )
    if bundle.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest():
        raise SystemExit("current model config checksum does not match the selected model bundle")
    if bundle.get("lineage", {}).get("data_manifest_sha256") != manifest_hash:
        raise SystemExit("current data manifest hash does not match the selected model bundle")
    if float(bundle.get("threshold")) != 0.15:
        raise SystemExit("the frozen final-test model must retain the previously selected 0.15 threshold")
    bundle_artifact = mlflow.artifacts.download_artifacts(
        run_id=str(bundle["mlflow_run_id"]), artifact_path="deployment_bundle.json"
    )
    frozen = {
        "freeze_version": "1.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "model_uri": str(bundle["model_uri"]),
        "mlflow_run_id": str(bundle["mlflow_run_id"]),
        "deployment_bundle_sha256": _sha256(Path(bundle_artifact)),
        "config_version": str(bundle["config_version"]),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "threshold": float(bundle["threshold"]),
        "target_version": str(bundle["target_version"]),
        "feature_schema_version": str(bundle["feature_schema_version"]),
        "feature_columns": list(bundle["feature_columns"]),
        "training_data_manifest_sha256": manifest_hash,
        "train_prevalence": float(bundle["train_prevalence"]),
        "champion_alias_version_at_freeze": str(champion.version),
        "git_state": _git_state(),
        "test_evaluation_status": "not_started",
    }
    _atomic_json(output, frozen)
    print(json.dumps({"status": "frozen", "output": str(output), "bundle_sha256": frozen["deployment_bundle_sha256"], "model_uri": frozen["model_uri"], "threshold": frozen["threshold"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
