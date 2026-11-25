"""Run scoped ProfAssist worker operations from a trusted JSON job envelope.

The caller owns authorization and places all inputs/artifacts in a private
per-job directory. Paths in the envelope are intentionally not accepted.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON number")))
    if not isinstance(value, dict) or set(value) != {"schema_version", "operation", "context", "payload"}:
        raise ValueError("input must be a versioned worker envelope")
    if value["schema_version"] != 1 or not isinstance(value["context"], dict) or not isinstance(value["payload"], dict):
        raise ValueError("unsupported worker envelope")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.link(temp, path)
    except FileExistsError:
        if temp.read_bytes() != path.read_bytes():
            raise ValueError("output path already contains a different result")
    finally:
        temp.unlink(missing_ok=True)


def _required(context: dict[str, Any], field: str) -> str:
    value = context.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"context.{field} is required")
    return value


def _safe_work_file(work: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("artifact names must be relative paths inside the job directory")
    resolved = (work / name).resolve()
    if work.resolve() not in resolved.parents:
        raise ValueError("artifact path escapes work directory")
    return resolved


def _input_file(work: Path, primary: str, alternate: str | None = None) -> Path:
    for name in (primary, alternate):
        if name:
            candidate = _safe_work_file(work, name)
            if candidate.is_file():
                return candidate
    return _safe_work_file(work, primary)


def _materialize_files(envelope: dict[str, Any], work: Path) -> None:
    files = envelope["payload"].get("files")
    if files is None:
        return
    allowed = {"snapshots.csv", "outcomes.csv", "manifest.json",
               "raw/lms_snapshots.csv", "raw/student_outcomes.csv"}
    if not isinstance(files, dict) or set(files) - allowed:
        raise ValueError("payload.files contains an unsupported filename")
    total = 0
    for name, item in files.items():
        if not isinstance(item, dict) or set(item) != {"encoding", "content"} or item["encoding"] != "base64":
            raise ValueError("each payload.files entry must use base64 encoding and content")
        content = base64.b64decode(item["content"], validate=True)
        total += len(content)
        if total > 128 * 1024 * 1024:
            raise ValueError("worker input files exceed the 128 MiB job limit")
        target = _safe_work_file(work, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise ValueError("job input file already exists with different content")
        if not target.exists():
            target.write_bytes(content)


def _scope_snapshot_root(scope: dict[str, Any]) -> Path:
    root_value = os.environ.get("PROFASSIST_WORKER_SCOPE_DIR")
    if not root_value:
        raise ValueError("PROFASSIST_WORKER_SCOPE_DIR must point at the private shared worker volume")
    root = Path(root_value).resolve()
    if root.is_symlink():
        raise ValueError("worker scope storage root cannot be a symlink")
    scope_key = _sha(_canonical(scope))
    target = root / "plagiarism-snapshots" / scope_key
    target.mkdir(parents=True, exist_ok=True)
    return target


def _scope_snapshot_dir(scope: dict[str, Any], manifest_sha256: str) -> Path:
    if len(manifest_sha256) != 64 or any(char not in "0123456789abcdef" for char in manifest_sha256):
        raise ValueError("snapshot manifest checksum must be lowercase SHA-256")
    return _scope_snapshot_root(scope) / manifest_sha256


def _plagiarism(operation: str, envelope: dict[str, Any], work: Path) -> dict[str, Any]:
    from plagiarism.platform_worker import run_score, run_snapshot, run_verify_release
    context, payload = envelope["context"], envelope["payload"]
    in_path = _safe_work_file(work, "worker-input.json")
    out_path = _safe_work_file(work, "worker-result.json")
    in_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    args = SimpleNamespace(input=str(in_path), output=str(out_path),
                           tracking_uri=context.get("tracking_uri"),
                           model_name=context.get("model_name"),
                           version=context.get("version"),
                           encoder_dir=context.get("encoder_dir", "/app/models/minilm"),
                           fixture=bool(context.get("fixture", False)))
    if operation in {"plagiarism-score", "plagiarism-verify-release"}:
        args.tracking_uri = _required(context, "tracking_uri")
        args.model_name = _required(context, "model_name")
        args.version = _required(context, "version")
    if operation == "plagiarism-score":
        args.index = str(_safe_work_file(work, "index"))
        args.index_manifest = str(_safe_work_file(work, "index-manifest.json"))
        result = run_score(args)
    elif operation == "plagiarism-snapshot":
        args.snapshot_dir = str(_safe_work_file(work, "snapshot-staging"))
        args.validation_fraction = 0.2
        args.test_fraction = 0.0
        args.seed = 903
        result = run_snapshot(args)
        manifest_hash = result["manifest_sha256"]
        published = _scope_snapshot_dir(payload["scope"], manifest_hash)
        if published.exists():
            existing_manifest = published / "manifest.json"
            if not existing_manifest.is_file() or _sha(existing_manifest.read_bytes()) != manifest_hash:
                raise ValueError("scoped snapshot checksum path contains different content")
            shutil.rmtree(args.snapshot_dir, ignore_errors=True)
        else:
            temporary = published.with_name(f".{published.name}.{os.getpid()}.tmp")
            shutil.copytree(args.snapshot_dir, temporary)
            try:
                os.rename(temporary, published)
            except FileExistsError:
                shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(args.snapshot_dir, ignore_errors=True)
        result["scope_snapshot_key"] = _sha(_canonical(payload["scope"]))
    else:
        result = run_verify_release(args)
    return result


def _dropout_score(envelope: dict[str, Any], work: Path) -> dict[str, Any]:
    from dropout.score import load_config, score_file
    ctx = envelope["context"]
    source = _input_file(work, "snapshots.csv", "raw/lms_snapshots.csv")
    output = _safe_work_file(work, "predictions.csv")
    config_path = ctx.get("config_path", ROOT / "config/model.yaml")
    config = load_config(config_path)
    if _required(ctx, "model_name") != config["model_name"]:
        raise ValueError("requested dropout model name does not match the trusted scorer config")
    bundle = _verify_dropout_scope(ctx, config["model_name"])
    summary = score_file(source, output, _required(ctx, "version"),
                         tracking_uri=_required(ctx, "tracking_uri"),
                         config_path=config_path)
    import pandas as pd
    frame = pd.read_csv(output, dtype=str, keep_default_na=False)
    return {"summary": {**{k: v for k, v in summary.items() if k != "output_path"},
                         "platform_scope": bundle["lineage"]["platform_scope"]},
            "predictions": frame.to_dict(orient="records"),
            "artifacts": {"predictions.csv": _sha(output.read_bytes())}}


def _dropout_monitor(envelope: dict[str, Any], work: Path) -> dict[str, Any]:
    from dropout.monitor import monitor_file, write_report
    from dropout.score import load_config
    ctx = envelope["context"]
    source = _input_file(work, "snapshots.csv", "raw/lms_snapshots.csv")
    report_path = _safe_work_file(work, "monitor-report.json")
    outcomes = _input_file(work, "outcomes.csv", "raw/student_outcomes.csv")
    config_path = ctx.get("config_path", ROOT / "config/model.yaml")
    config = load_config(config_path)
    if _required(ctx, "model_name") != config["model_name"]:
        raise ValueError("requested dropout model name does not match the trusted monitor config")
    bundle = _verify_dropout_scope(ctx, config["model_name"])
    report = monitor_file(source, _required(ctx, "version"),
                          tracking_uri=_required(ctx, "tracking_uri"),
                          config_path=config_path,
                          report_path=report_path,
                          outcomes_path=outcomes if outcomes.is_file() else None)
    report["platform_scope"] = bundle["lineage"]["platform_scope"]
    report.pop("report_path", None)
    write_report(report_path, report)
    return {"summary": report, "artifacts": {"monitor-report.json": _sha(report_path.read_bytes())}}


def _verify_dropout_scope(context: dict[str, Any], model_name: str) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    from dropout.model_io import load_verified_bundle
    scope = _scope_for_dropout(context)
    uri, version = _required(context, "tracking_uri"), _required(context, "version")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    _model, bundle = load_verified_bundle(client, model_name, version)
    lineage = bundle.get("lineage") or {}
    if lineage.get("platform_scope") != scope:
        raise ValueError("verified dropout model scope does not match the requested tenant")
    if bool(lineage.get("synthetic_fixture")) != bool(context.get("fixture", False)):
        raise ValueError("synthetic fixture models require an explicit fixture operation mode")
    return bundle


def _dropout_verify(envelope: dict[str, Any]) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    from dropout.model_io import load_verified_bundle
    ctx = envelope["context"]
    uri, name, version = (_required(ctx, key) for key in ("tracking_uri", "model_name", "version"))
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    _model, bundle = load_verified_bundle(client, name, version)
    scope = envelope["payload"].get("scope")
    if not isinstance(scope, dict) or set(scope) != {"tenant_id"}:
        raise ValueError("dropout release scope must contain tenant_id")
    lineage = bundle.get("lineage") or {}
    if lineage.get("synthetic_fixture"):
        raise ValueError("synthetic fixture candidates cannot be verified as production releases")
    if lineage.get("platform_scope") != scope:
        raise ValueError("verified model scope does not match the requested tenant")
    gate_path = Path(mlflow.artifacts.download_artifacts(run_id=bundle["mlflow_run_id"], artifact_path="gate_report.json"))
    gate_bytes = gate_path.read_bytes()
    bundle_path = Path(mlflow.artifacts.download_artifacts(run_id=bundle["mlflow_run_id"], artifact_path="deployment_bundle.json"))
    bundle_sha256 = _sha(bundle_path.read_bytes())
    gate = json.loads(gate_bytes)
    lineage = bundle["lineage"]
    if (gate.get("platform_scope") != scope
            or gate.get("platform_snapshot_sha256") != lineage.get("platform_snapshot_sha256")
            or gate.get("passed") is not True or gate.get("decision") != "passed"):
        raise ValueError("dropout release gate is not bound to the verified scope and training snapshot")
    return {"model_name": name, "model_version": version,
            "model_uri": bundle["model_uri"], "gate_passed": True,
            "model_bundle_sha256": bundle_sha256,
            "platform_scope": scope,
            "platform_snapshot_sha256": lineage.get("platform_snapshot_sha256"),
            "training_manifest_sha256": lineage.get("data_manifest_sha256"),
            "gate_report_sha256": _sha(gate_bytes)}


def _dropout_train(envelope: dict[str, Any], work: Path) -> dict[str, Any]:
    """Launch the existing gated Metaflow flow against a private export directory."""
    from dropout.score import load_config
    ctx, payload = envelope["context"], envelope["payload"]
    config_path = ctx.get("config_path", ROOT / "config/model.yaml")
    config = load_config(config_path)
    if _required(ctx, "model_name") != config["model_name"]:
        raise ValueError("requested dropout model name does not match the trusted training config")
    data_dir = work
    result_path = _safe_work_file(work, "train-result.json")
    args = [sys.executable, str(ROOT / "flows/train_flow.py"), "run",
            "--data-dir", str(data_dir),
            "--config-path", str(config_path),
            "--tracking-uri", _required(ctx, "tracking_uri"),
            "--experiment-name", str(ctx.get("experiment_name", "student-dropout-platform")),
            "--result-path", str(result_path), "--data-mode", "platform-fixture" if ctx.get("fixture") else "platform",
            "--platform-scope", json.dumps(_scope_for_dropout(ctx), sort_keys=True),
            "--train-cohorts", ",".join(payload["train_cohorts"]),
            "--validation-cohorts", ",".join(payload["validation_cohorts"]),
            "--test-cohorts", ",".join(payload.get("test_cohorts", []))]
    if ctx.get("version") is not None:
        args.extend(["--active-model-version", _required(ctx, "version")])
    child_env = os.environ.copy()
    child_env["PROFASSIST_WORK_DIR"] = str(work)
    child_env.setdefault("METAFLOW_USER", "profassist-worker")
    child_env["METAFLOW_DATASTORE_SYSROOT_LOCAL"] = str(_safe_work_file(work, "metaflow"))
    completed = subprocess.run(args, cwd=ROOT, env=child_env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"dropout training flow failed with exit code {completed.returncode}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("report_dir", None)
    monitoring_profile = None
    candidate_version = result.get("candidate_version")
    if result.get("gate_passed") is True and candidate_version:
        from dropout.monitor import create_and_register_reference_profile

        profile_path = _safe_work_file(
            work, f"monitoring/reference-profile-v{candidate_version}.json"
        )
        monitoring_profile = create_and_register_reference_profile(
            data_dir,
            str(candidate_version),
            tracking_uri=_required(ctx, "tracking_uri"),
            model_name=config["model_name"],
            output_path=profile_path,
        )
        result["monitoring_profile"] = {
            "status": monitoring_profile["status"],
            "sha256": monitoring_profile["sha256"],
            "artifact": monitoring_profile["artifact"],
        }
    result["model_name"] = config["model_name"]
    result["platform_scope"] = _scope_for_dropout(ctx)
    result["bootstrap_candidate"] = ctx.get("version") is None
    artifact_hashes = {}
    for name in ("manifest.json", "gate_report.json", "validation_metrics.json", "deployment_bundle.json"):
        matches = list((work / "reports").glob(f"*/{name}"))
        if matches:
            artifact = max(matches, key=lambda path: path.stat().st_mtime_ns)
            rel = artifact.relative_to(work).as_posix()
            artifact_hashes[rel] = _sha(artifact.read_bytes())
    if monitoring_profile is not None:
        artifact_hashes[profile_path.relative_to(work).as_posix()] = _sha(profile_path.read_bytes())
    return {"summary": result, "artifacts": artifact_hashes}


def _plagiarism_train(envelope: dict[str, Any], work: Path) -> dict[str, Any]:
    ctx, payload = envelope["context"], envelope["payload"]
    scope = ctx.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("context.scope is required for plagiarism training")
    snapshot = _scope_snapshot_dir(scope, _required(payload, "snapshot_manifest_sha256"))
    result_path = _safe_work_file(work, "plagiarism-train-result.json")
    args = [sys.executable, str(ROOT / "flows/train_plagiarism_flow.py"), "run",
            "--data-dir", str(snapshot),
            "--config-path", str(ctx.get("config_path", ROOT / "config/plagiarism.yaml")),
            "--tracking-uri", _required(ctx, "tracking_uri"),
            "--cache-dir", str(_safe_work_file(work, "plagiarism-cache")),
            "--result-path", str(result_path),
            "--active-model-name", _required(ctx, "model_name"),
            "--active-model-version", _required(ctx, "version"),
            "--platform-snapshot-sha256", _required(payload, "snapshot_manifest_sha256"),
            "--encoder-dir", str(ctx.get("encoder_dir", "/app/models/minilm"))]
    child_env = os.environ.copy()
    child_env["PROFASSIST_WORK_DIR"] = str(work)
    child_env.setdefault("METAFLOW_USER", "profassist-worker")
    child_env["METAFLOW_DATASTORE_SYSROOT_LOCAL"] = str(_safe_work_file(work, "metaflow"))
    completed = subprocess.run(args, cwd=ROOT, env=child_env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"plagiarism training flow failed with exit code {completed.returncode}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("report_dir", None)
    artifacts = {}
    for path in work.rglob("*"):
        if path.is_file() and path != result_path:
            artifacts[path.relative_to(work).as_posix()] = _sha(path.read_bytes())
    result.pop("report_dir", None)
    return {"summary": result, "scope_snapshot_key": _sha(_canonical(scope)), "artifacts": artifacts}


def _scope_for_dropout(context: dict[str, Any]) -> dict[str, str]:
    scope = context.get("scope")
    if not isinstance(scope, dict) or set(scope) != {"tenant_id"}:
        raise ValueError("context.scope must contain tenant_id")
    return {"tenant_id": str(scope["tenant_id"])}


def execute(operation: str, envelope: dict[str, Any], work: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if operation == "plagiarism-train":
        result = _plagiarism_train(envelope, work)
        artifacts = result.pop("artifacts", {})
        return result, artifacts
    elif operation.startswith("plagiarism-"):
        result = _plagiarism(operation, envelope, work)
        artifacts = {}
        for path in work.rglob("*"):
            if path.is_file() and path.name not in {"worker-input.json", "worker-result.json"}:
                artifacts[path.relative_to(work).as_posix()] = _sha(path.read_bytes())
        return result, artifacts
    if operation == "dropout-score":
        result = _dropout_score(envelope, work)
    elif operation == "dropout-monitor":
        result = _dropout_monitor(envelope, work)
    elif operation == "dropout-verify-release":
        result = _dropout_verify(envelope)
    elif operation == "dropout-train":
        result = _dropout_train(envelope, work)
    else:
        raise ValueError("unsupported worker operation")
    artifacts = result.pop("artifacts", {})
    return result, artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plagiarism-score", "plagiarism-snapshot", "plagiarism-train",
        "plagiarism-verify-release", "dropout-score", "dropout-train", "dropout-verify-release", "dropout-monitor"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args(argv)
    try:
        work = Path(args.work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        envelope = _read(Path(args.input))
        if envelope["operation"] != args.operation:
            raise ValueError("operation argument does not match the input envelope")
        _materialize_files(envelope, work)
        result, artifacts = execute(args.operation, envelope, work)
        response = {"schema_version": 1, "operation": args.operation, "status": "completed",
                    "result": result, "artifacts": artifacts,
                    "input_sha256": _sha(_canonical(envelope))}
        response["result_sha256"] = _sha(_canonical(response))
        _write(Path(args.output), response)
        print(json.dumps({"schema_version": 1, "operation": args.operation, "status": "completed"}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"worker failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
