"""Local, linear Metaflow training and gated MLflow candidate-registration flow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import importlib.metadata
import sys
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode
from mlflow.models import infer_signature
from metaflow import FlowSpec, Parameter, card, current, step

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dropout.features import make_features
from dropout.gates import evaluate_release_gate
from dropout.lineage import collect_source_lineage, verify_dvc_directory
from dropout.model_io import load_verified_bundle
from dropout.training import (
    assert_feature_schema,
    build_pipeline,
    choose_threshold,
    evaluate_probabilities,
    positive_class_probability,
    threshold_choice_dict,
)
from dropout.validation import FEATURE_COLUMNS, TEST_COHORTS, TRAIN_COHORTS, VALIDATION_COHORTS, load_training_data, verify_manifest


def _json_dump(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _result_path(path_value: str | Path | None) -> Path | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(path_value)
    candidate = Path(os.path.abspath(path if path.is_absolute() else ROOT / path))
    worker_root_value = os.environ.get("PROFASSIST_WORK_DIR")
    reports_root = Path(os.path.abspath(ROOT / "reports"))
    worker_root = Path(os.path.abspath(worker_root_value)) if worker_root_value else None
    inside_worker = worker_root is not None and candidate != worker_root and worker_root in candidate.parents
    if not inside_worker and (candidate == reports_root or reports_root not in candidate.parents):
        raise ValueError("result-path must be a file inside the project reports/ directory")
    if reports_root.is_symlink() and not inside_worker:
        raise ValueError("result-path reports/ directory cannot be a symlink")
    cursor = candidate.parent
    root_boundary = worker_root if inside_worker else reports_root
    while cursor != root_boundary and root_boundary in cursor.parents:
        if cursor.exists() and cursor.is_symlink():
            raise ValueError("result-path cannot traverse a symlink inside reports/")
        cursor = cursor.parent
    if candidate.is_symlink():
        raise ValueError("result-path cannot be a symlink")
    resolved = candidate.resolve()
    if not inside_worker and reports_root.resolve() not in resolved.parents:
        raise ValueError("result-path resolves outside the project reports/ directory")
    if inside_worker and worker_root.resolve() not in resolved.parents:
        raise ValueError("result-path resolves outside the private worker directory")
    if candidate.exists():
        raise FileExistsError(f"result-path already exists: {candidate}")
    return candidate


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create a result without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as result_file:
            json.dump(payload, result_file, indent=2, sort_keys=True, allow_nan=False)
            result_file.write("\n")
            result_file.flush()
            os.fsync(result_file.fileno())
        # link() is an atomic create-if-absent operation on this filesystem.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _result_payload(
    *,
    status: str,
    candidate_version: str | None,
    mlflow_run_id: str | None,
    metaflow_run_id: str | None,
    gate_passed: bool,
    error: BaseException | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "gate_passed": bool(gate_passed),
        "candidate_version": candidate_version,
        "mlflow_run_id": mlflow_run_id,
        "metaflow_run_id": metaflow_run_id,
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    payload = resolved.read_bytes()
    config = yaml.safe_load(payload)
    if not isinstance(config, dict):
        raise ValueError("model configuration must be a YAML mapping")
    expected = tuple(config.get("feature_schema", {}).get("columns", ()))
    if expected != FEATURE_COLUMNS:
        raise ValueError("configured feature schema does not match the explicit model feature allowlist")
    return config, hashlib.sha256(payload).hexdigest()


def _set_tracking(uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _experiment(client: MlflowClient, name: str, tracking_uri: str) -> str:
    found = client.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    if tracking_uri.startswith(("sqlite:", "file:")):
        artifact_root = (ROOT / "reports" / "mlflow" / "artifacts").resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        return client.create_experiment(name, artifact_location=artifact_root.as_uri())
    return client.create_experiment(name)


def _in_run(run_id: str, operation) -> Any:
    """Reopen one existing run for a step; keep it RUNNING until flow end."""
    mlflow.start_run(run_id=run_id)
    try:
        result = operation()
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="RUNNING")
        return result


def _failure_guard(function):
    """Mark this flow's only MLflow run failed if any step work raises."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except BaseException as exc:
            run_id = getattr(self, "mlflow_run_id", None)
            if run_id:
                try:
                    _set_tracking(getattr(self, "resolved_tracking_uri", self.tracking_uri)).set_terminated(run_id, status="FAILED")
                except Exception:
                    pass
            try:
                result_path = getattr(self, "resolved_result_path", None)
                if result_path is not None:
                    gate_report = getattr(self, "gate_report", {}) or {}
                    _write_result(
                        result_path,
                        _result_payload(
                            status="failed",
                            candidate_version=getattr(self, "registered_version", None),
                            mlflow_run_id=run_id,
                            metaflow_run_id=getattr(self, "flow_run_id", None),
                            gate_passed=bool(gate_report.get("passed", False)),
                            error=exc,
                        ),
                    )
            except Exception:
                pass
            raise
    return wrapped


def _data_lineage(data_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    hashes = {name: info["sha256"] for name, info in manifest.get("files", {}).items()}
    pointer = ROOT / "data.dvc"
    required = {"data/raw/lms_snapshots.csv", "data/raw/student_outcomes.csv"}
    complete = required.issubset(hashes) and all(
        isinstance(hashes.get(name), str) and re.fullmatch(r"[0-9a-fA-F]{64}", hashes[name])
        for name in required
    )
    manifest_files = set(hashes)
    expected_paths = {name.removeprefix("data/") for name in manifest_files} | {"manifest.json"}
    dvc_verification = verify_dvc_directory(
        pointer,
        data_dir,
        expected_path="data",
        expected_files=expected_paths,
    )
    return {
        "dvc_pointer_path": str(pointer.relative_to(ROOT)),
        "dvc_pointer_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest() if pointer.is_file() else None,
        "data_file_sha256": hashes,
        "complete_data_checksums": bool(complete),
        "dvc_directory_verification": dvc_verification,
        "data_manifest_sha256": hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest(),
    }


def _plot_confusion(metrics: dict[str, Any], path: Path) -> Path | None:
    matrix = metrics.get("confusion_matrix")
    if matrix is None:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["retained", "withdrawn"], yticklabels=["retained", "withdrawn"], xlabel="Predicted", ylabel="Observed", title="Validation confusion matrix")
    for (row, col), value in np.ndenumerate(np.asarray(matrix)):
        ax.text(col, row, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_reliability(metrics: dict[str, Any], path: Path) -> Path:
    bins = metrics["reliability_bins"]
    occupied = [entry for entry in bins if entry["count"]]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect agreement")
    if occupied:
        ax.plot([entry["mean_probability"] for entry in occupied], [entry["observed_rate"] for entry in occupied], marker="o", label="validation")
        for entry in occupied:
            ax.annotate(str(entry["count"]), (entry["mean_probability"], entry["observed_rate"]), xytext=(3, 4), textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed withdrawal rate", title="Validation reliability (labels are counts)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _download_bundle(client: MlflowClient, run_id: str) -> dict[str, Any]:
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="deployment_bundle.json")
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


def _resolve_champion_bundle(client: MlflowClient, model_name: str):
    """Return the current champion and its bundle; only absence of alias is optional."""
    try:
        registered = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(ErrorCode.INVALID_PARAMETER_VALUE) and "alias" in str(exc).lower() and "not found" in str(exc).lower():
            return None, None
        if exc.error_code == ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST):
            try:
                client.get_registered_model(model_name)
            except MlflowException as missing_model:
                if missing_model.error_code == ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST):
                    return None, None
                raise
            raise
        raise
    # Do not catch errors after alias resolution: damaged/inaccessible champion artifacts
    # must stop the gate instead of silently disabling challenger comparisons.
    return registered, _download_bundle(client, registered.run_id)


class TrainFlow(FlowSpec):
    data_dir = Parameter("data-dir", default="data", help="Read-only DVC-restored data directory")
    config_path = Parameter("config-path", default="config/model.yaml", help="Frozen model and release-gate configuration")
    tracking_uri = Parameter("tracking-uri", default="sqlite:///reports/mlflow/mlflow.db", help="MLflow tracking and registry URI")
    experiment_name = Parameter("experiment-name", default="student-dropout", help="MLflow experiment name")
    result_path = Parameter("result-path", default=None, help="Optional machine-readable run result JSON path")
    data_mode = Parameter("data-mode", default="synthetic", help="synthetic demo data or immutable platform export")
    platform_scope = Parameter("platform-scope", default="{}", help="JSON scope for platform export models")
    active_model_version = Parameter("active-model-version", default=None, help="Concrete active version for scoped candidate comparison")
    train_cohorts_param = Parameter("train-cohorts", default="", help="Comma-separated chronological training cohorts")
    validation_cohorts_param = Parameter("validation-cohorts", default="", help="Comma-separated validation cohorts")
    test_cohorts_param = Parameter("test-cohorts", default="", help="Comma-separated excluded test cohorts")

    @_failure_guard
    @card
    @step
    def start(self):
        self.flow_run_id = str(current.run_id)
        self.resolved_result_path = _result_path(self.result_path)
        self.resolved_data_dir = str(Path(self.data_dir).resolve())
        self.resolved_config_path = str(Path(self.config_path).resolve())
        self.resolved_tracking_uri = self.tracking_uri
        if self.tracking_uri.startswith("sqlite:///reports/"):
            relative_db = self.tracking_uri.removeprefix("sqlite:///")
            self.resolved_tracking_uri = "sqlite:///" + str((ROOT / relative_db).resolve())
        self.config, self.config_sha256 = _load_config(self.resolved_config_path)
        if self.data_mode not in {"synthetic", "platform", "platform-fixture"}:
            raise ValueError("data-mode must be synthetic, platform, or platform-fixture")
        self.platform_scope_value = json.loads(self.platform_scope)
        platform_mode = self.data_mode in {"platform", "platform-fixture"}
        if platform_mode and (not isinstance(self.platform_scope_value, dict) or set(self.platform_scope_value) != {"tenant_id"}):
            raise ValueError("platform-scope must be JSON with exactly tenant_id")
        if self.data_mode == "platform-fixture" and "fixture" not in str(self.config["model_name"]).lower():
            raise ValueError("platform-fixture mode requires an isolated registry model name containing 'fixture'")
        if not platform_mode and self.active_model_version:
            raise ValueError("active-model-version is available only in platform mode")
        if platform_mode and self.active_model_version and not str(self.active_model_version).isdecimal():
            raise ValueError("active-model-version must be a concrete positive registry version")
        self.train_cohorts = tuple(x for x in self.train_cohorts_param.split(",") if x) or TRAIN_COHORTS
        self.validation_cohorts = tuple(x for x in self.validation_cohorts_param.split(",") if x) or VALIDATION_COHORTS
        self.test_cohorts = tuple(x for x in self.test_cohorts_param.split(",") if x) or TEST_COHORTS
        all_cohorts = (*self.train_cohorts, *self.validation_cohorts, *self.test_cohorts)
        if len(set(all_cohorts)) != len(all_cohorts) or not self.train_cohorts or not self.validation_cohorts:
            raise ValueError("train, validation, and test cohorts must be disjoint with nonempty train and validation")
        self.seed = int(self.config["seed"])
        client = _set_tracking(self.resolved_tracking_uri)
        experiment_id = _experiment(client, self.experiment_name, self.resolved_tracking_uri)
        tags = {
            "synthetic": str(self.data_mode != "platform").lower(),
            "metaflow.flow": str(current.flow_name),
            "metaflow.run_id": self.flow_run_id,
            "feature_schema_version": str(self.config["feature_schema"]["version"]),
            "target_version": str(self.config["target"]["version"]),
        }
        run = client.create_run(experiment_id, run_name=f"dropout-{self.flow_run_id}", tags=tags)
        self.mlflow_run_id = run.info.run_id
        report_root = Path(os.environ.get("PROFASSIST_WORK_DIR", ROOT / "reports"))
        report_parent = report_root / "reports" if os.environ.get("PROFASSIST_WORK_DIR") else report_root
        self.report_dir = str((report_parent / self.flow_run_id).resolve())
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        self.next(self.validate)

    @_failure_guard
    @card
    @step
    def validate(self):
        client = _set_tracking(self.resolved_tracking_uri)
        try:
            platform_mode = self.data_mode in {"platform", "platform-fixture"}
            manifest = verify_manifest(self.resolved_data_dir, mode=self.data_mode,
                                       expected_scope=self.platform_scope_value if platform_mode else None)
            if platform_mode:
                manifest_split = manifest["split"]
                if (tuple(manifest_split["train"]) != self.train_cohorts
                        or tuple(manifest_split["validation"]) != self.validation_cohorts
                        or tuple(manifest_split.get("test", [])) != self.test_cohorts):
                    raise ValueError("requested cohort splits do not match the immutable export manifest")
            training_data = load_training_data(self.resolved_data_dir, train_cohorts=self.train_cohorts,
                                               validation_cohorts=self.validation_cohorts,
                                               test_cohorts=self.test_cohorts)
            snapshots = training_data.snapshots
            cohorts = snapshots["cohort_id"].astype(str)
            train_mask = cohorts.isin(self.train_cohorts).to_numpy()
            validation_mask = cohorts.isin(self.validation_cohorts).to_numpy()
            if not train_mask.any() or not validation_mask.any():
                raise ValueError("both train and validation cohorts must have mature supervised labels")

            # Keep only allowed cohorts from this point forward. Test rows and labels are discarded.
            train_snapshots = snapshots.loc[train_mask].reset_index(drop=True)
            validation_snapshots = snapshots.loc[validation_mask].reset_index(drop=True)
            self.X_train = make_features(train_snapshots, validate=False)
            self.X_validation = make_features(validation_snapshots, validate=False)
            self.y_train = training_data.labels.loc[train_mask].reset_index(drop=True).astype(int)
            self.y_validation = training_data.labels.loc[validation_mask].reset_index(drop=True).astype(int)
            assert_feature_schema(self.X_train)
            assert_feature_schema(self.X_validation)
            if len(self.X_train) != len(self.y_train) or len(self.X_validation) != len(self.y_validation):
                raise ValueError("features and labels are not aligned")
            self.exclusions = dict(training_data.exclusions)
            self.split_metadata = {
                "train": {"cohorts": list(self.train_cohorts), "rows": int(len(self.X_train)), "positive_count": int(self.y_train.sum())},
                "validation": {"cohorts": list(self.validation_cohorts), "rows": int(len(self.X_validation)), "positive_count": int(self.y_validation.sum())},
                "test": {"cohorts": list(self.test_cohorts), "rows": int(training_data.split_counts["test"])},
                "student_overlap": training_data.student_overlap,
                "exclusions": self.exclusions,
            }
            self.manifest = manifest
            if self.data_mode == "synthetic":
                self.lineage = {**collect_source_lineage(ROOT), **_data_lineage(Path(self.resolved_data_dir), manifest)}
            else:
                manifest_bytes = (Path(self.resolved_data_dir) / "manifest.json").read_bytes()
                self.lineage = {
                    **collect_source_lineage(ROOT),
                    "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "data_file_sha256": {name: info["sha256"] for name, info in manifest["files"].items()},
                    "platform_export_id": manifest["export_id"],
                    "platform_scope": self.platform_scope_value,
                    "platform_snapshot_sha256": manifest["files"]["raw/lms_snapshots.csv"]["sha256"],
                    "platform_outcomes_sha256": manifest["files"]["raw/student_outcomes.csv"]["sha256"],
                    "cohorts": list(manifest["cohorts"]),
                    "split": {"train": list(self.train_cohorts), "validation": list(self.validation_cohorts), "test": list(self.test_cohorts)},
                    "verified_platform_export": self.data_mode == "platform",
                    "synthetic_fixture": self.data_mode == "platform-fixture",
                }
            dvc_verification = self.lineage.get("dvc_directory_verification", {"passed": True, "mode": "platform_export"})
            if not dvc_verification["passed"]:
                lineage_path = _json_dump(Path(self.report_dir) / "lineage.json", self.lineage)

                def log_lineage_failure():
                    mlflow.set_tag("dvc.directory.verified", "false")
                    mlflow.log_dict(dvc_verification, "dvc_verification.json")
                    mlflow.log_artifact(str(lineage_path), artifact_path="provenance")

                _in_run(self.mlflow_run_id, log_lineage_failure)
                raise ValueError(f"DVC-tracked data directory verification failed: {dvc_verification.get('reason')}")
            self.dependency_versions = {
                "python": sys.version.split()[0],
                "pandas": importlib.metadata.version("pandas"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
                "mlflow": importlib.metadata.version("mlflow"),
                "metaflow": importlib.metadata.version("metaflow"),
            }

            def log_validation():
                mlflow.log_params({
                    "seed": self.seed,
                    "feature_schema_version": self.config["feature_schema"]["version"],
                    "target_version": self.config["target"]["version"],
                    "config_sha256": self.config_sha256,
                    "data_manifest_sha256": self.lineage["data_manifest_sha256"],
                    "dvc_pointer_sha256": self.lineage.get("dvc_pointer_sha256") or "unavailable",
                    "train_rows": len(self.X_train),
                    "validation_rows": len(self.X_validation),
                    "train_prevalence": round(float(self.y_train.mean()), 8),
                    "validation_prevalence": round(float(self.y_validation.mean()), 8),
                    "python_version": self.dependency_versions["python"],
                    "pandas_version": self.dependency_versions["pandas"],
                    "scikit_learn_version": self.dependency_versions["scikit_learn"],
                    "mlflow_version": self.dependency_versions["mlflow"],
                    "metaflow_version": self.dependency_versions["metaflow"],
                })
                mlflow.set_tags({
                    "git.commit": self.lineage["git_commit"] or "unavailable",
                    "git.dirty": str(self.lineage["git_dirty"]),
                    "git.commit.source": str(self.lineage["git_commit_source"]),
                    "source.revision.source": str(self.lineage["source_revision_source"]),
                    "source.tree.sha256": str(self.lineage["source_tree_sha256"] or "unavailable"),
                    "source.tree.origin": str(self.lineage["source_tree_sha256_source"]),
                    "docker.image.id": str(self.lineage["docker_image_id"] or "unavailable"),
                    "docker.image.id.source": str(self.lineage["docker_image_id_source"]),
                    "dvc.pointer.sha256": str(self.lineage.get("dvc_pointer_sha256") or "unavailable"),
                    "dvc.directory.verified": str(self.lineage.get("dvc_directory_verification", {}).get("passed", False)).lower(),
                })
                mlflow.log_params({
                    "git_commit": self.lineage["git_commit"] or "unavailable",
                    "git_dirty": self.lineage["git_dirty"],
                    "git_commit_source": self.lineage["git_commit_source"],
                    "source_revision_source": self.lineage["source_revision_source"],
                    "source_tree_sha256": self.lineage["source_tree_sha256"] or "unavailable",
                    "docker_image_id": self.lineage["docker_image_id"] or "unavailable",
                    "dvc_pointer_sha256": self.lineage.get("dvc_pointer_sha256") or "unavailable",
                })
                _json_dump(Path(self.report_dir) / "manifest.json", self.manifest)
                _json_dump(Path(self.report_dir) / "lineage.json", self.lineage)
                _json_dump(Path(self.report_dir) / "split_metadata.json", self.split_metadata)
                _json_dump(Path(self.report_dir) / "dependency_versions.json", self.dependency_versions)
                for artifact in ("manifest.json", "lineage.json", "split_metadata.json", "dependency_versions.json"):
                    mlflow.log_artifact(str(Path(self.report_dir) / artifact), artifact_path="provenance")
                mlflow.log_artifact(self.resolved_config_path, artifact_path="provenance")

            _in_run(self.mlflow_run_id, log_validation)
        except BaseException:
            client.set_terminated(self.mlflow_run_id, status="FAILED")
            raise
        self.next(self.prepare)

    @_failure_guard
    @card
    @step
    def prepare(self):
        # All feature construction happened in validate via the shared allowlisted feature function.
        self.train_prevalence = float(self.y_train.mean())
        self.validation_prevalence = float(self.y_validation.mean())
        self.feature_columns = list(self.X_train.columns)
        self.next(self.train)

    @_failure_guard
    @card
    @step
    def train(self):
        _set_tracking(self.resolved_tracking_uri)
        model = build_pipeline(seed=self.seed)
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.train_parameters = {key: str(value) for key, value in model.get_params(deep=True).items()}
        self.training_probability_summary = {
            "rows": int(len(self.X_train)),
            "classes": [int(value) for value in model.classes_],
            "train_prevalence": self.train_prevalence,
        }

        def log_training():
            mlflow.log_params(self.train_parameters)
            example = self.X_train.head(5)
            example_output = np.asarray(model.predict_proba(example), dtype=float)
            signature = infer_signature(example, example_output)
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=example,
                signature=signature,
                pip_requirements=["scikit-learn==1.9.1", "pandas>=2.3,<3"],
                metadata={"target_version": self.config["target"]["version"], "feature_schema_version": self.config["feature_schema"]["version"]},
                skops_trusted_types=["numpy.dtype"],
            )
            return model_info.model_uri

        self.logged_model_uri = _in_run(self.mlflow_run_id, log_training)
        self.next(self.evaluate)

    @_failure_guard
    @card
    @step
    def evaluate(self):
        _set_tracking(self.resolved_tracking_uri)
        probabilities = positive_class_probability(self.model, self.X_validation)
        threshold_config = self.config["threshold_selection"]
        choice = choose_threshold(
            self.y_validation,
            probabilities,
            minimum_precision=float(threshold_config["minimum_precision"]),
            lower=float(threshold_config["lower"]),
            upper=float(threshold_config["upper"]),
            step=float(threshold_config["step"]),
        )
        self.threshold_choice = threshold_choice_dict(choice)
        self.threshold = choice.threshold
        self.validation_metrics = evaluate_probabilities(
            self.y_validation,
            probabilities,
            choice.threshold,
            train_prevalence=self.train_prevalence,
        )
        self.candidate_metadata = {
            **self.validation_metrics,
            "target_version": self.config["target"]["version"],
            "feature_schema_version": self.config["feature_schema"]["version"],
            "threshold_choice": self.threshold_choice,
            "train_prevalence": self.train_prevalence,
            "config_version": self.config["config_version"],
            "config_sha256": self.config_sha256,
            "feature_columns": self.feature_columns,
            "mlflow_run_id": self.mlflow_run_id,
            "metaflow_run_id": self.flow_run_id,
            "lineage": self.lineage,
            "split_metadata": self.split_metadata,
        }
        evaluation_path = _json_dump(Path(self.report_dir) / "validation_metrics.json", self.candidate_metadata)
        confusion_plot = _plot_confusion(self.validation_metrics, Path(self.report_dir) / "confusion_matrix.png")
        reliability_plot = _plot_reliability(self.validation_metrics, Path(self.report_dir) / "reliability.png")

        client = _set_tracking(self.resolved_tracking_uri)

        def log_evaluation():
            metric_values = {
                key: float(value)
                for key, value in self.validation_metrics.items()
                if key in {"average_precision", "roc_auc", "precision", "recall", "f1", "brier_score", "train_prevalence_brier_score", "prevalence"}
                and value is not None
            }
            metric_values.update({"validation_rows": self.validation_metrics["support"]})
            if self.threshold is not None:
                metric_values["threshold"] = self.threshold
            mlflow.log_metrics(metric_values)
            mlflow.log_dict(self.candidate_metadata, "validation_metrics.json")
            mlflow.log_dict({"threshold_choice": self.threshold_choice, "metrics": self.validation_metrics}, "evaluation.json")
            mlflow.log_artifact(str(evaluation_path))
            mlflow.log_artifact(str(reliability_plot))
            if confusion_plot is not None:
                mlflow.log_artifact(str(confusion_plot))

        _in_run(self.mlflow_run_id, log_evaluation)
        self.next(self.gate)

    @_failure_guard
    @card
    @step
    def gate(self):
        client = _set_tracking(self.resolved_tracking_uri)
        champion_metadata = None
        champion_version = None
        if self.data_mode in {"platform", "platform-fixture"} and self.active_model_version:
            registered = client.get_model_version(self.config["model_name"], str(self.active_model_version))
            old_bundle = _download_bundle(client, registered.run_id)
            if old_bundle.get("lineage", {}).get("platform_scope") != self.platform_scope_value:
                raise ValueError("active model scope does not match the requested platform scope")
        elif self.data_mode == "synthetic":
            registered, old_bundle = _resolve_champion_bundle(client, self.config["model_name"])
        else:
            # A tenant with no activated scoped release starts as an explicit
            # bootstrap candidate; never compare against another tenant's alias.
            registered, old_bundle = None, None
        if registered is not None:
            champion_version = str(registered.version)
            champion_metadata = {
                "target_version": old_bundle.get("target_version"),
                "feature_schema_version": old_bundle.get("feature_schema_version"),
            }
            if (
                champion_metadata["target_version"] == self.candidate_metadata["target_version"]
                and champion_metadata["feature_schema_version"] == self.candidate_metadata["feature_schema_version"]
            ):
                champion_model, old_bundle = load_verified_bundle(
                    client,
                    self.config["model_name"],
                    champion_version,
                    expected_target_version=self.candidate_metadata["target_version"],
                    expected_feature_schema_version=self.candidate_metadata["feature_schema_version"],
                )
                champion_probabilities = positive_class_probability(champion_model, self.X_validation)
                champion_metrics = evaluate_probabilities(
                    self.y_validation,
                    champion_probabilities,
                    old_bundle["threshold"],
                    train_prevalence=float(old_bundle["train_prevalence"]),
                )
                champion_metadata.update(champion_metrics)
                champion_metadata["target_version"] = old_bundle["target_version"]
                champion_metadata["feature_schema_version"] = old_bundle["feature_schema_version"]
            else:
                champion_metadata["version"] = champion_version

        self.gate_report = evaluate_release_gate(
            self.candidate_metadata,
            self.config,
            lineage_passed=bool(
                (self.data_mode == "platform" and self.lineage.get("verified_platform_export")
                 and self.lineage.get("data_manifest_sha256")
                 and self.lineage.get("platform_snapshot_sha256")
                 and self.lineage.get("platform_outcomes_sha256"))
                or (self.data_mode == "platform-fixture" and self.lineage.get("synthetic_fixture")
                    and self.lineage.get("data_manifest_sha256")
                    and self.lineage.get("platform_snapshot_sha256")
                    and self.lineage.get("platform_outcomes_sha256"))
                or (self.data_mode == "synthetic" and self.lineage.get("dvc_pointer_sha256")
                    and self.lineage.get("complete_data_checksums")
                    and self.lineage.get("data_manifest_sha256")
                    and self.lineage.get("dvc_directory_verification", {}).get("passed"))
            ),
            champion=champion_metadata,
        )
        self.gate_report["champion_version"] = champion_version
        if self.data_mode in {"platform", "platform-fixture"}:
            self.gate_report["platform_scope"] = self.platform_scope_value
            self.gate_report["platform_snapshot_sha256"] = self.lineage["platform_snapshot_sha256"]
            self.gate_report["bootstrap_candidate"] = self.active_model_version is None
        self.gate_report["mlflow_run_id"] = self.mlflow_run_id
        self.gate_report["metaflow_run_id"] = self.flow_run_id
        gate_path = _json_dump(Path(self.report_dir) / "gate_report.json", self.gate_report)

        def log_gate():
            mlflow.set_tag("release_gate", self.gate_report["decision"])
            mlflow.log_dict(self.gate_report, "gate_report.json")
            mlflow.log_artifact(str(gate_path))
            mlflow.log_metric("release_gate_passed", int(self.gate_report["passed"]))

        _in_run(self.mlflow_run_id, log_gate)
        self.next(self.register)

    @_failure_guard
    @card
    @step
    def register(self):
        self.registered_version = None
        self.registration_status = "blocked_by_release_gate"
        if self.gate_report["passed"]:
            client = _set_tracking(self.resolved_tracking_uri)
            registered = mlflow.register_model(self.logged_model_uri, self.config["model_name"], await_registration_for=0)
            self.registered_version = str(registered.version)
            self.registration_status = "registered_candidate"
            bundle = {
                "bundle_version": "1.0",
                "model_name": self.config["model_name"],
                "model_version": self.registered_version,
                "model_uri": f"models:/{self.config['model_name']}/{self.registered_version}",
                "mlflow_run_id": self.mlflow_run_id,
                "metaflow_run_id": self.flow_run_id,
                "threshold": self.threshold,
                "target_version": self.config["target"]["version"],
                "target_description": self.config["target"]["description"],
                "feature_schema_version": self.config["feature_schema"]["version"],
                "feature_columns": self.feature_columns,
                "config_version": self.config["config_version"],
                "config_sha256": self.config_sha256,
                "train_prevalence": self.train_prevalence,
                "gate_passed": True,
                "gate_report_artifact": "gate_report.json",
                "validation_metrics": self.validation_metrics,
                "lineage": self.lineage,
                "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            bundle_path = _json_dump(Path(self.report_dir) / "deployment_bundle.json", bundle)
            client.log_artifact(self.mlflow_run_id, str(bundle_path))
            client.set_model_version_tag(self.config["model_name"], self.registered_version, "gate_passed", "true")
            client.set_model_version_tag(self.config["model_name"], self.registered_version, "feature_schema_version", str(bundle["feature_schema_version"]))
            client.set_model_version_tag(self.config["model_name"], self.registered_version, "target_version", str(bundle["target_version"]))
            client.set_model_version_tag(self.config["model_name"], self.registered_version, "bundle_artifact", "deployment_bundle.json")
        self.next(self.end)

    @_failure_guard
    @card
    @step
    def end(self):
        client = _set_tracking(self.resolved_tracking_uri)
        self.flow_summary = {
            **_result_payload(
                status="completed",
                candidate_version=self.registered_version,
                mlflow_run_id=self.mlflow_run_id,
                metaflow_run_id=self.flow_run_id,
                gate_passed=bool(self.gate_report["passed"]),
            ),
            "metaflow_run_id": self.flow_run_id,
            "mlflow_run_id": self.mlflow_run_id,
            "gate_decision": self.gate_report["decision"],
            "registration_status": self.registration_status,
            "model_version": self.registered_version,
            "report_dir": self.report_dir,
        }
        if self.resolved_result_path is not None:
            _write_result(self.resolved_result_path, self.flow_summary)
        client.set_terminated(self.mlflow_run_id, status="FINISHED")
        print(json.dumps(self.flow_summary, sort_keys=True))


if __name__ == "__main__":
    TrainFlow()
