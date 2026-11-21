from __future__ import annotations

import json
from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dropout.gates import evaluate_release_gate
from dropout.training import (
    build_pipeline,
    choose_threshold,
    evaluate_probabilities,
    positive_class_probability,
    reliability_bins,
)
from scripts.promote_model import promote
from scripts.rollback_model import rollback
import scripts.promote_model as promote_module
import scripts.rollback_model as rollback_module
import flows.train_flow as train_flow_module
from types import SimpleNamespace


def test_threshold_uses_inclusive_comparison_and_tie_breaks() -> None:
    choice = choose_threshold([0, 1, 1, 0], [0.2, 0.3, 0.6, 0.5], minimum_precision=0.5, lower=0.3, upper=0.6, step=0.3)
    assert choice.threshold == 0.3
    assert choice.precision == pytest.approx(2 / 3)
    assert choice.recall == 1.0


def test_threshold_reports_none_when_no_threshold_meets_precision() -> None:
    choice = choose_threshold([0, 0, 0, 1], [0.9, 0.8, 0.7, 0.01], minimum_precision=1.0, lower=0.05, upper=0.95, step=0.05)
    assert choice.threshold is None
    assert choice.precision is None
    assert choice.candidates_considered == 19


def test_pipeline_probability_uses_positive_class_and_metrics_name_average_precision() -> None:
    X = pd.DataFrame({"a": [0.0, 0.1, 0.8, 0.9], "b": [np.nan, 0.2, 0.7, 0.8]})
    y = pd.Series([0, 0, 1, 1])
    model = build_pipeline(903).fit(X, y)
    probabilities = positive_class_probability(model, X)
    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    metrics = evaluate_probabilities(y, probabilities, 0.5, train_prevalence=0.5)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert "pr_auc" not in metrics
    assert metrics["support"] == 4
    assert len(metrics["reliability_bins"]) == 10
    assert sum(item["count"] for item in metrics["reliability_bins"]) == 4


def test_reliability_bins_preserve_empty_counts() -> None:
    bins = reliability_bins([0, 1], [0.1, 0.9], bins=4)
    assert [entry["count"] for entry in bins] == [1, 0, 0, 1]


def _candidate() -> dict[str, object]:
    return {
        "average_precision": 0.40,
        "prevalence": 0.20,
        "recall": 0.70,
        "precision": 0.35,
        "brier_score": 0.15,
        "train_prevalence_brier_score": 0.16,
        "target_version": "target-v1",
        "feature_schema_version": "1.0",
        "support": 100,
    }


def _gate_settings() -> dict[str, object]:
    return {
        "gates": {
            "average_precision_above_prevalence": 0.05,
            "minimum_recall": 0.60,
            "minimum_precision": 0.30,
            "maximum_recall_drop": 0.05,
            "maximum_precision_drop": 0.05,
            "maximum_brier_increase": 0.02,
        }
    }


def test_gate_passes_initial_and_champion_delta_checks() -> None:
    champion = {**_candidate(), "recall": 0.72, "precision": 0.37, "brier_score": 0.14}
    report = evaluate_release_gate(_candidate(), _gate_settings(), lineage_passed=True, champion=champion)
    assert report["passed"] is True
    assert report["champion_comparison"]["status"] == "compared"


def test_gate_fails_missing_lineage_and_incompatible_champion() -> None:
    champion = {**_candidate(), "feature_schema_version": "2.0"}
    report = evaluate_release_gate(_candidate(), _gate_settings(), lineage_passed=False, champion=champion)
    assert report["passed"] is False
    assert {item["name"] for item in report["checks"] if not item["passed"]} >= {"input_lineage", "champion_compatibility"}
    assert report["champion_comparison"]["status"] == "incompatible"


def test_champion_lookup_only_swallows_missing_alias_and_surfaces_broken_bundle(monkeypatch) -> None:
    missing_alias = MlflowException("Registered model alias champion not found.", ErrorCode.INVALID_PARAMETER_VALUE)

    class NoChampion:
        def get_model_version_by_alias(self, *_args):
            raise missing_alias

    assert train_flow_module._resolve_champion_bundle(NoChampion(), "student-dropout") == (None, None)

    class BrokenChampion:
        def get_model_version_by_alias(self, *_args):
            return SimpleNamespace(version="4", run_id="run-4")

    artifact_error = MlflowException("deployment bundle artifact unavailable", ErrorCode.RESOURCE_DOES_NOT_EXIST)

    def fail_bundle(*_args):
        raise artifact_error

    monkeypatch.setattr(train_flow_module, "_download_bundle", fail_bundle)
    with pytest.raises(MlflowException, match="deployment bundle artifact unavailable"):
        train_flow_module._resolve_champion_bundle(BrokenChampion(), "student-dropout")


def test_explicit_promotion_and_rollback_are_idempotent(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "registry.db"
    tracking_uri = f"sqlite:///{database}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = "student-dropout-test"
    config = tmp_path / "model.yaml"
    config.write_text(yaml.safe_dump({"model_name": model_name}), encoding="utf-8")
    release_log = tmp_path / "releases.json"

    # Build two valid versions and use the concrete versions returned by MLflow.
    for config_version in ("1.0", "2.0"):
        experiment = client.get_experiment_by_name("promotion-test")
        experiment_id = experiment.experiment_id if experiment else client.create_experiment("promotion-test")
        model = LogisticRegression().fit(np.array([[0], [1], [2], [3]]), np.array([0, 0, 1, 1]))
        with mlflow.start_run(experiment_id=experiment_id) as run:
            model_info = mlflow.sklearn.log_model(model, name="model", pip_requirements=["scikit-learn==1.9.1"])
        pending = mlflow.register_model(model_info.model_uri, model_name, await_registration_for=0)
        bundle = {
            "bundle_version": "1.0", "model_name": model_name,
            "model_version": str(pending.version), "model_uri": f"models:/{model_name}/{pending.version}",
            "mlflow_run_id": run.info.run_id, "threshold": 0.5,
            "target_version": "target-v1", "feature_schema_version": "1.0",
            "config_version": config_version, "config_sha256": f"sha-{config_version}", "gate_passed": True,
        }
        candidate = {"target_version": "target-v1", "feature_schema_version": "1.0"}
        mlflow.log_dict(bundle, "deployment_bundle.json", run_id=run.info.run_id)
        mlflow.log_dict({"passed": True, "decision": "passed", "mlflow_run_id": run.info.run_id, "candidate": candidate}, "gate_report.json", run_id=run.info.run_id)
        client.set_model_version_tag(model_name, pending.version, "gate_passed", "true")
        client.set_model_version_tag(model_name, pending.version, "target_version", "target-v1")
        client.set_model_version_tag(model_name, pending.version, "feature_schema_version", "1.0")

    release_log.write_text("{invalid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        promote("1", tracking_uri=tracking_uri, config_path=config, release_log=release_log)
    with pytest.raises(Exception):
        client.get_model_version_by_alias(model_name, "champion")
    release_log.write_text("[]\n", encoding="utf-8")

    assert promote("1", tracking_uri=tracking_uri, config_path=config, release_log=release_log)["action"] == "promoted"
    assert promote("1", tracking_uri=tracking_uri, config_path=config, release_log=release_log)["action"] == "already_champion"
    real_write = promote_module.write_release_log
    monkeypatch.setattr(promote_module, "write_release_log", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only audit")))
    with pytest.raises(OSError, match="read-only audit"):
        promote("2", tracking_uri=tracking_uri, config_path=config, release_log=release_log)
    assert str(client.get_model_version_by_alias(model_name, "champion").version) == "1"
    monkeypatch.setattr(promote_module, "write_release_log", real_write)
    assert promote("2", tracking_uri=tracking_uri, config_path=config, release_log=release_log)["action"] == "promoted"
    real_rollback_write = rollback_module.write_release_log
    monkeypatch.setattr(rollback_module, "write_release_log", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only audit")))
    with pytest.raises(OSError, match="read-only audit"):
        rollback(tracking_uri=tracking_uri, config_path=config, release_log=release_log)
    assert str(client.get_model_version_by_alias(model_name, "champion").version) == "2"
    monkeypatch.setattr(rollback_module, "write_release_log", real_rollback_write)
    assert rollback(tracking_uri=tracking_uri, config_path=config, release_log=release_log)["action"] == "rolled_back"
    assert rollback(tracking_uri=tracking_uri, config_path=config, release_log=release_log)["action"] == "already_rolled_back"
    assert str(client.get_model_version_by_alias(model_name, "champion").version) == "1"
    events = json.loads(release_log.read_text(encoding="utf-8"))
    assert events[0]["promoted"]["config_version"] == "1.0"
    assert events[1]["previous"]["config_version"] == "1.0"
    assert events[-1]["restored"]["config_version"] == "1.0"


def test_future_gate_policy_is_80_percent_and_historical_config_bytes_are_preserved() -> None:
    import hashlib

    historical_bytes = (ROOT / "config/model-v1.yaml").read_bytes()
    assert hashlib.sha256(historical_bytes).hexdigest() == "b71fb714b1657910e64ba155aa57563f400cde3551822538466e760edbf69f9b"
    historical = yaml.safe_load(historical_bytes)
    future = yaml.safe_load((ROOT / "config/model.yaml").read_text(encoding="utf-8"))
    assert historical["config_version"] == "1.0"
    assert historical["gates"]["minimum_recall"] == 0.60
    assert future["config_version"] == "1.1"
    assert future["gates"]["minimum_recall"] == 0.80

    candidate = {**_candidate(), "recall": 0.75}
    old_report = evaluate_release_gate(candidate, historical, lineage_passed=True)
    future_report = evaluate_release_gate(candidate, future, lineage_passed=True)
    assert old_report["passed"] is True
    assert future_report["passed"] is False
    recall_check = next(check for check in future_report["checks"] if check["name"] == "recall")
    assert recall_check["expected"] == 0.80
    assert recall_check["passed"] is False
