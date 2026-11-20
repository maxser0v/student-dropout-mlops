from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dropout.score import load_verified_bundle, score_frame, write_csv_atomically
from dropout.validation import FEATURE_COLUMNS


class _FakeModel:
    classes_ = np.asarray([1, 0])
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(np.asarray([[0.70, 0.30]]), (len(features), 1))


def _bundle() -> dict[str, object]:
    return {
        "bundle_version": "1.0",
        "model_name": "student-dropout",
        "model_version": "7",
        "model_uri": "models:/student-dropout/7",
        "mlflow_run_id": "run-123",
        "threshold": 0.65,
        "target_version": "withdrawal-after-day-35-by-course-end-v1",
        "feature_schema_version": "1.0",
        "feature_columns": list(FEATURE_COLUMNS),
        "gate_passed": True,
        "lineage": {"data_manifest_sha256": "a" * 64},
    }


def _gate_report(threshold: float = 0.65) -> dict[str, object]:
    return {
        "passed": True,
        "decision": "passed",
        "mlflow_run_id": "run-123",
        "candidate": {
            "threshold": threshold,
            "threshold_choice": {"threshold": threshold},
            "target_version": "withdrawal-after-day-35-by-course-end-v1",
            "feature_schema_version": "1.0",
        },
    }


@pytest.fixture
def snapshots() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/examples/scoring_input.csv", dtype=str, keep_default_na=False)


def test_score_frame_uses_probability_of_positive_class_and_complete_contract(snapshots: pd.DataFrame) -> None:
    result = score_frame(
        snapshots,
        _FakeModel(),
        _bundle(),
        "7",
        predicted_at="2026-11-20T12:00:00Z",
        batch_id="batch-test",
    )

    assert len(result) == len(snapshots)
    assert list(result.columns) == [
        "tenant_id", "student_key", "course_id", "cohort_id", "snapshot_at",
        "predicted_at", "dropout_probability", "risk_band", "prediction_status",
        "model_version", "feature_schema_version", "training_data_hash", "batch_id",
    ]
    assert np.allclose(result["dropout_probability"], 0.70)
    assert set(result["risk_band"]) == {"review"}
    assert set(result["prediction_status"]) == {"scored"}
    assert set(result["model_version"]) == {"7"}
    assert set(result["training_data_hash"]) == {"a" * 64}
    assert set(result["batch_id"]) == {"batch-test"}


def test_score_frame_rejects_invalid_batch_without_partial_rows() -> None:
    invalid = pd.read_csv(ROOT / "data/fixtures/invalid_snapshot.csv", dtype=str, keep_default_na=False)
    with pytest.raises(ValueError, match="active_days_14d"):
        score_frame(invalid, _FakeModel(), _bundle(), "7")


def test_atomic_csv_write_and_input_path_guard(tmp_path: Path, snapshots: pd.DataFrame) -> None:
    target = tmp_path / "nested" / "predictions.csv"
    result = score_frame(snapshots, _FakeModel(), _bundle(), "7")
    assert write_csv_atomically(result, target) == target.resolve()
    round_trip = pd.read_csv(target, dtype=str)
    assert len(round_trip) == len(snapshots)
    assert not list(target.parent.glob(".*.tmp"))


def test_verified_bundle_binds_threshold_config_and_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = _bundle()
    bundle_path = tmp_path / "deployment_bundle.json"
    gate_path = tmp_path / "gate_report.json"
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "MLmodel").write_text(
        "run_id: run-123\nflavors:\n  sklearn:\n    serialization_format: skops\n    skops_trusted_types:\n    - numpy.dtype\n    pickled_model: model.skops\nmetadata:\n  target_version: withdrawal-after-day-35-by-course-end-v1\n  feature_schema_version: '1.0'\n",
        encoding="utf-8",
    )
    (model_path / "model.skops").write_text("fixture", encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    gate_path.write_text(json.dumps(_gate_report()), encoding="utf-8")

    class Registered:
        run_id = "run-123"
        tags = {
            "gate_passed": "true",
            "target_version": bundle["target_version"],
            "feature_schema_version": bundle["feature_schema_version"],
        }

    class Client:
        def get_model_version(self, model_name: str, version: str) -> Registered:
            assert (model_name, version) == ("student-dropout", "7")
            return Registered()

    downloads = iter((bundle_path, gate_path, model_path))
    monkeypatch.setattr("dropout.model_io.mlflow.artifacts.download_artifacts", lambda **kwargs: str(next(downloads)))
    monkeypatch.setattr("dropout.model_io.skops.io.get_untrusted_types", lambda **kwargs: ["numpy.dtype"])
    monkeypatch.setattr("dropout.model_io.skops.io.load", lambda path, trusted: _FakeModel())

    model, loaded = load_verified_bundle(
        Client(),
        "student-dropout",
        "7",
        expected_target_version="withdrawal-after-day-35-by-course-end-v1",
        expected_feature_schema_version="1.0",
    )
    assert isinstance(model, _FakeModel)
    assert loaded == bundle


@pytest.mark.parametrize(
    "bundle_edits, gate, expected_target, expected_schema, message",
    [
        ({"threshold": 0.55}, _gate_report(), None, None, "threshold does not match"),
        ({}, _gate_report(0.55), None, None, "threshold does not match"),
        ({}, _gate_report(), "another-target", None, "target version does not match"),
        ({}, _gate_report(), None, "2.0", "feature schema version does not match"),
    ],
)
def test_verified_bundle_rejects_release_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bundle_edits: dict[str, object],
    gate: dict[str, object],
    expected_target: str | None,
    expected_schema: str | None,
    message: str,
) -> None:
    bundle = {**_bundle(), **bundle_edits}
    bundle_path = tmp_path / "deployment_bundle.json"
    gate_path = tmp_path / "gate_report.json"
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "MLmodel").write_text(
        "run_id: run-123\nflavors:\n  sklearn:\n    serialization_format: skops\n    skops_trusted_types:\n    - numpy.dtype\n    pickled_model: model.skops\nmetadata:\n  target_version: withdrawal-after-day-35-by-course-end-v1\n  feature_schema_version: '1.0'\n",
        encoding="utf-8",
    )
    (model_path / "model.skops").write_text("fixture", encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    class Registered:
        run_id = "run-123"
        tags = {
            "gate_passed": "true",
            "target_version": _bundle()["target_version"],
            "feature_schema_version": _bundle()["feature_schema_version"],
        }

    class Client:
        def get_model_version(self, model_name: str, version: str) -> Registered:
            return Registered()

    downloads = iter((bundle_path, gate_path))
    monkeypatch.setattr("dropout.model_io.mlflow.artifacts.download_artifacts", lambda **kwargs: str(next(downloads)))

    with pytest.raises(ValueError, match=message):
        load_verified_bundle(
            Client(),
            "student-dropout",
            "7",
            expected_target_version=expected_target,
            expected_feature_schema_version=expected_schema,
        )
