from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dropout.features import make_features
from dropout.monitor import (
    MONITORED_FEATURES,
    PROFILE_TAG,
    _json_bytes,
    _outcome_metrics,
    calculate_psi,
    create_reference_profile,
    load_verified_reference_profile,
    monitor_file,
    monitor_frame,
    profile_from_snapshot_export,
)
from dropout.validation import FEATURE_COLUMNS, KEY_COLUMNS, TRAIN_COHORTS


class FakeProbabilityModel:
    classes_ = np.asarray([1, 0])
    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(np.asarray([[0.7, 0.3]], dtype=float), (len(features), 1))


def _bundle() -> dict[str, object]:
    manifest_path = ROOT / "data" / "manifest.json"
    return {
        "model_name": "student-dropout",
        "model_version": "1",
        "model_uri": "models:/student-dropout/1",
        "mlflow_run_id": "training-run-1",
        "threshold": 0.65,
        "target_version": "withdrawal-after-day-35-by-course-end-v1",
        "feature_schema_version": "1.0",
        "feature_columns": list(FEATURE_COLUMNS),
        "lineage": {"data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
    }


def _train_snapshots() -> pd.DataFrame:
    snapshots = pd.read_csv(ROOT / "data/raw/lms_snapshots.csv", dtype=str, keep_default_na=False)
    return snapshots.loc[snapshots["cohort_id"].isin(TRAIN_COHORTS)].reset_index(drop=True)


def _profile() -> dict[str, object]:
    features = make_features(_train_snapshots())
    bundle = _bundle()
    return create_reference_profile(
        features,
        model_name=str(bundle["model_name"]),
        model_version=str(bundle["model_version"]),
        mlflow_run_id=str(bundle["mlflow_run_id"]),
        data_manifest_sha256=str(bundle["lineage"]["data_manifest_sha256"]),
        target_version=str(bundle["target_version"]),
        feature_schema_version=str(bundle["feature_schema_version"]),
    )


def _profile_sha256(profile: dict[str, object]) -> str:
    return hashlib.sha256(_json_bytes(profile)).hexdigest()


def test_psi_equal_distributions_shift_and_empty_bins_are_finite() -> None:
    assert calculate_psi([0.2, 0.3, 0.5], [20, 30, 50]) == pytest.approx(0.0)
    assert calculate_psi([0.5, 0.5], [100, 0]) > 0.2
    assert np.isfinite(calculate_psi([0.5, 0.5, 0.0], [0, 10, 0]))


def test_profile_uses_training_snapshots_only_and_records_fixed_bins() -> None:
    bundle = _bundle()
    profile = profile_from_snapshot_export(ROOT / "data", bundle)
    assert profile["train_cohorts"] == list(TRAIN_COHORTS)
    assert profile["train_rows"] == 2400
    assert profile["data_manifest_sha256"] == bundle["lineage"]["data_manifest_sha256"]
    assert set(profile["features"]) == set(MONITORED_FEATURES)
    for feature in MONITORED_FEATURES:
        summary = profile["features"][feature]
        assert sum(summary["bin_counts"]) == summary["valid_reference_count"]
        assert sum(summary["bin_proportions"]) == pytest.approx(1.0)


def test_profile_uses_scoped_manifest_train_cohorts_and_hashes() -> None:
    fixture_root = ROOT / "fixtures" / "profassist" / "dropout-training"
    manifest_path = fixture_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = {
        **_bundle(),
        "lineage": {
            "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "platform_scope": manifest["platform_scope"],
            "platform_export_id": manifest["export_id"],
            "split": manifest["split"],
        },
    }

    profile = profile_from_snapshot_export(fixture_root, bundle)

    assert profile["train_cohorts"] == manifest["split"]["train"] == ["2024_S1"]
    assert profile["train_rows"] == 40
    assert profile["platform_scope"] == manifest["platform_scope"]
    assert profile["platform_export_id"] == manifest["export_id"]
    assert profile["data_manifest_sha256"] == bundle["lineage"]["data_manifest_sha256"]


def test_clean_reference_distribution_has_negligible_psi() -> None:
    train = _train_snapshots()
    profile = _profile()
    report = monitor_frame(train, FakeProbabilityModel(), _bundle(), "1", profile, profile_sha256=_profile_sha256(profile), minimum_rows=100)
    assert report["drift_status"] == "clear"
    assert report["drift_alert"] is False
    assert all(report["features"][name]["psi"] == pytest.approx(0.0) for name in MONITORED_FEATURES)
    assert report["outcomes"]["status"] == "data_not_ready"
    assert report["retraining_recommendation"]["state"] == "data_not_ready"


def test_drift_fixture_alerts_and_small_batch_is_insufficient() -> None:
    profile = _profile()
    drift = pd.read_csv(ROOT / "data/fixtures/drifted_scoring_input.csv", dtype=str, keep_default_na=False)
    report = monitor_frame(drift, FakeProbabilityModel(), _bundle(), "1", profile, profile_sha256=_profile_sha256(profile))
    assert report["drift_status"] == "alert"
    assert report["drift_alert"] is True
    assert any(report["features"][name]["psi"] > 0.2 for name in MONITORED_FEATURES)
    assert report["outcomes"]["status"] == "data_not_ready"
    assert report["retraining_recommendation"]["state"] == "data_not_ready"
    assert "collect mature outcomes" in report["retraining_recommendation"]["reason"]
    small = pd.read_csv(ROOT / "data/examples/scoring_input.csv", dtype=str, keep_default_na=False)
    small_report = monitor_frame(small, FakeProbabilityModel(), _bundle(), "1", profile, profile_sha256=_profile_sha256(profile))
    assert small_report["drift_status"] == "insufficient_data"
    assert small_report["drift_alert"] is None
    assert all(item["psi"] is None for item in small_report["features"].values())


def test_invalid_batch_fails_before_any_distribution_report() -> None:
    invalid = pd.read_csv(ROOT / "data/fixtures/invalid_snapshot.csv", dtype=str, keep_default_na=False)
    with pytest.raises(ValueError, match="active_days_14d"):
        profile = _profile()
        monitor_frame(invalid, FakeProbabilityModel(), _bundle(), "1", profile, profile_sha256=_profile_sha256(profile))


def test_missing_unknown_and_unmatured_outcomes_remain_pending() -> None:
    snapshots = pd.read_csv(ROOT / "data/examples/scoring_input.csv", dtype=str, keep_default_na=False).iloc[:3].copy()
    rows = []
    for index in (0, 1):
        row = snapshots.iloc[index]
        entry = {key: row[key] for key in KEY_COLUMNS}
        if index == 0:
            entry.update(final_outcome="withdrawn", withdrawn_at="2025-10-07T00:00:00Z", outcome_observed_at="2026-01-01T00:00:00Z")
        else:
            entry.update(final_outcome="unknown", withdrawn_at="", outcome_observed_at="2026-01-01T00:00:00Z")
        rows.append(entry)
    outcomes = pd.DataFrame(rows)
    predictions = pd.DataFrame(snapshots.loc[:, KEY_COLUMNS].copy())
    predictions["dropout_probability"] = 0.7
    predictions["risk_band"] = "review"
    result = _outcome_metrics(snapshots, predictions, outcomes, _bundle())
    assert result["status"] == "reported"
    assert result["support"] == 1
    assert result["positive_support"] == 1
    assert result["pending_count"] == 2
    assert result["simulated"] is True


def test_profile_loader_verifies_checksum_and_release_bindings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = _profile()
    payload = (json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path = tmp_path / "reference_profile.json"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    class Registered:
        run_id = "training-run-1"
        tags = {PROFILE_TAG: digest}

    class Client:
        def get_model_version(self, model_name: str, version: str) -> Registered:
            assert (model_name, version) == ("student-dropout", "1")
            return Registered()

    monkeypatch.setattr("dropout.monitor.mlflow.artifacts.download_artifacts", lambda **kwargs: str(path))
    loaded, actual_digest = load_verified_reference_profile(Client(), "student-dropout", "1", _bundle())
    assert loaded == profile
    assert actual_digest == digest


def test_monitor_report_cannot_overwrite_input_or_outcomes(tmp_path: Path) -> None:
    source = ROOT / "data/examples/scoring_input.csv"
    with pytest.raises(ValueError, match="must differ"):
        monitor_file(source, "1", tracking_uri="http://127.0.0.1:5000", config_path=ROOT / "config/model.yaml", report_path=source)
    outcomes = tmp_path / "outcomes.csv"
    outcomes.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must differ"):
        monitor_file(source, "1", tracking_uri="http://127.0.0.1:5000", config_path=ROOT / "config/model.yaml", report_path=outcomes, outcomes_path=outcomes)
