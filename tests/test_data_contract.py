import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from dropout.features import FEATURE_COLUMNS, make_features
from dropout.validation import (
    DataContractError,
    load_training_data,
    validate_snapshot_frame,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def snapshots():
    return pd.read_csv(DATA / "raw" / "lms_snapshots.csv", dtype=str, keep_default_na=False)


def test_clean_bundle_and_manifest_pass(snapshots):
    assert len(validate_snapshot_frame(snapshots)) == 3600
    manifest = verify_manifest(DATA)
    assert manifest["synthetic"] is True


def test_manifest_must_hash_both_supervised_training_inputs(tmp_path):
    copied_data = tmp_path / "data"
    shutil.copytree(DATA, copied_data)
    manifest_path = copied_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["data/raw/student_outcomes.csv"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="student_outcomes.csv"):
        verify_manifest(copied_data)


def test_invalid_snapshot_fixture_fails():
    invalid = pd.read_csv(DATA / "fixtures" / "invalid_snapshot.csv", dtype=str, keep_default_na=False)
    with pytest.raises(DataContractError, match="active_days_14d"):
        validate_snapshot_frame(invalid)


@pytest.mark.parametrize("mutate, message", [
    (lambda f: f.__setitem__("active_days_14d", "19"), "active_days_14d"),
    (lambda f: pd.concat([f, f], ignore_index=True), "duplicate join keys"),
    (lambda f: f.__setitem__("last_activity_at", "2025-01-01T00:00:00Z"), "last_activity_at"),
    (lambda f: f.__setitem__("activity_coverage_start_at", "2025-01-01T00:00:00Z"), "coverage"),
    (lambda f: f.__setitem__("mean_score_pct_to_date", ""), "mean_score_pct_to_date"),
    (lambda f: f.__setitem__("source_system", ""), "source_system is required"),
    (lambda f: f.__setitem__("dropout_label", "1"), "unexpected columns"),
])
def test_contract_rejects_bad_rows(snapshots, mutate, message):
    row = snapshots.iloc[[0]].copy()
    bad = mutate(row)
    if bad is not None:
        row = bad
    with pytest.raises(DataContractError, match=message):
        validate_snapshot_frame(row)


def test_feature_values_schema_and_no_row_drops(snapshots):
    batch = snapshots.iloc[:2].copy()
    result = make_features(batch)
    assert tuple(result.columns) == FEATURE_COLUMNS
    assert len(result) == len(batch)
    for i, row in batch.iterrows():
        expected_missed = int(row["assignments_due_to_date"]) - int(row["assignments_submitted_to_date"])
        expected_change = int(row["active_days_14d"]) - int(row["active_days_previous_14d"])
        assert result.loc[i, "missed_assignments"] == expected_missed
        assert result.loc[i, "activity_change"] == expected_change
        if row["last_activity_at"] == "":
            assert result.loc[i, "days_since_last_activity"] == 35
            assert result.loc[i, "no_activity_history"] == 1
        else:
            cutoff = pd.Timestamp(row["snapshot_at"])
            last = pd.Timestamp(row["last_activity_at"])
            expected_age = min(35, (cutoff - last).total_seconds() / 86400)
            assert result.loc[i, "days_since_last_activity"] == pytest.approx(expected_age)
            assert result.loc[i, "no_activity_history"] == 0
    if batch.iloc[0]["mean_score_pct_to_date"]:
        assert result.loc[0, "mean_score_pct_to_date"] == float(batch.iloc[0]["mean_score_pct_to_date"])
    else:
        assert pd.isna(result.loc[0, "mean_score_pct_to_date"])


def test_outcome_join_and_temporal_split_contract():
    training = load_training_data(DATA)
    assert len(training.snapshots) == len(training.labels) == 3600
    assert training.split_counts == {"train": 2400, "validation": 600, "test": 600}
    assert training.exclusions == {"unknown_outcome": 0, "immature_outcome": 0}
    assert training.student_overlap == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert set(training.labels.unique()) == {0, 1}
    assert "final_outcome" not in training.snapshots
    assert "dropout_label" not in training.snapshots


def _write_mini_dataset(tmp_path, snapshots, outcomes):
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    snapshots.to_csv(raw / "lms_snapshots.csv", index=False)
    outcomes.to_csv(raw / "student_outcomes.csv", index=False)
    return root


def test_join_uses_keys_when_outcome_rows_are_reordered(tmp_path, snapshots):
    sample = snapshots.iloc[:3].copy()
    outcomes = pd.read_csv(DATA / "raw" / "student_outcomes.csv", dtype=str, keep_default_na=False)
    outcomes = outcomes[outcomes["student_key"].isin(sample["student_key"])].iloc[::-1].reset_index(drop=True)
    data = load_training_data(_write_mini_dataset(tmp_path, sample, outcomes))
    assert data.labels.tolist() == [
        int(outcomes.set_index("student_key").loc[key, "final_outcome"] == "withdrawn")
        for key in sample["student_key"]
    ]


def test_unknown_and_immature_outcomes_are_excluded(tmp_path, snapshots):
    sample = snapshots.iloc[:4].copy()
    outcomes = pd.read_csv(DATA / "raw" / "student_outcomes.csv", dtype=str, keep_default_na=False)
    outcomes = outcomes[outcomes["student_key"].isin(sample["student_key"])].reset_index(drop=True)
    ends = pd.to_datetime(sample["course_end_at"], utc=True)
    outcomes.loc[0, "final_outcome"] = "unknown"
    outcomes.loc[0, "withdrawn_at"] = ""
    outcomes.loc[1, "final_outcome"] = "completed"
    outcomes.loc[1, "withdrawn_at"] = ""
    outcomes.loc[1, "outcome_observed_at"] = (ends.iloc[1] + pd.Timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    outcomes.loc[2, "final_outcome"] = "failed"
    outcomes.loc[2, "withdrawn_at"] = ""
    outcomes.loc[2, "outcome_observed_at"] = (ends.iloc[2] + pd.Timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
    outcomes.loc[3, "final_outcome"] = "withdrawn"
    outcomes.loc[3, "withdrawn_at"] = (pd.to_datetime(sample.iloc[3]["snapshot_at"], utc=True) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    outcomes.loc[3, "outcome_observed_at"] = (ends.iloc[3] + pd.Timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = load_training_data(_write_mini_dataset(tmp_path, sample, outcomes))
    assert data.exclusions == {"unknown_outcome": 1, "immature_outcome": 1}
    assert data.labels.tolist() == [0, 1]


def test_duplicate_unmatched_and_out_of_window_withdrawal_fail(tmp_path, snapshots):
    sample = snapshots.iloc[:2].copy()
    outcomes = pd.read_csv(DATA / "raw" / "student_outcomes.csv", dtype=str, keep_default_na=False)
    outcomes = outcomes[outcomes["student_key"].isin(sample["student_key"])].reset_index(drop=True)
    duplicate = pd.concat([outcomes, outcomes.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate join keys"):
        load_training_data(_write_mini_dataset(tmp_path / "dup", sample, duplicate))
    with pytest.raises(DataContractError, match="one-to-one"):
        load_training_data(_write_mini_dataset(tmp_path / "unmatched", sample, outcomes.iloc[:1]))
    outcomes.loc[0, "final_outcome"] = "withdrawn"
    outcomes.loc[0, "withdrawn_at"] = sample.iloc[0]["snapshot_at"]
    with pytest.raises(DataContractError, match="strictly after cutoff"):
        load_training_data(_write_mini_dataset(tmp_path / "window", sample, outcomes))


def test_invalid_utf8_fails_as_data_contract_error(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "lms_snapshots.csv").write_bytes(b"\xff")
    (raw / "student_outcomes.csv").write_text("", encoding="utf-8")
    with pytest.raises(DataContractError, match="cannot read dataset"):
        load_training_data(tmp_path)


def test_withdrawal_cannot_be_confirmed_before_it_occurs(tmp_path, snapshots):
    sample = snapshots.iloc[[0]].copy()
    outcomes = pd.read_csv(DATA / "raw" / "student_outcomes.csv", dtype=str, keep_default_na=False)
    outcomes = outcomes[outcomes["student_key"].isin(sample["student_key"])].reset_index(drop=True)
    cutoff = pd.to_datetime(sample.iloc[0]["snapshot_at"], utc=True)
    outcomes.loc[0, "final_outcome"] = "withdrawn"
    outcomes.loc[0, "withdrawn_at"] = (cutoff + pd.Timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    outcomes.loc[0, "outcome_observed_at"] = (cutoff + pd.Timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with pytest.raises(DataContractError, match="cannot precede withdrawn_at"):
        load_training_data(_write_mini_dataset(tmp_path, sample, outcomes))
