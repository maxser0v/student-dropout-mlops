from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.features import FeaturePipeline
from plagiarism.platform_scoring import build_assignment_index, load_verified_assignment_index, score_assignment_reports
from plagiarism.platform_types import AssignmentScope, ReportVersion
from plagiarism.retrieval import FixtureEncoder, chunking_config


class _Model:
    classes_ = np.asarray([1, 0])

    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[self.score, 1.0 - self.score]]), (len(features), 1))


def _bundle(threshold: float = 0.7) -> dict:
    return {"model_name": "fixture", "model_version": "3", "model_uri": "models:/fixture/3",
            "threshold": threshold, "chunking_config": chunking_config(), "retrieval_top_k": 10}


def test_platform_types_validate_and_freeze_scope_and_report() -> None:
    scope = AssignmentScope("inst", "course", "assignment")
    item = ReportVersion(scope, "report", "version-1", "some text")
    with pytest.raises((AttributeError, TypeError)):
        item.text = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="nonempty"):
        AssignmentScope(" ", "course", "assignment")
    with pytest.raises(ValueError, match="nonempty"):
        ReportVersion(scope, "", "version-1", "some text")


def test_scoped_index_checks_version_scope_source_hashes_and_immutable_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    scope = AssignmentScope("inst", "course", "assignment")
    reports = [ReportVersion(scope, "r1", "v1", "a sufficiently useful report passage")]
    encoder = FixtureEncoder()
    index_path, manifest_path = tmp_path / "index.npz", tmp_path / "manifest.json"
    manifest = build_assignment_index(reports, scope, index_path, manifest_path, encoder)
    assert len(manifest["index_version"]) == 64
    loaded = load_verified_assignment_index(index_path, manifest_path, expected_scope=scope,
        expected_index_version=manifest["index_version"], encoder=encoder,
        expected_chunking_config=chunking_config(), expected_reports=reports)
    assert loaded.manifest["index_version"] == manifest["index_version"]
    monkeypatch.setattr(encoder, "encode_texts", lambda texts: pytest.fail("idempotent build re-encoded source text"))
    assert build_assignment_index(reports, scope, index_path, manifest_path, encoder) == manifest

    other_scope = AssignmentScope("inst", "course", "other-assignment")
    with pytest.raises(ValueError, match="scope"):
        load_verified_assignment_index(index_path, manifest_path, expected_scope=other_scope,
            expected_index_version=manifest["index_version"], encoder=encoder,
            expected_chunking_config=chunking_config())
    with pytest.raises(ValueError, match="source versions or content hashes"):
        load_verified_assignment_index(index_path, manifest_path, expected_scope=scope,
            expected_index_version=manifest["index_version"], encoder=encoder,
            expected_chunking_config=chunking_config(),
            expected_reports=[ReportVersion(scope, "r1", "v1", "changed text")])
    with pytest.raises(ValueError, match="immutable"):
        build_assignment_index([ReportVersion(scope, "r1", "v2", "changed text")], scope,
                               index_path, manifest_path, encoder)
    index_path.write_bytes(index_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="cache checksum"):
        load_verified_assignment_index(index_path, manifest_path, expected_scope=scope,
            expected_index_version=manifest["index_version"], encoder=encoder,
            expected_chunking_config=chunking_config())


def test_same_batch_scoring_excludes_all_versions_but_keeps_duplicate_report_evidence(tmp_path: Path) -> None:
    scope = AssignmentScope("inst", "course", "assignment")
    copied = "A unique copied passage with several distinctive words."
    reports = [ReportVersion(scope, "alpha", "v1", copied),
               ReportVersion(scope, "alpha", "v2", copied + " Updated."),
               ReportVersion(scope, "beta", "v1", "Student says: " + copied),
               ReportVersion(scope, "gamma", "v1", copied)]
    encoder = FixtureEncoder()
    index_path, manifest_path = tmp_path / "index.npz", tmp_path / "manifest.json"
    manifest = build_assignment_index(reports, scope, index_path, manifest_path, encoder)
    pipeline = FeaturePipeline().fit([copied, "another unrelated sample document", "third distinct source"])
    records, metrics = score_assignment_reports(reports, expected_scope=scope,
        expected_index_version=manifest["index_version"], index_path=index_path, manifest_path=manifest_path,
        model=_Model(0.9), bundle=_bundle(), feature_pipeline=pipeline, encoder=encoder)
    by_identity = {(row["report_id"], row["version_id"]): row for row in records}
    assert metrics["report_version_count"] == len(reports)
    assert len(records) == len(reports)
    for row in records:
        assert all(flag["source_report_id"] != row["report_id"] for flag in row["flags"])
    beta_row = by_identity[("beta", "v1")]
    beta_matches = beta_row["evidence"]
    # Alpha versions and gamma are separate evidence even when their text is identical.
    assert {flag["source_report_id"] for flag in beta_matches} >= {"alpha", "gamma"}
    assert len({flag["evidence_id"] for flag in beta_matches}) == len(beta_matches)
    assert len(beta_row["flags"]) < len(beta_matches)
    assert sum(flag["evidence_count"] for flag in beta_row["flags"]) == len(beta_matches)


def test_below_threshold_sample_is_bounded_and_platform_model_scope_checked(tmp_path: Path) -> None:
    scope = AssignmentScope("inst", "course", "assignment")
    reports = [ReportVersion(scope, "a", "v1", "A passage to compare against another document."),
               ReportVersion(scope, "b", "v1", "A passage to compare against another document.")]
    encoder = FixtureEncoder()
    index_path, manifest_path = tmp_path / "index.npz", tmp_path / "manifest.json"
    manifest = build_assignment_index(reports, scope, index_path, manifest_path, encoder)
    pipeline = FeaturePipeline().fit(["one training sample document", "second training sample text"])
    wrong_scope = {"model_name": "fixture", "model_version": "3", "model_uri": "models:/fixture/3",
                   "threshold": 0.7, "chunking_config": chunking_config(), "retrieval_top_k": 10,
                   "lineage": {"platform_scope": {"institution_id": "inst", "course_id": "course",
                                                      "assignment_id": "different"}}}
    with pytest.raises(ValueError, match="model platform scope"):
        score_assignment_reports(reports, expected_scope=scope, expected_index_version=manifest["index_version"],
            index_path=index_path, manifest_path=manifest_path, model=_Model(0.2), bundle=wrong_scope,
            feature_pipeline=pipeline, encoder=encoder)
    records, _ = score_assignment_reports(reports, expected_scope=scope,
        expected_index_version=manifest["index_version"], index_path=index_path, manifest_path=manifest_path,
        model=_Model(0.2), bundle=_bundle(), feature_pipeline=pipeline, encoder=encoder,
        below_threshold_limit=1)
    assert all(not row["flags"] for row in records)
    assert all(len(row["below_threshold_sample"]) <= 1 for row in records)
