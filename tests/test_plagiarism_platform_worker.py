from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.features import FeaturePipeline
from plagiarism.platform_worker import main, run_score, run_snapshot
from plagiarism.retrieval import FixtureEncoder, chunking_config


class _Model:
    classes_ = np.asarray([1, 0])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.95, 0.05]]), (len(features), 1))


def _scope() -> dict[str, str]:
    return {"institution_id": "inst-1", "course_id": "course-1", "assignment_id": "assignment-1"}


def _reports() -> list[dict[str, str]]:
    text = "A sufficiently distinctive passage appears in this report for matching."
    return [{"report_id": "r1", "version_id": "v1", "text": text},
            {"report_id": "r2", "version_id": "v1", "text": "Student copy: " + text}]


def _balanced_snapshot_payload() -> dict:
    reports: list[dict[str, str]] = []
    reviews: list[dict] = []
    for index in range(24):
        reused = index % 2 == 0
        shared = "Distinctive reviewed passage"
        left_text = f"Submission {index}: {shared}." if reused else f"Independent submission text {index}."
        right_text = f"Archive {index}: {shared}." if reused else f"Unrelated source material {index}."
        left_id, right_id = f"left-{index}", f"right-{index}"
        reports.extend([{"report_id": left_id, "version_id": "v1", "text": left_text},
                        {"report_id": right_id, "version_id": "v1", "text": right_text}])
        spans = ([{"left_start": left_text.index(shared), "left_end": left_text.index(shared) + len(shared),
                   "right_start": right_text.index(shared), "right_end": right_text.index(shared) + len(shared)}]
                 if reused else [])
        reviews.append({"review_id": f"review-{index}", "revision": 1,
                        "left": {"report_id": left_id, "version_id": "v1"},
                        "right": {"report_id": right_id, "version_id": "v1"},
                        "coverage": "full", "reused": reused, "spans": spans})
    return {"scope": _scope(), "reports": reports, "reviews": reviews}


def _write_input(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_snapshot_cli_builds_scoped_private_result_and_retry_is_idempotent(tmp_path: Path) -> None:
    payload = _balanced_snapshot_payload()
    input_path, output_path = tmp_path / "input.json", tmp_path / "result.json"
    _write_input(input_path, payload)
    args = ["snapshot", "--input", str(input_path), "--output", str(output_path),
            "--snapshot-dir", str(tmp_path / "snapshot")]
    assert main(args) == 0
    first_bytes = output_path.read_bytes()
    result = json.loads(first_bytes)
    assert result["operation"] == "snapshot"
    assert result["scope"] == _scope()
    assert len(result["manifest_sha256"]) == 64
    assert len(result["dataset_sha256"]) == 64
    assert main(args) == 0
    assert output_path.read_bytes() == first_bytes


def test_snapshot_rejects_unknown_fields_and_conflicting_result_path(tmp_path: Path) -> None:
    payload = {**_balanced_snapshot_payload(), "output_path": "/tmp/escape"}
    input_path = tmp_path / "input.json"
    _write_input(input_path, payload)
    assert main(["snapshot", "--input", str(input_path), "--output", str(tmp_path / "result.json"),
                 "--snapshot-dir", str(tmp_path / "snapshot")]) == 2
    assert not (tmp_path / "snapshot").exists()


def test_score_uses_verified_loader_contract_and_publishes_deterministic_scoped_json(tmp_path: Path, monkeypatch) -> None:
    import plagiarism.platform_worker as worker

    scope = _scope()
    payload = {"scope": scope, "reports": _reports()}
    input_path = tmp_path / "input.json"
    _write_input(input_path, payload)
    bundle = {"model_name": "passage-reuse", "model_version": "4", "model_uri": "models:/passage-reuse/4",
              "threshold": 0.7, "chunking_config": chunking_config(target_tokens=12, overlap_tokens=3, max_tokens=20), "retrieval_top_k": 10,
              "lineage": {"platform_scope": scope}}
    pipeline = FeaturePipeline().fit(["sample training report text", "a second sample text"])
    monkeypatch.setattr(worker, "_load_registry", lambda _args: (object(), _Model(), bundle, pipeline, FixtureEncoder()))
    output = tmp_path / "score.json"
    index, manifest = tmp_path / "index.npz", tmp_path / "index-manifest.json"
    namespace = type("Args", (), {"input": input_path, "output": output, "index": index,
                                    "index_manifest": manifest})()
    first = run_score(namespace)
    byte_content = output.read_bytes()
    assert first["scope"] == scope
    assert first["model"]["version"] == "4"
    assert first["index"]["version"]
    assert json.loads(manifest.read_text(encoding="utf-8"))["chunking_config"] == bundle["chunking_config"]
    assert all("query_excerpt" in match for row in first["reports"] for match in row["evidence"])
    assert first["display_flags"]
    assert all(len(flag["evidence_ids"]) >= 1 for flag in first["display_flags"])
    assert "embedding_seconds" not in first["metrics"]
    second = run_score(namespace)
    assert second == first
    assert output.read_bytes() == byte_content
    assert hashlib.sha256(byte_content).hexdigest()


def test_worker_does_not_print_exception_text_or_report_content(tmp_path: Path, capsys) -> None:
    secret = "PRIVATE_CANONICAL_REPORT_SENTINEL"
    payload = {"scope": _scope(), "reports": [{"report_id": "r1", "version_id": "v1", "text": secret}],
               "unexpected": secret}
    input_path = tmp_path / "input.json"
    _write_input(input_path, payload)
    assert main(["score", "--input", str(input_path), "--output", str(tmp_path / "out.json"),
                 "--tracking-uri", "sqlite:///unused", "--model-name", "passage-reuse", "--version", "1",
                 "--index", str(tmp_path / "index.npz"), "--index-manifest", str(tmp_path / "manifest.json")]) == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "platform worker failed (ValueError)" in captured.err


def test_verify_release_binds_gate_to_exact_scope_and_training_snapshot(tmp_path: Path, monkeypatch) -> None:
    import plagiarism.platform_worker as worker

    input_path = tmp_path / "input.json"
    _write_input(input_path, {"scope": _scope()})
    snapshot_hash = "c" * 64
    bundle = {"model_name": "passage-reuse", "model_version": "4", "model_uri": "models:/passage-reuse/4",
              "mlflow_run_id": "run-4", "target_version": "target-v1", "feature_schema_version": "1.0",
              "threshold": 0.7, "lineage": {"platform_scope": _scope(),
                                                "training_corpus_sha256": snapshot_hash}}
    monkeypatch.setattr(worker, "_load_registry", lambda _args: (object(), object(), bundle, object(), object()))
    gate_path = tmp_path / "gate.json"
    gate = {"passed": True, "decision": "passed", "mlflow_run_id": "run-4",
            "platform_scope": _scope(), "platform_snapshot_sha256": snapshot_hash,
            "candidate": {"target_version": "target-v1", "feature_schema_version": "1.0", "threshold": 0.7}}
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    import plagiarism.score as score_module
    monkeypatch.setattr(score_module, "_download_run_artifact", lambda *_args, **_kwargs: gate_path)
    args = SimpleNamespace(input=input_path, output=tmp_path / "release.json")
    result = worker.run_verify_release(args)
    assert result["operation"] == "verify-release"
    assert result["release"]["gate_passed"] is True
    assert result["release"]["platform_scope"] == _scope()
    assert result["release"]["training_corpus_sha256"] == snapshot_hash
    assert args.output.exists()

    gate["platform_scope"] = {**_scope(), "assignment_id": "other-assignment"}
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="passing gate report"):
        worker._release_evidence(SimpleNamespace(), worker.AssignmentScope(**_scope()))
