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
from plagiarism.retrieval import Chunk, FixtureEncoder, chunking_config, save_source_cache
from plagiarism.score import (
    FIXTURE_MODEL_NAME,
    TARGET_VERSION,
    load_verified_release,
    parse_submissions,
    score_submissions,
    write_jsonl_atomically,
)


class _ReviewModel:
    classes_ = np.asarray([1, 0])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.85, 0.15]]), (len(features), 1))


def test_score_submissions_returns_offsets_excerpts_and_versioned_evidence() -> None:
    encoder = FixtureEncoder()
    source_text = "A copied passage has exact source wording."
    source = Chunk("source-a:0:43", "source-a", source_text, 0, len(source_text), 0, 8)
    vectors = encoder.encode_texts([source_text])
    pipeline = FeaturePipeline().fit([source_text, "an unrelated training document", "another training record"])
    bundle = {
        "model_name": FIXTURE_MODEL_NAME,
        "model_version": "4",
        "model_uri": f"models:/{FIXTURE_MODEL_NAME}/4",
        "threshold": 0.7,
        "feature_schema_version": "1.0",
        "source_manifest_hash": "a" * 64,
        "source_index_sha256": "b" * 64,
        "chunking_config": chunking_config(),
        "retrieval_top_k": 10,
    }
    submission_text = "Here: A copied passage has exact source wording. End."
    records = score_submissions(
        [{"submission_id": "submission-a", "text": submission_text}], _ReviewModel(), bundle,
        pipeline, [source], vectors, encoder,
    )
    repeat = score_submissions(
        [{"submission_id": "submission-a", "text": submission_text}], _ReviewModel(), bundle,
        pipeline, [source], vectors, encoder,
    )
    evidence_hash = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert evidence_hash == hashlib.sha256(
        json.dumps(repeat, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert len(evidence_hash) == 64
    assert len(records) == 1
    record = records[0]
    assert record["model_version"] == "4"
    assert record["source_index_version"] == "a" * 64
    assert record["flag_count"] == 1
    flag = record["flags"][0]
    assert flag["submission_excerpt"] == submission_text[flag["submission_start"]:flag["submission_end"]]
    assert flag["source_excerpt"] == source_text
    assert flag["review_score"] == pytest.approx(0.85)
    assert (flag["source_start"], flag["source_end"]) == (0, len(source_text))


def test_parse_submissions_and_atomic_jsonl_output(tmp_path: Path) -> None:
    inputs = [json.dumps({"submission_id": "s1", "text": "some text"})]
    assert parse_submissions(inputs) == [{"submission_id": "s1", "text": "some text"}]
    with pytest.raises(ValueError, match="duplicate"):
        parse_submissions(inputs * 2)
    with pytest.raises(ValueError, match="only submission_id and text"):
        parse_submissions([json.dumps({"submission_id": "s1", "text": "x", "other": 1})])

    target = tmp_path / "nested" / "reviews.jsonl"
    record = {"submission_id": "s1", "flags": []}
    assert write_jsonl_atomically([record], target) == target.resolve()
    assert json.loads(target.read_text(encoding="utf-8")) == record
    assert not list(target.parent.glob(".*.tmp"))


def test_verified_fixture_release_checks_artifact_hashes_and_namespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    encoder = FixtureEncoder()
    source_text = "fixture source text"
    source_chunk = Chunk("src:0:18", "src", source_text, 0, len(source_text), 0, 3)
    source_vectors = encoder.encode_texts([source_text])
    index_path = tmp_path / "source_index.npz"
    config = chunking_config()
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    save_source_cache(index_path, [source_chunk], source_vectors,
                      source_manifest_hash=manifest_hash,
                      encoder_revision=encoder.revision, config=config)
    features_path = FeaturePipeline().fit([source_text, "another training document"]).save(tmp_path / "features.json")
    chunking_path = tmp_path / "chunking.json"
    chunking_path.write_text(json.dumps(config), encoding="utf-8")
    encoder_manifest_path = tmp_path / "encoder_manifest.json"
    encoder_manifest_path.write_text(json.dumps({"mode": "fixture", "revision": encoder.revision}), encoding="utf-8")
    gate_path = tmp_path / "gate_report.json"
    gate_path.write_text(json.dumps({
        "passed": True, "decision": "passed", "mlflow_run_id": "run-fixture",
        "candidate": {"target_version": TARGET_VERSION, "feature_schema_version": "1.0",
                      "threshold": 0.5, "threshold_choice": {"threshold": 0.5}},
    }), encoding="utf-8")

    artifact_paths = {
        "plagiarism_bundle/source_index.npz": index_path,
        "plagiarism_bundle/features.json": features_path,
        "plagiarism_bundle/chunking.json": chunking_path,
        "plagiarism_bundle/source_manifest.json": manifest,
        "plagiarism_bundle/encoder_manifest.json": encoder_manifest_path,
        "gate_report.json": gate_path,
    }
    bundle = {
        "bundle_version": "1.0",
        "model_name": FIXTURE_MODEL_NAME,
        "model_version": "2",
        "model_uri": f"models:/{FIXTURE_MODEL_NAME}/2",
        "mlflow_run_id": "run-fixture",
        "gate_passed": True,
        "target_version": TARGET_VERSION,
        "feature_schema_version": "1.0",
        "feature_columns": ["minilm_cosine", "tfidf_cosine", "shared_word_ngram_overlap", "rare_term_overlap", "length_ratio"],
        "threshold": 0.5,
        "source_cache_artifact": "plagiarism_bundle/source_index.npz",
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "features_artifact": "plagiarism_bundle/features.json",
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "chunking_artifact": "plagiarism_bundle/chunking.json",
        "chunking_sha256": hashlib.sha256(chunking_path.read_bytes()).hexdigest(),
        "encoder_manifest_artifact": "plagiarism_bundle/encoder_manifest.json",
        "encoder_manifest_sha256": hashlib.sha256(encoder_manifest_path.read_bytes()).hexdigest(),
        "source_manifest_artifact": "plagiarism_bundle/source_manifest.json",
        "chunking_config": config,
        "source_manifest_hash": manifest_hash,
        "retrieval_top_k": 10,
        "encoder_revision": encoder.revision,
        "encoder_provenance": {"kind": "fixture", "revision": encoder.revision},
    }
    registered = SimpleNamespace(run_id="run-fixture")
    registry_loader = lambda client, name, version: (registered, bundle)
    model = _ReviewModel()

    def downloader(*, run_id: str, artifact_path: str) -> str:
        assert run_id == "run-fixture"
        return str(artifact_paths[artifact_path])

    loaded = load_verified_release(
        object(), FIXTURE_MODEL_NAME, "2", fixture=True,
        source_manifest_path=manifest, downloader=downloader,
        model_loader=lambda uri: model, registry_loader=registry_loader,
    )
    assert loaded[0] is model
    assert loaded[1] is bundle
    assert loaded[3] == [source_chunk]
    assert isinstance(loaded[5], FixtureEncoder)

    bundle["source_index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source index artifact checksum"):
        load_verified_release(object(), FIXTURE_MODEL_NAME, "2", fixture=True,
                              source_manifest_path=manifest, downloader=downloader,
                              model_loader=lambda uri: model, registry_loader=registry_loader)


def test_fixture_registry_namespace_cannot_be_loaded_without_fixture_flag() -> None:
    with pytest.raises(ValueError, match="fixture releases require --fixture"):
        load_verified_release(object(), FIXTURE_MODEL_NAME, "1", fixture=False)
