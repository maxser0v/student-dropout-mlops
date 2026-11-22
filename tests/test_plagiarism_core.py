from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.features import FEATURE_NAMES, FeaturePipeline, TextPair
from plagiarism.retrieval import (
    Chunk,
    FixtureEncoder,
    chunk_text,
    chunking_config,
    load_source_cache,
    retrieve,
    save_source_cache,
)
from plagiarism.training import train_classifier


def test_fixture_chunking_preserves_exact_offsets_and_special_token_budget() -> None:
    encoder = FixtureEncoder()
    text = "one two three four five six seven"
    chunks = chunk_text(text, "submission-a", encoder.tokenizer, target_tokens=5,
                        overlap_tokens=2, max_tokens=7)

    assert [(chunk.start_char, chunk.end_char, chunk.text) for chunk in chunks] == [
        (0, 23, "one two three four five"),
        (14, 33, "four five six seven"),
    ]
    assert all(len(encoder.tokenizer(chunk.text)["input_ids"]) + 2 <= 7 for chunk in chunks)
    assert chunks[0].chunk_id == "submission-a:0:23"
    assert chunks[1].token_start == 3


def test_retrieval_exact_top_k_has_stable_ties_and_pair_filtering() -> None:
    query = Chunk("q:0:1", "q", "query", 0, 1, 0, 1)
    sources = [
        Chunk("z:0:1", "z", "z", 0, 1, 0, 1),
        Chunk("a:0:1", "a", "a", 0, 1, 0, 1),
        Chunk("b:0:1", "b", "b", 0, 1, 0, 1),
    ]
    result = retrieve([query], sources, np.array([[1.0, 0.0]]),
                      np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), top_k=2)
    assert [item.source_chunk.document_id for item in result] == ["a", "z"]
    filtered = retrieve([query], sources, np.array([[1.0, 0.0]]),
                        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                        top_k=2, allowed_source_ids={"b", "z"})
    assert [item.source_chunk.document_id for item in filtered] == ["z", "b"]


def test_source_cache_round_trip_and_rejects_mismatched_or_tampered_state(tmp_path: Path) -> None:
    encoder = FixtureEncoder()
    chunks = [Chunk("source:0:3", "source", "one two", 0, 7, 0, 2)]
    vectors = encoder.encode_texts([chunks[0].text])
    path = tmp_path / "index.npz"
    config = chunking_config(target_tokens=180, overlap_tokens=40)
    save_source_cache(path, chunks, vectors, source_manifest_hash="a" * 64,
                      encoder_revision=encoder.revision, config=config)
    restored, restored_vectors = load_source_cache(path, source_manifest_hash="a" * 64,
                                                    encoder_revision=encoder.revision, config=config)
    assert restored == chunks
    assert np.allclose(restored_vectors, vectors)
    with pytest.raises(ValueError, match="metadata"):
        load_source_cache(path, source_manifest_hash="b" * 64,
                          encoder_revision=encoder.revision, config=config)

    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload["chunks"] = np.asarray(json.dumps([{**chunks[0].__dict__, "text": "changed"}]))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    with pytest.raises(ValueError, match="chunk integrity"):
        load_source_cache(path, source_manifest_hash="a" * 64,
                          encoder_revision=encoder.revision, config=config)


def test_feature_pipeline_fits_only_training_text_and_round_trips(tmp_path: Path) -> None:
    train_texts = ["the river flows through town", "a river crosses the valley", "another topic about astronomy"]
    pipeline = FeaturePipeline().fit(train_texts)
    pair = TextPair("river flows through town", "river flows through the town", 0.8)
    matrix = pipeline.transform_pairs([pair])
    assert matrix.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()
    assert "validationonlytoken" not in pipeline.vectorizer.vocabulary_
    path = pipeline.save(tmp_path / "features.json")
    restored = FeaturePipeline.load(path)
    assert np.array_equal(restored.transform_pairs([pair]), matrix)
    assert restored.rare_terms == pipeline.rare_terms


def test_training_uses_real_optional_xgboost_or_reports_missing_runtime() -> None:
    features = np.asarray([[0.1, 0.0, 0.0, 0.0, 0.1], [0.9, 0.9, 0.9, 0.9, 1.0]], dtype=np.float32)
    labels = [0, 1]
    try:
        model = train_classifier(np.vstack([features, features]), labels * 2, n_estimators=2)
    except RuntimeError as exc:
        assert "XGBoost is not installed" in str(exc)
        return
    assert set(model.classes_) == {0, 1}
    assert model.get_params()["tree_method"] == "hist"
    assert model.get_params()["device"] == "cpu"
