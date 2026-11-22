from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.evaluation import (candidate_labels, choose_shared_recall_thresholds,
                                   evaluate_release_gate, merge_alerts,
                                   retrieval_link_recall)
from plagiarism.retrieval import Chunk


def _candidate(q0, q1, s0, s1, *, qid="q", sid="s"):
    q = Chunk(f"{qid}:{q0}:{q1}", qid, "q" * (q1-q0), q0, q1, 0, 1)
    s = Chunk(f"{sid}:{s0}:{s1}", sid, "s" * (s1-s0), s0, s1, 0, 1)
    return {"query_chunk": q, "source_chunk": s, "query_text": q.text,
            "source_text": s.text, "cosine": .8}


def _ann(ident="span-1", q0=10, q1=20, s0=30, s1=40):
    return SimpleNamespace(id=ident, submission_id="q", source_id="s",
                           submission_start=q0, submission_end=q1,
                           source_start=s0, source_end=s1)


def test_candidate_labels_distinguish_positive_ambiguous_and_clear_negative():
    rows = [_candidate(8, 22, 28, 42), _candidate(0, 12, 28, 42),
            _candidate(0, 5, 0, 5)]
    assert candidate_labels(rows, [_ann()]).tolist() == [1, -1, 0]


def test_recall_counts_annotation_ids_and_reports_retrieval_misses():
    rows = [_candidate(8, 22, 28, 42)]
    result = retrieval_link_recall(rows, [_ann(), _ann("span-2", 50, 60, 70, 80)])
    assert result["links"] == 2
    assert result["retrieved_links"] == 1
    assert result["missed_links"] == ["span-2"]
    assert result["recall_at_k"] == .5


def test_alert_merge_bridges_components_and_counts_unmatched_alert_per_alert():
    # First and third windows overlap indirectly through the second on both axes.
    rows = [_candidate(8, 14, 28, 34), _candidate(12, 18, 32, 38),
            _candidate(16, 22, 36, 42), _candidate(60, 70, 60, 70)]
    result = merge_alerts(rows, [.9, .8, .7, .99], .5, [_ann()], eligible_submissions=["q", "q2"])
    assert result["alert_count"] == 2
    assert result["true_alerts"] == 1
    assert result["false_alerts"] == 1
    assert result["precision"] == .5
    assert result["recall"] == 1.0
    assert result["false_alerts_per_100_submissions"] == 50


def test_thresholds_use_span_linked_merged_alerts_and_release_gate_is_strict():
    rows = [_candidate(8, 22, 28, 42), _candidate(0, 5, 0, 5),
            _candidate(50, 60, 70, 80)]
    annotations = [_ann(), _ann("span-2", 50, 60, 70, 80)]
    shared, choices = choose_shared_recall_thresholds(
        annotations,
        {"minilm_cosine": (rows, [.8, .7, .6]),
         "minilm_xgboost": (rows, [.9, .2, .8])},
        requested_recall=1.0, minimum_recall=.5, eligible_submissions=["q"],
    )
    assert shared == 1.0
    assert choices["minilm_xgboost"]["recall"] == 1
    candidate = {"target_version": "v1", "feature_schema_version": "1.0",
                 "shared_recall_target": shared, "recall": choices["minilm_xgboost"]["recall"],
                 "minilm_cosine_recall": choices["minilm_cosine"]["recall"],
                 "false_alerts": 0, "minilm_cosine_false_alerts": 1}
    settings = {"gates": {"minimum_recall": .5, "maximum_recall_drop": .05}}
    gate = evaluate_release_gate(candidate, settings, data_checks_passed=True,
                                 artifact_checks_passed=True)
    assert gate["passed"]
    candidate["false_alerts"] = 1
    assert not evaluate_release_gate(candidate, settings, data_checks_passed=True,
                                     artifact_checks_passed=True)["passed"]


def test_threshold_selection_fails_if_retrieval_ceiling_is_below_minimum():
    rows = [_candidate(0, 2, 0, 2)]
    annotations = [_ann()]
    with pytest.raises(ValueError, match="shared recall"):
        choose_shared_recall_thresholds(
            annotations,
            {"minilm_cosine": (rows, [.1]), "minilm_xgboost": (rows, [.2])},
            requested_recall=.8, minimum_recall=.5, eligible_submissions=["q"])
