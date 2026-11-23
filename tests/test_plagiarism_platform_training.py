from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.evaluation import (choose_shared_recall_thresholds,
                                   evaluate_at_threshold, evaluate_release_gate)
from plagiarism.retrieval import Chunk


def _row(q0: int, q1: int, s0: int, s1: int) -> dict:
    q = Chunk(f"q:{q0}:{q1}", "q", "q" * (q1 - q0), q0, q1, 0, 1)
    s = Chunk(f"s:{s0}:{s1}", "s", "s" * (s1 - s0), s0, s1, 0, 1)
    return {"query_chunk": q, "source_chunk": s, "query_text": q.text,
            "source_text": s.text, "cosine": .8}


def _span(ident: str, q0: int, q1: int, s0: int, s1: int):
    return SimpleNamespace(id=ident, submission_id="q", source_id="s",
                           submission_start=q0, submission_end=q1,
                           source_start=s0, source_end=s1)


def test_active_model_joins_shared_recall_selection_and_keeps_deployed_threshold():
    rows = [_row(8, 22, 28, 42), _row(0, 5, 0, 5), _row(50, 60, 70, 80)]
    labels = [_span("one", 10, 20, 30, 40), _span("two", 52, 58, 72, 78)]
    _target, choices = choose_shared_recall_thresholds(
        labels,
        {"minilm_cosine": (rows, [.8, .7, .6]),
         "minilm_xgboost": (rows, [.95, .05, .85]),
         "active_xgboost": (rows, [.9, .1, .8])},
        requested_recall=1.0, minimum_recall=.75, eligible_submissions=["q"])
    assert choices["minilm_xgboost"]["recall"] == choices["active_xgboost"]["recall"] == 1
    deployed = evaluate_at_threshold(rows, [.9, .1, .8], .85, labels,
                                     eligible_submissions=["q"])
    assert deployed["threshold"] == .85
    assert deployed["recall"] == .5


def _gate_candidate(candidate_false_alerts=0, candidate_recall=.9,
                    active_false_alerts=1, active_recall=.88):
    return {
        "shared_recall_target": .85,
        "recall": candidate_recall,
        "false_alerts": candidate_false_alerts,
        "minilm_cosine_recall": .9,
        "minilm_cosine_false_alerts": 2,
        "active_model_comparison": {
            "shared_recall": {"recall": active_recall,
                               "false_alerts": active_false_alerts}
        },
    }


def test_platform_gate_requires_active_model_recall_and_false_alert_improvement():
    settings = {"gates": {"minimum_recall": .75, "maximum_recall_drop": .05,
                           "minimum_false_alert_reduction": 1}}
    passed = evaluate_release_gate(_gate_candidate(), settings,
                                   data_checks_passed=True, artifact_checks_passed=True,
                                   active_comparison_required=True)
    assert passed["passed"] is True
    assert passed["active_model_comparison"] == _gate_candidate()["active_model_comparison"]

    recall_regression = evaluate_release_gate(
        _gate_candidate(candidate_recall=.82), settings,
        data_checks_passed=True, artifact_checks_passed=True,
        active_comparison_required=True)
    assert not recall_regression["passed"]
    assert any(check["name"] == "active_recall_preservation" and not check["passed"]
               for check in recall_regression["checks"])

    no_reduction = evaluate_release_gate(
        _gate_candidate(candidate_false_alerts=1), settings,
        data_checks_passed=True, artifact_checks_passed=True,
        active_comparison_required=True)
    assert not no_reduction["passed"]
    assert any(check["name"] == "active_false_alert_improvement" and not check["passed"]
               for check in no_reduction["checks"])


def test_platform_gate_fails_closed_when_active_comparison_is_missing():
    settings = {"gates": {"minimum_recall": .75, "maximum_recall_drop": .05,
                           "minimum_false_alert_reduction": 1}}
    candidate = _gate_candidate()
    candidate["active_model_comparison"] = {"error": "active bundle is incompatible"}
    report = evaluate_release_gate(candidate, settings, data_checks_passed=True,
                                   artifact_checks_passed=True, active_comparison_required=True)
    assert not report["passed"]
    assert any(check["name"] == "active_comparison_available" and not check["passed"]
               for check in report["checks"])
    assert report["active_comparison_error"] == "active bundle is incompatible"


def test_shared_recall_gate_preserves_existing_non_platform_behavior():
    settings = {"gates": {"minimum_recall": .75, "maximum_recall_drop": .05,
                           "minimum_false_alert_reduction": 1}}
    candidate = _gate_candidate()
    candidate.pop("active_model_comparison")
    report = evaluate_release_gate(candidate, settings,
                                   data_checks_passed=True, artifact_checks_passed=True)
    assert report["passed"]
    assert not any(check["name"].startswith("active_") for check in report["checks"])
