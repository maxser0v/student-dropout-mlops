"""Pure evaluation, threshold selection, and release gate calculations.

These functions deliberately accept small plain Python records so the decision
logic can be exercised without importing or fitting XGBoost.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


def _overlap(start: int, end: int, span_start: int, span_end: int) -> float:
    intersection = max(0, min(end, span_end) - max(start, span_start))
    return intersection / max(1, span_end - span_start)


def candidate_labels(candidates: Sequence[dict[str, Any]], annotations: Iterable[Any], *,
                     min_span_coverage: float = 0.5) -> np.ndarray:
    """Return 1 for aligned reuse, 0 for clear negatives, -1 for partial spans.

    Partial intersection is intentionally marked ambiguous so the caller can
    remove that training row rather than silently teaching it as a negative.
    """
    if not 0 < min_span_coverage <= 1:
        raise ValueError("min_span_coverage must be in (0, 1]")
    by_link: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for ann in annotations:
        by_link[(ann.submission_id, ann.source_id)].append(ann)
    labels = []
    for row in candidates:
        q, s = row["query_chunk"], row["source_chunk"]
        linked = by_link.get((q.document_id, s.document_id), ())
        positive = any(
            _overlap(q.start_char, q.end_char, ann.submission_start, ann.submission_end) >= min_span_coverage
            and _overlap(s.start_char, s.end_char, ann.source_start, ann.source_end) >= min_span_coverage
            for ann in by_link.get((q.document_id, s.document_id), ())
        )
        partial = any(
            _overlap(q.start_char, q.end_char, ann.submission_start, ann.submission_end) > 0
            or _overlap(s.start_char, s.end_char, ann.source_start, ann.source_end) > 0
            for ann in linked
        )
        labels.append(1 if positive else -1 if partial else 0)
    return np.asarray(labels, dtype=np.int8)


def retrieval_link_recall(candidates: Sequence[dict[str, Any]], annotations: Iterable[Any], *,
                          min_span_coverage: float = 0.5) -> dict[str, Any]:
    """Share of annotated links with at least one retrieved span-linked candidate."""
    annotations = list(annotations)
    labels = candidate_labels(candidates, annotations, min_span_coverage=min_span_coverage)
    reached = set()
    for i, value in enumerate(labels):
        if value == 1:
            q, s = candidates[i]["query_chunk"], candidates[i]["source_chunk"]
            for ann in annotations:
                if (q.document_id, s.document_id) == (ann.submission_id, ann.source_id) \
                        and _overlap(q.start_char, q.end_char, ann.submission_start, ann.submission_end) >= min_span_coverage \
                        and _overlap(s.start_char, s.end_char, ann.source_start, ann.source_end) >= min_span_coverage:
                    reached.add(ann.id)
    ids = {a.id for a in annotations}
    found = ids & reached
    return {"links": len(ids), "retrieved_links": len(found),
            "missed_links": sorted(ids - found),
            "recall_at_k": len(found) / len(ids) if ids else None}


def merge_alerts(candidates: Sequence[dict[str, Any]], probabilities: Sequence[float], threshold: float,
                 annotations: Iterable[Any], *, min_span_coverage: float = 0.5,
                 eligible_submissions: Iterable[str] | None = None) -> dict[str, Any]:
    """Merge overlapping flagged windows per document pair and link alerts once."""
    if len(candidates) != len(probabilities):
        raise ValueError("candidate and probability counts differ")
    annotations = list(annotations)
    selected = []
    for row, prob in zip(candidates, probabilities):
        probability = float(prob)
        if not np.isfinite(probability):
            raise ValueError("scores must be finite")
        if probability >= threshold:
            q, s = row["query_chunk"], row["source_chunk"]
            selected.append({"submission_id": q.document_id, "source_id": s.document_id,
                             "q_start": q.start_char, "q_end": q.end_char,
                             "s_start": s.start_char, "s_end": s.end_char,
                             "probability": probability})
    # Connected interval unions require overlap on both sides; touching windows
    # are joined to avoid duplicate alerts at chunk boundaries.
    selected.sort(key=lambda x: (x["submission_id"], x["source_id"], x["q_start"], x["s_start"]))
    merged: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for alert in selected:
        group = by_pair[(alert["submission_id"], alert["source_id"])]
        group.append(dict(alert))
        changed = True
        while changed:
            changed = False
            for i in range(len(group) - 1):
                left = group[i]
                for j in range(i + 1, len(group)):
                    right = group[j]
                    q_overlap = left["q_start"] <= right["q_end"] and right["q_start"] <= left["q_end"]
                    s_overlap = left["s_start"] <= right["s_end"] and right["s_start"] <= left["s_end"]
                    if q_overlap and s_overlap:
                        left["q_start"] = min(left["q_start"], right["q_start"])
                        left["q_end"] = max(left["q_end"], right["q_end"])
                        left["s_start"] = min(left["s_start"], right["s_start"])
                        left["s_end"] = max(left["s_end"], right["s_end"])
                        left["probability"] = max(left["probability"], right["probability"])
                        group.pop(j)
                        changed = True
                        break
                if changed:
                    break
    merged = [alert for group in by_pair.values() for alert in group]
    matched_annotation_ids: set[str] = set()
    false_alerts = 0
    true_alerts = 0
    for alert in merged:
        matches = [a for a in annotations if (alert["submission_id"], alert["source_id"]) == (a.submission_id, a.source_id)
               and _overlap(alert["q_start"], alert["q_end"], a.submission_start, a.submission_end) >= min_span_coverage
               and _overlap(alert["s_start"], alert["s_end"], a.source_start, a.source_end) >= min_span_coverage]
        if matches:
            true_alerts += 1
            # One merged alert can match multiple spans when it truly covers them.
            matched_annotation_ids.update(a.id for a in matches)
        else:
            false_alerts += 1
    positives = {a.id for a in annotations}
    submissions = set(eligible_submissions or ())
    if not submissions:
        submissions = {a.submission_id for a in annotations} | {x["submission_id"] for x in merged}
    true_positive_links = len(matched_annotation_ids)
    return {"alerts": merged, "alert_count": len(merged), "true_alerts": true_alerts,
            "recovered_spans": true_positive_links,
            "false_alerts": false_alerts, "positive_links": len(positives),
            "precision": true_alerts / len(merged) if merged else 0.0,
            "recall": true_positive_links / len(positives) if positives else None,
            "false_alerts_per_100_submissions": false_alerts * 100.0 / max(1, len(submissions))}


def choose_shared_recall_thresholds(annotations: Iterable[Any],
                                    score_sets: dict[str, tuple[Sequence[dict[str, Any]], Sequence[float]]], *,
                                    requested_recall: float, minimum_recall: float,
                                    eligible_submissions: Iterable[str],
                                    min_span_coverage: float = 0.5) -> tuple[float, dict[str, Any]]:
    """Choose each validation operating point using merged, span-level outcomes."""
    annotations = list(annotations)
    if not annotations:
        raise ValueError("threshold selection requires annotated positive spans")
    arrays = {name: (list(rows), np.asarray(values, dtype=float))
              for name, (rows, values) in score_sets.items()}
    if len(arrays) < 2 or any(len(a) != len(rows) or len(a) == 0 or not np.isfinite(a).all()
                              for rows, a in arrays.values()):
        raise ValueError("all score arrays must be finite and align with candidates")
    eligible_submissions = list(eligible_submissions)
    choices: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    max_recall: dict[str, float] = {}
    for name, (rows, values) in arrays.items():
        points = []
        # Above-maximum threshold represents an empty review list.
        unique = np.unique(values)
        if len(unique) > 101:
            unique = np.unique(np.quantile(unique, np.linspace(0, 1, 101)))
        thresholds = sorted(set(map(float, unique)), reverse=True) + [float(np.max(values)) + 1e-12]
        for threshold in thresholds:
            metrics = merge_alerts(rows, values, threshold, annotations,
                                   eligible_submissions=eligible_submissions,
                                   min_span_coverage=min_span_coverage)
            points.append((threshold, metrics))
        choices[name] = points
        max_recall[name] = max(float(row.get("recall") or 0.0) for _, row in points)
    shared = min(float(requested_recall), min(max_recall.values()))
    if shared < minimum_recall:
        raise ValueError(f"no shared recall operating point meets minimum {minimum_recall:.3f}; feasible={shared:.3f}")
    result: dict[str, Any] = {}
    for name, points in choices.items():
        available = [(row["false_alerts"], -float(row.get("recall") or 0), threshold, row)
                     for threshold, row in points if float(row.get("recall") or 0) + 1e-12 >= shared]
        if not available:
            raise ValueError(f"no valid threshold for {name} at shared recall {shared:.3f}")
        false_alerts, _, threshold, metrics = min(available)
        result[name] = {"threshold": threshold, **{k: v for k, v in metrics.items() if k != "alerts"}}
    baseline = result.get("minilm_cosine")
    candidate = result.get("minilm_xgboost")
    result["shared_recall_target"] = shared
    result["feasible_recall_by_method"] = max_recall
    result["strict_false_alert_improvement"] = bool(
        baseline is not None and candidate is not None and candidate["false_alerts"] < baseline["false_alerts"])
    return shared, result


def evaluate_release_gate(candidate: dict[str, Any], settings: dict[str, Any], *,
                          data_checks_passed: bool, artifact_checks_passed: bool) -> dict[str, Any]:
    """Release gate requiring validation thresholds and strict alert reduction."""
    gates = settings["gates"]
    minimum_false_alert_reduction = gates.get("minimum_false_alert_reduction", 1)
    if (isinstance(minimum_false_alert_reduction, bool)
            or not isinstance(minimum_false_alert_reduction, int)
            or minimum_false_alert_reduction < 1):
        raise ValueError("minimum_false_alert_reduction must be a positive integer")
    checks = []
    def add(name: str, actual: Any, comparator: str, expected: Any, passed: bool) -> None:
        checks.append({"name": name, "actual": actual, "comparator": comparator,
                       "expected": expected, "passed": bool(passed)})
    add("data_checks", data_checks_passed, "is", True, data_checks_passed)
    add("artifact_checks", artifact_checks_passed, "is", True, artifact_checks_passed)
    add("shared_recall_target", candidate.get("shared_recall_target"), ">=", gates["minimum_recall"],
        candidate.get("shared_recall_target") is not None and candidate["shared_recall_target"] >= gates["minimum_recall"])
    recall = candidate.get("recall")
    baseline_recall = candidate.get("minilm_cosine_recall")
    tolerance = float(gates["maximum_recall_drop"])
    add("recall_preservation", recall, ">= baseline - tolerance",
        None if baseline_recall is None else baseline_recall - tolerance,
        recall is not None and baseline_recall is not None and recall >= baseline_recall - tolerance)
    base_fa, model_fa = candidate.get("minilm_cosine_false_alerts"), candidate.get("false_alerts")
    add("false_alert_improvement", model_fa,
        f"<= minilm_cosine - {minimum_false_alert_reduction}",
        None if base_fa is None else base_fa - minimum_false_alert_reduction,
        model_fa is not None and base_fa is not None
        and model_fa <= base_fa - minimum_false_alert_reduction)
    passed = all(x["passed"] for x in checks)
    return {"report_version": "1.0", "decision": "passed" if passed else "failed",
            "passed": passed, "candidate": candidate, "settings": gates, "checks": checks,
            "champion_comparison": None}
