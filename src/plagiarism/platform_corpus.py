"""Build immutable training corpora from platform-authorized pair reviews.

This module is deliberately an adapter, not an authorization layer. Callers
must resolve report and review records inside the caller's authorized scope
before passing them here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable
from urllib.parse import quote

from .data import (
    Annotation,
    Corpus,
    DataContractError,
    Document,
    Pair,
    assign_splits,
    corpus_to_dict,
    validate_corpus,
)
from .platform_types import AssignmentScope, ReportVersion


@dataclass(frozen=True)
class ReviewSpan:
    """Confirmed aligned character offsets in the review's left/right reports."""

    left_start: int
    left_end: int
    right_start: int
    right_end: int


@dataclass(frozen=True)
class PairReview:
    """One immutable revision of a report-pair judgment.

    ``coverage`` is ``"full"``, ``"partial"``, or ``"uncertain"``. Only a
    full review with a boolean ``reused`` verdict is eligible for training.
    A positive full review must include every confirmed aligned span; a
    negative full review must have no spans.
    """

    review_id: str
    revision: int
    scope: AssignmentScope
    left: ReportVersion
    right: ReportVersion
    coverage: str
    reused: bool | None
    spans: tuple[ReviewSpan, ...] = ()


def _report_key(report: ReportVersion) -> tuple[str, str]:
    return report.report_id, report.version_id


def _role_id(report: ReportVersion, role: str) -> str:
    return f"report:{quote(report.report_id, safe='')}@{quote(report.version_id, safe='')}:{role}"


def _logical_group(report: ReportVersion) -> str:
    # All revisions and role surrogates for one logical report stay together.
    return f"platform-report:{quote(report.report_id, safe='')}"


def _report_pair_key(left: ReportVersion, right: ReportVersion) -> tuple[tuple[str, str], tuple[str, str]]:
    return tuple(sorted((_report_key(left), _report_key(right))))  # type: ignore[return-value]


def _canonical_review(review: PairReview) -> tuple[tuple[str, str], tuple[str, str], tuple[tuple[int, int, int, int], ...]]:
    """Return stable undirected pair and span order, swapping reversed input."""
    left_key, right_key = _report_key(review.left), _report_key(review.right)
    if left_key <= right_key:
        spans = tuple(sorted((s.left_start, s.left_end, s.right_start, s.right_end) for s in review.spans))
        return left_key, right_key, spans
    spans = tuple(sorted((s.right_start, s.right_end, s.left_start, s.left_end) for s in review.spans))
    return right_key, left_key, spans


def _validate_span(span: ReviewSpan, left: ReportVersion, right: ReportVersion, review_id: str) -> None:
    for start, end, text, side in (
        (span.left_start, span.left_end, left.text, "left"),
        (span.right_start, span.right_end, right.text, "right"),
    ):
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
            raise DataContractError(f"{side} review offsets must be integers: {review_id}")
        if start < 0 or end <= start or end > len(text):
            raise DataContractError(f"{side} review offsets out of bounds: {review_id}")


def _canonical_json_bytes(corpus: Corpus) -> bytes:
    return (json.dumps(corpus_to_dict(corpus), indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def build_platform_corpus(
    scope: AssignmentScope,
    reports: Iterable[ReportVersion],
    reviews: Iterable[PairReview],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 903,
) -> Corpus:
    """Build a deterministic, grouped corpus from full pair reviews only.

    Review revision histories are collapsed to the latest revision per
    ``review_id``. Duplicate revision numbers and review threads that change
    scope or pair are rejected. Different current review threads for the same
    logical pair must agree on both verdict and aligned spans.
    """
    if not isinstance(scope, AssignmentScope):
        raise DataContractError("scope must be an AssignmentScope")
    report_map: dict[tuple[str, str], ReportVersion] = {}
    for report in reports:
        if not isinstance(report, ReportVersion):
            raise DataContractError("reports must contain ReportVersion values")
        if report.scope != scope:
            raise DataContractError(f"report is outside the requested assignment scope: {report.report_id}")
        if not report.report_id or not report.version_id:
            raise DataContractError("report_id and version_id must be non-empty")
        key = _report_key(report)
        previous = report_map.get(key)
        if previous is not None:
            if previous != report:
                raise DataContractError(f"conflicting canonical report version: {key}")
            raise DataContractError(f"duplicate canonical report version: {key}")
        report_map[key] = report

    revisions: dict[tuple[str, int], PairReview] = {}
    thread_identity: dict[str, tuple[AssignmentScope, tuple[str, str]]] = {}
    for review in reviews:
        if not isinstance(review, PairReview):
            raise DataContractError("reviews must contain PairReview values")
        if not review.review_id or not isinstance(review.revision, int) or isinstance(review.revision, bool) or review.revision < 1:
            raise DataContractError("review_id must be non-empty and revision must be a positive integer")
        if review.scope != scope:
            raise DataContractError(f"review is outside the requested assignment scope: {review.review_id}")
        if not isinstance(review.left, ReportVersion) or not isinstance(review.right, ReportVersion):
            raise DataContractError(f"review reports must be ReportVersion values: {review.review_id}")
        if review.left.scope != scope or review.right.scope != scope:
            raise DataContractError(f"review report is outside the requested assignment scope: {review.review_id}")
        left_key, right_key = _report_key(review.left), _report_key(review.right)
        if left_key == right_key or review.left.report_id == review.right.report_id:
            raise DataContractError(f"self-pair review is invalid: {review.review_id}")
        for key, given in ((left_key, review.left), (right_key, review.right)):
            canonical = report_map.get(key)
            if canonical is None or canonical != given:
                raise DataContractError(f"review references a missing or conflicting canonical report version: {key}")
        pair_key = tuple(sorted((review.left.report_id, review.right.report_id)))
        identity = (review.scope, pair_key)
        old_identity = thread_identity.setdefault(review.review_id, identity)
        if old_identity != identity:
            raise DataContractError(f"review thread changed scope or report pair: {review.review_id}")
        revision_key = (review.review_id, review.revision)
        if revision_key in revisions:
            raise DataContractError(f"duplicate review revision: {revision_key}")
        if review.coverage not in {"full", "partial", "uncertain"}:
            raise DataContractError(f"unknown review coverage: {review.coverage}")
        if review.coverage == "full":
            if not isinstance(review.reused, bool):
                raise DataContractError(f"full review must have a boolean verdict: {review.review_id}")
            if review.reused and not review.spans:
                raise DataContractError(f"positive full review must include confirmed spans: {review.review_id}")
            if not review.reused and review.spans:
                raise DataContractError(f"negative full review cannot include spans: {review.review_id}")
            for span in review.spans:
                _validate_span(span, review.left, review.right, review.review_id)
        revisions[revision_key] = review

    latest: dict[str, PairReview] = {}
    for (review_id, _revision), review in sorted(revisions.items()):
        current = latest.get(review_id)
        if current is None or review.revision > current.revision:
            latest[review_id] = review

    eligible: dict[tuple[tuple[str, str], tuple[str, str]], tuple[bool, tuple[tuple[int, int, int, int], ...]]] = {}
    logical_versions: dict[tuple[str, str], tuple[tuple[str, str], tuple[str, str]]] = {}
    for review in latest.values():
        if review.coverage != "full" or review.reused is None:
            continue
        pair_key = _report_pair_key(review.left, review.right)
        logical_pair = tuple(sorted((review.left.report_id, review.right.report_id)))
        prior_versions = logical_versions.get(logical_pair)
        if prior_versions is not None and prior_versions != pair_key:
            raise DataContractError(f"conflicting report versions in current reviews: {logical_pair}")
        logical_versions[logical_pair] = pair_key
        _, _, normalized_spans = _canonical_review(review)
        judgment = (review.reused, normalized_spans)
        previous = eligible.get(pair_key)
        if previous is not None and previous != judgment:
            raise DataContractError(f"conflicting current full-pair reviews: {pair_key}")
        eligible[pair_key] = judgment

    if not eligible:
        raise DataContractError("no full-pair reviews are eligible for training")

    if not 0 < validation_fraction < 1:
        raise DataContractError("validation_fraction must be between zero and one")
    if not 0 <= test_fraction < 1 - validation_fraction:
        raise DataContractError("test_fraction must leave room for training after validation")
    requested_partitions = 3 if test_fraction else 2
    positive_count = sum(label for label, _spans in eligible.values())
    negative_count = len(eligible) - positive_count
    if positive_count < requested_partitions or negative_count < requested_partitions:
        raise DataContractError("reviewed data cannot provide both pair classes in every requested split")

    docs: dict[str, Document] = {}
    annotations: list[Annotation] = []
    pairs: list[Pair] = []
    for pair_key in sorted(eligible):
        left_key, right_key = pair_key
        left, right = report_map[left_key], report_map[right_key]
        # The stable report ID order gives every undirected review one role
        # assignment, regardless of the order supplied by the platform.
        submission, source = left, right
        submission_id = _role_id(submission, "submission")
        source_id = _role_id(source, "source")
        for doc_id, report, role in (
            (submission_id, submission, "submission"),
            (source_id, source, "source"),
        ):
            docs.setdefault(doc_id, Document(
                id=doc_id,
                text=report.text,
                role=role,
                content_hash=hashlib.sha256(report.text.encode("utf-8")).hexdigest(),
                group_id=_logical_group(report),
            ))
        reused, spans = eligible[pair_key]
        pairs.append(Pair(submission_id, source_id, int(reused)))
        if reused:
            for span_index, (submission_start, submission_end, source_start, source_end) in enumerate(spans):
                annotations.append(Annotation(
                    id=f"{submission_id}:{source_id}:span:{span_index}",
                    submission_id=submission_id,
                    source_id=source_id,
                    submission_start=submission_start,
                    submission_end=submission_end,
                    source_start=source_start,
                    source_end=source_end,
                ))

    base = Corpus(docs, tuple(annotations), tuple(pairs), lineage={
        "evaluation_protocol": "platform-reviewed-grouped",
        "scope": {
            "institution_id": scope.institution_id,
            "course_id": scope.course_id,
            "assignment_id": scope.assignment_id,
        },
        "review_ids": sorted(review.review_id for review in latest.values()
                              if review.coverage == "full" and review.reused is not None
                              and _report_pair_key(review.left, review.right) in eligible),
        "review_revisions": sorted(
            ({"review_id": review.review_id, "revision": review.revision}
             for review in latest.values()
             if review.coverage == "full" and review.reused is not None
             and _report_pair_key(review.left, review.right) in eligible),
            key=lambda row: (row["review_id"], row["revision"]),
        ),
        "review_revision_policy": "latest-per-review-id; conflicting-current-pairs-rejected",
        "split_seed_requested": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    })
    validate_corpus(base)

    # `assign_splits` drops negative comparisons whose reports land in
    # different partitions. Search a deterministic sequence derived from the
    # requested seed so each required partition retains both classes.
    needed = ("train", "validation", "test") if test_fraction else ("train", "validation")
    assigned: Corpus | None = None
    for candidate_seed in range(seed, seed + 128):
        try:
            candidate = assign_splits(base, validation_fraction=validation_fraction,
                                      test_fraction=test_fraction, seed=candidate_seed)
        except DataContractError:
            continue
        if all({pair.label for pair in candidate.pairs if pair.split == split} == {0, 1} for split in needed):
            assigned = candidate
            break
    if assigned is None:
        raise DataContractError("reviewed data cannot provide both pair classes in every requested split")
    lineage = dict(assigned.lineage)
    lineage["split_seed_requested"] = seed
    lineage["split_seed"] = candidate_seed
    assigned = Corpus(assigned.documents, assigned.annotations, assigned.pairs, assigned.splits, lineage)
    validate_corpus(assigned)

    # Bind a content hash to the deterministic corpus payload, excluding the
    # hash field itself. The writer returns the separate exact manifest hash.
    payload = corpus_to_dict(assigned)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    final_lineage = dict(assigned.lineage)
    final_lineage["dataset_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    result = Corpus(assigned.documents, assigned.annotations, assigned.pairs, assigned.splits, final_lineage)
    validate_corpus(result)
    return result


def write_platform_snapshot(
    corpus: Corpus,
    output_dir: str | Path,
    *,
    manifest_name: str = "manifest.json",
) -> str:
    """Write a deterministic immutable manifest and return its exact SHA-256.

    Reusing a destination with identical bytes is idempotent. A destination
    containing different bytes fails closed and is never overwritten.
    """
    validate_corpus(corpus)
    if corpus.lineage.get("evaluation_protocol") != "platform-reviewed-grouped":
        raise DataContractError("snapshot is not a platform-reviewed grouped corpus")
    if Path(manifest_name).name != manifest_name or manifest_name in {"", ".", ".."}:
        raise DataContractError("manifest_name must be a simple filename")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / manifest_name
    content = _canonical_json_bytes(corpus)
    digest = hashlib.sha256(content).hexdigest()
    if target.exists():
        if target.read_bytes() != content:
            raise DataContractError(f"refusing to replace a conflicting platform snapshot: {target}")
        return digest

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output, prefix=f".{manifest_name}.", delete=False) as temp:
            temp_name = temp.name
            temp.write(content)
            temp.flush()
            os.fsync(temp.fileno())
        try:
            os.link(temp_name, target)
        except FileExistsError:
            if target.read_bytes() != content:
                raise DataContractError(f"refusing to replace a conflicting platform snapshot: {target}")
        return digest
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
