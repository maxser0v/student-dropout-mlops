from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.data import DataContractError, load_corpus
from plagiarism.platform_corpus import PairReview, ReviewSpan, build_platform_corpus, write_platform_snapshot
from plagiarism.platform_types import AssignmentScope, ReportVersion


SCOPE = AssignmentScope("inst-a", "course-a", "assignment-a")


def _reports_and_reviews(*, count: int = 24):
    reports: list[ReportVersion] = []
    reviews: list[PairReview] = []
    reused_text = "Introductory notes. Κοινό κείμενο appears here. Closing notes."
    copied = "Student preface. Κοινό κείμενο appears here. Student conclusion."
    for index in range(count):
        label = index % 2 == 0
        left_id, right_id = f"query-{index:02}", f"source-{index:02}"
        left_text = f"Submission {index}: {copied}" if label else f"Independent submission text {index} with distinct ideas."
        right_text = f"Archive {index}: {reused_text}" if label else f"Unrelated source material {index} discusses another topic."
        left = ReportVersion(SCOPE, left_id, "v1", left_text)
        right = ReportVersion(SCOPE, right_id, "v1", right_text)
        reports.extend((left, right))
        spans = (ReviewSpan(left_text.index("Κοινό κείμενο"), left_text.index("Κοινό κείμενο") + len("Κοινό κείμενο"),
                            right_text.index("Κοινό κείμενο"), right_text.index("Κοινό κείμενο") + len("Κοινό κείμενο")),) if label else ()
        reviews.append(PairReview(f"review-{index:02}", 1, SCOPE, left, right, "full", label, spans))
    return reports, reviews


def test_platform_corpus_is_deterministic_grouped_and_snapshot_ready(tmp_path: Path) -> None:
    reports, reviews = _reports_and_reviews()
    first = build_platform_corpus(SCOPE, reports, reviews, seed=120)
    second = build_platform_corpus(SCOPE, reversed(reports), reversed(reviews), seed=120)
    assert first == second
    assert first.lineage["evaluation_protocol"] == "platform-reviewed-grouped"
    assert first.lineage["scope"] == {
        "institution_id": "inst-a", "course_id": "course-a", "assignment_id": "assignment-a"
    }
    assert first.lineage["review_revisions"] == [
        {"review_id": f"review-{index:02}", "revision": 1} for index in range(24)
    ]
    for split in ("train", "validation"):
        assert {pair.label for pair in first.pairs if pair.split == split} == {0, 1}
    for annotation in first.annotations:
        submission = first.documents[annotation.submission_id]
        source = first.documents[annotation.source_id]
        assert submission.text[annotation.submission_start:annotation.submission_end] == "Κοινό κείμενο"
        assert source.text[annotation.source_start:annotation.source_end] == "Κοινό κείμενο"
        assert submission.group_id == source.group_id

    digest = write_platform_snapshot(first, tmp_path)
    assert digest == write_platform_snapshot(second, tmp_path)
    loaded = load_corpus(tmp_path / "manifest.json")
    assert loaded == first


def test_latest_partial_or_uncertain_review_removes_old_full_judgment() -> None:
    reports, reviews = _reports_and_reviews()
    old = reviews[0]
    reviews[0] = PairReview(old.review_id, 2, SCOPE, old.left, old.right, "partial", None)
    corpus = build_platform_corpus(SCOPE, reports, reviews, seed=120)
    assert all({pair.submission_id, pair.source_id} != {
        f"report:{old.left.report_id}@v1:submission", f"report:{old.right.report_id}@v1:source"
    } for pair in corpus.pairs)
    assert all(pair.submission_id != f"report:{old.left.report_id}@v1:submission" for pair in corpus.pairs)


def test_conflicting_full_pair_reviews_and_duplicate_revision_are_rejected() -> None:
    reports, reviews = _reports_and_reviews()
    original = reviews[0]
    reviews.append(PairReview("second-thread", 1, SCOPE, original.left, original.right, "full", not original.reused))
    with pytest.raises(DataContractError, match="conflicting current full-pair"):
        build_platform_corpus(SCOPE, reports, reviews)

    reports, reviews = _reports_and_reviews()
    reviews.append(reviews[0])
    with pytest.raises(DataContractError, match="duplicate review revision"):
        build_platform_corpus(SCOPE, reports, reviews)


def test_invalid_offsets_self_pairs_and_cross_scope_inputs_fail_closed() -> None:
    reports, reviews = _reports_and_reviews()
    review = reviews[0]
    reviews[0] = PairReview(review.review_id, 1, SCOPE, review.left, review.right, "full", True,
                            (ReviewSpan(0, len(review.left.text) + 1, 0, 1),))
    with pytest.raises(DataContractError, match="offsets out of bounds"):
        build_platform_corpus(SCOPE, reports, reviews)

    reports, reviews = _reports_and_reviews()
    review = reviews[0]
    reviews[0] = PairReview(review.review_id, 1, SCOPE, review.left, review.left, "full", False)
    with pytest.raises(DataContractError, match="self-pair"):
        build_platform_corpus(SCOPE, reports, reviews)

    reports, reviews = _reports_and_reviews()
    outsider = ReportVersion(AssignmentScope("other", "course-a", "assignment-a"), "outsider", "v1", "text")
    with pytest.raises(DataContractError, match="outside the requested"):
        build_platform_corpus(SCOPE, reports + [outsider], reviews)


def test_new_report_version_supersedes_prior_thread_and_snapshot_conflicts_fail(tmp_path: Path) -> None:
    reports, reviews = _reports_and_reviews()
    old = reviews[0]
    revised_left = ReportVersion(SCOPE, old.left.report_id, "v2", old.left.text + " Revised.")
    reports.append(revised_left)
    revised = PairReview(old.review_id, 2, SCOPE, revised_left, old.right, "full", old.reused,
                         (ReviewSpan(revised_left.text.index("Κοινό κείμενο"),
                                     revised_left.text.index("Κοινό κείμενο") + len("Κοινό κείμενο"),
                                     old.right.text.index("Κοινό κείμενο"),
                                     old.right.text.index("Κοινό κείμενο") + len("Κοινό κείμενο")),))
    reviews.append(revised)
    corpus = build_platform_corpus(SCOPE, reports, reviews, seed=120)
    assert any("@v2:submission" in key for key in corpus.documents)
    assert {row["review_id"]: row["revision"] for row in corpus.lineage["review_revisions"]}[old.review_id] == 2
    write_platform_snapshot(corpus, tmp_path)
    altered = build_platform_corpus(SCOPE, reports, reviews, seed=121)
    if altered != corpus:
        with pytest.raises(DataContractError, match="conflicting platform snapshot"):
            write_platform_snapshot(altered, tmp_path)
