"""Regenerate the small authored corpus fixture used by offline checks."""

from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.data import Annotation, Corpus, Document, Pair, assign_splits, save_corpus


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text.casefold())
    return {tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))}


def _reordered_hard_negative(query: str, seed: int) -> str:
    """Create an authored lexical stress control by permuting query terms."""
    terms = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", query)
    expected = _trigrams(query)
    rng = random.Random(seed)
    for _ in range(10000):
        shuffled = list(terms)
        rng.shuffle(shuffled)
        candidate = " ".join(shuffled) + "."
        if not (_trigrams(candidate) & expected):
            return candidate
    raise RuntimeError("could not produce a no-shared-trigram stress control")


def main() -> None:
    rows = [
        ("train-a", "A reliable experiment keeps a final test set untouched. The test set estimates performance on examples that played no role in fitting the model.", "", "submission-train-a", "In this report the panel discusses a long list of examples, model fitting, study design, review practice, and quality control. A reliable experiment keeps a final test set untouched. The test set estimates performance on examples that played no role in fitting the model. It then considers author history, the scoring record, and the material submitted for assessment.", "A reliable experiment keeps a final test set untouched. The test set estimates performance on examples that played no role in fitting the model.", "development"),
        ("train-b", "Researchers should report both retrieval failures and scoring errors. Retrieval determines which passages reach the reviewer, while scoring ranks the retrieved candidates.", "", "submission-train-b", "The chapter reviews several evaluation details, the candidate pool, model settings, and the reviewer workflow. Researchers should report both retrieval failures and scoring errors. Retrieval determines which passages reach the reviewer, while scoring ranks the retrieved candidates. Separate paragraphs describe test coverage and annotation quality.", "Researchers should report both retrieval failures and scoring errors. Retrieval determines which passages reach the reviewer, while scoring ranks the retrieved candidates.", "development"),
        ("dev-a", "A passage alignment records the start and end positions on both sides of a reused segment. The original text is retained so a reviewer can verify the evidence.", "", "submission-dev-a", "The assignment considers several review examples, passage records, evidence checks, annotation steps, and score summaries. A passage alignment records the start and end positions on both sides of a reused segment. The original text is retained so a reviewer can verify the evidence. It also describes how the team stores the final report.", "A passage alignment records the start and end positions on both sides of a reused segment. The original text is retained so a reviewer can verify the evidence.", "development"),
        ("dev-b", "A student can compare several passages before deciding which one deserves closer review. Similar vocabulary alone is not proof of copied wording.", "", "submission-dev-b", "When the question concerns the same subject, a student can compare several passages before deciding which one deserves closer review. Similar vocabulary alone is not proof of copied wording. The surrounding discussion covers examples, evidence, and the review process.", "student can compare several passages before deciding which one deserves closer review. Similar vocabulary alone is not proof of copied wording.", "development"),
        ("test-a", "A good evaluation reserves examples that were unavailable during model fitting. This helps reveal whether predictions carry over to new work.", "", "submission-test-a", "The final sample includes discussion about evaluation, model fitting, data coverage, predictions, independent review, and the study's limits. A good evaluation reserves examples that were unavailable during model fitting. This helps reveal whether predictions carry over to new work. It closes with notes about the annotation process.", "A good evaluation reserves examples that were unavailable during model fitting. This helps reveal whether predictions carry over to new work.", "test"),
    ]
    docs: dict[str, Document] = {}
    annotations: list[Annotation] = []
    pairs: list[Pair] = []
    import hashlib
    for name, source, hard_negative, submission_id, submission, reused_span, split in rows:
        source_id = f"source-{name}"
        negative_id = f"source-{name}-independent"
        group_id = f"test:{name}" if split == "test" else f"fixture:{name}"
        docs[source_id] = Document(source_id, source, "source", hashlib.sha256(source.encode()).hexdigest(), "evaluation-data", group_id)
        hard_negative = _reordered_hard_negative(submission, seed=sum(name.encode()))
        docs[negative_id] = Document(negative_id, hard_negative, "source", hashlib.sha256(hard_negative.encode()).hexdigest(), "evaluation-data", group_id)
        docs[submission_id] = Document(submission_id, submission, "submission", hashlib.sha256(submission.encode()).hexdigest(), "evaluation-data", group_id)
        pairs.extend((Pair(submission_id, source_id, 1, group_id, split), Pair(submission_id, negative_id, 0, group_id, split)))
        if reused_span not in source or reused_span not in submission:
            raise RuntimeError(f"expected authored reuse span missing for {submission_id}")
        sub_start = submission.index(reused_span)
        src_start = source.index(reused_span)
        annotations.append(Annotation(
            f"{submission_id}:{source_id}:0", submission_id, source_id,
            sub_start, sub_start + len(reused_span), src_start, src_start + len(reused_span),
        ))
    corpus = assign_splits(Corpus(docs, tuple(annotations), tuple(pairs), lineage={"dataset": "authored-offline-fixture", "license": "CC0-1.0", "same_topic_stress": True, "note": "synthetic demonstration labels; negatives are deterministic bag-of-words stress controls, not a benchmark"}), validation_fraction=0.25, seed=903)
    save_corpus(corpus, ROOT / "fixtures" / "plagiarism" / "fixture.json")


if __name__ == "__main__":
    main()
