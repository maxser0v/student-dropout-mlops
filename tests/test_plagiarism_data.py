from __future__ import annotations

import io
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.data import (
    Annotation,
    Corpus,
    DataContractError,
    Document,
    Pair,
    assign_splits,
    filter_corpus,
    load_corpus,
    load_pan,
    validate_corpus,
)
from plagiarism.features import FeaturePipeline, TextPair
from plagiarism.retrieval import FixtureEncoder


FIXTURE = ROOT / "fixtures" / "plagiarism" / "fixture.json"


def test_authored_fixture_has_both_classes_in_every_split_and_exact_offsets() -> None:
    corpus = load_corpus(FIXTURE)
    for split in ("train", "validation", "test"):
        labels = {pair.label for pair in corpus.pairs if pair.split == split}
        assert labels == {0, 1}
    assert corpus.lineage["same_topic_stress"] is True
    assert len({doc.topic_id for doc in corpus.documents.values()}) == 1
    for annotation in corpus.annotations:
        query = corpus.documents[annotation.submission_id].text
        source = corpus.documents[annotation.source_id].text
        assert query[annotation.submission_start:annotation.submission_end]
        assert query[annotation.submission_start:annotation.submission_end] == source[annotation.source_start:annotation.source_end]


def test_training_loader_can_exclude_heldout_document_text() -> None:
    training = load_corpus(FIXTURE, include_splits=("train", "validation"))
    assert set(training.splits) == {"train", "validation"}
    assert all(pair.split != "test" for pair in training.pairs)
    assert all("test" not in doc_id for doc_id in training.documents)


def test_custom_holdout_is_deterministic_and_preserves_linked_document_groups() -> None:
    source = load_corpus(FIXTURE, include_splits=("train", "validation"))
    split = assign_splits(source, validation_fraction=0.25, test_fraction=0.25, seed=903)
    duplicate = assign_splits(source, validation_fraction=0.25, test_fraction=0.25, seed=903)
    assert split == duplicate
    assert split.lineage["test_fraction"] == 0.25
    for name in ("train", "validation", "test"):
        documents = set(split.splits[name])
        assert documents
        assert {pair.label for pair in split.pairs if pair.split == name} == {0, 1}
        assert all({pair.submission_id, pair.source_id} <= documents
                   for pair in split.pairs if pair.split == name)
    with pytest.raises(DataContractError, match="reserved test split"):
        assign_splits(load_corpus(FIXTURE), test_fraction=0.2)


def test_same_topic_hard_negatives_outscore_positives_in_lexical_retrieval() -> None:
    corpus = load_corpus(FIXTURE)
    training_texts = [corpus.documents[key].text for key in corpus.splits["train"]]
    pipeline = FeaturePipeline().fit(training_texts)
    encoder = FixtureEncoder()
    for split in ("train", "validation", "test"):
        positive = next(pair for pair in corpus.pairs if pair.split == split and pair.label == 1)
        negative = next(pair for pair in corpus.pairs if pair.split == split and pair.label == 0)
        query = corpus.documents[positive.submission_id].text
        assert positive.submission_id == negative.submission_id
        positive_text = corpus.documents[positive.source_id].text
        negative_text = corpus.documents[negative.source_id].text
        positive_cosine = float((encoder.encode_texts([query]) @ encoder.encode_texts([positive_text]).T)[0, 0])
        negative_cosine = float((encoder.encode_texts([query]) @ encoder.encode_texts([negative_text]).T)[0, 0])
        assert negative_cosine > positive_cosine
        assert not _trigrams(query) & _trigrams(negative_text)
        positive_features = pipeline.transform_pairs([TextPair(query, positive_text, positive_cosine)])[0]
        negative_features = pipeline.transform_pairs([TextPair(query, negative_text, negative_cosine)])[0]
        assert positive_features[2] > negative_features[2]


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    import re

    words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text.casefold())
    return {tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))}


def test_invalid_offsets_and_group_leakage_fail_closed() -> None:
    corpus = load_corpus(FIXTURE)
    annotation = corpus.annotations[0]
    bad = Annotation(**{**vars(annotation), "submission_end": len(corpus.documents[annotation.submission_id].text) + 1})
    with pytest.raises(DataContractError, match="offsets out of bounds"):
        validate_corpus(Corpus(corpus.documents, (bad,), corpus.pairs, corpus.splits, corpus.lineage))

    docs = dict(corpus.documents)
    train_id = corpus.splits["train"][0]
    val_id = corpus.splits["validation"][0]
    docs[train_id] = Document(**{**vars(docs[train_id]), "group_id": "leaked"})
    docs[val_id] = Document(**{**vars(docs[val_id]), "group_id": "leaked"})
    with pytest.raises(DataContractError, match="group leakage"):
        validate_corpus(Corpus(docs, corpus.annotations, corpus.pairs, corpus.splits, corpus.lineage))


@pytest.mark.parametrize("extracted", [False, True])
def test_pan_preserves_newlines_offsets_and_judged_negatives(tmp_path: Path, extracted: bool) -> None:
    source = b"first line\r\nsecond shared line\r\n"
    submission = b"new introduction\r\nsecond shared line\r\n"
    inner_bytes = io.BytesIO()
    with zipfile.ZipFile(inner_bytes, "w") as inner:
        inner.writestr("source-documents/source-document1.txt", source)
        inner.writestr("suspicious-documents/suspicious-document1.txt", submission)
        inner.writestr("suspicious-documents/suspicious-document2.txt", b"independent text\r\n")
        inner.writestr("pairs", "suspicious-document1.txt source-document1.txt\nsuspicious-document2.txt source-document1.txt\n")
        inner.writestr("category/pairs", "suspicious-document1.txt source-document1.txt\n")
        inner.writestr(
            "suspicious-document1-source-document1.xml",
            '<document reference="suspicious-document1.txt"><feature name="plagiarism" this_offset="18" this_length="20" source_reference="source-document1.txt" source_offset="12" source_length="20"/></document>',
        )
        inner.writestr("suspicious-document2-source-document1.xml", '<document reference="suspicious-document2.txt"/>')
    outer_path = tmp_path / "pan.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.writestr("pan13-text-alignment-test-corpus.zip", inner_bytes.getvalue())
    if extracted:
        with zipfile.ZipFile(inner_bytes) as inner:
            inner.extractall(tmp_path / "test-corpus")
    corpus = load_pan(tmp_path if extracted else outer_path)
    assert corpus.documents["test/source-document1.txt"].text == source.decode("utf-8")
    assert corpus.annotations[0].submission_start == 18
    assert corpus.documents["test/suspicious-document1.txt"].text[18:38] == "second shared line\r\n"
    assert {(pair.label, pair.split) for pair in corpus.pairs} == {(0, "test"), (1, "test")}
    assert len(corpus.pairs) == 2
    assert corpus.lineage["duplicate_pair_rows_removed"] == 1
    split = assign_splits(corpus)
    assert split.splits["test"]
    assert {pair.split for pair in split.pairs} == {"test"}
