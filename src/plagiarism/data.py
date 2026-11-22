"""Corpus import, split, and validation helpers for passage reuse experiments.

PAN text is read as bytes and decoded without universal-newline conversion so
that character offsets in its XML remain valid. The importer accepts either a
downloaded outer PAN 2013 archive (containing nested training/test archives) or
an extracted corpus directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import random
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET
import zipfile


class DataContractError(ValueError):
    """Raised when corpus documents, labels, or splits violate the contract."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    role: str
    content_hash: str
    topic_id: str | None = None
    group_id: str | None = None


@dataclass(frozen=True)
class Annotation:
    """One immutable aligned reuse span, using half-open character offsets."""

    id: str
    submission_id: str
    source_id: str
    submission_start: int
    submission_end: int
    source_start: int
    source_end: int


@dataclass(frozen=True)
class Pair:
    """An explicitly supplied submission/source comparison and its label."""

    submission_id: str
    source_id: str
    label: int
    group_id: str | None = None
    split: str = ""


@dataclass(frozen=True)
class Corpus:
    documents: Mapping[str, Document]
    annotations: tuple[Annotation, ...]
    pairs: tuple[Pair, ...]
    splits: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    lineage: Mapping[str, Any] = field(default_factory=dict)


_SUSPICIOUS = re.compile(r"^suspicious-document", re.IGNORECASE)
_SOURCE = re.compile(r"^source-document", re.IGNORECASE)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _basename(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).name


def _role(name: str) -> str | None:
    base = _basename(name)
    if _SUSPICIOUS.match(base):
        return "submission"
    if _SOURCE.match(base):
        return "source"
    return None


def _read_document(data: bytes, encoding: str, name: str, *, prefix: str = "") -> Document:
    role = _role(name)
    if role is None:
        raise DataContractError(f"unrecognized PAN text document name: {name}")
    try:
        text = data.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise DataContractError(f"cannot decode {name} as {encoding}: {exc}") from exc
    doc_id = f"{prefix}/{_basename(name)}" if prefix else _basename(name)
    return Document(doc_id, text, role, _digest(text))


def _read_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise DataContractError(f"pairs line {line_no} must contain submission and source names")
        submission, source = map(_basename, fields[:2])
        if _role(submission) != "submission" or _role(source) != "source":
            raise DataContractError(f"pairs line {line_no} has invalid document roles")
        pairs.append((submission, source))
    return pairs


def _parse_annotations(xml_bytes: bytes, submission_name: str, *, prefix: str = "") -> list[Annotation]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise DataContractError(f"invalid PAN annotation XML {submission_name}: {exc}") from exc
    refs = {"reference": _basename(root.attrib.get("reference", submission_name))}
    out: list[Annotation] = []
    for i, element in enumerate(root.iter("feature")):
        # PAN files can include non-plagiarism metadata features.
        if element.attrib.get("name", "plagiarism") != "plagiarism":
            continue
        try:
            submission_start = int(element.attrib["this_offset"])
            submission_length = int(element.attrib["this_length"])
            source_id = _basename(element.attrib["source_reference"])
            source_start = int(element.attrib["source_offset"])
            source_length = int(element.attrib["source_length"])
        except (KeyError, ValueError) as exc:
            raise DataContractError(f"invalid aligned span in {submission_name}: {exc}") from exc
        submission_id = f"{prefix}/{refs['reference']}" if prefix else refs["reference"]
        source_id = f"{prefix}/{source_id}" if prefix else source_id
        out.append(Annotation(
            id=f"{submission_id}:{source_id}:{i}",
            submission_id=submission_id,
            source_id=source_id,
            submission_start=submission_start,
            submission_end=submission_start + submission_length,
            source_start=source_start,
            source_end=source_start + source_length,
        ))
    return out


def _import_one_archive(archive: zipfile.ZipFile, *, split: str, encoding: str) -> tuple[list[Document], list[Annotation], list[Pair]]:
    names = [name for name in archive.namelist() if not name.endswith("/")]
    document_paths: dict[str, str] = {}
    annotations: list[Annotation] = []
    pair_paths = [n for n in names if _basename(n).lower() == "pairs"]
    for name in names:
        base = _basename(name)
        if base.lower().endswith(".txt") and _role(base):
            if base in document_paths:
                raise DataContractError(f"duplicate PAN document basename: {base}")
            document_paths[base] = name
    docs = [_read_document(archive.read(path), encoding, path, prefix=split) for _, path in sorted(document_paths.items())]
    docs_by_id = {doc.id: doc for doc in docs}
    pair_defs: list[tuple[str, str]] = []
    for pair_path in pair_paths:
        pair_defs.extend(
            (f"{split}/{submission}", f"{split}/{source}")
            for submission, source in _read_pairs(archive.read(pair_path).decode("utf-8-sig"))
        )
    xml_names = [name for name in names if _basename(name).lower().endswith(".xml")]
    for xml_name in xml_names:
        sub_guess = _basename(xml_name).split("-source-document", 1)[0] + ".txt"
        parsed = _parse_annotations(archive.read(xml_name), sub_guess, prefix=split)
        annotations.extend(parsed)
        if not pair_paths and parsed:
            pair_defs.extend((a.submission_id, a.source_id) for a in parsed)
    pairs: list[Pair] = []
    positive_links = {(a.submission_id, a.source_id) for a in annotations}
    for submission_id, source_id in pair_defs:
        if submission_id not in docs_by_id or source_id not in docs_by_id:
            raise DataContractError(f"PAN pair references missing documents: {submission_id}, {source_id}")
        pairs.append(Pair(submission_id, source_id, int((submission_id, source_id) in positive_links), split=split))
    for ann in annotations:
        if ann.submission_id not in docs_by_id or ann.source_id not in docs_by_id:
            raise DataContractError(f"annotation references missing documents: {ann.submission_id}, {ann.source_id}")
    return docs, annotations, pairs


def load_pan(path: str | Path, *, encoding: str = "utf-8", training_archive_only: bool = False) -> Corpus:
    """Load a PAN13 outer archive or an extracted PAN corpus directory.

    It does not download data. If an outer archive contains nested train/test
    zips, nested archive names containing ``test`` become held-out test data;
    the remaining nested archive(s) are development data. The training-only
    option imports just the named training archive from that official release.
    """
    input_path = Path(path)
    docs: list[Document] = []
    annotations: list[Annotation] = []
    pairs: list[Pair] = []
    lineage: dict[str, Any] = {"dataset": "PAN 2013 Text Alignment", "input": str(input_path), "encoding": encoding}
    if training_archive_only and not zipfile.is_zipfile(input_path):
        raise DataContractError("training-archive-only import requires the official outer PAN ZIP")
    if input_path.is_dir():
        files = list(input_path.rglob("*"))
        text_files = [f for f in files if f.is_file() and f.suffix.lower() == ".txt" and _role(f.name)]
        if not text_files:
            raise DataContractError(f"no PAN text documents found under {input_path}")
        all_pair_files = sorted(f for f in files if f.is_file() and f.name.lower() == "pairs")
        pair_roots = {f.parent for f in all_pair_files}
        pair_files = [f for f in all_pair_files if not pair_roots.intersection(f.parent.parents)]
        pair_defs: list[tuple[str, str, str]] = []
        for pair_file in pair_files:
            split = "test" if "test" in pair_file.as_posix().lower() else "development"
            same_partition = [p for p in pair_files if ("test" in p.as_posix().lower()) == (split == "test")]
            prefix = split if len(same_partition) == 1 else f"{split}-{hashlib.sha1(pair_file.parent.as_posix().encode()).hexdigest()[:8]}"
            # A pairs file scopes its own document and annotation tree. The
            # prefix avoids collisions where releases reuse the same basename.
            for file in sorted(f for f in files if f.is_file() and pair_file.parent in f.parents):
                if file.suffix.lower() == ".txt" and _role(file.name):
                    docs.append(_read_document(file.read_bytes(), encoding, file.name, prefix=prefix))
                elif file.suffix.lower() == ".xml":
                    sub_guess = file.name.split("-source-document", 1)[0] + ".txt"
                    annotations.extend(_parse_annotations(file.read_bytes(), sub_guess, prefix=prefix))
            for scoped_pair_file in all_pair_files:
                if pair_file.parent not in scoped_pair_file.parents:
                    continue
                pair_defs.extend(
                    (f"{prefix}/{sid}", f"{prefix}/{srcid}", split)
                    for sid, srcid in _read_pairs(scoped_pair_file.read_text(encoding="utf-8-sig"))
                )
        if not pair_files:
            for file in sorted(text_files):
                docs.append(_read_document(file.read_bytes(), encoding, file.name, prefix="development"))
            for xml_file in sorted(f for f in files if f.is_file() and f.suffix.lower() == ".xml"):
                sub_guess = xml_file.name.split("-source-document", 1)[0] + ".txt"
                annotations.extend(_parse_annotations(xml_file.read_bytes(), sub_guess, prefix="development"))
        docs_by_id = {doc.id: doc for doc in docs}
        positive_links = {(a.submission_id, a.source_id) for a in annotations}
        for sid, srcid, split in pair_defs:
            if sid not in docs_by_id or srcid not in docs_by_id:
                raise DataContractError(f"PAN pair references missing documents: {sid}, {srcid}")
            pairs.append(Pair(sid, srcid, int((sid, srcid) in positive_links), split=split))
        if not pair_files:
            pairs.extend(Pair(a.submission_id, a.source_id, 1, split="development") for a in annotations)
    elif zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as outer:
            nested = [n for n in outer.namelist() if n.lower().endswith(".zip")]
            if training_archive_only:
                nested = [name for name in nested if "training-corpus" in _basename(name).lower()]
                if len(nested) != 1:
                    raise DataContractError("expected one official PAN training-corpus ZIP in the outer archive")
                lineage.update(input=input_path.name, pan_training_archive=nested[0],
                               evaluation_protocol="pan13-official-training-grouped-holdout")
            if nested:
                for name in sorted(nested):
                    split = "test" if "test" in name.lower() else "development"
                    with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                        d, a, p = _import_one_archive(inner, split=split, encoding=encoding)
                    docs.extend(d)
                    annotations.extend(a)
                    pairs.extend(p)
            else:
                d, a, p = _import_one_archive(outer, split="development", encoding=encoding)
                docs.extend(d)
                annotations.extend(a)
                pairs.extend(p)
        lineage["archive_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    else:
        raise DataContractError(f"PAN input must be an archive or directory: {input_path}")
    if len({doc.id for doc in docs}) != len(docs):
        raise DataContractError("document IDs must be unique across PAN partitions")
    # PAN lists each comparison in the aggregate and its category pairs file.
    # Keep its original ordering and preserve all distinct aligned spans.
    unique_pairs = tuple(dict.fromkeys(pairs))
    lineage["duplicate_pair_rows_removed"] = len(pairs) - len(unique_pairs)
    result = Corpus({doc.id: doc for doc in docs}, tuple(annotations), unique_pairs, lineage=lineage)
    validate_corpus(result)
    return result


def validate_corpus(corpus: Corpus) -> None:
    """Fail closed on invalid references, offsets, labels, or split leakage."""
    docs = corpus.documents
    for doc_id, doc in docs.items():
        if doc.id != doc_id or doc.role not in {"submission", "source"}:
            raise DataContractError(f"invalid document identity or role: {doc_id}")
        if doc.content_hash != _digest(doc.text):
            raise DataContractError(f"content hash mismatch: {doc_id}")
    annotation_keys: set[str] = set()
    for annotation in corpus.annotations:
        if annotation.id in annotation_keys:
            raise DataContractError(f"duplicate annotation id: {annotation.id}")
        annotation_keys.add(annotation.id)
        sub = docs.get(annotation.submission_id)
        src = docs.get(annotation.source_id)
        if sub is None or src is None:
            raise DataContractError(f"annotation references missing documents: {annotation.id}")
        if sub.role != "submission" or src.role != "source":
            raise DataContractError(f"annotation roles are invalid: {annotation.id}")
        for start, end, text, label in (
            (annotation.submission_start, annotation.submission_end, sub.text, "submission"),
            (annotation.source_start, annotation.source_end, src.text, "source"),
        ):
            if start < 0 or end <= start or end > len(text):
                raise DataContractError(f"{label} annotation offsets out of bounds: {annotation.id}")
    annotated_links = {(a.submission_id, a.source_id) for a in corpus.annotations}
    pair_keys: set[tuple[str, str]] = set()
    for pair in corpus.pairs:
        key = (pair.submission_id, pair.source_id)
        if key in pair_keys:
            raise DataContractError(f"duplicate pair: {key}")
        pair_keys.add(key)
        sub, src = docs.get(pair.submission_id), docs.get(pair.source_id)
        if sub is None or src is None or sub.role != "submission" or src.role != "source":
            raise DataContractError(f"pair references missing or invalid documents: {key}")
        if pair.label not in {0, 1}:
            raise DataContractError(f"pair label must be 0 or 1: {key}")
        annotated = key in annotated_links
        if pair.label != int(annotated):
            raise DataContractError(f"pair label disagrees with annotations: {key}")
    split_by_doc: dict[str, str] = {}
    for split, doc_ids in corpus.splits.items():
        if split not in {"train", "validation", "test", "excluded"}:
            raise DataContractError(f"unknown split: {split}")
        for doc_id in doc_ids:
            if doc_id not in docs:
                raise DataContractError(f"split references missing document: {doc_id}")
            previous = split_by_doc.setdefault(doc_id, split)
            if previous != split:
                raise DataContractError(f"document appears in multiple splits: {doc_id}")
    group_splits: dict[str, str] = {}
    for doc_id, split in split_by_doc.items():
        group_id = docs[doc_id].group_id or docs[doc_id].content_hash
        previous = group_splits.setdefault(group_id, split)
        if previous != split:
            raise DataContractError(f"document-group leakage between {previous} and {split}")


def assign_splits(corpus: Corpus, *, validation_fraction: float = 0.2, test_fraction: float = 0.0,
                  seed: int = 903) -> Corpus:
    """Assign linked/exact duplicate document groups before splitting dev data."""
    if not 0 < validation_fraction < 1:
        raise DataContractError("validation_fraction must be between zero and one")
    if not 0 <= test_fraction < 1 - validation_fraction:
        raise DataContractError("test_fraction must leave room for training after validation")
    docs = dict(corpus.documents)
    parent = {doc_id: doc_id for doc_id in docs}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    content_owner: dict[str, str] = {}
    explicit_group_owner: dict[str, str] = {}
    for doc in docs.values():
        previous = content_owner.setdefault(doc.content_hash, doc.id)
        union(previous, doc.id)
        if doc.group_id:
            previous_group = explicit_group_owner.setdefault(doc.group_id, doc.id)
            union(previous_group, doc.id)
    for annotation in corpus.annotations:
        union(annotation.submission_id, annotation.source_id)
    # Only positive source links define a reuse group. Negative comparisons are
    # judgments, not evidence that documents belong to the same content group.
    components: dict[str, list[str]] = {}
    for doc_id in docs:
        components.setdefault(find(doc_id), []).append(doc_id)
    heldout_docs = {doc.id for doc in docs.values() if doc.group_id and doc.group_id.startswith("test:")}
    # A real PAN import marks archives in per-pair split metadata; explicit
    # `test` pair groups are propagated to documents below.
    test_pairs = [(p.submission_id, p.source_id) for p in corpus.pairs if p.split == "test"]
    for submission_id, source_id in test_pairs:
        heldout_docs.update((submission_id, source_id))
    if test_fraction and heldout_docs:
        raise DataContractError("cannot create a custom test split from a corpus with a reserved test split")
    test_roots = {find(doc_id) for doc_id in heldout_docs}
    # Duplicate hashes or positive links connecting a benchmark item to dev
    # material stay with the upstream test group.
    dev_roots = {find(doc_id) for doc_id in docs} - test_roots
    rng = random.Random(seed)
    ordered = sorted(dev_roots)
    rng.shuffle(ordered)
    validation_n = round(len(ordered) * validation_fraction)
    validation_roots = set(ordered[:validation_n])
    if test_fraction:
        test_n = round(len(ordered) * test_fraction)
        if min(validation_n, test_n, len(ordered) - validation_n - test_n) < 1:
            raise DataContractError("custom split requires at least one document group in every partition")
        test_roots.update(ordered[validation_n:validation_n + test_n])
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": [], "excluded": []}
    split_for_root = {root: ("test" if root in test_roots else "validation" if root in validation_roots else "train") for root in components}
    # Ensure linked docs always share an explicit, stable group ID.
    group_for_root = {root: f"group:{root}" for root in components}
    updated: dict[str, Document] = {}
    for doc_id, doc in docs.items():
        root = find(doc_id)
        split = split_for_root[root]
        updated[doc_id] = Document(doc.id, doc.text, doc.role, doc.content_hash, doc.topic_id, group_for_root[root])
        splits[split].append(doc_id)
    # Cross-split negative comparisons have no safe training/evaluation role;
    # drop and count them. Positive links were grouped above and remain intact.
    excluded_negative_pairs = 0
    assigned_pairs: list[Pair] = []
    for pair in corpus.pairs:
        sub_root, source_root = find(pair.submission_id), find(pair.source_id)
        sub_split, source_split = split_for_root[sub_root], split_for_root[source_root]
        if pair.label == 0 and sub_split != source_split:
            excluded_negative_pairs += 1
            continue
        assigned_pairs.append(Pair(pair.submission_id, pair.source_id, pair.label, group_for_root[sub_root], sub_split))
    result = Corpus(
        updated,
        corpus.annotations,
        tuple(assigned_pairs),
        {k: tuple(sorted(v)) for k, v in splits.items()},
        dict(corpus.lineage, split_seed=seed, validation_fraction=validation_fraction,
             excluded_cross_split_negative_pairs=excluded_negative_pairs,
             **({"test_fraction": test_fraction} if test_fraction else {})),
    )
    validate_corpus(result)
    return result


def corpus_to_dict(corpus: Corpus) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "documents": [vars(corpus.documents[key]) for key in sorted(corpus.documents)],
        "annotations": [vars(row) for row in corpus.annotations],
        "pairs": [vars(row) for row in corpus.pairs],
        "splits": {key: list(value) for key, value in corpus.splits.items()},
        "lineage": dict(corpus.lineage),
    }


def save_corpus(corpus: Corpus, path: str | Path) -> None:
    validate_corpus(corpus)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus_to_dict(corpus), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="")


def load_corpus(path: str | Path, *, include_splits: Iterable[str] | None = None) -> Corpus:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise DataContractError("unsupported corpus schema version")
        documents = {row["id"]: Document(**row) for row in raw["documents"]}
        annotations = tuple(Annotation(**row) for row in raw["annotations"])
        pairs = tuple(Pair(**row) for row in raw["pairs"])
        splits = {key: tuple(value) for key, value in raw.get("splits", {}).items()}
        corpus = Corpus(documents, annotations, pairs, splits, raw.get("lineage", {}))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataContractError(f"cannot read corpus manifest {path}: {exc}") from exc
    validate_corpus(corpus)
    if include_splits is None:
        return corpus
    return filter_corpus(corpus, include_splits=include_splits)


def filter_corpus(corpus: Corpus, *, include_splits: Iterable[str]) -> Corpus:
    """Return a subset without exposing records from unselected splits."""
    selected = set(include_splits)
    unknown = selected - set(corpus.splits)
    if unknown:
        raise DataContractError(f"requested splits are missing: {sorted(unknown)}")
    allowed_ids = {doc_id for split in selected for doc_id in corpus.splits[split]}
    docs = {doc_id: doc for doc_id, doc in corpus.documents.items() if doc_id in allowed_ids}
    annotations = tuple(a for a in corpus.annotations if a.submission_id in docs and a.source_id in docs)
    pairs = tuple(p for p in corpus.pairs if p.submission_id in docs and p.source_id in docs and p.split in selected)
    result = Corpus(docs, annotations, pairs, {split: tuple(doc_id for doc_id in corpus.splits[split] if doc_id in docs) for split in selected}, corpus.lineage)
    validate_corpus(result)
    return result
