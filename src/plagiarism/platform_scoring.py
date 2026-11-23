"""Assignment-scoped, immutable source indexes and platform scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import fcntl
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import numpy as np

from .features import FeaturePipeline
from .platform_types import AssignmentScope, ReportVersion
from .retrieval import Chunk, chunk_text, chunking_config, load_source_cache, retrieve, save_source_cache
from .training import positive_class_probability


@dataclass(frozen=True)
class AssignmentIndex:
    scope: AssignmentScope
    index_version: str
    manifest: dict[str, Any]
    chunks: tuple[Chunk, ...]
    vectors: np.ndarray
    report_documents: dict[str, dict[str, str]]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scope_record(scope: AssignmentScope) -> dict[str, str]:
    return asdict(scope)


def _validate_reports(reports: Sequence[ReportVersion], scope: AssignmentScope) -> list[ReportVersion]:
    if not isinstance(scope, AssignmentScope):
        raise ValueError("expected_scope must be an AssignmentScope")
    if not reports:
        raise ValueError("assignment index requires at least one report version")
    items = list(reports)
    seen: set[tuple[str, str, str, str, str]] = set()
    for report in items:
        if not isinstance(report, ReportVersion):
            raise ValueError("reports must contain ReportVersion values")
        if report.scope != scope:
            raise ValueError("report version is outside the expected assignment scope")
        if report.identity in seen:
            raise ValueError("duplicate report version identity in assignment input")
        seen.add(report.identity)
    return sorted(items, key=lambda item: item.identity)


def _combine_reports(reports: Sequence[ReportVersion], extra: Sequence[ReportVersion],
                    scope: AssignmentScope) -> list[ReportVersion]:
    combined: dict[tuple[str, str, str, str, str], ReportVersion] = {}
    for report in [*reports, *extra]:
        if not isinstance(report, ReportVersion) or report.scope != scope:
            raise ValueError("report version is outside the expected assignment scope")
        previous = combined.get(report.identity)
        if previous is not None and previous.text != report.text:
            raise ValueError("duplicate report version identity has conflicting text")
        combined[report.identity] = report
    return _validate_reports(list(combined.values()), scope)


def _source_manifest(reports: Sequence[ReportVersion], scope: AssignmentScope,
                     encoder_revision: str, config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    sources = [{"report_id": report.report_id, "version_id": report.version_id,
                "content_sha256": hashlib.sha256(report.text.encode("utf-8")).hexdigest()}
               for report in reports]
    digest = hashlib.sha256(_canonical({"scope": _scope_record(scope), "sources": sources,
                                        "encoder_revision": encoder_revision,
                                        "chunking_config": config})).hexdigest()
    return digest, sources


def _document_id(report: ReportVersion) -> str:
    return "platform-report:" + hashlib.sha256(_canonical(report.identity)).hexdigest()


def build_assignment_index(
    reports: Sequence[ReportVersion], scope: AssignmentScope, index_path: str | Path,
    manifest_path: str | Path, encoder: Any, *, target_tokens: int = 180,
    overlap_tokens: int = 40, max_tokens: int = 256,
    source_reports: Sequence[ReportVersion] = (),
) -> dict[str, Any]:
    """Serialize publishers per path; flock releases automatically on process exit."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(manifest_path) + ".lock")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        return _build_assignment_index_locked(
            reports, scope, index_path, manifest_path, encoder,
            target_tokens=target_tokens, overlap_tokens=overlap_tokens,
            max_tokens=max_tokens, source_reports=source_reports,
        )


def _build_assignment_index_locked(
    reports: Sequence[ReportVersion], scope: AssignmentScope, index_path: str | Path,
    manifest_path: str | Path, encoder: Any, *, target_tokens: int,
    overlap_tokens: int, max_tokens: int, source_reports: Sequence[ReportVersion],
) -> dict[str, Any]:
    """Build and atomically publish an immutable index plus its integrity manifest.

    The returned manifest contains a content-derived ``index_version``. Callers
    should persist that value as the trusted version expected by scoring jobs.
    """
    ordered = _combine_reports(reports, source_reports, scope)
    config = chunking_config(target_tokens=target_tokens, overlap_tokens=overlap_tokens,
                             max_tokens=max_tokens)
    source_hash, sources = _source_manifest(ordered, scope, str(encoder.revision), config)
    index_path, manifest_path = Path(index_path), Path(manifest_path)
    input_core = {"schema_version": 1, "scope": _scope_record(scope), "sources": sources,
                  "encoder_revision": str(encoder.revision), "chunking_config": config,
                  "source_manifest_hash": source_hash}
    if index_path.exists() or manifest_path.exists():
        if not (index_path.exists() and manifest_path.exists()):
            raise ValueError("assignment index snapshot path is partially published; use a new path")
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing assignment index snapshot is invalid; use a new path") from exc
        existing_core = {key: existing.get(key) for key in input_core}
        if existing_core != input_core:
            raise ValueError("assignment index snapshot path is immutable; use a new path")
        complete_core = {**input_core, "index_sha256": existing.get("index_sha256")}
        expected_version = hashlib.sha256(_canonical(complete_core)).hexdigest()
        if (existing.get("index_version") != expected_version
                or _sha256(index_path) != existing.get("index_sha256")):
            raise ValueError("existing assignment index snapshot failed integrity verification")
        cached_chunks, cached_vectors = load_source_cache(index_path, source_manifest_hash=source_hash,
            encoder_revision=str(encoder.revision), config=config)
        expected_documents = {_document_id(report) for report in ordered}
        if any(chunk.document_id not in expected_documents for chunk in cached_chunks):
            raise ValueError("existing assignment index contains an unknown report version")
        return existing
    chunks: list[Chunk] = []
    for report in ordered:
        document_id = _document_id(report)
        chunks.extend(chunk_text(report.text, document_id, encoder.tokenizer, **config))
    if not chunks:
        raise ValueError("assignment reports produced no indexable chunks")
    vectors = encoder.encode_texts([chunk.text for chunk in chunks])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Stage both files first, then publish under a filesystem lock. Existing
    # snapshots are immutable: an identical build is idempotent; any change
    # requires a new job-owned path.
    index_fd, staged_index_name = tempfile.mkstemp(prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent)
    os.close(index_fd)
    staged_index = Path(staged_index_name)
    save_source_cache(staged_index, chunks, vectors, source_manifest_hash=source_hash,
                      encoder_revision=str(encoder.revision), config=config)
    cache_hash = _sha256(staged_index)
    core = {"schema_version": 1, "scope": _scope_record(scope), "sources": sources,
            "encoder_revision": str(encoder.revision), "chunking_config": config,
            "source_manifest_hash": source_hash, "index_sha256": cache_hash}
    index_version = hashlib.sha256(_canonical(core)).hexdigest()
    manifest = {**core, "index_version": index_version}
    fd, temporary = tempfile.mkstemp(prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if index_path.exists() or manifest_path.exists():
            raise ValueError("assignment index snapshot path is immutable; use a new path")
        os.replace(staged_index, index_path)
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
        if staged_index.exists():
            staged_index.unlink()
    return manifest


def load_verified_assignment_index(
    index_path: str | Path, manifest_path: str | Path, *, expected_scope: AssignmentScope,
    expected_index_version: str, encoder: Any, expected_chunking_config: dict[str, Any],
    expected_reports: Sequence[ReportVersion] | None = None,
) -> AssignmentIndex:
    """Verify scope, pinned version, source set (when supplied), encoder and cache."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("assignment index manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("assignment index manifest schema is unsupported")
    if manifest.get("scope") != _scope_record(expected_scope):
        raise ValueError("assignment index scope does not match expected assignment scope")
    if manifest.get("encoder_revision") != str(encoder.revision):
        raise ValueError("assignment index encoder revision does not match scorer encoder")
    if manifest.get("chunking_config") != expected_chunking_config:
        raise ValueError("assignment index chunking configuration does not match scorer")
    if not isinstance(expected_index_version, str) or manifest.get("index_version") != expected_index_version:
        raise ValueError("assignment index version does not match trusted expected version")
    core = {key: manifest.get(key) for key in ("schema_version", "scope", "sources", "encoder_revision",
                                               "chunking_config", "source_manifest_hash", "index_sha256")}
    if hashlib.sha256(_canonical(core)).hexdigest() != expected_index_version:
        raise ValueError("assignment index manifest version checksum is invalid")
    if _sha256(index_path) != manifest.get("index_sha256"):
        raise ValueError("assignment index cache checksum does not match manifest")
    if expected_reports is not None:
        ordered = _validate_reports(expected_reports, expected_scope)
        _, sources = _source_manifest(ordered, expected_scope, str(encoder.revision), expected_chunking_config)
        if sources != manifest.get("sources"):
            raise ValueError("assignment index source versions or content hashes do not match expected reports")
    chunks, vectors = load_source_cache(index_path, source_manifest_hash=manifest["source_manifest_hash"],
                                        encoder_revision=str(encoder.revision), config=expected_chunking_config)
    by_document = { _document_id(report): {"report_id": report.report_id, "version_id": report.version_id}
                    for report in (expected_reports or ()) }
    if expected_reports is None:
        # Recover only report IDs from the manifest; index chunks remain opaque.
        # These IDs are needed to exclude every version of the query report.
        for source in manifest["sources"]:
            matching = hashlib.sha256(_canonical((expected_scope.institution_id, expected_scope.course_id,
                expected_scope.assignment_id, source["report_id"], source["version_id"]))).hexdigest()
            by_document[f"platform-report:{matching}"] = {"report_id": source["report_id"], "version_id": source["version_id"]}
    if any(chunk.document_id not in by_document for chunk in chunks):
        raise ValueError("assignment index contains a chunk absent from its source manifest")
    return AssignmentIndex(expected_scope, expected_index_version, manifest, tuple(chunks), vectors, by_document)


def score_assignment_reports(
    reports: Sequence[ReportVersion], *, expected_scope: AssignmentScope, expected_index_version: str,
    index_path: str | Path, manifest_path: str | Path, model: Any, bundle: dict[str, Any],
    feature_pipeline: FeaturePipeline, encoder: Any, top_k: int | None = None,
    below_threshold_limit: int = 3, source_reports: Sequence[ReportVersion] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score all reports against the same assignment snapshot, excluding all own versions."""
    ordered = _validate_reports(reports, expected_scope)
    if below_threshold_limit < 0:
        raise ValueError("below_threshold_limit cannot be negative")
    lineage = bundle.get("lineage") or {}
    if not isinstance(lineage, dict):
        raise ValueError("registered model lineage metadata is invalid")
    platform_scope = lineage.get("platform_scope")
    if platform_scope is not None and platform_scope != _scope_record(expected_scope):
        raise ValueError("registered model platform scope does not match expected assignment scope")
    if bundle.get("chunking_config") is None:
        raise ValueError("registered model has no chunking configuration")
    all_sources = _combine_reports(ordered, source_reports, expected_scope)
    index = load_verified_assignment_index(index_path, manifest_path, expected_scope=expected_scope,
        expected_index_version=expected_index_version, encoder=encoder,
        expected_chunking_config=bundle["chunking_config"], expected_reports=all_sources)
    frozen_top_k = bundle.get("retrieval_top_k")
    if isinstance(frozen_top_k, bool) or not isinstance(frozen_top_k, int) or frozen_top_k < 1:
        raise ValueError("deployment bundle has no valid frozen retrieval_top_k")
    if top_k is not None and top_k != frozen_top_k:
        raise ValueError("requested top_k does not match frozen model configuration")
    threshold = float(bundle["threshold"])
    outputs: list[dict[str, Any]] = []
    embed_seconds = 0.0
    candidate_count = 0
    all_scores: list[float] = []
    for report in ordered:
        own_doc_ids = {_document_id(item) for item in all_sources if item.report_id == report.report_id}
        query_doc_id = _document_id(report)
        query_chunks = chunk_text(report.text, query_doc_id, encoder.tokenizer,
                                  **bundle["chunking_config"])
        started = time.perf_counter()
        query_vectors = encoder.encode_texts([chunk.text for chunk in query_chunks])
        embed_seconds += time.perf_counter() - started
        source_rows = [i for i, chunk in enumerate(index.chunks) if chunk.document_id not in own_doc_ids]
        source_chunks = [index.chunks[i] for i in source_rows]
        source_vectors = index.vectors[source_rows]
        candidates = retrieve(query_chunks, source_chunks, query_vectors, source_vectors, top_k=frozen_top_k)
        candidate_count += len(candidates)
        probabilities = positive_class_probability(model, feature_pipeline.transform_pairs(candidates)) if candidates else []
        all_scores.extend(float(item) for item in probabilities)
        matches: list[dict[str, Any]] = []
        below: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for candidate, probability in zip(candidates, probabilities):
            src = index.report_documents[candidate.source_chunk.document_id]
            key = (src["report_id"], src["version_id"], candidate.query_chunk.start_char,
                   candidate.query_chunk.end_char, candidate.source_chunk.start_char,
                   candidate.source_chunk.end_char)
            if key in seen:
                continue
            seen.add(key)
            record = {"source_report_id": src["report_id"], "source_version_id": src["version_id"],
                "query_start": candidate.query_chunk.start_char, "query_end": candidate.query_chunk.end_char,
                "query_excerpt": report.text[candidate.query_chunk.start_char:candidate.query_chunk.end_char],
                "source_start": candidate.source_chunk.start_char, "source_end": candidate.source_chunk.end_char,
                "source_excerpt": candidate.source_chunk.text, "review_score": float(probability),
                "minilm_cosine": float(candidate.cosine)}
            record["evidence_id"] = hashlib.sha256(_canonical({"scope": _scope_record(expected_scope),
                "query_report_id": report.report_id, "query_version_id": report.version_id,
                "source_report_id": src["report_id"], "source_version_id": src["version_id"],
                "query_start": record["query_start"], "query_end": record["query_end"],
                "source_start": record["source_start"], "source_end": record["source_end"]})).hexdigest()
            record["display_group_id"] = hashlib.sha256(_canonical({"scope": _scope_record(expected_scope),
                "query_report_id": report.report_id, "query_version_id": report.version_id,
                "query_start": record["query_start"], "query_end": record["query_end"],
                "source_excerpt": " ".join(record["source_excerpt"].casefold().split())})).hexdigest()
            if float(probability) >= threshold:
                matches.append(record)
            elif len(below) < below_threshold_limit:
                below.append(record)
        matches.sort(key=lambda row: (-row["review_score"], row["source_report_id"], row["query_start"], row["source_start"]))
        display_groups: dict[str, list[dict[str, Any]]] = {}
        for evidence in matches:
            display_groups.setdefault(evidence["display_group_id"], []).append(evidence)
        display_flags = []
        for group_id, evidence_rows in sorted(display_groups.items()):
            representative = dict(evidence_rows[0])
            representative["evidence_ids"] = [row["evidence_id"] for row in evidence_rows]
            representative["evidence_count"] = len(evidence_rows)
            display_flags.append(representative)
        outputs.append({"report_id": report.report_id, "version_id": report.version_id,
            "institution_id": expected_scope.institution_id, "course_id": expected_scope.course_id,
            "assignment_id": expected_scope.assignment_id, "model_name": bundle["model_name"],
            "model_version": str(bundle["model_version"]), "model_uri": bundle["model_uri"],
            "threshold": threshold, "index_version": index.index_version,
            "flag_count": len(display_flags), "flags": display_flags, "evidence": matches,
            "below_threshold_sample": below})
    return outputs, {"scope": _scope_record(expected_scope), "index_version": index.index_version,
        "model_name": bundle["model_name"], "model_version": str(bundle["model_version"]),
        "report_version_count": len(ordered), "candidate_count": candidate_count,
        "embedding_seconds": embed_seconds, "scored_candidate_count": len(all_scores),
        "flag_count": sum(len(output["flags"]) for output in outputs),
        "score_min": min(all_scores) if all_scores else None,
        "score_max": max(all_scores) if all_scores else None}
