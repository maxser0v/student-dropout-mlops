"""Review flagged passage pairs from a verified immutable MLflow release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any, Iterable, Sequence

from .features import FeaturePipeline
from .retrieval import (
    DEFAULT_REVISION,
    FixtureEncoder,
    MiniLMEncoder,
    chunk_text,
    load_source_cache,
    retrieve,
)
from .training import load_logged_classifier, positive_class_probability


FIXTURE_MODEL_NAME = "passage-reuse-fixture"
TARGET_VERSION = "passage-reuse-pair-classifier-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_artifact_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"deployment bundle is missing {name}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"deployment bundle has an unsafe {name}")
    return str(parsed)


def _verify_sha(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise ValueError(f"{label} artifact checksum does not match deployment bundle")


def _download_run_artifact(run_id: str, artifact_path: str, *, downloader: Any | None = None) -> Path:
    if downloader is None:
        import mlflow

        downloader = mlflow.artifacts.download_artifacts
    return Path(downloader(run_id=run_id, artifact_path=artifact_path)).resolve()


def load_verified_release(
    client: Any,
    model_name: str,
    version: str,
    *,
    fixture: bool = False,
    encoder_dir: str | Path = "/app/models/minilm",
    source_manifest_path: str | Path | None = None,
    downloader: Any | None = None,
    model_loader: Any | None = None,
    registry_loader: Any | None = None,
) -> tuple[Any, dict[str, Any], FeaturePipeline, list[Any], Any, Any]:
    """Resolve one version, validate its gate/artifacts, and load scoring assets.

    The return tuple is `(model, bundle, feature_pipeline, source_chunks,
    source_vectors, encoder)`. Alias resolution belongs to the caller and is
    intentionally not performed here.
    """
    if not version.isdecimal() or int(version) < 1:
        raise ValueError("release loading requires one positive concrete version")
    if fixture != (model_name == FIXTURE_MODEL_NAME):
        raise ValueError("fixture releases require --fixture and the passage-reuse-fixture registry namespace")
    if registry_loader is None:
        from dropout.registry import version_bundle

        registry_loader = version_bundle
    registered, bundle = registry_loader(client, model_name, version)
    checks = {
        "bundle_version": "1.0",
        "model_version": str(version),
        "model_name": model_name,
        "model_uri": f"models:/{model_name}/{version}",
        "mlflow_run_id": str(registered.run_id),
        "gate_passed": True,
        "target_version": TARGET_VERSION,
        "feature_schema_version": "1.0",
    }
    for field, expected in checks.items():
        if bundle.get(field) != expected:
            raise ValueError(f"deployment bundle field {field!r} does not match the verified registry version")
    from .features import FEATURE_NAMES

    if tuple(bundle.get("feature_columns", ())) != FEATURE_NAMES:
        raise ValueError("deployment bundle feature order does not match the scorer schema")
    threshold = bundle.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("deployment bundle threshold must be finite and in [0, 1]")

    run_id = str(registered.run_id)
    gate_path = _download_run_artifact(run_id, "gate_report.json", downloader=downloader)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    candidate = gate.get("candidate") or {}
    if (gate.get("passed") is not True or gate.get("decision") != "passed"
            or str(gate.get("mlflow_run_id")) != run_id
            or candidate.get("target_version") != bundle["target_version"]
            or candidate.get("feature_schema_version") != bundle["feature_schema_version"]):
        raise ValueError("gate report is not a passing report bound to the registered release")
    approved_threshold = candidate.get("threshold")
    if (isinstance(approved_threshold, bool) or not isinstance(approved_threshold, (int, float))
            or float(approved_threshold) != float(threshold)):
        raise ValueError("deployment threshold does not match gate-approved threshold")
    choice = candidate.get("threshold_choice")
    if isinstance(choice, dict) and choice.get("threshold") != approved_threshold:
        raise ValueError("gate threshold choice does not match its approved threshold")

    index_artifact = _safe_artifact_path(bundle.get("source_cache_artifact"), "source_cache_artifact")
    features_artifact = _safe_artifact_path(bundle.get("features_artifact"), "features_artifact")
    chunking_artifact = _safe_artifact_path(bundle.get("chunking_artifact"), "chunking_artifact")
    encoder_artifact = _safe_artifact_path(bundle.get("encoder_manifest_artifact"), "encoder_manifest_artifact")
    index_path = _download_run_artifact(run_id, index_artifact, downloader=downloader)
    feature_path = _download_run_artifact(run_id, features_artifact, downloader=downloader)
    chunking_path = _download_run_artifact(run_id, chunking_artifact, downloader=downloader)
    encoder_manifest_path = _download_run_artifact(run_id, encoder_artifact, downloader=downloader)
    _verify_sha(index_path, bundle.get("source_index_sha256"), "source index")
    _verify_sha(feature_path, bundle.get("features_sha256"), "fitted feature pipeline")
    _verify_sha(chunking_path, bundle.get("chunking_sha256"), "chunking configuration")
    _verify_sha(encoder_manifest_path, bundle.get("encoder_manifest_sha256"), "encoder manifest")

    chunking = json.loads(chunking_path.read_text(encoding="utf-8"))
    expected_chunking = bundle.get("chunking_config")
    if not isinstance(expected_chunking, dict) or chunking != expected_chunking:
        raise ValueError("chunking artifact does not match deployment bundle")
    source_manifest_hash = bundle.get("source_manifest_hash")
    manifest_artifact = _safe_artifact_path(bundle.get("source_manifest_artifact"), "source_manifest_artifact")
    manifest_path = _download_run_artifact(run_id, manifest_artifact, downloader=downloader)
    if _sha256(manifest_path) != source_manifest_hash:
        raise ValueError("frozen source manifest artifact checksum does not match deployment bundle")
    if source_manifest_path is not None and _sha256(source_manifest_path) != source_manifest_hash:
        raise ValueError("local source manifest checksum does not match deployment bundle")
    if not isinstance(source_manifest_hash, str) or len(source_manifest_hash) != 64:
        raise ValueError("deployment bundle has no source manifest SHA-256")
    encoder_manifest = json.loads(encoder_manifest_path.read_text(encoding="utf-8"))

    encoder_revision = bundle.get("encoder_revision")
    if fixture:
        encoder = FixtureEncoder()
        if encoder_revision != encoder.revision:
            raise ValueError("fixture encoder revision does not match the frozen release")
        if (not isinstance(encoder_manifest, dict)
                or encoder_manifest.get("kind", encoder_manifest.get("mode")) != "fixture"
                or encoder_manifest.get("revision") != encoder.revision):
            raise ValueError("fixture encoder provenance is missing or incompatible")
    else:
        if encoder_revision != DEFAULT_REVISION:
            raise ValueError("deployment bundle encoder revision is not the pinned MiniLM revision")
        encoder = MiniLMEncoder(encoder_dir, encoder_revision)
        provenance = bundle.get("encoder_provenance")
        manifest_fingerprint = hashlib.sha256(
            json.dumps(encoder.snapshot_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if encoder_manifest != encoder.snapshot_manifest:
            raise ValueError("staged encoder manifest artifact does not match the verified local snapshot")
        if provenance is not None and provenance != encoder.snapshot_manifest and not (
            isinstance(provenance, dict) and provenance.get("manifest_sha256") == manifest_fingerprint
        ):
            raise ValueError("staged encoder manifest does not match the frozen deployment bundle")

    source_chunks, source_vectors = load_source_cache(
        index_path,
        source_manifest_hash=source_manifest_hash,
        encoder_revision=encoder_revision,
        config=expected_chunking,
    )
    if not source_chunks:
        raise ValueError("verified source index contains no chunks")
    feature_pipeline = FeaturePipeline.load(feature_path)
    if tuple(bundle.get("feature_columns", ())) != tuple(getattr(feature_pipeline, "feature_names", ()) or FEATURE_NAMES):
        raise ValueError("fitted preprocessing feature schema does not match deployment bundle")

    if model_loader is None:
        model_loader = load_logged_classifier
    model = model_loader(bundle["model_uri"])
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("registered XGBoost model does not provide predict_proba")
    if 1 not in list(getattr(model, "classes_", [])):
        raise ValueError("registered model does not expose positive class 1")
    return model, bundle, feature_pipeline, source_chunks, source_vectors, encoder


def parse_submissions(lines: Iterable[str]) -> list[dict[str, str]]:
    submissions: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"input JSONL line {line_no} is invalid: {exc}") from exc
        if not isinstance(record, dict) or set(record) != {"submission_id", "text"}:
            raise ValueError(f"input JSONL line {line_no} must contain only submission_id and text")
        submission_id, text = record["submission_id"], record["text"]
        if not isinstance(submission_id, str) or not submission_id.strip() or not isinstance(text, str):
            raise ValueError(f"input JSONL line {line_no} has invalid submission_id or text")
        if submission_id in seen:
            raise ValueError(f"duplicate submission_id in batch: {submission_id}")
        seen.add(submission_id)
        submissions.append({"submission_id": submission_id, "text": text})
    if not submissions:
        raise ValueError("input JSONL contains no submissions")
    return submissions


def score_submissions(
    submissions: Sequence[dict[str, str]],
    model: Any,
    bundle: dict[str, Any],
    feature_pipeline: FeaturePipeline,
    source_chunks: Sequence[Any],
    source_vectors: Any,
    encoder: Any,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Run the shared retrieval/features/model path and produce review records."""
    records, _ = score_submissions_with_metrics(
        submissions, model, bundle, feature_pipeline, source_chunks, source_vectors,
        encoder, top_k=top_k,
    )
    return records


def score_submissions_with_metrics(
    submissions: Sequence[dict[str, str]],
    model: Any,
    bundle: dict[str, Any],
    feature_pipeline: FeaturePipeline,
    source_chunks: Sequence[Any],
    source_vectors: Any,
    encoder: Any,
    *,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score a batch once and return excerpt-free operational metrics."""
    frozen_top_k = bundle.get("retrieval_top_k")
    if isinstance(frozen_top_k, bool) or not isinstance(frozen_top_k, int) or frozen_top_k < 1:
        raise ValueError("deployment bundle has no valid frozen retrieval_top_k")
    if top_k is not None and top_k != frozen_top_k:
        raise ValueError("requested top_k does not match the frozen release configuration")
    config = bundle["chunking_config"]
    threshold = float(bundle["threshold"])
    outputs: list[dict[str, Any]] = []
    query_chunk_count = 0
    candidate_count = 0
    embedding_seconds = 0.0
    candidate_scores: list[float] = []
    for submission in submissions:
        query_chunks = chunk_text(submission["text"], submission["submission_id"], encoder.tokenizer, **config)
        query_chunk_count += len(query_chunks)
        embedding_started = time.perf_counter()
        query_vectors = encoder.encode_texts([chunk.text for chunk in query_chunks])
        embedding_seconds += time.perf_counter() - embedding_started
        candidates = retrieve(query_chunks, source_chunks, query_vectors, source_vectors, top_k=frozen_top_k)
        candidate_count += len(candidates)
        if candidates:
            matrix = feature_pipeline.transform_pairs(candidates)
            probabilities = positive_class_probability(model, matrix)
        else:
            probabilities = []
        candidate_scores.extend(float(probability) for probability in probabilities)
        flags: list[dict[str, Any]] = []
        for candidate, probability in zip(candidates, probabilities):
            if float(probability) < threshold:
                continue
            query, source = candidate.query_chunk, candidate.source_chunk
            flags.append({
                "source_id": source.document_id,
                "submission_start": query.start_char,
                "submission_end": query.end_char,
                "submission_excerpt": submission["text"][query.start_char:query.end_char],
                "source_start": source.start_char,
                "source_end": source.end_char,
                "source_excerpt": source.text,
                "review_score": float(probability),
                "minilm_cosine": float(candidate.cosine),
            })
        flags.sort(key=lambda flag: (-flag["review_score"], flag["source_id"], flag["submission_start"], flag["source_start"]))
        # Duplicate identical windows can arise when a source has repeated text;
        # retain one strongest evidence row for the same exact span pair.
        deduplicated: list[dict[str, Any]] = []
        fingerprints: set[tuple[Any, ...]] = set()
        for flag in flags:
            fingerprint = (flag["source_id"], flag["submission_start"], flag["submission_end"], flag["source_start"], flag["source_end"])
            if fingerprint not in fingerprints:
                fingerprints.add(fingerprint)
                deduplicated.append(flag)
        outputs.append({
            "submission_id": submission["submission_id"],
            "model_name": bundle["model_name"],
            "model_version": str(bundle["model_version"]),
            "model_uri": bundle["model_uri"],
            "threshold": threshold,
            "feature_schema_version": bundle["feature_schema_version"],
            "source_index_version": bundle["source_manifest_hash"],
            "source_index_sha256": bundle["source_index_sha256"],
            "flag_count": len(deduplicated),
            "flags": deduplicated,
        })
    alert_count = sum(int(record["flag_count"]) for record in outputs)
    flagged_submission_count = sum(int(record["flag_count"] > 0) for record in outputs)
    score_count = len(candidate_scores)
    metrics = {
        "model_name": bundle["model_name"],
        "model_version": str(bundle["model_version"]),
        "threshold": threshold,
        "source_index_version": bundle["source_manifest_hash"],
        "submission_count": len(submissions),
        "query_chunk_count": query_chunk_count,
        "candidate_count": candidate_count,
        "embedding_seconds": embedding_seconds,
        "score_min": min(candidate_scores) if score_count else None,
        "score_max": max(candidate_scores) if score_count else None,
        "score_mean": sum(candidate_scores) / score_count if score_count else None,
        "scored_candidate_count": score_count,
        "flagged_submission_count": flagged_submission_count,
        "flagged_submission_rate": flagged_submission_count / len(submissions) if submissions else 0.0,
        "alert_count": alert_count,
        "alerts_per_submission": alert_count / len(submissions) if submissions else 0.0,
    }
    return outputs, metrics


def write_jsonl_atomically(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Serialize complete output before atomically replacing the destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target.resolve()


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL with submission_id and text")
    parser.add_argument("--output", required=True, help="atomic JSONL review output")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument("--model-name", default="passage-reuse")
    parser.add_argument("--version", default="champion", help="champion alias or concrete positive version")
    parser.add_argument("--encoder-dir", default="/app/models/minilm")
    parser.add_argument("--source-manifest", help="optional local manifest cross-check; the release manifest is downloaded and verified")
    parser.add_argument("--fixture", action="store_true", help="explicitly score from the fixture-only registry namespace")
    parser.add_argument("--top-k", type=int, help="optional assertion; must equal the frozen release retrieval_top_k")
    args = parser.parse_args()

    if args.fixture and args.model_name != FIXTURE_MODEL_NAME:
        parser.error(f"--fixture requires --model-name {FIXTURE_MODEL_NAME}")
    if not args.fixture and args.model_name == FIXTURE_MODEL_NAME:
        parser.error("fixture registry namespace requires --fixture")
    if args.version == "champion":
        from dropout.registry import configure_client

        client = configure_client(args.tracking_uri)
        # Resolve exactly once; every subsequent artifact/model request uses this version.
        registered = client.get_model_version_by_alias(args.model_name, "champion")
        concrete_version = str(registered.version)
    else:
        concrete_version = args.version
        from dropout.registry import configure_client

        client = configure_client(args.tracking_uri)
    model, bundle, pipeline, source_chunks, source_vectors, encoder = load_verified_release(
        client, args.model_name, concrete_version, fixture=args.fixture,
        encoder_dir=args.encoder_dir, source_manifest_path=args.source_manifest,
    )
    with Path(args.input).open(encoding="utf-8") as stream:
        submissions = parse_submissions(stream)
    output, metrics = score_submissions_with_metrics(
        submissions, model, bundle, pipeline, source_chunks,
        source_vectors, encoder, top_k=args.top_k,
    )
    output_path = write_jsonl_atomically(output, args.output)
    print(json.dumps({"status": "scored", "output_path": str(output_path), **metrics},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
