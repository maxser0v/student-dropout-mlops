"""Trusted backend worker for private assignment scoring and review snapshots.

This module is an integration boundary, not an HTTP service or authorization
layer. The caller must resolve all IDs in an authorized assignment scope and
provide job-owned input and output paths on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from .platform_types import AssignmentScope, ReportVersion


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_payload(path: str | Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("worker input contains a duplicate JSON field")
            result[key] = value
        return result

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                           parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("invalid JSON number")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("worker input is missing or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("worker input must be a JSON object")
    return value


def _exact_keys(value: dict[str, Any], required: set[str], optional: set[str] = frozenset()) -> None:
    if not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("worker input has missing or unsupported fields")


def _scope(raw: Any) -> AssignmentScope:
    if not isinstance(raw, dict) or set(raw) != {"institution_id", "course_id", "assignment_id"}:
        raise ValueError("scope must contain institution_id, course_id, and assignment_id")
    return AssignmentScope(**raw)


def _reports(raw: Any, scope: AssignmentScope, field: str) -> list[ReportVersion]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a nonempty array")
    reports: list[ReportVersion] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"report_id", "version_id", "text"}:
            raise ValueError(f"{field} entries must contain report_id, version_id, and text")
        reports.append(ReportVersion(scope, **item))
    return reports


def _atomic_json(path: str | Path, value: dict[str, Any]) -> str:
    """Atomically publish deterministic result bytes, allowing exact retries."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8") + b"\n"
    digest = _sha256(content)
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError("job result path already contains a different result")
        return digest
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != content:
                raise ValueError("job result path already contains a different result")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return digest


def _load_registry(args: argparse.Namespace) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    from dropout.registry import configure_client
    from .score import load_verified_model

    client = configure_client(args.tracking_uri)
    model, bundle, feature_pipeline, encoder = load_verified_model(
        client, args.model_name, args.version, fixture=args.fixture, encoder_dir=args.encoder_dir,
    )
    return client, model, bundle, feature_pipeline, encoder


def _release_evidence(args: argparse.Namespace, scope: AssignmentScope) -> dict[str, Any]:
    from .score import _download_run_artifact

    _client, _model, bundle, _features, _encoder = _load_registry(args)
    lineage = bundle.get("lineage")
    expected_scope = {"institution_id": scope.institution_id, "course_id": scope.course_id,
                      "assignment_id": scope.assignment_id}
    if not isinstance(lineage, dict) or lineage.get("platform_scope") != expected_scope:
        raise ValueError("candidate model does not have matching assignment scope lineage")
    snapshot_hash = lineage.get("training_corpus_sha256")
    if (not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64
            or any(char not in "0123456789abcdef" for char in snapshot_hash)):
        raise ValueError("candidate model has no valid training corpus checksum")
    gate_path = _download_run_artifact(str(bundle["mlflow_run_id"]), "gate_report.json")
    gate_bytes = gate_path.read_bytes()
    try:
        gate = json.loads(gate_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("verified gate report is invalid") from exc
    candidate = gate.get("candidate") if isinstance(gate, dict) else None
    if (not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("decision") != "passed"
            or str(gate.get("mlflow_run_id")) != str(bundle.get("mlflow_run_id"))
            or gate.get("platform_scope") != expected_scope
            or gate.get("platform_snapshot_sha256") != snapshot_hash
            or not isinstance(candidate, dict)
            or candidate.get("target_version") != bundle.get("target_version")
            or candidate.get("feature_schema_version") != bundle.get("feature_schema_version")
            or candidate.get("threshold") != bundle.get("threshold")):
        raise ValueError("passing gate report is not bound to the verified concrete model")
    gate_hash = _sha256(gate_bytes)
    return {"model_name": bundle["model_name"], "model_version": str(bundle["model_version"]),
            "model_uri": bundle["model_uri"], "model_bundle_sha256": _sha256(_canonical(bundle)),
            "gate_report_artifact": "gate_report.json", "gate_report_sha256": gate_hash,
            "gate_passed": True, "platform_scope": expected_scope,
            "training_corpus_sha256": lineage["training_corpus_sha256"]}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    from .platform_scoring import build_assignment_index, score_assignment_reports

    payload = _read_payload(args.input)
    _exact_keys(payload, {"scope", "reports"}, {"source_reports"})
    scope = _scope(payload["scope"])
    reports = _reports(payload["reports"], scope, "reports")
    source_reports = _reports(payload["source_reports"], scope, "source_reports") if "source_reports" in payload else []
    _client, model, bundle, feature_pipeline, encoder = _load_registry(args)
    expected_scope = {"institution_id": scope.institution_id, "course_id": scope.course_id,
                      "assignment_id": scope.assignment_id}
    # A legacy unscoped seed may be selected by trusted backend policy for
    # scoring, but a scoped model must always match the requested assignment.
    lineage = bundle.get("lineage") or {}
    model_scope = lineage.get("platform_scope") if isinstance(lineage, dict) else None
    if model_scope is not None and model_scope != expected_scope:
        raise ValueError("verified model scope does not match the scoring assignment")
    manifest = build_assignment_index(reports, scope, args.index, args.index_manifest, encoder,
                                      source_reports=source_reports, **bundle["chunking_config"])
    results, metrics = score_assignment_reports(
        reports, expected_scope=scope, expected_index_version=manifest["index_version"],
        index_path=args.index, manifest_path=args.index_manifest, model=model, bundle=bundle,
        feature_pipeline=feature_pipeline, encoder=encoder, source_reports=source_reports,
    )
    # A passage pair can be returned from both query directions. Keep the
    # directional evidence in each report and add one assignment-level row
    # for the unordered report/version/span pair for review display.
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in results:
        for evidence in row["evidence"]:
            if float(evidence["review_score"]) < float(bundle["threshold"]):
                continue
            query = (row["report_id"], row["version_id"], evidence["query_start"], evidence["query_end"])
            source = (evidence["source_report_id"], evidence["source_version_id"],
                      evidence["source_start"], evidence["source_end"])
            key = tuple(sorted((query, source)))
            grouped.setdefault(key, []).append({**evidence, "query_report_id": row["report_id"],
                                                "query_version_id": row["version_id"]})
    display_flags = []
    for key, evidence_rows in sorted(grouped.items()):
        representative = max(evidence_rows, key=lambda item: (item["review_score"], item["evidence_id"]))
        display_flags.append({"report_a": {"report_id": key[0][0], "version_id": key[0][1],
                                             "start": key[0][2], "end": key[0][3]},
                              "report_b": {"report_id": key[1][0], "version_id": key[1][1],
                                             "start": key[1][2], "end": key[1][3]},
                              "representative": representative,
                              "evidence_ids": sorted({item["evidence_id"] for item in evidence_rows}),
                              "evidence_count": len({item["evidence_id"] for item in evidence_rows})})
    # Timings vary between retries and are deliberately excluded from the
    # immutable result, so same-input retries publish byte-identical output.
    metrics.pop("embedding_seconds", None)
    result = {"schema_version": 1, "operation": "score", "scope": expected_scope,
              "model": {"name": bundle["model_name"], "version": str(bundle["model_version"]),
                        "uri": bundle["model_uri"], "bundle_sha256": _sha256(_canonical(bundle))},
              "index": {"version": manifest["index_version"],
                        "manifest_sha256": _sha256(_canonical(manifest))},
              "input_sha256": _sha256(_canonical(payload)), "metrics": metrics,
              "display_flags": display_flags, "reports": results}
    result["result_sha256"] = _sha256(_canonical(result))
    _atomic_json(args.output, result)
    return result


def _resolve_report(ref: Any, by_key: dict[tuple[str, str], ReportVersion]) -> ReportVersion:
    if not isinstance(ref, dict) or set(ref) != {"report_id", "version_id"}:
        raise ValueError("review report references must contain report_id and version_id")
    try:
        return by_key[(ref["report_id"], ref["version_id"])]
    except (KeyError, TypeError) as exc:
        raise ValueError("review references a report version absent from the snapshot input") from exc


def run_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    from .platform_corpus import PairReview, ReviewSpan, build_platform_corpus, write_platform_snapshot

    payload = _read_payload(args.input)
    _exact_keys(payload, {"scope", "reports", "reviews"})
    scope = _scope(payload["scope"])
    reports = _reports(payload["reports"], scope, "reports")
    by_key = {(item.report_id, item.version_id): item for item in reports}
    if len(by_key) != len(reports):
        raise ValueError("snapshot input contains duplicate report versions")
    raw_reviews = payload["reviews"]
    if not isinstance(raw_reviews, list) or not raw_reviews:
        raise ValueError("reviews must be a nonempty array")
    reviews = []
    for item in raw_reviews:
        if not isinstance(item, dict):
            raise ValueError("review entries must be JSON objects")
        _exact_keys(item, {"review_id", "revision", "left", "right", "coverage", "reused", "spans"})
        spans = item["spans"]
        if not isinstance(spans, list):
            raise ValueError("review spans must be an array")
        parsed_spans = []
        for span in spans:
            if not isinstance(span, dict) or set(span) != {"left_start", "left_end", "right_start", "right_end"}:
                raise ValueError("review spans have unsupported fields")
            parsed_spans.append(ReviewSpan(**span))
        reviews.append(PairReview(
            review_id=item["review_id"], revision=item["revision"], scope=scope,
            left=_resolve_report(item["left"], by_key), right=_resolve_report(item["right"], by_key),
            coverage=item["coverage"], reused=item["reused"], spans=tuple(parsed_spans),
        ))
    corpus = build_platform_corpus(
        scope, reports, reviews,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        test_fraction=getattr(args, "test_fraction", 0.0),
        seed=getattr(args, "seed", 903),
    )
    dataset_sha = str(corpus.lineage["dataset_sha256"])
    manifest_sha = write_platform_snapshot(corpus, args.snapshot_dir)
    result = {"schema_version": 1, "operation": "snapshot",
              "scope": {"institution_id": scope.institution_id, "course_id": scope.course_id,
                        "assignment_id": scope.assignment_id},
              "dataset_sha256": dataset_sha, "manifest_sha256": manifest_sha,
              "report_version_count": len(reports), "review_count": len(reviews),
              "input_sha256": _sha256(_canonical(payload))}
    result["result_sha256"] = _sha256(_canonical(result))
    _atomic_json(args.output, result)
    return result


def run_verify_release(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_payload(args.input)
    _exact_keys(payload, {"scope"})
    scope = _scope(payload["scope"])
    result = {"schema_version": 1, "operation": "verify-release",
              "scope": {"institution_id": scope.institution_id, "course_id": scope.course_id,
                        "assignment_id": scope.assignment_id},
              "release": _release_evidence(args, scope)}
    result["input_sha256"] = _sha256(_canonical(payload))
    result["result_sha256"] = _sha256(_canonical(result))
    _atomic_json(args.output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("score", "verify-release"):
        command = sub.add_parser(name)
        command.add_argument("--input", required=True, help="private job input JSON path")
        command.add_argument("--output", required=True, help="private job result JSON path")
        command.add_argument("--tracking-uri", required=True, help="trusted MLflow registry URI")
        command.add_argument("--model-name", required=True, help="trusted concrete registry namespace")
        command.add_argument("--version", required=True, help="one concrete positive registry version")
        command.add_argument("--encoder-dir", default="/app/models/minilm")
        command.add_argument("--fixture", action="store_true")
        if name == "score":
            command.add_argument("--index", required=True, help="job-owned private index path")
            command.add_argument("--index-manifest", required=True, help="job-owned private index manifest path")
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--input", required=True, help="private job input JSON path")
    snapshot.add_argument("--output", required=True, help="private job result JSON path")
    snapshot.add_argument("--snapshot-dir", required=True, help="job-owned private corpus snapshot directory")
    snapshot.add_argument("--validation-fraction", type=float, default=0.2)
    snapshot.add_argument("--test-fraction", type=float, default=0.0)
    snapshot.add_argument("--seed", type=int, default=903)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "score":
            run_score(args)
        elif args.operation == "snapshot":
            run_snapshot(args)
        else:
            run_verify_release(args)
    except Exception as exc:
        # Do not echo arbitrary exception text: lower layers may include
        # source excerpts or local paths. The trusted scheduler gets a stable
        # failure code; detailed diagnostics belong in restricted worker logs.
        print(f"platform worker failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(json.dumps({"status": "completed", "operation": args.operation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
