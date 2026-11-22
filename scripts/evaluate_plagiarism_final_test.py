"""Run the frozen passage-reuse model once against the held-out PAN test split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import mlflow
import numpy as np
import yaml
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.candidates import prepare_split_candidates, prepare_tfidf_candidates
from plagiarism.data import filter_corpus, load_corpus, load_pan
from plagiarism.evaluation import merge_alerts, retrieval_link_recall
from plagiarism.features import FeaturePipeline
from plagiarism.retrieval import FixtureEncoder, MiniLMEncoder
from plagiarism.training import load_logged_classifier, positive_class_probability


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_corpus(path: Path):
    if path.is_dir():
        json_files = [x for x in (path / "pan13.json", path / "fixture.json", path / "manifest.json") if x.is_file()]
        if json_files:
            corpus = load_corpus(json_files[0])
        else:
            archives = sorted(path.glob("*.zip"))
            if not archives:
                raise ValueError(f"no held-out corpus manifest or archive under {path}")
            corpus = load_pan(archives[0])
    elif path.suffix.lower() == ".json":
        corpus = load_corpus(path)
    else:
        corpus = load_pan(path)
    if "test" not in corpus.splits or not corpus.splits["test"]:
        raise ValueError("corpus has no explicit held-out test split")
    return corpus


def _source_file(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("pan13.json", "fixture.json", "manifest.json"):
        if (path / name).is_file():
            return path / name
    archives = sorted(path.glob("*.zip"))
    if not archives:
        raise ValueError(f"cannot identify corpus source file under {path}")
    return archives[0]


def evaluate_frozen_test(labels: list[int], probabilities: list[float], candidates,
                          annotations, threshold: float, eligible_submissions: list[str]) -> dict[str, Any]:
    """A small pure helper ensures test scoring cannot tune its threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("frozen XGBoost threshold must be in [0, 1]")
    return merge_alerts(candidates, probabilities, threshold, annotations,
                        eligible_submissions=eligible_submissions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--model-name", default="passage-reuse")
    parser.add_argument("--version", required=True, help="Concrete registered version, never an alias")
    parser.add_argument("--config", default="config/plagiarism.yaml")
    parser.add_argument("--encoder-dir", default=None)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--freeze-record", default="reports/plagiarism/frozen_final_model.json")
    parser.add_argument("--attempt-record", default="reports/plagiarism/final_test_attempt.json")
    parser.add_argument("--output", default="reports/plagiarism/final_test_metrics.json")
    args = parser.parse_args()
    freeze_path, attempt_path, output = (Path(x).expanduser().resolve()
                                         for x in (args.freeze_record, args.attempt_record, args.output))
    if len({freeze_path, attempt_path, output}) != 3 or any(x.exists() for x in (freeze_path, attempt_path, output)):
        raise SystemExit("freeze, attempt, and output paths must be distinct and unused; final test is one-time")
    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if (args.fixture and args.model_name != config["data"]["fixture_model_name"]):
        raise SystemExit("fixture final evaluation requires the passage-reuse-fixture namespace")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    version = client.get_model_version(args.model_name, str(args.version))
    if (version.tags or {}).get("gate_passed", "").lower() != "true":
        raise SystemExit("selected model version did not pass its release gate")
    run_id = version.run_id
    bundle_file = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="deployment_bundle.json"))
    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    gate_file = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="gate_report.json"))
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    if not gate.get("passed") or gate.get("decision") != "passed" or gate.get("mlflow_run_id") != run_id:
        raise SystemExit("registry version has no matching passing gate report")
    if (bundle.get("model_name") != args.model_name or str(bundle.get("model_version")) != str(args.version)
            or bundle.get("model_uri") != f"models:/{args.model_name}/{args.version}"
            or bundle.get("mlflow_run_id") != run_id or not bundle.get("gate_passed")):
        raise SystemExit("registered model and deployment bundle do not match")
    if bundle.get("source_manifest_artifact"):
        source_artifact = Path(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=bundle["source_manifest_artifact"]))
        if _sha(source_artifact) != bundle.get("source_manifest_hash"):
            raise SystemExit("registered source manifest does not match its frozen checksum")
    if hashlib.sha256(config_bytes).hexdigest() != bundle.get("config_sha256"):
        raise SystemExit("config changed since candidate registration")
    if not bundle.get("validation_metrics", {}).get("threshold_choices"):
        raise SystemExit("deployment bundle is missing frozen validation thresholds")
    thresholds = bundle["validation_metrics"]["threshold_choices"]
    frozen = {
        "status": "frozen", "model_name": args.model_name, "model_version": str(args.version),
        "model_uri": bundle["model_uri"], "mlflow_run_id": run_id,
        "config_sha256": bundle["config_sha256"], "threshold": bundle["threshold"],
        "threshold_choices": thresholds, "source_manifest_hash": bundle["source_manifest_hash"],
        "target_version": bundle["target_version"],
        "feature_schema_version": bundle["feature_schema_version"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluation_status": "not_started",
    }
    _create_once(freeze_path, frozen)
    _create_once(attempt_path, {"status": "started", "model_version": str(args.version),
                                "model_uri": bundle["model_uri"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
                                "freeze_record_sha256": _sha(freeze_path)})
    # Held-out labels are loaded only after model version and thresholds freeze.
    corpus = _read_corpus(Path(args.data_dir).expanduser().resolve())
    corpus_source_sha = _sha(_source_file(Path(args.data_dir).expanduser().resolve()))
    if corpus_source_sha != bundle.get("lineage", {}).get("training_corpus_sha256"):
        raise SystemExit("test corpus input checksum differs from the frozen training run")
    test_corpus = filter_corpus(corpus, include_splits=("test",))
    encoder = FixtureEncoder() if args.fixture else MiniLMEncoder(args.encoder_dir or "", bundle["encoder_revision"])
    feature_file = Path(mlflow.artifacts.download_artifacts(run_id=run_id,
                                   artifact_path=bundle["features_artifact"]))
    if _sha(feature_file) != bundle["features_sha256"]:
        raise SystemExit("fitted feature transforms failed checksum verification")
    pipeline = FeaturePipeline.load(feature_file)
    model = load_logged_classifier(bundle["model_uri"])
    settings = config["chunking"]
    rows = prepare_split_candidates(test_corpus, encoder, split="test",
                                    top_k=int(bundle["retrieval_top_k"]),
                                    target_tokens=int(bundle["chunking_config"]["target_tokens"]),
                                    overlap_tokens=int(bundle["chunking_config"]["overlap_tokens"]),
                                    max_tokens=int(bundle["chunking_config"]["max_tokens"]),
                                    source_manifest_hash=bundle["source_manifest_hash"])[0]
    tfidf_rows = prepare_tfidf_candidates(test_corpus, encoder, pipeline.vectorizer,
                                          split="test", top_k=int(bundle["retrieval_top_k"]),
                                          target_tokens=int(bundle["chunking_config"]["target_tokens"]),
                                          overlap_tokens=int(bundle["chunking_config"]["overlap_tokens"]),
                                          max_tokens=int(bundle["chunking_config"]["max_tokens"]))
    scores = positive_class_probability(model, pipeline.transform_pairs(rows))
    eligible = [doc.id for doc in test_corpus.documents.values() if doc.role == "submission"]
    annotations = list(test_corpus.annotations)
    evaluation = {}
    for method, candidates, values in (
        ("tfidf_cosine", tfidf_rows, [r["tfidf_cosine"] for r in tfidf_rows]),
        ("minilm_cosine", rows, [r["cosine"] for r in rows]),
        ("minilm_xgboost", rows, scores),
    ):
        threshold = float(thresholds[method]["threshold"])
        metrics = merge_alerts(candidates, values, threshold, annotations,
                               eligible_submissions=eligible)
        evaluation[method] = {"threshold": threshold, **{k: v for k, v in metrics.items() if k != "alerts"}}
    link_recall = retrieval_link_recall(rows, annotations,
                                       min_span_coverage=float(settings["min_span_coverage"]))
    evaluation_protocol = test_corpus.lineage.get(
        "evaluation_protocol", "authored-fixture" if args.fixture else "pan13-official-test")
    report = {
        "evaluation_split": "test", "conditional_on_pan_supplied_document_pairs": True,
        "evaluation_protocol": evaluation_protocol, "corpus_lineage": dict(test_corpus.lineage),
        "model_name": args.model_name, "model_version": str(args.version),
        "model_uri": bundle["model_uri"], "mlflow_run_id": run_id,
        "target_version": bundle["target_version"],
        "feature_schema_version": bundle["feature_schema_version"],
        "frozen_thresholds": thresholds,
        "raw_sample_counts": {"documents": len(test_corpus.documents),
                              "submissions": len(eligible),
                              "sources": sum(d.role == "source" for d in test_corpus.documents.values()),
                              "pairs": len(test_corpus.pairs),
                              "annotations": len(annotations),
                              "minilm_candidates": len(rows), "tfidf_candidates": len(tfidf_rows)},
        "retrieval_recall_at_k": link_recall,
        "methods": evaluation,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output, report)
    frozen["test_evaluation_status"] = "completed"
    frozen["test_metrics_sha256"] = _sha(output)
    frozen["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(freeze_path, frozen)
    _atomic_json(attempt_path, {"status": "completed", "model_version": str(args.version),
                                "metrics_sha256": _sha(output), "completed_at_utc": frozen["completed_at_utc"]})
    experiment_name = f"{args.model_name}-final-evaluation"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
    run = client.create_run(experiment_id, run_name=f"final-test-{args.version}", tags={
        "evaluation.split": "test", "model.version": str(args.version),
        "evaluation.protocol": evaluation_protocol,
        "conditional_on_pan_pairs": "true"})
    client.log_artifact(run.info.run_id, str(output))
    client.log_metric(run.info.run_id, "retrieval_recall_at_k", float(link_recall["recall_at_k"] or 0.0))
    client.set_terminated(run.info.run_id, status="FINISHED")
    print(json.dumps({"status": "completed", "model_version": str(args.version),
                      "output": str(output), "raw_sample_counts": report["raw_sample_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
