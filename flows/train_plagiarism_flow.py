"""Reproducible, gated passage-reuse training using Metaflow and MLflow."""

from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from metaflow import FlowSpec, Parameter, card, current, step
import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from plagiarism.candidates import (build_source_index, manifest_sha256,
                                   prepare_split_candidates, prepare_tfidf_candidates)
from plagiarism.data import assign_splits, filter_corpus, load_corpus, load_pan
from dropout.lineage import collect_source_lineage, verify_dvc_directory
from plagiarism.evaluation import (candidate_labels, choose_shared_recall_thresholds,
                                   evaluate_release_gate, merge_alerts,
                                   retrieval_link_recall)
from plagiarism.features import FEATURE_NAMES, FeaturePipeline
from plagiarism.retrieval import (FixtureEncoder, MiniLMEncoder, chunking_config,
                                  source_cache_key)
from plagiarism.training import positive_class_probability, train_classifier


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _config(path: str) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("plagiarism config must be a YAML mapping")
    if tuple(value.get("feature_schema", {}).get("columns", ())) != FEATURE_NAMES:
        raise ValueError("configured feature columns differ from the model feature allowlist")
    return value, hashlib.sha256(raw).hexdigest()


def _load_data(path: Path, validation_fraction: float, seed: int, *, fixture: bool = False):
    if path.is_dir():
        names = ("fixture.json", "manifest.json") if fixture else ("pan13.json", "manifest.json")
        manifest = next((path / name for name in names if (path / name).is_file()), None)
        if manifest is not None:
            corpus = load_corpus(manifest)
        else:
            candidates = sorted(path.glob("*.zip"))
            if not candidates:
                raise ValueError(f"no corpus JSON manifest or PAN zip found under {path}")
            manifest = candidates[0]
            corpus = load_pan(manifest)
    elif path.suffix.lower() == ".json":
        manifest = path
        corpus = load_corpus(path)
    else:
        manifest = None
        corpus = load_pan(path)
    if not corpus.splits:
        corpus = assign_splits(corpus, validation_fraction=validation_fraction, seed=seed)
    if not corpus.splits.get("train") or not corpus.splits.get("validation"):
        raise ValueError("corpus needs non-empty train and validation groups")
    # The training flow does not carry held-out documents, spans, or labels
    # across Metaflow steps. The test manifest remains available for its single
    # separately invoked final evaluation.
    return filter_corpus(corpus, include_splits=("train", "validation")), manifest


def _split_data(corpus, split: str):
    from plagiarism.data import Corpus
    ids = set(corpus.splits[split])
    docs = {key: value for key, value in corpus.documents.items() if key in ids}
    return Corpus(docs, tuple(a for a in corpus.annotations if a.submission_id in ids and a.source_id in ids),
                  tuple(p for p in corpus.pairs if p.split == split), {split: tuple(sorted(ids))}, corpus.lineage)


def _artifact_sha(path: Path) -> str:
    return _sha256(path)


def _failed_run_guard(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except BaseException:
            run_id = getattr(self, "mlflow_run_id", None)
            if run_id:
                try:
                    MlflowClient(tracking_uri=self.tracking_uri).set_terminated(run_id, status="FAILED")
                except Exception:
                    pass
            raise
    return wrapped


class PlagiarismTrainingFlow(FlowSpec):
    data_dir = Parameter("data-dir", default="plagiarism_data", help="DVC-restored PAN or fixture directory")
    config_path = Parameter("config-path", default="config/plagiarism.yaml", help="Experiment and release configuration")
    tracking_uri = Parameter("tracking-uri", default="http://127.0.0.1:5000", help="MLflow tracking and registry URI")
    cache_dir = Parameter("cache-dir", default="artifacts/plagiarism/cache", help="Writable local encoder cache")
    result_path = Parameter("result-path", default=None, help="Optional run summary JSON path")
    encoder_dir = Parameter("encoder-dir", default=None, help="Path to staged frozen MiniLM snapshot")
    fixture = Parameter("fixture", default=False, type=bool, help="Use deterministic fixture encoder")

    @_failed_run_guard
    @step
    def start(self):
        self.flow_run_id = str(current.run_id)
        self.config, self.config_sha256 = _config(self.config_path)
        self.data_path = str(Path(self.data_dir).resolve())
        self.config_path_resolved = str(Path(self.config_path).resolve())
        self.cache_root = str(Path(self.cache_dir).resolve())
        Path(self.cache_root).mkdir(parents=True, exist_ok=True)
        self.report_root = str((Path(self.cache_root).parent / "reports" / self.flow_run_id).resolve())
        Path(self.report_root).mkdir(parents=True, exist_ok=True)
        self.encoder_revision = "fixture-hash-encoder-v1" if self.fixture else str(self.config.get("encoder", {}).get("revision", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"))
        self.corpus, self.manifest_path = _load_data(Path(self.data_path),
                                                     float(self.config["data"]["validation_fraction"]),
                                                     int(self.config["data"]["split_seed"]),
                                                     fixture=bool(self.fixture))
        is_fixture_data = bool(self.corpus.lineage.get("fixture")) or "fixture" in str(self.corpus.lineage.get("dataset", "")).lower()
        if is_fixture_data and not self.fixture:
            raise ValueError("fixture corpus requires explicit --fixture encoder mode")
        if self.fixture and self.config["model_name"] != self.config["data"]["fixture_model_name"]:
            raise ValueError("fixture runs must use the isolated passage-reuse-fixture model namespace")
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment_name = str(self.config.get("experiment_name", self.config["model_name"]))
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
        run = client.create_run(experiment_id, run_name=f"plagiarism-{self.flow_run_id}", tags={
            "metaflow.flow": str(current.flow_name), "metaflow.run_id": self.flow_run_id,
            "model.namespace": self.config["model_name"], "fixture": str(bool(self.fixture)).lower(),
            "evaluation.protocol": self.corpus.lineage.get(
                "evaluation_protocol", "authored-fixture" if self.fixture else "pan13-official-test"),
            "target_version": self.config["target"]["version"],
            "feature_schema_version": self.config["feature_schema"]["version"],
        })
        self.mlflow_run_id = run.info.run_id
        self.next(self.validate)

    @_failed_run_guard
    @step
    def validate(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        self.split_metadata = {name: len(ids) for name, ids in self.corpus.splits.items()}
        self.input_manifest_sha256 = manifest_sha256(self.manifest_path) if self.manifest_path else None
        self.train_corpus = _split_data(self.corpus, "train")
        self.validation_corpus = _split_data(self.corpus, "validation")
        pair_labels = {
            split: {str(label): sum(pair.label == label for pair in corpus.pairs) for label in (0, 1)}
            for split, corpus in (("train", self.train_corpus), ("validation", self.validation_corpus))
        }
        train_validation = set(self.corpus.splits.get("train", ())) | set(self.corpus.splits.get("validation", ()))
        source_records = [{"id": doc.id, "content_hash": doc.content_hash}
                          for doc in sorted(self.corpus.documents.values(), key=lambda item: item.id)
                          if doc.role == "source" and doc.id in train_validation]
        source_manifest_local = Path(self.report_root) / "source_manifest.json"
        _json(source_manifest_local, {"schema_version": 1, "sources": source_records})
        self.source_manifest_artifact_local = str(source_manifest_local)
        self.source_manifest_hash = _sha256(source_manifest_local)
        self.source_manifest_path = "plagiarism_bundle/source_manifest.json"
        self.source_lineage = collect_source_lineage(ROOT)
        if self.fixture:
            self.dvc_verification = {"passed": True, "fixture_exempt": True}
        else:
            data_root = Path(self.data_path)
            if data_root.is_file():
                data_root = data_root.parent
            pointer = data_root.parent / f"{data_root.name}.dvc"
            expected_files = {item.relative_to(data_root).as_posix() for item in data_root.rglob("*")
                              if item.is_file() and not item.is_symlink()}
            self.dvc_verification = verify_dvc_directory(
                pointer, data_root, expected_path=data_root.name, expected_files=expected_files)
            if not self.dvc_verification.get("passed"):
                raise ValueError(f"DVC data verification failed: {self.dvc_verification.get('reason')}")
        self.data_checks = {
            "input_manifest_sha256": self.input_manifest_sha256,
            "corpus_lineage": dict(self.corpus.lineage),
            "source_manifest_sha256": self.source_manifest_hash,
            "source_lineage": self.source_lineage,
            "dvc_verification": self.dvc_verification,
            "document_count": len(self.corpus.documents),
            "annotation_count": len(self.corpus.annotations),
            "pair_count": len(self.corpus.pairs),
            "pair_label_counts": pair_labels,
            "split_document_counts": self.split_metadata,
            "group_leakage_checked": True,
            "test_documents_excluded_from_development": True,
        }
        mlflow.start_run(run_id=self.mlflow_run_id)
        try:
            mlflow.log_params({"config_sha256": self.config_sha256, "source_manifest_sha256": self.source_manifest_hash,
                               "encoder_revision": self.encoder_revision, "seed": self.config["seed"],
                               "train_documents": len(self.train_corpus.documents),
                               "validation_documents": len(self.validation_corpus.documents),
                               "train_pairs": len(self.train_corpus.pairs),
                               "validation_pairs": len(self.validation_corpus.pairs)})
            mlflow.set_tags({"source.manifest.path": self.source_manifest_path,
                             "encoder.revision": self.encoder_revision})
            mlflow.log_dict(self.data_checks, "provenance/data_checks.json")
            mlflow.log_artifact(self.config_path_resolved, artifact_path="provenance")
        finally:
            mlflow.end_run(status="RUNNING")
        for split, counts in pair_labels.items():
            if not all(counts.values()):
                raise ValueError(
                    f"{split} requires judged positive and negative pairs after leakage checks "
                    f"(positive={counts['1']}, negative={counts['0']})"
                )
        self.next(self.prepare)

    @_failed_run_guard
    @step
    def prepare(self):
        settings = self.config["chunking"]
        # The encoder object, especially the PyTorch model, stays local to this
        # step and is never serialized as a Metaflow artifact.
        encoder = FixtureEncoder() if self.fixture else MiniLMEncoder(self.encoder_dir or "", self.encoder_revision)
        self.encoder_manifest = getattr(encoder, "snapshot_manifest", None) or {
            "mode": "fixture", "revision": self.encoder_revision,
            "encoder": "FixtureEncoder", "not_a_benchmark": True}
        index_cfg = chunking_config(target_tokens=int(settings["target_tokens"]),
                                    overlap_tokens=int(settings["overlap_tokens"]),
                                    max_tokens=int(settings["max_tokens"]))
        cache_id = source_cache_key(self.source_manifest_hash, self.encoder_revision, index_cfg)
        cache_path = Path(self.cache_root) / f"source_index-{cache_id}.npz"
        # All non-test source passages are fixed into the deployment index.
        non_test_splits = tuple(name for name in self.corpus.splits if name != "test")
        self.index_chunks, self.index_vectors, self.index_metadata = build_source_index(
            self.corpus, encoder, split_names=non_test_splits,
            cache_path=cache_path, source_manifest_hash=self.source_manifest_hash,
            target_tokens=int(settings["target_tokens"]), overlap_tokens=int(settings["overlap_tokens"]),
            max_tokens=int(settings["max_tokens"]))
        self.index_path = str(cache_path)
        self.train_rows, self.train_retrieval_stats = prepare_split_candidates(
            self.train_corpus, encoder, split="train", top_k=int(settings["top_k"]),
            target_tokens=int(settings["target_tokens"]), overlap_tokens=int(settings["overlap_tokens"]),
            max_tokens=int(settings["max_tokens"]), source_manifest_hash=self.source_manifest_hash,
            source_index=(self.index_chunks, self.index_vectors))
        labels = candidate_labels(self.train_rows, self.train_corpus.annotations,
                                  min_span_coverage=float(settings["min_span_coverage"]))
        keep = labels >= 0
        self.train_rows = [r for r, valid in zip(self.train_rows, keep) if valid]
        self.train_labels = labels[keep].astype(int).tolist()
        if not self.train_labels or set(self.train_labels) != {0, 1}:
            raise ValueError("retrieved training set must contain both clear negative and span-linked positive candidates")
        self.ambiguous_training_candidates = int(np.sum(~keep))
        self.training_texts = [doc.text for doc in self.train_corpus.documents.values()]
        self.feature_pipeline = FeaturePipeline().fit(self.training_texts)
        self.train_features = self.feature_pipeline.transform_pairs(self.train_rows)
        self.validation_rows, self.validation_retrieval_stats = prepare_split_candidates(
            self.validation_corpus, encoder, split="validation", top_k=int(settings["top_k"]),
            target_tokens=int(settings["target_tokens"]), overlap_tokens=int(settings["overlap_tokens"]),
            max_tokens=int(settings["max_tokens"]), source_manifest_hash=self.source_manifest_hash,
            source_index=(self.index_chunks, self.index_vectors))
        self.validation_tfidf_rows = prepare_tfidf_candidates(
            self.validation_corpus, encoder, self.feature_pipeline.vectorizer,
            split="validation", top_k=int(settings["top_k"]),
            target_tokens=int(settings["target_tokens"]), overlap_tokens=int(settings["overlap_tokens"]),
            max_tokens=int(settings["max_tokens"]))
        self.validation_link_recall = retrieval_link_recall(
            self.validation_rows, self.validation_corpus.annotations,
            min_span_coverage=float(settings["min_span_coverage"]))
        self.next(self.train)

    @_failed_run_guard
    @step
    def train(self):
        params = self.config.get("training", {})
        self.model = train_classifier(self.train_features, self.train_labels,
                                      seed=int(self.config["seed"]),
                                      n_estimators=int(params.get("n_estimators", 80)),
                                      max_depth=int(params.get("max_depth", 4)),
                                      learning_rate=float(params.get("learning_rate", 0.08)))
        # Store JSON/state-only transforms rather than an arbitrary-code pickle.
        artifact_dir = Path(self.report_root) / "plagiarism_bundle"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.feature_path = str(self.feature_pipeline.save(artifact_dir / "features.json"))
        shutil.copy2(self.index_path, artifact_dir / "source_index.npz")
        self.index_path = str(artifact_dir / "source_index.npz")
        self.chunking_config = {key: int(self.config["chunking"][key])
                                for key in ("target_tokens", "overlap_tokens", "max_tokens")}
        self.chunking_path = str(_json(artifact_dir / "chunking.json", self.chunking_config))
        encoder_manifest = self.encoder_manifest
        self.encoder_manifest_path = str(_json(artifact_dir / "encoder_manifest.json", encoder_manifest))
        self.artifact_checks = {
            "features_sha256": _sha256(Path(self.feature_path)),
            "chunking_sha256": _sha256(Path(self.chunking_path)),
            "source_index_sha256": _sha256(Path(self.index_path)),
            "encoder_manifest_sha256": _sha256(Path(self.encoder_manifest_path)),
            "source_manifest_sha256": _sha256(Path(self.source_manifest_artifact_local)),
        }
        if self.artifact_checks["source_manifest_sha256"] != self.source_manifest_hash:
            raise ValueError("source manifest artifact checksum changed after validation")
        self.next(self.log_model)

    @_failed_run_guard
    @step
    def log_model(self):
        """Log and reload the XGBoost flavor before applying the release gate."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.start_run(run_id=self.mlflow_run_id)
        try:
            example = self.train_features[: min(5, len(self.train_features))]
            requirements = [f"{package}=={importlib.metadata.version(package)}"
                            for package in ("xgboost", "scikit-learn", "numpy")]
            info = mlflow.xgboost.log_model(self.model, name="model", input_example=example,
                                            pip_requirements=requirements,
                                            metadata={"target_version": self.config["target"]["version"],
                                                      "feature_schema_version": self.config["feature_schema"]["version"],
                                                      "feature_names": list(FEATURE_NAMES)})
            self.logged_model_uri = info.model_uri
            for file_path in (self.feature_path, self.index_path, self.chunking_path,
                              self.encoder_manifest_path, self.source_manifest_artifact_local):
                mlflow.log_artifact(file_path, artifact_path="plagiarism_bundle")
            loaded = mlflow.xgboost.load_model(self.logged_model_uri)
            sample = self.feature_pipeline.transform_pairs(self.validation_rows[:5])
            expected = positive_class_probability(self.model, sample)
            actual = positive_class_probability(loaded, sample)
            if not np.allclose(actual, expected, rtol=0, atol=1e-7):
                raise ValueError("logged XGBoost model failed artifact round-trip prediction")
            self.artifact_checks["logged_xgboost_round_trip"] = True
            mlflow.log_dict(self.artifact_checks, "artifact_checks.json")
            mlflow.log_dict({"source_index": self.index_metadata,
                             "train_retrieval": self.train_retrieval_stats,
                             "validation_retrieval": self.validation_retrieval_stats},
                            "retrieval_timings.json")
            mlflow.log_metrics({
                "cold_index_seconds": (float(self.index_metadata["source_index_seconds"])
                                       if not self.index_metadata["cache_hit"] else 0.0),
                "source_index_load_seconds": float(self.index_metadata["source_index_seconds"]),
                "source_index_cache_hit": float(bool(self.index_metadata["cache_hit"])),
                "train_candidate_count": float(self.train_retrieval_stats["candidate_count"]),
                "validation_candidate_count": float(self.validation_retrieval_stats["candidate_count"]),
                "validation_query_encode_seconds": float(self.validation_retrieval_stats["query_encode_seconds"]),
                "validation_retrieval_seconds": float(self.validation_retrieval_stats["retrieval_seconds"]),
            })
        finally:
            mlflow.end_run(status="RUNNING")
        self.next(self.evaluate)

    @_failed_run_guard
    @step
    def evaluate(self):
        settings = self.config["chunking"]
        threshold_cfg = self.config["threshold_selection"]
        self.validation_labels = candidate_labels(
            self.validation_rows, self.validation_corpus.annotations,
            min_span_coverage=float(settings["min_span_coverage"]))
        cached_score_start = time.perf_counter()
        validation_features = self.feature_pipeline.transform_pairs(self.validation_rows)
        model_scores = positive_class_probability(self.model, validation_features)
        self.validation_cached_batch_seconds = time.perf_counter() - cached_score_start
        score_sets = {
            "tfidf_cosine": (self.validation_tfidf_rows, [row["tfidf_cosine"] for row in self.validation_tfidf_rows]),
            "minilm_cosine": (self.validation_rows, [row["cosine"] for row in self.validation_rows]),
            "minilm_xgboost": (self.validation_rows, model_scores),
        }
        eligible = [doc.id for doc in self.validation_corpus.documents.values() if doc.role == "submission"]
        self.threshold_selection_error = None
        try:
            shared, choices = choose_shared_recall_thresholds(
                self.validation_corpus.annotations, score_sets,
                requested_recall=float(threshold_cfg["requested_shared_recall"]),
                minimum_recall=float(self.config["gates"]["minimum_recall"]),
                eligible_submissions=eligible,
                min_span_coverage=float(settings["min_span_coverage"]))
            self.threshold_choices = choices
            self.threshold = float(choices["minilm_xgboost"]["threshold"])
            self.baseline_metrics = choices["minilm_cosine"]
            self.validation_metrics = choices["minilm_xgboost"]
            self.candidate_metrics = {
                "target_version": self.config["target"]["version"],
                "feature_schema_version": self.config["feature_schema"]["version"],
                "shared_recall_target": shared,
                "recall": self.validation_metrics["recall"],
                "false_alerts": self.validation_metrics["false_alerts"],
                "false_alerts_per_100_submissions": self.validation_metrics["false_alerts_per_100_submissions"],
                "minilm_cosine_recall": self.baseline_metrics["recall"],
                "minilm_cosine_false_alerts": self.baseline_metrics["false_alerts"],
                "threshold": self.threshold,
                "threshold_choices": choices,
                "retrieval_recall_at_10": self.validation_link_recall["recall_at_k"],
                "retrieval_missed_links": self.validation_link_recall["missed_links"],
                "validation_support": len(self.validation_corpus.pairs),
                "validation_candidate_count": len(self.validation_rows),
                "validation_submission_count": len(eligible),
                "ambiguous_training_candidates_excluded": self.ambiguous_training_candidates,
            }
        except ValueError as exc:
            self.threshold_selection_error = str(exc)
            self.threshold_choices = {}
            self.threshold = None
            self.baseline_metrics = {}
            self.validation_metrics = {}
            self.candidate_metrics = {
                "target_version": self.config["target"]["version"],
                "feature_schema_version": self.config["feature_schema"]["version"],
                "shared_recall_target": None, "recall": None, "false_alerts": None,
                "minilm_cosine_recall": None, "minilm_cosine_false_alerts": None,
                "threshold_selection_error": self.threshold_selection_error,
                "retrieval_recall_at_10": self.validation_link_recall["recall_at_k"],
                "retrieval_missed_links": self.validation_link_recall["missed_links"],
            }
        self.next(self.gate)

    @_failed_run_guard
    @step
    def gate(self):
        self.gate_report = evaluate_release_gate(
            self.candidate_metrics, self.config,
            data_checks_passed=True,
            artifact_checks_passed=all(bool(v) for v in self.artifact_checks.values()))
        self.gate_report["mlflow_run_id"] = self.mlflow_run_id
        self.gate_report["metaflow_run_id"] = self.flow_run_id
        self.gate_report["fixture"] = bool(self.fixture)
        report_path = _json(Path(self.report_root) / "gate_report.json", self.gate_report)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.start_run(run_id=self.mlflow_run_id)
        try:
            mlflow.set_tag("release_gate", self.gate_report["decision"])
            mlflow.log_dict(self.gate_report, "gate_report.json")
            mlflow.log_artifact(str(report_path))
            mlflow.log_metric("release_gate_passed", int(self.gate_report["passed"]))
            mlflow.log_metric("validation_cached_batch_scoring_seconds",
                              float(self.validation_cached_batch_seconds))
            if self.threshold is not None:
                mlflow.log_metric("validation_threshold", self.threshold)
                for key in ("recall", "false_alerts", "false_alerts_per_100_submissions", "retrieval_recall_at_10"):
                    value = self.candidate_metrics.get(key)
                    if isinstance(value, (int, float)) and value is not None:
                        mlflow.log_metric("validation_" + key, float(value))
                for method, metrics in self.threshold_choices.items():
                    if not isinstance(metrics, dict):
                        continue
                    for key in ("threshold", "precision", "recall", "false_alerts",
                                "false_alerts_per_100_submissions"):
                        value = metrics.get(key)
                        if isinstance(value, (int, float)) and value is not None:
                            mlflow.log_metric(f"validation_{method}_{key}", float(value))
        finally:
            mlflow.end_run(status="RUNNING")
        self.next(self.register)

    @_failed_run_guard
    @step
    def register(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        self.registered_version = None
        if self.gate_report["passed"]:
            client = MlflowClient(tracking_uri=self.tracking_uri)
            version = mlflow.register_model(self.logged_model_uri, self.config["model_name"], await_registration_for=0)
            self.registered_version = str(version.version)
            index_rel = "plagiarism_bundle/source_index.npz"
            bundle = {
                "bundle_version": "1.0", "model_name": self.config["model_name"],
                "model_version": self.registered_version,
                "model_uri": f"models:/{self.config['model_name']}/{self.registered_version}",
                "mlflow_run_id": self.mlflow_run_id, "metaflow_run_id": self.flow_run_id,
                "threshold": self.threshold, "target_version": self.config["target"]["version"],
                "target_description": self.config["target"]["description"],
                "feature_schema_version": self.config["feature_schema"]["version"],
                "feature_columns": list(FEATURE_NAMES),
                "config_version": self.config["config_version"], "config_sha256": self.config_sha256,
                "train_prevalence": float(np.mean(self.train_labels)), "gate_passed": True,
                "gate_report_artifact": "gate_report.json", "validation_metrics": self.candidate_metrics,
                "lineage": {"source_manifest_path": self.source_manifest_path,
                            "source_manifest_hash": self.source_manifest_hash,
                            "training_corpus_sha256": self.input_manifest_sha256,
                            "source_lineage": self.source_lineage,
                            "encoder_revision": self.encoder_revision,
                            "fixture": bool(self.fixture)},
                "source_manifest_hash": self.source_manifest_hash,
                "source_manifest_artifact": "plagiarism_bundle/source_manifest.json",
                "source_cache_artifact": index_rel,
                "source_index_sha256": self.artifact_checks["source_index_sha256"],
                "features_artifact": "plagiarism_bundle/features.json",
                "features_sha256": self.artifact_checks["features_sha256"],
                "chunking_artifact": "plagiarism_bundle/chunking.json",
                "chunking_sha256": self.artifact_checks["chunking_sha256"],
                "encoder_manifest_artifact": "plagiarism_bundle/encoder_manifest.json",
                "encoder_manifest_sha256": self.artifact_checks["encoder_manifest_sha256"],
                "encoder_revision": self.encoder_revision,
                "encoder_provenance": self.encoder_manifest,
                "retrieval_top_k": int(self.config["chunking"]["top_k"]),
                "chunking_config": self.chunking_config,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            bundle_path = _json(Path(self.report_root) / "deployment_bundle.json", bundle)
            client.log_artifact(self.mlflow_run_id, str(bundle_path))
            for tag, value in (("gate_passed", "true"), ("target_version", bundle["target_version"]),
                               ("feature_schema_version", bundle["feature_schema_version"]),
                               ("bundle_artifact", "deployment_bundle.json")):
                client.set_model_version_tag(self.config["model_name"], self.registered_version, tag, str(value))
            self.bundle = bundle
        else:
            self.bundle = None
        self.next(self.end)

    @_failed_run_guard
    @step
    def end(self):
        summary = {"status": "completed", "gate_passed": bool(self.gate_report["passed"]),
                   "candidate_version": self.registered_version, "mlflow_run_id": self.mlflow_run_id,
                   "metaflow_run_id": self.flow_run_id, "gate_decision": self.gate_report["decision"],
                   "threshold_selection_error": self.threshold_selection_error,
                   "report_dir": self.report_root}
        if self.result_path:
            _json(Path(self.result_path), summary)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.set_terminated(self.mlflow_run_id, status="FINISHED")
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    PlagiarismTrainingFlow()
