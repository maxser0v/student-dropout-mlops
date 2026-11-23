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
from plagiarism.data import assign_splits, filter_corpus, load_corpus, load_corpus_bytes, load_pan
from dropout.lineage import collect_source_lineage, verify_dvc_directory
from plagiarism.evaluation import (candidate_labels, choose_shared_recall_thresholds,
                                   evaluate_at_threshold, evaluate_release_gate, merge_alerts,
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


def _load_data(path: Path, validation_fraction: float, seed: int, *, fixture: bool = False,
               verified_manifest_bytes: bytes | None = None):
    if path.is_dir():
        names = ("fixture.json", "manifest.json") if fixture else ("pan13.json", "manifest.json")
        manifest = next((path / name for name in names if (path / name).is_file()), None)
        if manifest is not None:
            corpus = (load_corpus_bytes(verified_manifest_bytes, source=str(manifest))
                      if verified_manifest_bytes is not None else load_corpus(manifest))
        else:
            candidates = sorted(path.glob("*.zip"))
            if not candidates:
                raise ValueError(f"no corpus JSON manifest or PAN zip found under {path}")
            manifest = candidates[0]
            corpus = load_pan(manifest)
    elif path.suffix.lower() == ".json":
        manifest = path
        corpus = (load_corpus_bytes(verified_manifest_bytes, source=str(path))
                  if verified_manifest_bytes is not None else load_corpus(path))
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


def _json_manifest_for(path: Path, *, fixture: bool) -> Path | None:
    if path.is_dir():
        names = ("fixture.json", "manifest.json") if fixture else ("pan13.json", "manifest.json")
        return next((path / name for name in names if (path / name).is_file()), None)
    return path if path.suffix.lower() == ".json" else None


def _platform_scope(lineage: dict[str, Any]) -> dict[str, str]:
    raw = lineage.get("scope")
    fields = ("institution_id", "course_id", "assignment_id")
    if not isinstance(raw, dict) or any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in fields):
        raise ValueError("platform-reviewed corpus lineage requires institution, course, and assignment scope")
    return {key: raw[key] for key in fields}


def _validate_sha256(value: str | None, option: str) -> str | None:
    if value is None:
        return None
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"{option} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _validate_active_bundle(bundle: dict[str, Any], *, model_name: str, model_version: str,
                            target_version: str, feature_schema_version: str,
                            encoder_revision: str, chunking: dict[str, int], top_k: int,
                            fixture_model_name: str, fixture: bool,
                            platform_scope: dict[str, str] | None) -> dict[str, Any]:
    if bundle.get("model_name") != model_name or str(bundle.get("model_version")) != model_version:
        raise ValueError("verified active bundle does not match the requested model version")
    lineage = bundle.get("lineage", {})
    lineage = lineage if isinstance(lineage, dict) else {}
    active_scope = lineage.get("platform_scope")
    if active_scope is not None:
        fields = ("institution_id", "course_id", "assignment_id")
        if (not isinstance(active_scope, dict)
                or any(not isinstance(active_scope.get(key), str) or not active_scope[key].strip()
                       for key in fields)):
            raise ValueError("active model has malformed platform_scope lineage")
        if platform_scope is None:
            raise ValueError("platform-scoped active model cannot be compared in a non-platform run")
        if active_scope != platform_scope:
            raise ValueError("active model platform_scope does not match this reviewed assignment")
    if (bool(lineage.get("fixture", False)) != fixture
            or (model_name == fixture_model_name) != fixture):
        raise ValueError("active model fixture namespace is incompatible with this training run")
    if (bundle.get("target_version") != target_version
            or bundle.get("feature_schema_version") != feature_schema_version):
        raise ValueError("active model target or feature schema is incompatible with the candidate")
    if (bundle.get("encoder_revision") != encoder_revision
            or bundle.get("chunking_config") != chunking
            or bundle.get("retrieval_top_k") != top_k):
        raise ValueError("active model encoder or retrieval configuration is incompatible with the candidate")
    threshold = bundle.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError("verified active model has no deployed numeric threshold")
    return {"model_name": model_name, "model_version": model_version,
            "deployed_threshold": float(threshold), "platform_scope": active_scope}


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
    active_model_name = Parameter("active-model-name", default=None, help="Registered active model namespace for comparison")
    active_model_version = Parameter("active-model-version", default=None, help="Concrete active registry version for comparison")
    platform_snapshot_sha256 = Parameter("platform-snapshot-sha256", default=None,
                                         help="Trusted SHA-256 of the immutable platform snapshot manifest")

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
        self.normalized_active_model_name = self.active_model_name
        self.normalized_active_model_version = self.active_model_version
        self.verified_platform_snapshot_sha256 = _validate_sha256(
            self.platform_snapshot_sha256, "--platform-snapshot-sha256")
        if (self.normalized_active_model_name is None) != (self.normalized_active_model_version is None):
            raise ValueError("--active-model-name and --active-model-version must be supplied together")
        if self.normalized_active_model_name is not None:
            if not isinstance(self.normalized_active_model_name, str) or not self.normalized_active_model_name.strip():
                raise ValueError("--active-model-name must be a concrete non-empty model name")
            self.normalized_active_model_name = self.normalized_active_model_name.strip()
            if (not isinstance(self.normalized_active_model_version, str) or not self.normalized_active_model_version.isdecimal()
                    or int(self.normalized_active_model_version) < 1):
                raise ValueError("--active-model-version must be a concrete positive registry version")
            self.normalized_active_model_version = str(int(self.normalized_active_model_version))
        manifest_before_parse = _json_manifest_for(Path(self.data_path), fixture=bool(self.fixture))
        self.is_platform_run = False
        self.platform_scope = None
        if manifest_before_parse is not None:
            # Read only the lineage header first. If it declares platform input, verify
            # the exact bytes against the trusted worker argument before loading corpus data.
            manifest_bytes = manifest_before_parse.read_bytes()
            if (self.verified_platform_snapshot_sha256 is not None
                    and hashlib.sha256(manifest_bytes).hexdigest() != self.verified_platform_snapshot_sha256):
                raise ValueError("platform snapshot manifest does not match trusted --platform-snapshot-sha256")
            header = json.loads(manifest_bytes)
            lineage_header = header.get("lineage", {}) if isinstance(header, dict) else {}
            protocol = lineage_header.get("evaluation_protocol") if isinstance(lineage_header, dict) else None
            self.is_platform_run = protocol == "platform-reviewed-grouped"
            if self.is_platform_run:
                if self.verified_platform_snapshot_sha256 is None:
                    raise ValueError("platform-reviewed training requires --platform-snapshot-sha256")
                self.platform_scope = _platform_scope(lineage_header)
                if self.fixture:
                    raise ValueError("platform-reviewed training cannot use the fixture encoder namespace")
                if self.normalized_active_model_name is None:
                    raise ValueError("platform-reviewed training requires explicit active model name and version")
            elif self.verified_platform_snapshot_sha256 is not None:
                raise ValueError("--platform-snapshot-sha256 is only valid for platform-reviewed snapshots")
        elif self.verified_platform_snapshot_sha256 is not None:
            raise ValueError("--platform-snapshot-sha256 requires a JSON platform snapshot manifest")
        self.corpus, self.manifest_path = _load_data(Path(self.data_path),
                                                     float(self.config["data"]["validation_fraction"]),
                                                     int(self.config["data"]["split_seed"]),
                                                     fixture=bool(self.fixture),
                                                     verified_manifest_bytes=(manifest_bytes if self.is_platform_run else None))
        is_fixture_data = bool(self.corpus.lineage.get("fixture")) or "fixture" in str(self.corpus.lineage.get("dataset", "")).lower()
        if is_fixture_data and not self.fixture:
            raise ValueError("fixture corpus requires explicit --fixture encoder mode")
        if self.fixture and self.config["model_name"] != self.config["data"]["fixture_model_name"]:
            raise ValueError("fixture runs must use the isolated passage-reuse-fixture model namespace")
        if self.is_platform_run and self.corpus.lineage.get("evaluation_protocol") != "platform-reviewed-grouped":
            raise ValueError("platform snapshot lineage changed during corpus loading")
        if self.is_platform_run and self.manifest_path and _sha256(Path(self.manifest_path)) != self.verified_platform_snapshot_sha256:
            raise ValueError("platform snapshot manifest changed after its trusted hash check")
        self.candidate_model_name = self.config["model_name"]
        if self.is_platform_run:
            scope_key = json.dumps(self.platform_scope, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.candidate_model_name = f"{self.config['model_name']}-platform-{hashlib.sha256(scope_key).hexdigest()[:12]}"
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment_name = (self.candidate_model_name if self.is_platform_run
                           else str(self.config.get("experiment_name", self.candidate_model_name)))
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
        run = client.create_run(experiment_id, run_name=f"plagiarism-{self.flow_run_id}", tags={
            "metaflow.flow": str(current.flow_name), "metaflow.run_id": self.flow_run_id,
            "model.namespace": self.candidate_model_name, "fixture": str(bool(self.fixture)).lower(),
            "evaluation.protocol": self.corpus.lineage.get(
                "evaluation_protocol", "authored-fixture" if self.fixture else "pan13-official-test"),
            "target_version": self.config["target"]["version"],
            "feature_schema_version": self.config["feature_schema"]["version"],
            "training.origin": "platform" if self.is_platform_run else "repository",
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
        elif self.is_platform_run:
            current_sha = _sha256(Path(self.manifest_path))
            if current_sha != self.verified_platform_snapshot_sha256:
                raise ValueError("platform snapshot manifest changed before data validation")
            self.dvc_verification = {"passed": True, "method": "trusted-platform-manifest-sha256",
                                     "manifest_sha256": current_sha}
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
            "platform_snapshot_sha256": self.verified_platform_snapshot_sha256 if self.is_platform_run else None,
            "platform_scope": self.platform_scope,
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
                               "active_model_name": self.normalized_active_model_name or "",
                               "active_model_version": self.normalized_active_model_version or "",
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
        self.active_model_details = None
        self.active_model_error = None
        self.active_deployed_metrics = None
        if self.normalized_active_model_name is not None:
            try:
                from plagiarism.score import load_verified_model

                mlflow.set_tracking_uri(self.tracking_uri)
                client = MlflowClient(tracking_uri=self.tracking_uri)
                active_model, active_bundle, active_features, _active_encoder = load_verified_model(
                    client, self.normalized_active_model_name, self.normalized_active_model_version,
                    fixture=bool(self.fixture), encoder_dir=self.encoder_dir or "/app/models/minilm")
                active_metadata = _validate_active_bundle(
                    active_bundle, model_name=self.normalized_active_model_name,
                    model_version=self.normalized_active_model_version,
                    target_version=self.config["target"]["version"],
                    feature_schema_version=self.config["feature_schema"]["version"],
                    encoder_revision=self.encoder_revision, chunking=self.chunking_config,
                    top_k=int(settings["top_k"]),
                    fixture_model_name=self.config["data"]["fixture_model_name"],
                    fixture=bool(self.fixture), platform_scope=self.platform_scope)
                active_scores = positive_class_probability(
                    active_model, active_features.transform_pairs(self.validation_rows))
                score_sets["active_xgboost"] = (self.validation_rows, active_scores)
                self.active_model_details = active_metadata
                self.active_deployed_metrics = evaluate_at_threshold(
                    self.validation_rows, active_scores, active_metadata["deployed_threshold"],
                    self.validation_corpus.annotations, eligible_submissions=eligible,
                    min_span_coverage=float(settings["min_span_coverage"]))
            except Exception as exc:
                # Preserve a machine-readable failed gate report for broken or
                # incompatible active releases; never silently compare another model.
                self.active_model_error = f"{type(exc).__name__}: {exc}"
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
            active_comparison = None
            if self.active_model_details is not None:
                active_shared = choices["active_xgboost"]
                active_comparison = {**self.active_model_details,
                                     "shared_recall": active_shared,
                                     "deployed_threshold_metrics": self.active_deployed_metrics}
            elif self.active_model_error:
                active_comparison = {"model_name": self.normalized_active_model_name,
                                     "model_version": self.normalized_active_model_version,
                                     "error": self.active_model_error}
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
                "active_model_comparison": active_comparison,
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
                "active_model_comparison": ({**self.active_model_details,
                                             "shared_recall": None,
                                             "deployed_threshold_metrics": self.active_deployed_metrics}
                                            if self.active_model_details else
                                            ({"model_name": self.normalized_active_model_name,
                                              "model_version": self.normalized_active_model_version,
                                              "error": self.active_model_error}
                                             if self.active_model_error else None)),
            }
        self.next(self.gate)

    @_failed_run_guard
    @step
    def gate(self):
        self.gate_report = evaluate_release_gate(
            self.candidate_metrics, self.config,
            data_checks_passed=True,
            artifact_checks_passed=all(bool(v) for v in self.artifact_checks.values()),
            active_comparison_required=bool(getattr(self, "is_platform_run", False)))
        self.gate_report["mlflow_run_id"] = self.mlflow_run_id
        self.gate_report["metaflow_run_id"] = self.flow_run_id
        self.gate_report["fixture"] = bool(self.fixture)
        self.gate_report["platform_scope"] = getattr(self, "platform_scope", None)
        self.gate_report["platform_snapshot_sha256"] = (
            getattr(self, "platform_snapshot_sha256", None)
            if getattr(self, "is_platform_run", False) else None)
        report_path = _json(Path(self.report_root) / "gate_report.json", self.gate_report)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.start_run(run_id=self.mlflow_run_id)
        try:
            mlflow.set_tag("release_gate", self.gate_report["decision"])
            mlflow.log_dict(self.gate_report, "gate_report.json")
            mlflow.log_artifact(str(report_path))
            if self.active_model_details:
                mlflow.set_tags({"active.model_name": self.active_model_details["model_name"],
                                 "active.model_version": self.active_model_details["model_version"]})
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
            if self.active_deployed_metrics:
                for key in ("precision", "recall", "false_alerts", "false_alerts_per_100_submissions"):
                    value = self.active_deployed_metrics.get(key)
                    if isinstance(value, (int, float)) and value is not None:
                        mlflow.log_metric(f"validation_active_deployed_{key}", float(value))
        finally:
            mlflow.end_run(status="RUNNING")
        self.next(self.register)

    @_failed_run_guard
    @step
    def register(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        registry_model_name = getattr(self, "candidate_model_name", self.config["model_name"])
        is_platform_run = bool(getattr(self, "is_platform_run", False))
        self.registered_version = None
        if self.gate_report["passed"]:
            client = MlflowClient(tracking_uri=self.tracking_uri)
            version = mlflow.register_model(self.logged_model_uri, registry_model_name, await_registration_for=0)
            self.registered_version = str(version.version)
            index_rel = "plagiarism_bundle/source_index.npz"
            bundle = {
                "bundle_version": "1.0", "model_name": registry_model_name,
                "model_version": self.registered_version,
                "model_uri": f"models:/{registry_model_name}/{self.registered_version}",
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
                            "fixture": bool(self.fixture),
                            **({"platform_scope": self.platform_scope} if is_platform_run else {})},
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
                client.set_model_version_tag(registry_model_name, self.registered_version, tag, str(value))
            if is_platform_run:
                client.set_model_version_tag(registry_model_name, self.registered_version,
                                             "platform.scope_sha256", hashlib.sha256(
                                                 json.dumps(self.platform_scope, sort_keys=True,
                                                            separators=(",", ":")).encode("utf-8")).hexdigest())
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
                   "report_dir": self.report_root,
                   "active_model_comparison": self.candidate_metrics.get("active_model_comparison"),
                   "platform_scope": self.platform_scope,
                   "platform_snapshot_sha256": self.verified_platform_snapshot_sha256 if self.is_platform_run else None}
        if self.result_path:
            _json(Path(self.result_path), summary)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.set_terminated(self.mlflow_run_id, status="FINISHED")
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    PlagiarismTrainingFlow()
