from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import flows.train_plagiarism_flow as training_flow


class _Client:
    def __init__(self, **kwargs):
        self.artifacts = []
        self.tags = []

    def log_artifact(self, run_id, path):
        self.artifacts.append((run_id, Path(path)))

    def set_model_version_tag(self, name, version, tag, value):
        self.tags.append((name, version, tag, value))


def test_passing_register_hands_encoder_manifest_into_deployment_bundle(monkeypatch, tmp_path):
    client = _Client()
    monkeypatch.setattr(training_flow, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(
        training_flow.mlflow,
        "register_model",
        lambda uri, name, await_registration_for: SimpleNamespace(version="7"),
    )
    flow = training_flow.PlagiarismTrainingFlow(use_cli=False)
    manifest = {"mode": "fixture", "revision": "fixture-hash-encoder-v1", "dimensions": 128}
    flow.tracking_uri = "http://mlflow.test"
    flow.gate_report = {"passed": True}
    flow.logged_model_uri = "runs:/run-123/model"
    flow.config = {
        "model_name": "passage-reuse-fixture",
        "target": {"version": "passage-reuse-pair-classifier-v1", "description": "fixture"},
        "feature_schema": {"version": "1.0"},
        "config_version": "1.0",
        "chunking": {"top_k": 10},
    }
    flow.config_sha256 = "c" * 64
    flow.train_labels = [0, 1]
    flow.gate_report = {"passed": True}
    flow.candidate_metrics = {"recall": 1.0}
    flow.mlflow_run_id = "run-123"
    flow.flow_run_id = "flow-456"
    flow.threshold = 0.5
    flow.source_manifest_path = "plagiarism_bundle/source_manifest.json"
    flow.source_manifest_hash = "a" * 64
    flow.input_manifest_sha256 = "b" * 64
    flow.source_lineage = {"dataset": "authored-offline-fixture"}
    flow.encoder_revision = manifest["revision"]
    flow.encoder_manifest = manifest
    flow.fixture = True
    flow.artifact_checks = {
        "source_index_sha256": "1" * 64,
        "features_sha256": "2" * 64,
        "chunking_sha256": "3" * 64,
        "encoder_manifest_sha256": "4" * 64,
    }
    flow.chunking_config = {"target_tokens": 180, "overlap_tokens": 40, "max_tokens": 256}
    flow.report_root = str(tmp_path)
    flow.next = lambda *args, **kwargs: None

    flow.register()

    bundle_path = tmp_path / "deployment_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["model_version"] == "7"
    assert bundle["model_uri"] == "models:/passage-reuse-fixture/7"
    assert bundle["mlflow_run_id"] == "run-123"
    assert bundle["encoder_revision"] == manifest["revision"]
    assert bundle["encoder_provenance"] == manifest
    assert ("passage-reuse-fixture", "7", "gate_passed", "true") in client.tags
