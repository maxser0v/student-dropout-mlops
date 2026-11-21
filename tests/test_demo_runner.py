from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_demo


def test_plain_env_parser_rejects_shell_and_unknown_settings(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEMO_MLFLOW_PORT=$(touch /tmp/nope)\n", encoding="utf-8")
    with pytest.raises(run_demo.DemoError, match="plain text"):
        run_demo._parse_plain_env(env_file)

    env_file.write_text("UNRECOGNIZED_SETTING=value\n", encoding="utf-8")
    with pytest.raises(run_demo.DemoError, match="unknown setting"):
        run_demo._parse_plain_env(env_file)


def test_runtime_identity_uses_host_ids_and_never_root():
    settings = {"RUNTIME_UID": "auto", "RUNTIME_GID": "auto"}
    assert run_demo.runtime_identity(settings, uid=501, gid=20) == (501, 20)
    with pytest.raises(run_demo.DemoError, match="non-root"):
        run_demo.runtime_identity(settings, uid=0, gid=0)
    settings["RUNTIME_UID"] = "0"
    with pytest.raises(run_demo.DemoError, match="nonzero"):
        run_demo.runtime_identity(settings, uid=501, gid=20)


def _synthetic_data(root: Path) -> None:
    data = root / "data"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(
        json.dumps({"synthetic": True, "files": {"data/raw/snapshots.csv": {"sha256": "a" * 64}}}),
        encoding="utf-8",
    )
    (root / "data.dvc").write_text("outs: []\n", encoding="utf-8")


def test_existing_non_synthetic_data_is_never_regenerated(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    original = data / "manifest.json"
    original.write_text(json.dumps({"synthetic": False}), encoding="utf-8")
    (tmp_path / "data.dvc").write_text("outs: []\n", encoding="utf-8")
    called: list[list[str]] = []

    with pytest.raises(run_demo.DemoError, match="left untouched"):
        run_demo.ensure_data(tmp_path, run=lambda command, **kwargs: called.append(command))

    assert called == []
    assert json.loads(original.read_text(encoding="utf-8")) == {"synthetic": False}


def test_existing_dataset_with_stale_dvc_pointer_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _synthetic_data(tmp_path)
    monkeypatch.setattr(run_demo, "verify_dvc_directory", lambda **_: {"passed": False, "reason": "hash_mismatch"})
    called: list[list[str]] = []

    with pytest.raises(run_demo.DemoError, match="DVC data binding check failed"):
        run_demo.ensure_data(tmp_path, run=lambda command, **kwargs: called.append(command))

    assert called == []


def test_missing_dataset_generates_only_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(run_demo, "verify_dvc_directory", lambda **_: {"passed": True, "pointer_hash": "ok"})
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[-1] == "scripts/generate_synthetic_data.py":
            _synthetic_data(tmp_path)

    result = run_demo.ensure_data(tmp_path, run=fake_run)
    assert result["passed"] is True
    assert len(calls) == 2
    assert calls[0][-1] == "scripts/generate_synthetic_data.py"
    assert calls[1][-1] == "scripts/verify_synthetic_data.py"


def test_training_result_requires_gate_pass(tmp_path: Path):
    result = {
        "status": "completed",
        "gate_passed": False,
        "candidate_version": "2",
        "mlflow_run_id": "run-id",
        "metaflow_run_id": "flow-id",
    }
    path = tmp_path / "run-demo-result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(run_demo.DemoError, match="gate-passing"):
        run_demo._read_training_result(path)


def test_compose_environment_overrides_auto_and_external_identity():
    computed = {
        "RUNTIME_UID": "501",
        "RUNTIME_GID": "20",
        "GIT_COMMIT": "a" * 40,
        "DockerImageID": "sha256:" + "b" * 64,
    }
    env = run_demo.compose_environment(
        {
            "RUNTIME_UID": "auto",
            "RUNTIME_GID": "auto",
            "GIT_COMMIT": "stale",
            "COMPOSE_FILE": "evil.yaml",
            "COMPOSE_PROFILES": "jobs",
            "COMPOSE_PROJECT_NAME": "shared",
        },
        computed,
    )
    assert {key: env[key] for key in computed} == computed
    assert not {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"} & set(env)
    assert "--file" in run_demo.compose_command("docker", Path("/project"), Path("/tmp/demo.env"), "demo")


def test_source_archive_records_unknown_git_without_hashing_runtime_files(tmp_path: Path):
    from scripts.demo_provenance import git_provenance

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('demo')\n", encoding="utf-8")
    (tmp_path / "src" / ".env").write_text("SECRET=x\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "output.json").write_text("{}", encoding="utf-8")
    provenance = git_provenance(tmp_path)
    assert provenance["git_commit"] == "unknown"
    assert provenance["git_dirty"] == "unknown"
    assert provenance["source_revision_source"] == "unknown"
    assert len(provenance["source_tree_sha256"]) == 64


def test_source_provenance_rejects_symlink_outside_source_root(tmp_path: Path):
    from scripts.demo_provenance import source_tree_sha256

    source = tmp_path / "src"
    source.mkdir()
    private = tmp_path.parent / "outside-secret.txt"
    private.write_text("secret\n", encoding="utf-8")
    (source / "external.txt").symlink_to(private)
    try:
        with pytest.raises(ValueError, match="symlinks are not accepted"):
            source_tree_sha256(tmp_path)
    finally:
        private.unlink(missing_ok=True)


def test_failed_demo_cleanup_stops_project_without_removing_volume(tmp_path: Path):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)

    compose = ["docker", "compose", "--project-name", "demo-123"]
    run_demo.cleanup_project(compose, cwd=tmp_path, env={"RUNTIME_UID": "501"}, run=fake_run)
    assert observed["command"] == compose + ["down"]
    assert "-v" not in observed["command"]
    assert observed["check"] is False


def test_project_output_directory_rejects_nested_symlinks(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "reports").mkdir()
    (root / "reports" / "demo_runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(run_demo.DemoError, match="project output symlink"):
        run_demo._safe_project_mkdir(root, root / "reports" / "demo_runs" / "new-run")
