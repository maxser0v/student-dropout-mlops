from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dropout.lineage import collect_source_lineage, verify_dvc_directory
from flows import train_flow

DATA = ROOT / "data"


def _expected_files(manifest: dict[str, object]) -> set[str]:
    file_map = manifest["files"]
    return {str(path).removeprefix("data/") for path in file_map} | {"manifest.json"}


def _copy_tracked_data(destination: Path) -> tuple[Path, Path, dict[str, object]]:
    destination.mkdir(parents=True)
    copied_data = destination / "data"
    shutil.copytree(DATA, copied_data)
    copied_pointer = destination / "data.dvc"
    shutil.copy2(ROOT / "data.dvc", copied_pointer)
    manifest = json.loads((copied_data / "manifest.json").read_text(encoding="utf-8"))
    return copied_pointer, copied_data, manifest


def test_real_dvc_pointer_matches_all_manifest_files() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    report = verify_dvc_directory(
        ROOT / "data.dvc",
        DATA,
        expected_path="data",
        expected_files=_expected_files(manifest),
    )
    assert report["passed"] is True
    assert report["pointer_hash"] == "58dd4f08e5a36015f5b265a093195256.dir"
    assert report["computed_hash"] == report["pointer_hash"]
    assert report["pointer_size"] == report["computed_size"] == 1_424_698
    assert report["pointer_nfiles"] == report["computed_nfiles"] == 6


def test_verifier_matches_fresh_public_dvc_add_output(tmp_path: Path) -> None:
    """Cross-check the documented serializer against the pinned DVC CLI."""
    dvc_cli = Path(sys.executable).with_name("dvc")
    if not dvc_cli.is_file():
        pytest.skip("DVC CLI is unavailable in this Python environment")
    project = tmp_path / "dvc-project"
    data = project / "data" / "nested"
    data.mkdir(parents=True)
    (data / "plain.txt").write_text("DVC directory hash fixture\n", encoding="utf-8")
    executable = data / "run.sh"
    executable.write_text("#!/bin/sh\necho fixture\n", encoding="utf-8")
    executable.chmod(0o755)

    env = os.environ.copy()
    env["DVC_SITE_CACHE_DIR"] = str(tmp_path / "dvc-site-cache")
    env["DVC_NO_ANALYTICS"] = "true"
    subprocess.run([str(dvc_cli), "init", "--no-scm"], cwd=project, env=env, check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([str(dvc_cli), "add", "data"], cwd=project, env=env, check=True, capture_output=True, text=True, timeout=60)

    report = verify_dvc_directory(
        project / "data.dvc",
        project / "data",
        expected_path="data",
        expected_files={"nested/plain.txt", "nested/run.sh"},
    )
    assert report["passed"] is True
    assert report["pointer_hash"] == report["computed_hash"]
    assert report["pointer_size"] == report["computed_size"]
    assert report["pointer_nfiles"] == report["computed_nfiles"] == 2


def test_changed_file_and_refreshed_manifest_still_fail_stale_dvc_pointer(tmp_path: Path) -> None:
    pointer, data, manifest = _copy_tracked_data(tmp_path / "changed")
    outcomes = data / "raw" / "student_outcomes.csv"
    outcomes.write_bytes(outcomes.read_bytes() + b"\n")
    manifest["files"]["data/raw/student_outcomes.csv"]["sha256"] = hashlib.sha256(outcomes.read_bytes()).hexdigest()
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = verify_dvc_directory(pointer, data, expected_files=_expected_files(manifest))
    assert report["passed"] is False
    assert report["reason"] == "dvc_directory_content_mismatch"
    assert report["computed_hash"] != report["pointer_hash"]


@pytest.mark.parametrize("mutation, expected_reason", [
    ("extra", "dvc_directory_file_set_mismatch"),
    ("missing", "dvc_directory_file_set_mismatch"),
])
def test_dvc_verifier_rejects_extra_or_missing_files(tmp_path: Path, mutation: str, expected_reason: str) -> None:
    pointer, data, manifest = _copy_tracked_data(tmp_path / mutation)
    expected = _expected_files(manifest)
    if mutation == "extra":
        (data / "unexpected.csv").write_text("untracked\n", encoding="utf-8")
    else:
        (data / "raw" / "student_outcomes.csv").unlink()
    report = verify_dvc_directory(pointer, data, expected_files=expected)
    assert report["passed"] is False
    assert report["reason"] == expected_reason


def test_dvc_verifier_rejects_wrong_pointer_path_and_hash_format(tmp_path: Path) -> None:
    pointer, data, manifest = _copy_tracked_data(tmp_path / "pointer")
    original = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    wrong_path = json.loads(json.dumps(original))
    wrong_path["outs"][0]["path"] = "other"
    pointer.write_text(yaml.safe_dump(wrong_path), encoding="utf-8")
    report = verify_dvc_directory(pointer, data, expected_files=_expected_files(manifest))
    assert report["reason"] == "dvc_pointer_path_mismatch"

    wrong_hash = json.loads(json.dumps(original))
    wrong_hash["outs"][0]["md5"] = "not-a-directory-hash"
    pointer.write_text(yaml.safe_dump(wrong_hash), encoding="utf-8")
    report = verify_dvc_directory(pointer, data, expected_files=_expected_files(manifest))
    assert report["reason"] == "invalid_dvc_directory_hash"


def test_dvc_verifier_rejects_unsupported_hash_and_multiple_outputs(tmp_path: Path) -> None:
    pointer, data, manifest = _copy_tracked_data(tmp_path / "unsupported")
    original = yaml.safe_load(pointer.read_text(encoding="utf-8"))

    unsupported_hash = json.loads(json.dumps(original))
    unsupported_hash["outs"][0]["hash"] = "sha256"
    pointer.write_text(yaml.safe_dump(unsupported_hash), encoding="utf-8")
    report = verify_dvc_directory(pointer, data, expected_files=_expected_files(manifest))
    assert report["reason"] == "unsupported_dvc_hash_algorithm"

    multiple_outputs = json.loads(json.dumps(original))
    multiple_outputs["outs"].append(dict(multiple_outputs["outs"][0]))
    pointer.write_text(yaml.safe_dump(multiple_outputs), encoding="utf-8")
    report = verify_dvc_directory(pointer, data, expected_files=_expected_files(manifest))
    assert report["reason"] == "expected_one_dvc_directory_output"


def test_dvc_verifier_rejects_a_symlink_data_root(tmp_path: Path) -> None:
    data_link = tmp_path / "data"
    data_link.symlink_to(DATA, target_is_directory=True)
    report = verify_dvc_directory(ROOT / "data.dvc", data_link, expected_files={"manifest.json"})
    assert report["reason"] == "data_directory_symlink_not_allowed"


def test_source_provenance_keeps_compose_sentinels_unknown(tmp_path: Path) -> None:
    provenance = collect_source_lineage(
        tmp_path,
        environ={
            "GIT_COMMIT": "unknown",
            "GIT_DIRTY": "unknown",
            "SOURCE_TREE_SHA256": "unknown",
            "SOURCE_REVISION_SOURCE": "unspecified",
            "DockerImageID": "unknown",
        },
    )
    assert provenance["git_commit"] is None
    assert provenance["git_commit_source"] == "unavailable"
    assert provenance["git_dirty"] == "unknown"
    assert provenance["source_revision_source"] == "unknown"
    assert provenance["source_tree_sha256"] is None
    assert provenance["docker_image_id"] is None


def test_source_provenance_records_and_validates_host_metadata(tmp_path: Path) -> None:
    provenance = collect_source_lineage(
        tmp_path,
        environ={
            "GIT_COMMIT": "a" * 40,
            "GIT_DIRTY": "true",
            "SOURCE_REVISION_SOURCE": "host-git",
            "SOURCE_TREE_SHA256": "b" * 64,
            "SOURCE_TREE_HASH_ALGORITHM": "sha256",
            "DockerImageID": "sha256:" + "c" * 64,
        },
    )
    assert provenance["git_commit"] == "a" * 40
    assert provenance["git_commit_source"] == "host-runner-env"
    assert provenance["git_dirty"] == "true"
    assert provenance["source_tree_sha256"] == "b" * 64
    assert provenance["docker_image_id"] == "sha256:" + "c" * 64
    assert provenance["source_revision_source"] == "host-git"
    assert provenance["source_tree_sha256_source"] == "host-runner-env"

    with pytest.raises(ValueError, match="GIT_COMMIT"):
        collect_source_lineage(tmp_path, environ={"GIT_COMMIT": "nope"})
    with pytest.raises(ValueError, match="DockerImageID"):
        collect_source_lineage(tmp_path, environ={"DockerImageID": "sha256:bad"})


def test_source_provenance_without_git_never_claims_clean_or_commit(tmp_path: Path) -> None:
    provenance = collect_source_lineage(tmp_path, environ={})
    assert provenance["git_commit"] is None
    assert provenance["git_dirty"] == "unknown"
    assert provenance["git_commit_source"] == "unavailable"


def test_result_path_is_confined_and_create_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train_flow, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    target = train_flow._result_path("reports/demo_runs/one/result.json")
    assert target == reports / "demo_runs" / "one" / "result.json"
    train_flow._write_result(target, train_flow._result_payload(
        status="completed", candidate_version="1", mlflow_run_id="mlf-1", metaflow_run_id="mf-1", gate_passed=True
    ))
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved == {
        "status": "completed", "gate_passed": True, "candidate_version": "1",
        "mlflow_run_id": "mlf-1", "metaflow_run_id": "mf-1",
    }
    with pytest.raises(FileExistsError):
        train_flow._result_path("reports/demo_runs/one/result.json")
    with pytest.raises(ValueError, match="inside the project reports"):
        train_flow._result_path("data/raw/student_outcomes.csv")
    with pytest.raises(ValueError, match="inside the project reports"):
        train_flow._result_path(str(tmp_path / "outside.json"))


def test_result_path_rejects_symlink_escape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train_flow, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (reports / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        train_flow._result_path("reports/escape/result.json")
