"""Project-scoped DVC directory verification and source provenance capture.

The data verifier follows DVC's documented directory cache format: a JSON array
of per-file MD5s and relative paths, itself MD5-addressed by the `.dvc` pointer.
It deliberately needs neither a Git checkout nor a DVC cache, so the same check
works in the training container after host-side DVC restoration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Mapping

import yaml


_HEX_32 = re.compile(r"^[0-9a-fA-F]{32}$")
_HEX_40_OR_64 = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _failed(reason: str, **details) -> dict[str, object]:
    return {"passed": False, "reason": reason, **details}


def verify_dvc_directory(
    pointer_file: str | Path,
    data_dir: str | Path,
    *,
    expected_path: str = "data",
    expected_files: set[str],
    dvcignore_file: str | Path | None = None,
) -> dict[str, object]:
    """Verify the complete directory contents against one DVC `.dir` pointer.

    This intentionally supports the project's pinned local MD5 directory
    pointer: one output at ``expected_path``, `hash: md5`, and no active
    `.dvcignore` patterns. Missing, extra, symlinked, or changed files fail.
    """
    pointer_input = Path(pointer_file)
    data_input = Path(data_dir)
    if pointer_input.is_symlink():
        return _failed("dvc_pointer_symlink_not_allowed", pointer_file=str(pointer_input))
    if data_input.is_symlink():
        return _failed("data_directory_symlink_not_allowed", data_dir=str(data_input))
    pointer = pointer_input.resolve()
    root = data_input.resolve()
    pointer_label = str(pointer)
    try:
        pointer_doc = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return _failed("cannot_read_dvc_pointer", pointer_file=pointer_label, detail=str(exc))
    if not isinstance(pointer_doc, dict) or pointer_doc.get("wdir", ".") not in {"", "."}:
        return _failed("unsupported_dvc_pointer_layout", pointer_file=pointer_label)
    outputs = pointer_doc.get("outs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        return _failed("expected_one_dvc_directory_output", pointer_file=pointer_label)
    output = outputs[0]
    output_path = output.get("path")
    expected_posix = str(PurePosixPath(expected_path))
    if output_path != expected_posix:
        return _failed(
            "dvc_pointer_path_mismatch",
            pointer_file=pointer_label,
            expected_path=expected_posix,
            pointer_path=output_path,
        )
    if (pointer.parent / expected_posix).resolve() != root:
        return _failed(
            "data_directory_does_not_match_pointer_path",
            pointer_file=pointer_label,
            expected_path=expected_posix,
            data_dir=str(root),
        )
    if output.get("hash", "md5") != "md5":
        return _failed("unsupported_dvc_hash_algorithm", hash_algorithm=output.get("hash"))
    pointer_digest = output.get("md5")
    if not isinstance(pointer_digest, str) or not pointer_digest.endswith(".dir") or not _HEX_32.fullmatch(pointer_digest[:-4]):
        return _failed("invalid_dvc_directory_hash", pointer_hash=pointer_digest)
    if isinstance(output.get("size"), bool) or not isinstance(output.get("size"), int):
        return _failed("invalid_dvc_directory_size", pointer_size=output.get("size"))
    if isinstance(output.get("nfiles"), bool) or not isinstance(output.get("nfiles"), int):
        return _failed("invalid_dvc_directory_file_count", pointer_nfiles=output.get("nfiles"))
    if not root.is_dir():
        return _failed("data_directory_missing", data_dir=str(root))

    ignore_path = Path(dvcignore_file).resolve() if dvcignore_file else pointer.parent / ".dvcignore"
    try:
        if ignore_path.is_file():
            active_patterns = [
                line.strip()
                for line in ignore_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if active_patterns:
                return _failed("unsupported_active_dvcignore_patterns", patterns=active_patterns)
        nested_ignores = list(root.rglob(".dvcignore"))
        if nested_ignores:
            return _failed(
                "unsupported_nested_dvcignore",
                paths=[path.relative_to(root).as_posix() for path in nested_ignores],
            )
    except (OSError, UnicodeDecodeError) as exc:
        return _failed("cannot_read_dvcignore", detail=str(exc))

    entries: list[dict[str, str]] = []
    total_size = 0
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        actual_paths: set[str] = set()
        for item in paths:
            if item.is_symlink():
                return _failed("symlink_in_dvc_directory", relative_path=item.relative_to(root).as_posix())
            if item.is_dir():
                continue
            if not item.is_file():
                return _failed("non_regular_file_in_dvc_directory", relative_path=item.relative_to(root).as_posix())
            relative = item.relative_to(root).as_posix()
            actual_paths.add(relative)
            file_md5 = hashlib.md5(item.read_bytes()).hexdigest()  # DVC's configured local cache hash is MD5.
            entries.append({"md5": file_md5, "relpath": relative})
            total_size += item.stat().st_size
    except OSError as exc:
        return _failed("cannot_hash_dvc_directory", detail=str(exc))

    # DVC 3.67.1 emits the documented `.dir` JSON array sorted by relative path.
    directory_hash = hashlib.md5(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()
    nfiles = len(entries)
    if actual_paths != expected_files:
        return _failed(
            "dvc_directory_file_set_mismatch",
            expected_files=sorted(expected_files),
            actual_files=sorted(actual_paths),
            pointer_hash=pointer_digest.lower(),
            computed_hash=f"{directory_hash}.dir",
            pointer_size=output["size"],
            computed_size=total_size,
            pointer_nfiles=output["nfiles"],
            computed_nfiles=nfiles,
        )
    matches = (
        directory_hash == pointer_digest[:-4].lower()
        and total_size == output["size"]
        and nfiles == output["nfiles"]
    )
    return {
        "passed": matches,
        "reason": None if matches else "dvc_directory_content_mismatch",
        "pointer_file": pointer_label,
        "pointer_path": output_path,
        "hash_algorithm": "md5",
        "pointer_hash": pointer_digest.lower(),
        "computed_hash": f"{directory_hash}.dir",
        "pointer_size": output["size"],
        "computed_size": total_size,
        "pointer_nfiles": output["nfiles"],
        "computed_nfiles": nfiles,
        "files": entries,
    }


def _git_value(root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_source_lineage(
    root: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Collect source/build IDs, preserving unknown values instead of guessing.

    Build-time metadata from the host runner takes precedence. Local Git is a
    fallback only for commit and dirty-state observations; missing commit,
    source-tree digest, or image ID remains explicitly unavailable.
    """
    env = os.environ if environ is None else environ
    project = Path(root).resolve()
    raw_commit = env.get("GIT_COMMIT")
    commit_sentinel = raw_commit.strip().lower() if raw_commit is not None else ""
    if commit_sentinel in {"unknown", "unavailable", "unspecified"}:
        commit = None
        commit_source = "unavailable"
    elif raw_commit is not None and raw_commit.strip():
        commit = raw_commit.strip().lower()
        if not _HEX_40_OR_64.fullmatch(commit):
            raise ValueError("GIT_COMMIT must be a full 40- or 64-character hexadecimal commit ID")
        commit_source = "host-runner-env"
    else:
        commit = _git_value(project, "rev-parse", "HEAD")
        if commit and not _HEX_40_OR_64.fullmatch(commit):
            raise ValueError("local Git returned an invalid full commit ID")
        commit_source = "local-git" if commit else "unavailable"

    raw_dirty = env.get("GIT_DIRTY")
    dirty_sentinel = raw_dirty.strip().lower() if raw_dirty is not None else ""
    if dirty_sentinel in {"unknown", "unavailable", "unspecified"}:
        dirty = "unknown"
        dirty_source = "host-runner-env"
    elif raw_dirty is not None and raw_dirty.strip():
        dirty = raw_dirty.strip().lower()
        if dirty not in {"true", "false"}:
            raise ValueError("GIT_DIRTY must be exactly true or false")
        dirty_source = "host-runner-env"
    else:
        status = _git_value(project, "status", "--porcelain")
        dirty = ("true" if status else "false") if status is not None else "unknown"
        dirty_source = "local-git" if status is not None else "unavailable"

    revision_source = env.get("SOURCE_REVISION_SOURCE")
    if revision_source is not None and revision_source.strip():
        revision_source = revision_source.strip()
        if revision_source.lower() == "unspecified":
            revision_source = "local-git" if commit_source == "local-git" else "unknown"
        elif revision_source.lower() in {"unknown", "unavailable"}:
            revision_source = "unknown"
        if not _LABEL.fullmatch(revision_source):
            raise ValueError("SOURCE_REVISION_SOURCE must be a short identifier")
        if revision_source != "unknown" and commit is None:
            raise ValueError("SOURCE_REVISION_SOURCE cannot claim a revision source without GIT_COMMIT")
        if revision_source == "host-git" and commit_source != "host-runner-env":
            raise ValueError("SOURCE_REVISION_SOURCE=host-git requires GIT_COMMIT from the host runner")
    else:
        revision_source = "local-git" if commit_source == "local-git" else "unknown"

    raw_tree_sha = env.get("SOURCE_TREE_SHA256")
    tree_sentinel = raw_tree_sha.strip().lower() if raw_tree_sha is not None else ""
    if tree_sentinel in {"unknown", "unavailable", "unspecified"}:
        source_tree_sha = None
        hash_algorithm = "sha256"
        source_tree_source = "unavailable"
    elif raw_tree_sha is not None and raw_tree_sha.strip():
        source_tree_sha = raw_tree_sha.strip().lower()
        if not _HEX_64.fullmatch(source_tree_sha):
            raise ValueError("SOURCE_TREE_SHA256 must be a 64-character hexadecimal SHA-256 digest")
        hash_algorithm = env.get("SOURCE_TREE_HASH_ALGORITHM", "sha256").strip().lower()
        if hash_algorithm != "sha256":
            raise ValueError("SOURCE_TREE_HASH_ALGORITHM must be sha256")
        source_tree_source = "host-runner-env"
    else:
        source_tree_sha = None
        hash_algorithm = "sha256"
        source_tree_source = "unavailable"

    raw_image_id = env.get("DockerImageID")
    image_sentinel = raw_image_id.strip().lower() if raw_image_id is not None else ""
    if image_sentinel in {"unknown", "unavailable", "unspecified"}:
        image_id = None
        image_source = "unavailable"
    elif raw_image_id is not None and raw_image_id.strip():
        image_id = raw_image_id.strip().lower()
        if not _IMAGE_ID.fullmatch(image_id):
            raise ValueError("DockerImageID must have the form sha256:<64 hexadecimal characters>")
        image_source = "host-runner-env"
    else:
        image_id = None
        image_source = "unavailable"

    return {
        "git_commit": commit,
        "git_commit_source": commit_source,
        "git_dirty": dirty,
        "git_dirty_source": dirty_source,
        "source_revision_source": revision_source,
        "source_tree_sha256": source_tree_sha,
        "source_tree_hash_algorithm": hash_algorithm,
        "source_tree_sha256_source": source_tree_source,
        "docker_image_id": image_id,
        "docker_image_id_source": image_source,
    }
