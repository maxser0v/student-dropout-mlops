"""Compute honest Git and source-tree provenance for local/CI container builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (".github", "config", "docs", "flows", "scripts", "src", "tests")
IGNORED_SOURCE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "node_modules", "reports", "outputs", ".metaflow", ".git"}
IGNORED_SOURCE_NAMES = {".env", ".env.local", ".DS_Store"}
ROOT_FILES = {
    ".dockerignore", ".env.example", ".gitignore", ".dvcignore", "data.dvc",
    "Dockerfile", "Makefile", "README.md", "compose.yaml", "pyproject.toml",
    "requirements.lock.txt", "uv.lock", ".python-version",
}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def source_paths(root: Path = ROOT) -> list[Path]:
    """Return current tracked and nonignored project source files in stable order."""
    try:
        raw = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    except (OSError, subprocess.CalledProcessError):
        found: list[Path] = []
        for directory in SOURCE_DIRS:
            base = root / directory
            if not base.is_dir():
                continue
            for candidate in base.rglob("*"):
                relative = candidate.relative_to(root)
                if candidate.is_symlink() and not (IGNORED_SOURCE_PARTS & set(relative.parts)):
                    raise ValueError(f"source tree symlinks are not accepted: {relative.as_posix()}")
                if candidate.is_file() and not (IGNORED_SOURCE_PARTS & set(relative.parts)) and candidate.name not in IGNORED_SOURCE_NAMES and candidate.suffix not in {".pyc", ".pem", ".key"}:
                    found.append(relative)
        for name in ROOT_FILES:
            candidate = root / name
            if candidate.is_symlink():
                raise ValueError(f"source tree symlinks are not accepted: {name}")
            if candidate.is_file():
                found.append(Path(name))
        return sorted(set(found), key=lambda path: path.as_posix())
    paths: list[Path] = []
    for item in raw.decode("utf-8", errors="strict").split("\0"):
        if not item:
            continue
        relative = Path(item)
        if relative.name in IGNORED_SOURCE_NAMES or relative.suffix in {".pyc", ".pem", ".key"}:
            continue
        first = relative.parts[0]
        if first not in SOURCE_DIRS and str(relative) not in ROOT_FILES:
            continue
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"source tree symlinks are not accepted: {relative.as_posix()}")
        if candidate.is_file() and not candidate.is_symlink():
            paths.append(relative)
    return sorted(set(paths), key=lambda path: path.as_posix())


def source_tree_sha256(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for relative in source_paths(root):
        path_bytes = relative.as_posix().encode("utf-8")
        content = (root / relative).read_bytes()
        digest.update(path_bytes)
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def git_provenance(root: Path = ROOT, source_label: str = "host-git") -> dict[str, Any]:
    try:
        commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
        if not commit or any(ch not in "0123456789abcdefABCDEF" for ch in commit):
            raise ValueError("Git HEAD did not resolve to a concrete hexadecimal commit")
        dirty: bool | str = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=all"))
    except (OSError, subprocess.CalledProcessError):
        # Source archives have no Git metadata. Preserve the unknown state;
        # never invent a commit or claim the archive is clean.
        commit = "unknown"
        dirty = "unknown"
        source_label = "unknown"
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_tree_sha256": source_tree_sha256(root),
        "source_revision_source": source_label,
        "source_tree_hash_algorithm": "sha256",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-label", default="host-git")
    parser.add_argument("--verify-commit", help="Fail unless checked-out HEAD equals this expected commit")
    parser.add_argument("--require-clean", action="store_true", help="Fail if Git reports modified or untracked source")
    parser.add_argument("--github-output", help="Append GitHub Actions step outputs to this file")
    args = parser.parse_args()
    result = git_provenance(source_label=args.source_label)
    if args.verify_commit and result["git_commit"] != args.verify_commit:
        raise SystemExit("checked-out Git HEAD does not match the expected revision")
    if args.require_clean and result["git_dirty"]:
        raise SystemExit("Git worktree is dirty; refusing to label this CI image clean")
    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as handle:
            for key, value in result.items():
                handle.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
