"""Verify the immutable, hash-manifested local MiniLM snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MANIFEST_NAME = "STAGING_MANIFEST.json"
PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REQUIRED_FILES = {"modules.json", "config.json", "tokenizer.json", "1_Pooling/config.json"}
WEIGHT_FILES = {"model.safetensors", "pytorch_model.bin"}
ALLOWED_FILES = REQUIRED_FILES | WEIGHT_FILES | {
    "config_sentence_transformers.json", "sentence_bert_config.json",
    "special_tokens_map.json", "tokenizer_config.json", "vocab.txt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_encoder_snapshot(
    model_dir: str | Path,
    expected_revision: str | None = PINNED_REVISION,
) -> dict[str, Any]:
    """Verify local model bytes and return the validated manifest.

    `expected_revision` should come from the model/configuration bundle when
    available. When omitted, the revision recorded in the verified manifest is
    returned for callers that resolve it from the snapshot itself.
    """
    root = Path(model_dir).resolve(strict=True)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError(f"unexpected encoder model_id: {manifest.get('model_id')!r}")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("encoder manifest must name an exact lowercase 40-character revision")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(f"encoder revision mismatch: expected {expected_revision}, got {revision}")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("encoder manifest has no files_sha256 entries")
    names = set(files)
    missing = sorted(REQUIRED_FILES - names)
    if missing:
        raise ValueError(f"encoder manifest is missing required files: {', '.join(missing)}")
    weights = names & WEIGHT_FILES
    if len(weights) != 1:
        raise ValueError("encoder manifest must contain exactly one supported model weights file")
    unexpected = sorted(names - ALLOWED_FILES)
    if unexpected:
        raise ValueError(f"encoder manifest contains non-allowlisted model files: {', '.join(unexpected)}")

    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("encoder manifest entries must map relative paths to SHA-256 strings")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe path in encoder manifest: {relative!r}")
        path = (root / relative_path).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError(f"encoder manifest path is not a file inside the snapshot: {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or _sha256(path) != expected_hash:
            raise ValueError(f"encoder snapshot file hash mismatch: {relative}")
    return manifest
