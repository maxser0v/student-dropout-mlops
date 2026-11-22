"""Download one immutable MiniLM snapshot into a Docker bind-mount directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Keep runtime artifacts limited to the files Sentence Transformers needs.
# In particular, do not stage ONNX, OpenVINO, TensorFlow, or alternate exports.
ALLOWED_FILES = {
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "model.safetensors",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Exact 40-character Hugging Face commit")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("revision must be an exact lowercase 40-character commit hash")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "STAGING_MANIFEST.json"
    existing = [item for item in output.iterdir() if item.name != manifest_path.name]
    if existing:
        raise SystemExit(f"refusing to stage into non-empty directory: {output}")

    weight_files = ("model.safetensors", "pytorch_model.bin")
    repo_files = set(list_repo_files(MODEL_ID, revision=args.revision))
    selected_weight = next((name for name in weight_files if name in repo_files), None)
    if selected_weight is None:
        raise SystemExit("model snapshot contains no supported model weights file")
    download_files = (ALLOWED_FILES - set(weight_files)) | {selected_weight}
    snapshot_download(
        repo_id=MODEL_ID,
        revision=args.revision,
        local_dir=output,
        local_files_only=False,
        allow_patterns=sorted(download_files),
    )
    required = ("modules.json", "config.json", "tokenizer.json", "1_Pooling/config.json")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"model snapshot is missing required files: {', '.join(missing)}")
    weights = [name for name in weight_files if (output / name).is_file()]
    if len(weights) != 1:
        raise SystemExit("model snapshot must contain exactly one supported model weights file")

    files = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path and ".cache" not in path.relative_to(output).parts
    }
    unexpected = sorted(set(files) - ALLOWED_FILES)
    if unexpected:
        raise SystemExit(f"model snapshot contains non-allowlisted files: {', '.join(unexpected)}")
    manifest_path.write_text(
        json.dumps(
            {"model_id": MODEL_ID, "revision": args.revision, "files_sha256": files},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Staged {MODEL_ID}@{args.revision} in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
