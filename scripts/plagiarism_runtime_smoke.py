"""Load the staged MiniLM encoder on CPU without contacting Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from plagiarism.encoder_provenance import verify_encoder_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", required=True, type=Path)
    args = parser.parse_args()
    model_dir = args.encoder_dir
    manifest = verify_encoder_snapshot(model_dir)

    encoder = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True)
    vectors = encoder.encode(
        ["A short passage for the offline encoder check."],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if vectors.shape != (1, encoder.get_sentence_embedding_dimension()):
        raise SystemExit(f"unexpected embedding shape: {vectors.shape}")
    print(
        json.dumps(
            {
                "model_id": manifest["model_id"],
                "revision": manifest["revision"],
                "device": "cpu",
                "embedding_shape": list(vectors.shape),
                "normalized": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
