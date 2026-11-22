"""Deterministic passage chunking, embedding, retrieval, and source-vector cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Protocol, Sequence

import numpy as np


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class Candidate:
    query_chunk: Chunk
    source_chunk: Chunk
    cosine: float

    @property
    def query_chunk_id(self) -> str:
        return self.query_chunk.chunk_id

    @property
    def source_chunk_id(self) -> str:
        return self.source_chunk.chunk_id


class TextEncoder(Protocol):
    revision: str

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def _tokenize_with_offsets(text: str, tokenizer: Any) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except (TypeError, KeyError, AttributeError) as exc:
        raise TypeError("chunk tokenizer must return input_ids and offset_mapping") from exc
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    return list(ids), [(int(a), int(b)) for a, b in offsets]


def _special_token_count(tokenizer: Any) -> int:
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if builder is None:
        return 0
    return len(builder([]))


def chunk_text(
    text: str,
    document_id: str,
    tokenizer: Any,
    *,
    target_tokens: int = 180,
    overlap_tokens: int = 40,
    max_tokens: int = 256,
) -> list[Chunk]:
    """Split text into token windows while preserving exact half-open char offsets.

    Target and overlap count ordinary tokenizer tokens. Special tokens are added
    separately for the MiniLM limit check.
    """
    if target_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("require target_tokens > overlap_tokens >= 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    token_ids, offsets = _tokenize_with_offsets(text, tokenizer)
    if len(token_ids) != len(offsets):
        raise ValueError("tokenizer returned misaligned token IDs and offsets")
    special_count = _special_token_count(tokenizer)
    window = min(target_tokens, max_tokens - special_count)
    if window <= 0:
        raise ValueError("special tokens consume the entire model token budget")
    if target_tokens + special_count > max_tokens:
        # Honor the requested target only when it fits; otherwise cap safely.
        window = max_tokens - special_count
    stride = window - min(overlap_tokens, window - 1)
    chunks: list[Chunk] = []
    token_start = 0
    while token_start < len(token_ids):
        token_end = min(len(token_ids), token_start + window)
        while True:
            visible = [(a, b) for a, b in offsets[token_start:token_end] if b > a]
            if visible:
                start_char, end_char = visible[0][0], visible[-1][1]
            else:
                start_char = end_char = 0
            chunk_value = text[start_char:end_char]
            # Slicing through a subword boundary can change tokenization. Recount
            # the actual substring and shrink until special tokens also fit.
            actual_ids, _ = _tokenize_with_offsets(chunk_value, tokenizer)
            if len(actual_ids) + special_count <= max_tokens:
                break
            token_end -= 1
            if token_end <= token_start:
                raise ValueError("tokenizer cannot produce a chunk within the model token budget")
        chunk_id = f"{document_id}:{start_char}:{end_char}"
        chunks.append(Chunk(chunk_id, str(document_id), chunk_value, start_char,
                            end_char, token_start, token_end))
        if token_end == len(token_ids):
            break
        token_start = max(token_start + 1, token_end - min(overlap_tokens, token_end - token_start - 1))
    return chunks


def chunking_config(*, target_tokens: int = 180, overlap_tokens: int = 40, max_tokens: int = 256) -> dict[str, int]:
    return {"target_tokens": target_tokens, "overlap_tokens": overlap_tokens, "max_tokens": max_tokens}


class FixtureEncoder:
    """Deterministic lexical hash encoder for offline fixtures; not MiniLM."""

    revision = "fixture-hash-encoder-v1"

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 8:
            raise ValueError("fixture encoder needs at least 8 dimensions")
        self.dimensions = dimensions
        self.tokenizer = FixtureTokenizer()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = re.findall(r"[A-Za-z0-9]+", text.casefold())
            for term in terms:
                digest = hashlib.sha256(term.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vectors[row, index] += 1.0 if digest[4] & 1 else -1.0
            norm = float(np.linalg.norm(vectors[row]))
            if norm:
                vectors[row] /= norm
        return vectors


class FixtureTokenizer:
    """Word tokenizer with offsets for deterministic fixture chunking."""

    def __call__(self, text: str, *, add_special_tokens: bool = False,
                 return_offsets_mapping: bool = False, truncation: bool = False) -> dict[str, Any]:
        matches = list(re.finditer(r"\S+", text))
        result: dict[str, Any] = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result

    @staticmethod
    def build_inputs_with_special_tokens(token_ids: list[int]) -> list[int]:
        return [101, *token_ids, 102]


class MiniLMEncoder:
    """Frozen, local-only MiniLM encoder. Model weights must already be staged."""

    def __init__(self, model_path: str | Path, revision: str, *, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("MiniLM requires the separate sentence-transformers runtime") from exc
        from .encoder_provenance import verify_encoder_snapshot

        self.snapshot_manifest = verify_encoder_snapshot(model_path, expected_revision=revision)
        self.revision = revision
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            str(model_path or MINILM_MODEL), device="cpu", revision=revision, local_files_only=True
        )
        self.tokenizer = self.model.tokenizer
        limit = int(getattr(self.model, "max_seq_length", 256))
        if limit < 256:
            raise ValueError(f"encoder limit is {limit}, expected at least 256 tokens")

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dimension = int(self.model.get_sentence_embedding_dimension())
            return np.empty((0, dimension), dtype=np.float32)
        values = self.model.encode(
            list(texts), batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(values, dtype=np.float32)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)


def retrieve(
    query_chunks: Sequence[Chunk],
    source_chunks: Sequence[Chunk],
    query_vectors: np.ndarray,
    source_vectors: np.ndarray,
    *,
    top_k: int = 10,
    allowed_source_ids: Iterable[str] | None = None,
) -> list[Candidate]:
    """Exact cosine top-k for each query chunk with stable source/chunk ties."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    qv, sv = _normalize_rows(query_vectors), _normalize_rows(source_vectors)
    if len(query_chunks) != len(qv) or len(source_chunks) != len(sv):
        raise ValueError("chunk and embedding row counts must align")
    if qv.shape[1] != sv.shape[1]:
        raise ValueError("query and source embedding dimensions must match")
    allow = None if allowed_source_ids is None else {str(item) for item in allowed_source_ids}
    source_order = [i for i, chunk in enumerate(source_chunks) if allow is None or chunk.document_id in allow]
    if not source_order:
        return []
    selected_vectors = sv[source_order]
    output: list[Candidate] = []
    for qi, query in enumerate(query_chunks):
        # One query row at a time bounds intermediate memory for large indexes.
        row_scores = selected_vectors @ qv[qi]
        ranked = sorted(
            ((float(row_scores[j]), source_chunks[source_index]) for j, source_index in enumerate(source_order)),
            key=lambda item: (-item[0], item[1].document_id, item[1].chunk_id),
        )[:top_k]
        output.extend(Candidate(query, source, score) for score, source in ranked)
    return output


def source_cache_key(source_manifest_hash: str, encoder_revision: str, config: dict[str, Any]) -> str:
    payload = {"source_manifest_hash": source_manifest_hash, "encoder_revision": encoder_revision,
               "chunking_config": config}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_source_cache(path: str | Path, chunks: Sequence[Chunk], vectors: np.ndarray, *,
                      source_manifest_hash: str, encoder_revision: str, config: dict[str, Any]) -> None:
    """Write cache arrays and verification metadata to a single compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_rows(vectors)
    if len(chunks) != len(normalized):
        raise ValueError("chunk and embedding row counts must align")
    chunk_records = [asdict(item) for item in chunks]
    vec_digest = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
    chunk_blob = json.dumps(chunk_records, sort_keys=True, separators=(",", ":"))
    metadata = {"schema_version": 1, "cache_key": source_cache_key(source_manifest_hash, encoder_revision, config),
                "source_manifest_hash": source_manifest_hash, "encoder_revision": encoder_revision,
                "chunking_config": config, "vector_sha256": vec_digest,
                "chunk_sha256": hashlib.sha256(chunk_blob.encode()).hexdigest()}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, chunks=np.asarray(chunk_blob), vectors=normalized,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_source_cache(path: str | Path, *, source_manifest_hash: str, encoder_revision: str,
                      config: dict[str, Any]) -> tuple[list[Chunk], np.ndarray]:
    """Load only if manifest, encoder, chunking, and vector integrity all match."""
    expected_key = source_cache_key(source_manifest_hash, encoder_revision, config)
    with np.load(Path(path), allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        if metadata.get("schema_version") != 1 or metadata.get("cache_key") != expected_key:
            raise ValueError("source embedding cache metadata does not match requested inputs")
        vectors = np.asarray(cache["vectors"], dtype=np.float32)
        if hashlib.sha256(vectors.tobytes(order="C")).hexdigest() != metadata.get("vector_sha256"):
            raise ValueError("source embedding cache failed integrity verification")
        chunk_blob = str(cache["chunks"].item())
        if hashlib.sha256(chunk_blob.encode()).hexdigest() != metadata.get("chunk_sha256"):
            raise ValueError("source embedding cache failed chunk integrity verification")
        chunks = [Chunk(**record) for record in json.loads(chunk_blob)]
    if len(chunks) != len(vectors):
        raise ValueError("source embedding cache rows do not align")
    return chunks, vectors
