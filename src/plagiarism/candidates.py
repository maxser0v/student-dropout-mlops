"""Prepare split-isolated MiniLM candidate rows and reusable source index."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data import Corpus
from .retrieval import (Chunk, TextEncoder, chunk_text, chunking_config,
                       load_source_cache, retrieve, save_source_cache)


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_source_index(corpus: Corpus, encoder: TextEncoder, *, split_names: tuple[str, ...],
                       cache_path: str | Path, source_manifest_hash: str,
                       target_tokens: int = 180, overlap_tokens: int = 40,
                       max_tokens: int = 256) -> tuple[list[Chunk], np.ndarray, dict[str, Any]]:
    """Build a non-test fixed source index and reuse it only on exact cache match."""
    cfg = chunking_config(target_tokens=target_tokens, overlap_tokens=overlap_tokens,
                          max_tokens=max_tokens)
    cache_path = Path(cache_path)
    documents = [corpus.documents[i] for split in split_names for i in corpus.splits.get(split, ())
                 if corpus.documents[i].role == "source"]
    documents = sorted({doc.id: doc for doc in documents}.values(), key=lambda doc: doc.id)
    started = time.perf_counter()
    try:
        chunks, vectors = load_source_cache(cache_path, source_manifest_hash=source_manifest_hash,
                                            encoder_revision=encoder.revision, config=cfg)
        cache_hit = True
    except (ValueError, OSError, KeyError, FileNotFoundError):
        chunks = [chunk for doc in documents for chunk in chunk_text(
            doc.text, doc.id, encoder.tokenizer, target_tokens=target_tokens,
            overlap_tokens=overlap_tokens, max_tokens=max_tokens)]
        vectors = encoder.encode_texts([chunk.text for chunk in chunks])
        save_source_cache(cache_path, chunks, vectors, source_manifest_hash=source_manifest_hash,
                          encoder_revision=encoder.revision, config=cfg)
        cache_hit = False
    return chunks, vectors, {"source_chunk_count": len(chunks), "source_document_count": len(documents),
                             "source_manifest_hash": source_manifest_hash,
                             "encoder_revision": encoder.revision, "chunking_config": cfg,
                             "cache_hit": cache_hit, "source_index_seconds": time.perf_counter() - started}


def prepare_split_candidates(corpus: Corpus, encoder: TextEncoder, *, split: str,
                             top_k: int = 10, target_tokens: int = 180,
                             overlap_tokens: int = 40, max_tokens: int = 256,
                             cache_path: str | Path | None = None,
                             source_manifest_hash: str | None = None,
                             source_index: tuple[list[Chunk], np.ndarray] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chunk and retrieve per PAN query using only its judged source documents.

    Sources are indexed once; each query is then restricted to the source IDs
    supplied in that split's pair manifest to avoid treating unknown pairings
    as negative judgments.
    """
    if split not in corpus.splits:
        raise ValueError(f"corpus has no {split!r} split")
    cfg = chunking_config(target_tokens=target_tokens, overlap_tokens=overlap_tokens,
                          max_tokens=max_tokens)
    started = time.perf_counter()
    source_documents = [corpus.documents[i] for i in corpus.splits[split]
                        if corpus.documents[i].role == "source"]
    source_manifest_hash = source_manifest_hash or hashlib.sha256(
        json.dumps([(doc.id, doc.content_hash) for doc in source_documents], separators=(",", ":")).encode()
    ).hexdigest()
    source_chunks: list[Chunk] = []
    source_vectors: np.ndarray
    cached = False
    cache_path = Path(cache_path) if cache_path is not None else None
    if source_index is not None:
        source_chunks, source_vectors = source_index
        cached = True
    elif cache_path is not None and cache_path.exists():
        try:
            source_chunks, source_vectors = load_source_cache(
                cache_path, source_manifest_hash=source_manifest_hash,
                encoder_revision=encoder.revision, config=cfg)
            cached = True
        except (ValueError, OSError, KeyError):
            # A caller may be sharing this path with another run. Rebuild in
            # memory and let the atomic cache writer replace it when safe.
            pass
    if not cached:
        for doc in source_documents:
            source_chunks.extend(chunk_text(doc.text, doc.id, encoder.tokenizer,
                                            target_tokens=target_tokens,
                                            overlap_tokens=overlap_tokens,
                                            max_tokens=max_tokens))
        source_vectors = encoder.encode_texts([chunk.text for chunk in source_chunks])
        if cache_path is not None:
            save_source_cache(cache_path, source_chunks, source_vectors,
                              source_manifest_hash=source_manifest_hash,
                              encoder_revision=encoder.revision, config=cfg)
    cold_seconds = time.perf_counter() - started

    query_pairs: dict[str, set[str]] = {}
    for pair in corpus.pairs:
        if pair.split == split:
            query_pairs.setdefault(pair.submission_id, set()).add(pair.source_id)
    rows: list[dict[str, Any]] = []
    encode_seconds = 0.0
    retrieve_seconds = 0.0
    for query_id in sorted(query_pairs):
        document = corpus.documents[query_id]
        query_chunks = chunk_text(document.text, query_id, encoder.tokenizer,
                                  target_tokens=target_tokens,
                                  overlap_tokens=overlap_tokens,
                                  max_tokens=max_tokens)
        encode_start = time.perf_counter()
        query_vectors = encoder.encode_texts([chunk.text for chunk in query_chunks])
        encode_seconds += time.perf_counter() - encode_start
        retrieve_start = time.perf_counter()
        # Restrict before normalization/matrix multiplication. PAN may have a
        # very large source pool but each query has only a small judged set.
        allowed = query_pairs[query_id]
        source_indices = [i for i, chunk in enumerate(source_chunks) if chunk.document_id in allowed]
        allowed_chunks = [source_chunks[i] for i in source_indices]
        allowed_vectors = source_vectors[source_indices]
        candidates = retrieve(query_chunks, allowed_chunks, query_vectors, allowed_vectors,
                              top_k=top_k)
        retrieve_seconds += time.perf_counter() - retrieve_start
        rows.extend({"query_chunk": item.query_chunk, "source_chunk": item.source_chunk,
                     "cosine": item.cosine, "minilm_cosine": item.cosine,
                     "query_text": item.query_chunk.text, "source_text": item.source_chunk.text,
                     "query_document_id": query_id,
                     "allowed_source_ids": sorted(query_pairs[query_id])}
                    for item in candidates)
    return rows, {"split": split, "query_count": len(query_pairs),
                  "candidate_count": len(rows), "source_chunk_count": len(source_chunks),
                  "source_manifest_hash": source_manifest_hash,
                  "encoder_revision": encoder.revision, "chunking_config": cfg,
                  "cache_hit": cached, "source_index_seconds": cold_seconds,
                  "query_encode_seconds": encode_seconds,
                  "retrieval_seconds": retrieve_seconds,
                  "source_cache_path": str(cache_path) if cache_path else None}


def prepare_tfidf_candidates(corpus: Corpus, encoder: TextEncoder, vectorizer: Any, *, split: str,
                             top_k: int = 10, target_tokens: int = 180,
                             overlap_tokens: int = 40, max_tokens: int = 256) -> list[dict[str, Any]]:
    """Independent exact TF-IDF top-k retrieval on each judged source pool."""
    source_docs = [corpus.documents[i] for i in corpus.splits[split]
                   if corpus.documents[i].role == "source"]
    source_chunks = [chunk for doc in source_docs for chunk in chunk_text(
        doc.text, doc.id, encoder.tokenizer, target_tokens=target_tokens,
        overlap_tokens=overlap_tokens, max_tokens=max_tokens)]
    by_query: dict[str, set[str]] = {}
    for pair in corpus.pairs:
        if pair.split == split:
            by_query.setdefault(pair.submission_id, set()).add(pair.source_id)
    output = []
    for query_id in sorted(by_query):
        doc = corpus.documents[query_id]
        query_chunks = chunk_text(doc.text, doc.id, encoder.tokenizer,
                                  target_tokens=target_tokens, overlap_tokens=overlap_tokens,
                                  max_tokens=max_tokens)
        sources = [chunk for chunk in source_chunks if chunk.document_id in by_query[query_id]]
        if not sources:
            continue
        vectors = vectorizer.transform([x.text for x in query_chunks] + [x.text for x in sources])
        n = len(query_chunks)
        similarities = cosine_similarity(vectors[:n], vectors[n:])
        for qidx, query in enumerate(query_chunks):
            order = sorted(range(len(sources)), key=lambda i: (-float(similarities[qidx, i]),
                                                                sources[i].document_id, sources[i].chunk_id))[:top_k]
            output.extend({"query_chunk": query, "source_chunk": sources[i],
                           "cosine": float(similarities[qidx, i]),
                           "tfidf_cosine": float(similarities[qidx, i]),
                           "query_text": query.text, "source_text": sources[i].text,
                           "query_document_id": query_id}
                          for i in order)
    return output
