"""Train-only text transforms and shared pair-feature generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import json
import os
from pathlib import Path
import tempfile
import re
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


FEATURE_NAMES = (
    "minilm_cosine",
    "tfidf_cosine",
    "shared_word_ngram_overlap",
    "rare_term_overlap",
    "length_ratio",
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", re.UNICODE)


def _words(text: str) -> list[str]:
    return [word.casefold().replace("’", "'") for word in _WORD_RE.findall(text)]


def _ngrams(words: Sequence[str], max_n: int = 3) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for n in range(1, max_n + 1) for i in range(max(0, len(words) - n + 1))}


def _overlap(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


@dataclass(frozen=True)
class TextPair:
    query_text: str
    source_text: str
    minilm_cosine: float = 0.0


class FeaturePipeline:
    """Fitted training-text transforms, reused unchanged for validation/scoring."""

    schema_version = "1.0"
    feature_names = FEATURE_NAMES

    def __init__(self, *, rare_document_fraction: float = 0.10, max_ngram: int = 3) -> None:
        if not 0 < rare_document_fraction <= 1:
            raise ValueError("rare_document_fraction must be in (0, 1]")
        if max_ngram < 1:
            raise ValueError("max_ngram must be positive")
        self.rare_document_fraction = rare_document_fraction
        self.max_ngram = max_ngram
        self.vectorizer: TfidfVectorizer | None = None
        self.rare_terms: frozenset[str] = frozenset()
        self.training_document_count = 0
        self.is_fitted = False

    def fit(self, training_texts: Iterable[str]) -> "FeaturePipeline":
        texts = [str(text) for text in training_texts]
        if not texts:
            raise ValueError("at least one training text is required")
        self.training_document_count = len(texts)
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\b\w\w+\b")
        try:
            self.vectorizer.fit(texts)
        except ValueError as exc:
            raise ValueError("training text must contain at least one usable term") from exc
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(_words(text)))
        cutoff = max(1, math.floor(len(texts) * self.rare_document_fraction))
        self.rare_terms = frozenset(term for term, count in document_frequency.items() if count <= cutoff)
        self.is_fitted = True
        return self

    def transform_pairs(self, pairs: Sequence[TextPair | Any]) -> np.ndarray:
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("FeaturePipeline.fit must be called before transform_pairs")
        normalized: list[TextPair] = []
        for pair in pairs:
            if isinstance(pair, TextPair):
                normalized.append(pair)
            elif hasattr(pair, "query_chunk") and hasattr(pair, "source_chunk"):
                normalized.append(TextPair(pair.query_chunk.text, pair.source_chunk.text, float(pair.cosine)))
            elif isinstance(pair, dict):
                normalized.append(TextPair(str(pair["query_text"]), str(pair["source_text"]),
                                           float(pair.get("minilm_cosine", pair.get("cosine", 0.0)))))
            else:
                q, s, *score = pair
                normalized.append(TextPair(str(q), str(s), float(score[0]) if score else 0.0))
        if not normalized:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        texts = [item.query_text for item in normalized] + [item.source_text for item in normalized]
        vectors = self.vectorizer.transform(texts)
        n = len(normalized)
        # L2-normalized TF-IDF makes pairwise cosine the row-wise sparse dot
        # product. This avoids materializing an N by N dense matrix.
        tfidf_cos = np.asarray(vectors[:n].multiply(vectors[n:]).sum(axis=1)).ravel()
        rows: list[list[float]] = []
        for i, pair in enumerate(normalized):
            qwords, swords = _words(pair.query_text), _words(pair.source_text)
            qgrams, sgrams = _ngrams(qwords, self.max_ngram), _ngrams(swords, self.max_ngram)
            qrare = {word for word in qwords if word in self.rare_terms}
            srare = {word for word in swords if word in self.rare_terms}
            qlen, slen = len(qwords), len(swords)
            rows.append([
                float(pair.minilm_cosine),
                float(tfidf_cos[i]),
                _overlap(qgrams, sgrams),
                _overlap(qrare, srare),
                float(min(qlen, slen) / max(qlen, slen)) if max(qlen, slen) else 0.0,
            ])
        result = np.asarray(rows, dtype=np.float32)
        if result.shape != (len(normalized), len(FEATURE_NAMES)) or not np.isfinite(result).all():
            raise ValueError("feature generation returned invalid values or schema")
        return result

    def save(self, path: str | Path) -> Path:
        """Persist fitted transforms as JSON state, without executable pickle."""
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted FeaturePipeline")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self.vectorizer is not None
        vectorizer = self.vectorizer
        payload = {
            "schema_version": self.schema_version,
            "feature_names": list(FEATURE_NAMES),
            "rare_document_fraction": self.rare_document_fraction,
            "max_ngram": self.max_ngram,
            "rare_terms": sorted(self.rare_terms),
            "training_document_count": self.training_document_count,
            "vectorizer": {
                "lowercase": bool(vectorizer.lowercase),
                "token_pattern": vectorizer.token_pattern,
                "ngram_range": list(vectorizer.ngram_range),
                "vocabulary": dict(sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])),
                "idf": [float(value) for value in vectorizer.idf_],
                "norm": vectorizer.norm,
                "use_idf": bool(vectorizer.use_idf),
                "smooth_idf": bool(vectorizer.smooth_idf),
                "sublinear_tf": bool(vectorizer.sublinear_tf),
            },
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=False, sort_keys=True, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("schema_version") != cls.schema_version or tuple(state.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("saved text preprocessing schema is incompatible")
        pipeline = cls(rare_document_fraction=state["rare_document_fraction"], max_ngram=state["max_ngram"])
        vectorizer_state = state["vectorizer"]
        vocabulary = vectorizer_state.get("vocabulary")
        idf = np.asarray(vectorizer_state.get("idf", []), dtype=np.float64)
        if not isinstance(vocabulary, dict) or len(vocabulary) != len(idf) or not np.isfinite(idf).all():
            raise ValueError("saved TF-IDF vocabulary and IDF state are invalid")
        vectorizer = TfidfVectorizer(
            lowercase=vectorizer_state["lowercase"],
            token_pattern=vectorizer_state["token_pattern"],
            ngram_range=tuple(vectorizer_state["ngram_range"]),
            vocabulary=vocabulary,
            norm=vectorizer_state["norm"],
            use_idf=vectorizer_state["use_idf"],
            smooth_idf=vectorizer_state["smooth_idf"],
            sublinear_tf=vectorizer_state["sublinear_tf"],
        )
        vectorizer._validate_vocabulary()
        vectorizer._tfidf = TfidfTransformer(
            norm=vectorizer.norm, use_idf=vectorizer.use_idf,
            smooth_idf=vectorizer.smooth_idf, sublinear_tf=vectorizer.sublinear_tf,
        )
        vectorizer._tfidf.idf_ = idf
        pipeline.vectorizer = vectorizer
        pipeline.rare_terms = frozenset(state["rare_terms"])
        pipeline.training_document_count = int(state["training_document_count"])
        pipeline.is_fitted = True
        return pipeline
