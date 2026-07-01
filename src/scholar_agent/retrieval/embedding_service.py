"""Embedding service for semantic retrieval.

Uses transformers AutoModel + mean pooling (bypasses sentence-transformers
which segfaults on this environment). Supports BGE-M3 and other HF models.
Disk caching avoids re-embedding the same papers across runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

from scholar_agent.models.schemas import Paper

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512
DEFAULT_CACHE_DIR = "data/cache/embeddings"


def _text_hash(text: str, model_name: str = "") -> str:
    return hashlib.md5(f"{model_name}:{text}".encode("utf-8")).hexdigest()


class EmbeddingService:
    """Lazy-loaded embedding service with disk caching.

    Uses transformers AutoModel + AutoTokenizer directly, with manual
    mean pooling (attention-mask weighted). This bypasses sentence-transformers
    which segfaults in certain environment configurations.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    LOGGER.info(
                        "Loading embedding model: %s on %s",
                        self.model_name,
                        self.device,
                    )
                    os.environ.setdefault(
                        "HF_HUB_DISABLE_SYMLINKS_WARNING", "1"
                    )
                    from transformers import AutoTokenizer, AutoModel
                    import torch

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name
                    )
                    self._model = AutoModel.from_pretrained(
                        self.model_name,
                        dtype=torch.float32,
                    )
                    self._model.to(self.device)
                    self._model.eval()
                    self._dim = self._model.config.hidden_size
                    LOGGER.info(
                        "Model loaded: %s, dim=%d",
                        type(self._model).__name__,
                        self._dim,
                    )
        return self._model

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._ensure_model()
        assert self._dim is not None
        return self._dim

    def _paper_text(self, paper: Paper) -> str:
        """Build text for embedding: title + abstract."""
        parts = [paper.title or ""]
        if paper.abstract:
            parts.append(paper.abstract)
        return " ".join(parts).strip()

    def _cache_path(self, text_hash: str) -> Path:
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{text_hash}.json"

    def _get_cached(self, text_hash: str) -> np.ndarray | None:
        path = self._cache_path(text_hash)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return np.array(data["embedding"], dtype=np.float32)
            except Exception:
                pass
        return None

    def _set_cached(self, text_hash: str, embedding: np.ndarray) -> None:
        path = self._cache_path(text_hash)
        try:
            path.write_text(
                json.dumps({"embedding": embedding.tolist()}),
                encoding="utf-8",
            )
        except Exception as e:
            LOGGER.warning("Failed to cache embedding: %s", e)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts using AutoModel + mean pooling."""
        import torch

        model = self._ensure_model()
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean pooling: attention-mask weighted average of token embeddings
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return embeddings.cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, dim) float32 array."""
        if not texts:
            dim = self._dim or 1024
            return np.array([], dtype=np.float32).reshape(0, dim)

        self._ensure_model()

        # Check cache first
        results: list[np.ndarray | None] = [None] * len(texts)
        hashes = [_text_hash(t, self.model_name) for t in texts]
        cache_hits = 0
        for i, h in enumerate(hashes):
            cached = self._get_cached(h)
            if cached is not None:
                results[i] = cached
                cache_hits += 1

        # Find uncached texts
        uncached_indices = [i for i, r in enumerate(results) if r is None]
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            LOGGER.info(
                "Embedding %d texts (cache hits: %d/%d)",
                len(uncached_texts),
                cache_hits,
                len(texts),
            )

            # Process in batches
            all_vecs: list[np.ndarray] = []
            for start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[start : start + self.batch_size]
                vecs = self._encode_batch(batch)
                all_vecs.append(vecs)

            dense_vecs = np.concatenate(all_vecs, axis=0)

            for idx, vec_idx in enumerate(uncached_indices):
                vec = np.array(dense_vecs[idx], dtype=np.float32)
                results[vec_idx] = vec
                self._set_cached(hashes[vec_idx], vec)

        return np.stack(results)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns (dim,) float32 array."""
        vecs = self.embed_texts([query])
        return vecs[0]

    def embed_papers(self, papers: list[Paper]) -> np.ndarray:
        """Embed a list of papers (title + abstract). Returns (N, dim) array."""
        texts = [self._paper_text(p) for p in papers]
        return self.embed_texts(texts)

    def embed_paper(self, paper: Paper) -> np.ndarray:
        """Embed a single paper. Returns (dim,) float32 array."""
        return self.embed_query(self._paper_text(paper))
