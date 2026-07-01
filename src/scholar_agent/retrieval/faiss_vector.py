"""FAISS vector search provider using BGE-M3 embeddings.

Loads pre-built FAISS index (568K arxiv papers) + BGE-M3 model for query encoding.
Provides semantic search to supplement keyword-based API providers — finds papers
that keyword search misses due to vocabulary mismatch.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from scholar_agent.infra.config import ProviderConfig
from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider

LOGGER = logging.getLogger(__name__)

# Default paths relative to project root
_DEFAULT_INDEX_DIR = "data/cache/vector_index"
_DEFAULT_DB_PATH = "data/pasa_local_fts.sqlite"
_DEFAULT_ID2PAPER_PATH = "data/id2paper.json"
_DEFAULT_MODEL_NAME = "C:/Users/33316/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"


class FaissVectorProvider(PaperProvider):
    """Semantic search via pre-built FAISS index + BGE-M3 query encoding."""

    name = "faiss_vector"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        base = config.base_url or "."
        self.index_dir = Path(base) / "data" / "cache" / "vector_index"
        if not (self.index_dir / "faiss.index").exists():
            self.index_dir = Path(_DEFAULT_INDEX_DIR)
        self.index_path = self.index_dir / "faiss.index"
        self.ids_path = self.index_dir / "arxiv_ids.npy"
        self.db_path = Path(base) / "data" / "pasa_local_fts.sqlite"
        if not self.db_path.exists():
            self.db_path = Path(_DEFAULT_DB_PATH)
        self.id2paper_path = Path(base) / "data" / "id2paper.json"
        if not self.id2paper_path.exists():
            self.id2paper_path = Path(_DEFAULT_ID2PAPER_PATH)
        self.model_name = config.model_name or _DEFAULT_MODEL_NAME
        self.device = "cuda"
        self.max_length = 256

        self._index: faiss.Index | None = None
        self._arxiv_ids: list[str] | None = None
        self._id2title: dict[str, str] = {}
        self._db_conn: sqlite3.Connection | None = None
        self._tokenizer = None
        self._model = None
        self._dim: int | None = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def is_available(self) -> bool:
        return self.index_path.exists() and self.ids_path.exists()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            LOGGER.info("Loading FAISS vector index from %s", self.index_path)
            self._index = faiss.read_index(str(self.index_path))
            self._arxiv_ids = np.load(
                str(self.ids_path), allow_pickle=True
            ).tolist()
            self._dim = self._index.d
            LOGGER.info(
                "FAISS index loaded: %d vectors, dim=%d", self._index.ntotal, self._dim
            )

            # Load id2paper.json for fast O(1) title lookup
            LOGGER.info("Loading id2paper.json from %s", self.id2paper_path)
            with open(self.id2paper_path, encoding="utf-8") as f:
                self._id2title = json.load(f)
            LOGGER.info("id2paper loaded: %d entries", len(self._id2title))

            # Open sqlite for abstract lookup (read-only, lazy)
            db_uri = f"file:{self.db_path}?mode=ro"
            self._db_conn = sqlite3.connect(db_uri, uri=True)

            # Load BGE-M3 model for query encoding
            LOGGER.info("Loading BGE-M3 model on %s", self.device)
            import torch
            from transformers import AutoTokenizer, AutoModel

            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(
                self.model_name, dtype=torch.float16
            )
            self._model.to(self.device)
            self._model.eval()
            LOGGER.info("BGE-M3 model loaded, dim=%d", self._dim)

            self._loaded = True

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a query string to a normalized vector using BGE-M3."""
        import torch

        encoded = self._tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = self._model(**encoded)

        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embedding = (outputs.last_hidden_state.float() * mask).sum(1) / mask.sum(1)
        vec = embedding.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def _batch_abstracts(self, arxiv_ids: list[str]) -> dict[str, str]:
        """Fetch abstracts for a small batch of arxiv_ids from sqlite.
        Only called for final top results (max 20) to avoid slow full-table scans."""
        if not arxiv_ids or not self._db_conn:
            return {}
        result: dict[str, str] = {}
        placeholders = ",".join("?" * len(arxiv_ids))
        cur = self._db_conn.cursor()
        try:
            cur.execute(
                f"SELECT c0, c2 FROM papers_fts_content "
                f"WHERE c0 IN ({placeholders})",
                arxiv_ids,
            )
            for row in cur:
                aid, abstract = row
                if abstract:
                    result[aid] = abstract
        except Exception as exc:
            LOGGER.warning("Abstract batch lookup failed: %s", exc)
        return result

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        """Semantic search: encode query → FAISS top-K → lookup metadata → return Papers."""
        self._ensure_loaded()
        if not self._index or not self._arxiv_ids:
            return []

        query_text = query.query
        if not query_text or len(query_text) < 3:
            return []

        try:
            q_vec = self._encode_query(query_text)
        except Exception as exc:
            LOGGER.warning("FAISS vector query encoding failed: %s", exc)
            return []

        # Search top-K (over-fetch to allow filtering)
        k = min(limit * 2, 100)
        scores, indices = self._index.search(q_vec, k)

        # Collect arxiv_ids with scores
        scored: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._arxiv_ids):
                continue
            aid = self._arxiv_ids[idx]
            scored.append((aid, float(score)))

        if not scored:
            return []

        # Fast title lookup via id2paper dict
        papers: list[Paper] = []
        for aid, score in scored:
            title = self._id2title.get(aid, "")
            if not title:
                continue
            paper = Paper(
                paper_id=f"arxiv:{aid}",
                title=title,
                arxiv_id=aid,
                source=self.name,
                retrieval_path=[
                    f"provider:{self.name}",
                    f"route:{query.route}",
                    f"semantic_score:{score:.4f}",
                ],
                metadata={
                    "vector_score": score,
                    "semantic_search": True,
                    "query_text": query_text,
                },
            )
            papers.append(paper)
            if len(papers) >= limit:
                break

        # Skip abstract lookup for speed — title + vector_score is enough for candidate pool
        # Abstracts can be fetched later by the pipeline if needed

        LOGGER.info(
            "FAISS vector search: query='%s' → %d papers (top score=%.4f)",
            query_text[:60],
            len(papers),
            scored[0][1] if scored else 0,
        )
        return papers

    def search_title_exact(self, title: str, limit: int = 10) -> list[Paper]:
        """Use vector search as a fallback for title matching."""
        self._ensure_loaded()
        if not self._index:
            return []
        try:
            q_vec = self._encode_query(title)
        except Exception:
            return []
        scores, indices = self._index.search(q_vec, limit)
        papers = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._arxiv_ids):
                continue
            aid = self._arxiv_ids[idx]
            title_str = self._id2title.get(aid, "")
            if not title_str:
                continue
            papers.append(Paper(
                paper_id=f"arxiv:{aid}",
                title=title_str,
                arxiv_id=aid,
                source=self.name,
                retrieval_path=[f"provider:{self.name}", "route:title_exact"],
                metadata={"vector_score": float(score)},
            ))
            if len(papers) >= limit:
                break
        return papers
