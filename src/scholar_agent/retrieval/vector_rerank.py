"""Vector semantic re-ranking using BGE-M3 + FAISS.

Provides semantic similarity search over the candidate pool,
bridging vocabulary mismatch that keyword search cannot handle.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import faiss
import numpy as np

from scholar_agent.models.schemas import Paper
from scholar_agent.retrieval.embedding_service import EmbeddingService
from scholar_agent.retrieval.rrf_fusion import _paper_key

LOGGER = logging.getLogger(__name__)


class VectorReranker:
    """Build a FAISS index from candidate papers and do semantic search."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedder = embedding_service
        self._index: faiss.IndexFlatIP | None = None
        self._papers: list[Paper] = []
        self._embeddings: np.ndarray | None = None

    def build_index(self, papers: list[Paper]) -> None:
        """Embed all papers and build a FAISS inner-product index."""
        if not papers:
            LOGGER.warning("VectorReranker: empty paper list, skipping index build")
            return

        t0 = time.time()
        LOGGER.info("VectorReranker: embedding %d papers for FAISS index", len(papers))
        self._papers = papers
        self._embeddings = self.embedder.embed_papers(papers)

        # Normalize for cosine similarity (inner product on normalized vectors)
        faiss.normalize_L2(self._embeddings)

        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)

        elapsed = time.time() - t0
        LOGGER.info(
            "VectorReranker: FAISS index built (%d papers, %d dim, %.1fs)",
            len(papers),
            dim,
            elapsed,
        )

    def search(
        self,
        query_text: str,
        limit: int = 50,
    ) -> list[tuple[Paper, float]]:
        """Search the index with a text query. Returns (paper, score) pairs."""
        if self._index is None or not self._papers:
            LOGGER.warning("VectorReranker: index not built, returning empty")
            return []

        query_vec = self.embedder.embed_query(query_text)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, min(limit, len(self._papers)))

        results: list[tuple[Paper, float]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            paper = self._papers[idx]
            results.append((paper, float(score)))

        return results

    def search_multi(
        self,
        query_texts: list[str],
        limit: int = 50,
    ) -> list[tuple[Paper, float]]:
        """Search with multiple query texts and aggregate scores.

        Each query is searched independently, scores are summed
        (equivalent to RRF fusion at the vector level).
        """
        if self._index is None or not self._papers:
            return []

        # Embed all queries
        query_vecs = self.embedder.embed_texts(query_texts)
        query_vecs = query_vecs.astype(np.float32)
        faiss.normalize_L2(query_vecs)

        # Search each query
        all_scores: dict[str, float] = {}
        paper_by_key: dict[str, Paper] = {}

        for q_idx in range(len(query_texts)):
            q_vec = query_vecs[q_idx : q_idx + 1]
            scores, indices = self._index.search(q_vec, min(limit, len(self._papers)))
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < 0:
                    continue
                paper = self._papers[idx]
                key = _paper_key(paper)
                # Use raw cosine similarity score
                all_scores[key] = all_scores.get(key, 0.0) + float(score)
                if key not in paper_by_key:
                    paper_by_key[key] = paper

        # Sort by aggregated score
        sorted_keys = sorted(all_scores.keys(), key=lambda k: -all_scores[k])
        return [(paper_by_key[k], all_scores[k]) for k in sorted_keys[:limit]]

    def rerank(
        self,
        papers: list[Paper],
        query_texts: list[str],
        limit: int = 100,
    ) -> list[Paper]:
        """Build index from papers and return vector-ranked papers.

        Args:
            papers: Candidate pool from keyword search.
            query_texts: List of query texts (original query, HyDE document, etc.)
            limit: Maximum papers to return.

        Returns:
            Papers sorted by vector similarity score (descending).
            Each paper's metadata["vector_score"] is set.
        """
        self.build_index(papers)
        if self._index is None:
            return papers[:limit]

        results = self.search_multi(query_texts, limit=limit)

        ranked: list[Paper] = []
        for paper, score in results:
            paper_copy = paper.model_copy(deep=False)
            paper_copy.metadata = {**paper.metadata, "vector_score": score}
            paper_copy.retrieval_path = paper.retrieval_path + [
                "provider:vector_rerank",
                "route:vector_semantic",
            ]
            ranked.append(paper_copy)

        LOGGER.info(
            "VectorReranker: reranked %d papers -> top score %.4f",
            len(papers),
            results[0][1] if results else 0.0,
        )
        return ranked
