"""Semantic Bridge: orchestrates HyDE + BGE-M3 vector search + RRF fusion.

This module bridges vocabulary mismatch by adding a semantic vector search
layer on top of keyword retrieval, following the qmd architecture pattern:
    BM25 keyword search + Vector semantic search + RRF fusion.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from scholar_agent.models.schemas import Paper
from scholar_agent.retrieval.embedding_service import EmbeddingService
from scholar_agent.retrieval.rrf_fusion import rrf_fuse
from scholar_agent.retrieval.vector_rerank import VectorReranker

LOGGER = logging.getLogger(__name__)

# Default weights for RRF fusion (keyword vs vector)
DEFAULT_KEYWORD_WEIGHT = 1.0
DEFAULT_VECTOR_WEIGHT = 1.5  # Vector slightly higher — it bridges vocabulary mismatch
DEFAULT_HYDE_WEIGHT = 1.0


def generate_hyde_document(
    query: str,
    llm_client: Any | None,
    query_plan: Any | None = None,
) -> str | None:
    """Use LLM to generate a hypothetical paper abstract for HyDE.

    The generated abstract describes what an ideal matching paper would look like,
    using standard academic terminology. This is then embedded and used for
    vector similarity search.
    """
    if llm_client is None:
        return None

    methods = getattr(query_plan, "methods", [])
    entities = getattr(query_plan, "entities", [])
    research_topic = getattr(query_plan, "research_topic", "")

    context_parts = []
    if research_topic:
        context_parts.append(f"Research topic: {research_topic}")
    if methods:
        context_parts.append(f"Methods: {', '.join(methods[:5])}")
    if entities:
        context_parts.append(f"Entities: {', '.join(entities[:5])}")
    context = "\n".join(context_parts) if context_parts else ""

    system_prompt = """You are an expert academic researcher. Given a research question, generate a hypothetical paper abstract (3-4 sentences) that would perfectly answer this question. Use precise academic terminology that the actual paper would use in its title and abstract. Do not use vague language — be specific about the methods, datasets, and contributions.

Return JSON: {"hypothetical_abstract": "your abstract here"}"""

    user_prompt = f"Research question: {query}"
    if context:
        user_prompt += f"\n\nContext:\n{context}"

    try:
        response = getattr(llm_client, "complete_json", lambda *_: None)(
            system_prompt, user_prompt, model_type="flash"
        )
        if isinstance(response, dict):
            return response.get("hypothetical_abstract")
        return None
    except Exception as e:
        LOGGER.warning("HyDE document generation failed: %s", e)
        return None


class SemanticBridge:
    """Orchrate HyDE + BGE-M3 vector search + RRF fusion.

    Usage:
        bridge = SemanticBridge(embedding_service)
        fused = bridge.apply(
            candidate_pool=all_candidates,
            original_query="...",
            llm_client=llm_client,
            query_plan=query_plan,
        )
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        hyde_weight: float = DEFAULT_HYDE_WEIGHT,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedding_service
        self.reranker = VectorReranker(embedding_service)
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.hyde_weight = hyde_weight
        self.rrf_k = rrf_k

    def apply(
        self,
        candidate_pool: list[Paper],
        original_query: str,
        llm_client: Any | None = None,
        query_plan: Any | None = None,
        enable_hyde: bool = True,
    ) -> list[Paper]:
        """Apply semantic bridging to the candidate pool.

        1. Generate HyDE document (if enabled)
        2. Embed all candidates → FAISS index
        3. Embed query (+ HyDE) → vector search
        4. RRF fuse keyword ranking + vector ranking
        5. Return fused results
        """
        if not candidate_pool:
            return candidate_pool

        t0 = time.time()
        LOGGER.info(
            "SemanticBridge: applying to %d candidates, query='%s'",
            len(candidate_pool),
            original_query[:80],
        )

        # 1. Generate HyDE document
        query_texts = [original_query]
        if enable_hyde and llm_client is not None:
            hyde_doc = generate_hyde_document(original_query, llm_client, query_plan)
            if hyde_doc:
                LOGGER.info("SemanticBridge: HyDE document generated (%d chars)", len(hyde_doc))
                query_texts.append(hyde_doc)
            else:
                LOGGER.info("SemanticBridge: HyDE generation failed, using query only")

        # Also add research_topic if available (often more specific than original query)
        research_topic = getattr(query_plan, "research_topic", "")
        if research_topic and research_topic != original_query:
            query_texts.append(research_topic)

        # 2-3. Vector re-ranking
        vector_ranked = self.reranker.rerank(
            candidate_pool,
            query_texts=query_texts,
            limit=len(candidate_pool),
        )

        # 4. RRF fusion: keyword ranking (candidate_pool) + vector ranking
        # The candidate_pool is already sorted by _get_rough_score (keyword-based)
        # The vector_ranked is sorted by vector similarity
        fused = rrf_fuse(
            [candidate_pool, vector_ranked],
            weights=[self.keyword_weight, self.vector_weight],
            k=self.rrf_k,
        )

        elapsed = time.time() - t0
        LOGGER.info(
            "SemanticBridge: done in %.1fs — %d fused papers "
            "(keyword: %d, vector: %d, queries: %d)",
            elapsed,
            len(fused),
            len(candidate_pool),
            len(vector_ranked),
            len(query_texts),
        )

        return fused
