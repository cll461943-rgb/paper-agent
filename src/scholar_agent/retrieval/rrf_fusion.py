"""Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

RRF formula: score(d) = sum(weight_i / (k + rank_i))
for each ranked list i where document d appears at rank_i (0-indexed).
"""
from __future__ import annotations

import logging
from typing import Any

from scholar_agent.models.schemas import Paper

LOGGER = logging.getLogger(__name__)

# Standard RRF k value (from the original paper)
DEFAULT_K = 60


def _paper_key(paper: Paper) -> str:
    """Generate a dedup key for a paper (arxiv_id > doi > normalized title)."""
    if paper.arxiv_id:
        aid = str(paper.arxiv_id).strip().lower().replace("arxiv:", "").strip()
        if aid.startswith("http"):
            aid = aid.rstrip("/").split("/")[-1]
        if "v" in aid and aid.split("v")[-1].isdigit():
            aid = aid.rsplit("v", 1)[0]
        return f"aid:{aid}"
    if paper.doi:
        return f"doi:{paper.doi.lower().strip()}"
    title = (paper.title or "").lower().strip().rstrip(".")
    return f"title:{title}"


def rrf_fuse(
    ranked_lists: list[list[Paper]],
    weights: list[float] | None = None,
    k: int = DEFAULT_K,
) -> list[Paper]:
    """Fuse multiple ranked paper lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked paper lists (each list is sorted by relevance).
        weights: Optional weight for each list (default: all 1.0).
        k: RRF constant (default 60, standard value).

    Returns:
        List of unique papers sorted by RRF score (descending).
        Each paper's metadata["rrf_score"] is set to its fused score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length {len(weights)} != ranked_lists length {len(ranked_lists)}"
        )

    # Accumulate RRF scores, track best paper instance, and merge metadata
    score_map: dict[str, float] = {}
    paper_map: dict[str, Paper] = {}
    merged_metadata: dict[str, dict[str, Any]] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, paper in enumerate(ranked_list):
            key = _paper_key(paper)
            contribution = weight / (k + rank)
            score_map[key] = score_map.get(key, 0.0) + contribution

            # Merge metadata from all instances (vector_score, rough_score, etc.)
            if key not in merged_metadata:
                merged_metadata[key] = {}
            merged_metadata[key].update(paper.metadata)

            # Keep the paper instance with the most fields populated
            if key not in paper_map:
                paper_map[key] = paper
            else:
                existing = paper_map[key]
                if (paper.abstract and not existing.abstract) or (
                    paper.arxiv_id and not existing.arxiv_id
                ):
                    paper_map[key] = paper

    # Build result sorted by RRF score
    result: list[Paper] = []
    for key, score in sorted(score_map.items(), key=lambda x: -x[1]):
        paper = paper_map[key]
        # Create a copy with merged metadata + rrf_score
        paper_copy = paper.model_copy(deep=False)
        paper_copy.metadata = {**merged_metadata[key], "rrf_score": score}
        result.append(paper_copy)

    LOGGER.info(
        "RRF fusion: %d lists, %d total papers -> %d unique papers",
        len(ranked_lists),
        sum(len(lst) for lst in ranked_lists),
        len(result),
    )
    return result


def rrf_fuse_with_vector(
    keyword_ranked: list[Paper],
    vector_ranked: list[Paper],
    keyword_weight: float = 1.0,
    vector_weight: float = 1.0,
    k: int = DEFAULT_K,
) -> list[Paper]:
    """Convenience: fuse keyword and vector rankings with RRF.

    Args:
        keyword_ranked: Papers ranked by keyword search.
        vector_ranked: Papers ranked by vector similarity.
        keyword_weight: Weight for keyword ranking.
        vector_weight: Weight for vector ranking.
        k: RRF constant.

    Returns:
        Fused paper list sorted by RRF score.
    """
    return rrf_fuse(
        [keyword_ranked, vector_ranked],
        weights=[keyword_weight, vector_weight],
        k=k,
    )
