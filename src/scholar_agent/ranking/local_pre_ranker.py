"""
Local Pre-Ranker: compress the candidate pool from ~466 down to 120-200
before sending to the LLM Evidence Selector.

Goal:
  - Filter out obvious noise using cheap local signals.
  - Keep gold papers from being mis-killed at this stage.
  - Missing features (e.g., no BGE score) do NOT default to 0.5; they are
    dropped and the remaining weights are renormalized.
"""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any

from scholar_agent.models.schemas import Paper, QueryPlan


STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "to", "for", "with", "on", "at",
    "by", "from", "that", "this", "these", "those", "paper", "papers",
    "study", "studies", "method", "methods", "model", "models",
    "using", "based", "via", "through", "approach", "approaches",
    "research", "work", "results", "analysis", "review", "survey",
}


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]+", text.lower())
        if len(t) >= 3 and t not in STOPWORDS
    }


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_score(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


def _title_similarity(query: str, title: str) -> float:
    q = re.sub(r"\s+", " ", query.lower()).strip()
    t = re.sub(r"\s+", " ", title.lower()).strip()
    if not q or not t:
        return 0.0

    q_tokens = _tokens(q)
    t_tokens = _tokens(t)
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)

    seq = SequenceMatcher(None, q, t).ratio()
    return max(overlap, seq * 0.8)


def _constraint_coverage(paper: Paper, plan: QueryPlan) -> float:
    text = f"{paper.title} {paper.abstract or ''}".lower()
    constraints = []

    constraints.extend(plan.datasets or [])
    constraints.extend(plan.methods or [])
    constraints.extend(plan.entities or [])
    constraints.extend(plan.must_have_constraints or [])

    # deduplicate while preserving order
    constraints = list(dict.fromkeys([c for c in constraints if c]))

    if not constraints:
        return 0.5

    hit = 0
    for c in constraints:
        c_low = c.lower()
        c_tokens = _tokens(c_low)
        if c_low in text:
            hit += 1
        elif c_tokens and c_tokens.issubset(_tokens(text)):
            hit += 1

    return hit / max(len(constraints), 1)


def _source_agreement(paper: Paper) -> float:
    providers = {
        p.replace("provider:", "")
        for p in (paper.retrieval_path or [])
        if p.startswith("provider:")
    }
    return min(len(providers) / 3.0, 1.0)


def _route_prior(paper: Paper) -> float:
    paths = paper.retrieval_path or []
    score = 0.0

    if any("title_exact" in p for p in paths):
        score += 0.20
    elif any("title_like" in p for p in paths):
        score += 0.06

    if any("citation_expansion" in p for p in paths):
        score += 0.04
    if any("reference_expansion" in p for p in paths):
        score += 0.03
    if any("related_work_expansion" in p for p in paths):
        score += 0.03

    return min(score, 0.25)


def _citation_score(paper: Paper) -> float:
    if paper.citation_count is None:
        return 0.0
    return min(math.log1p(max(0, paper.citation_count)) / math.log1p(500), 1.0)


def _recency_score(paper: Paper, current_year: int = 2026) -> float:
    if paper.year is None:
        return 0.4
    age = max(0, current_year - paper.year)
    return max(0.0, 1.0 - age * 0.08)


def local_pre_rank_score(
    paper: Paper,
    plan: QueryPlan,
    original_query: str,
) -> tuple[float, dict[str, float]]:
    """Compute a local pre-rank score for a single paper.

    Returns (final_score, subscores_dict).
    Missing features (None) are dropped and weights renormalized.
    """
    title_sim = _title_similarity(original_query, paper.title)
    constraint = _constraint_coverage(paper, plan)
    source = _source_agreement(paper)
    route = _route_prior(paper)
    citation = _citation_score(paper)
    recency = _recency_score(paper)

    # Optional: BGE / SPECTER scores, if downstream retrieval has populated them.
    # If absent, we do NOT fake a 0.5 — we drop the feature and renormalize.
    bge = _safe_float(paper.metadata.get("bge_score"), None)
    specter = _safe_float(paper.metadata.get("specter_score"), None)

    features = {
        "title_similarity": title_sim,
        "constraint_coverage": constraint,
        "source_agreement": source,
        "route_prior": route,
        "citation": citation,
        "recency": recency,
        "bge": bge,
        "specter": specter,
    }

    weights = {
        "title_similarity": 0.18,
        "constraint_coverage": 0.28,
        "source_agreement": 0.12,
        "route_prior": 0.10,
        "citation": 0.08,
        "recency": 0.04,
        "bge": 0.12,
        "specter": 0.08,
    }

    valid = {k: v for k, v in features.items() if v is not None}
    total_w = sum(weights[k] for k in valid)
    if total_w <= 0:
        return 0.0, {k: 0.0 for k in weights}

    score = sum(_norm_score(valid[k]) * weights[k] / total_w for k in valid)
    return score, {k: (_norm_score(v) if v is not None else -1.0) for k, v in features.items()}


def local_pre_rank(
    papers: list[Paper],
    plan: QueryPlan,
    original_query: str,
    topk: int = 200,
) -> list[Paper]:
    """Sort papers by local_pre_rank_score descending and return the top `topk`.

    Each paper gets `local_pre_rank_score` and `local_pre_rank_subscores`
    written into its metadata dict so downstream stages can inspect the
    breakdown.
    """
    scored: list[tuple[float, Paper]] = []
    for idx, p in enumerate(papers):
        score, subscores = local_pre_rank_score(p, plan, original_query)
        
        # Gold Shield: Protect papers ranked in top 50 in rough sorted pool (input papers).
        # These are highly likely to be key candidates retrieved by multi-route/BGE,
        # and should not be filtered out by simple keyword overlap pre-ranking.
        if idx < 50:
            score = max(score, 0.75)
            
        p.metadata["local_pre_rank_score"] = score
        p.metadata["local_pre_rank_subscores"] = subscores
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:topk]]
