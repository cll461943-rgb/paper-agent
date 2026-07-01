"""LLM Listwise Reranker — uses DeepSeek-V4-Pro to rank candidate papers listwise.

Effect-first mode: the LLM sees all top candidates at once and produces a
global ranking. This captures inter-paper comparisons that pointwise scoring
misses.

Retry strategy:
1. Top 40 papers, abstract[:300], Pro model
2. If fail: Top 20 papers, abstract[:300]
3. If fail: Top 20, title only + local_features
4. All fail: return None (final reranker uses local fallback)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from scholar_agent.models.schemas import Paper, QueryPlan, SelectionResult

LOGGER = logging.getLogger(__name__)


@dataclass
class ListwiseRerankResult:
    """Result of LLM listwise reranking.

    Carries both the legacy listwise_score (for fusion formula backward
    compatibility) and the new probability-based fields used by the
    F1-aware K controller.
    """
    scores: dict[str, float] = field(default_factory=dict)  # paper_id -> listwise_score (0-1)
    recommended_k: int = 10
    success: bool = False
    reason: str = ""
    stats: dict = field(default_factory=dict)
    # Task 2: probability-based fields for K controller
    relevance_probabilities: dict[str, float] = field(default_factory=dict)  # paper_id -> p(0-1)
    relevance_levels: dict[str, str] = field(default_factory=dict)  # paper_id -> "high"/"medium"/"low"/"irrelevant"
    relevance_confidences: dict[str, float] = field(default_factory=dict)  # paper_id -> confidence(0-1)
    matched_aspects: dict[str, list[str]] = field(default_factory=dict)  # paper_id -> matched query aspects
    should_include: dict[str, bool] = field(default_factory=dict)  # paper_id -> include in final output?
    recommended_k_min: int = 1
    recommended_k_max: int = 20


@dataclass
class ListwiseStats:
    """Track LLM listwise reranker call statistics."""
    attempts: int = 0
    successes: int = 0
    timeouts: int = 0
    parse_failures: int = 0
    errors: int = 0
    papers_input: int = 0
    papers_scored: int = 0
    fallback: bool = False
    strategy_used: str = ""

    def to_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "timeouts": self.timeouts,
            "parse_failures": self.parse_failures,
            "errors": self.errors,
            "papers_input": self.papers_input,
            "papers_scored": self.papers_scored,
            "fallback": self.fallback,
            "strategy_used": self.strategy_used,
        }


_last_stats: ListwiseStats = ListwiseStats()


def get_last_stats() -> dict:
    return _last_stats.to_dict()


def _is_timeout_exception(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("timeout", "timed out", "read time"))


def _build_listwise_prompt(
    plan: QueryPlan,
    papers: list[Paper],
    selections_map: dict[str, SelectionResult],
    original_query: str,
    abstract_limit: int = 300,
    include_local_features: bool = False,
) -> tuple[str, str]:
    """Build system and user prompts for listwise reranking.

    Returns (system_prompt, user_prompt).
    """
    system_prompt = (
        "You are an expert scholarly paper ranking system. "
        "Your goal is to MAXIMIZE F1 score by ranking papers by relevance to the query.\n"
        "You will see a list of candidate papers with their metadata and evidence.\n"
        "For EACH paper, you must provide:\n"
        "  - listwise_score (0.0-1.0): ranking score, top paper gets highest.\n"
        "  - relevance_probability (0.0-1.0): independent probability this paper is truly relevant.\n"
        "  - relevance_level: one of 'high', 'medium', 'low', 'irrelevant'.\n"
        "  - matched_aspects: list of query aspects/constraints this paper addresses.\n"
        "  - should_include (boolean): whether this paper should appear in the final output.\n"
        "  - confidence (0.0-1.0): your confidence in this assessment.\n"
        "  - reason: one-sentence justification.\n"
        "Also provide recommended_k_min, recommended_k_max (range for final output count), "
        "recommended_k (suggested optimal count), and overall_reason.\n"
        "Return VALID JSON ONLY."
    )

    paper_list = []
    for p in papers:
        sel = selections_map.get(p.paper_id)
        entry: dict[str, Any] = {
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": (p.abstract or "")[:abstract_limit],
            "year": p.year,
            "venue": p.venue,
            "citation_count": p.citation_count,
            "retrieval_path": p.retrieval_path,
        }
        if sel:
            entry["evidence_level"] = sel.relevance_level
            entry["evidence_score"] = sel.evidence_score
            entry["matched_constraints"] = sel.matched_constraints
            evidence_texts = []
            for ev in (sel.evidence or []):
                evidence_texts.append(f"[{ev.field}] {ev.text}")
            if evidence_texts:
                entry["evidence_snippets"] = evidence_texts
        if include_local_features:
            entry["local_pre_rank_score"] = p.metadata.get("local_pre_rank_score", 0.0)
        paper_list.append(entry)

    user_prompt = (
        'Output format (JSON only):\n'
        '{"ranked_papers": [{"paper_id": "string", "rank": 1, "listwise_score": 0.0-1.0, '
        '"relevance_probability": 0.0-1.0, "relevance_level": "high|medium|low|irrelevant", '
        '"matched_aspects": ["aspect1", "aspect2"], "should_include": true, '
        '"confidence": 0.0-1.0, "reason": "string"}], '
        '"recommended_k": int, "recommended_k_min": int, "recommended_k_max": int, '
        '"overall_reason": "string"}\n\n'
        f"Original Query: {original_query}\n"
        f"QueryPlan: {plan.model_dump_json()}\n"
        f"Candidate Papers ({len(papers)} total):\n"
        f"{json.dumps(paper_list, ensure_ascii=False)}"
    )

    return system_prompt, user_prompt


def _parse_listwise_response(
    response: Any,
    paper_ids: set[str],
) -> dict[str, Any] | None:
    """Parse LLM listwise response.

    Returns a dict with keys:
      scores, recommended_k, reason,
      relevance_probabilities, relevance_levels, relevance_confidences,
      matched_aspects, should_include,
      recommended_k_min, recommended_k_max
    or None on failure.
    """
    if response is None or not isinstance(response, dict):
        return None

    ranked = response.get("ranked_papers")
    if not isinstance(ranked, list):
        return None

    scores: dict[str, float] = {}
    rel_probs: dict[str, float] = {}
    rel_levels: dict[str, str] = {}
    rel_confs: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    should_inc: dict[str, bool] = {}

    for item in ranked:
        if not isinstance(item, dict):
            continue
        pid = item.get("paper_id")
        if not pid or pid not in paper_ids:
            continue
        try:
            score = max(0.0, min(1.0, float(item.get("listwise_score", 0.0))))
            scores[pid] = score
        except (TypeError, ValueError):
            scores[pid] = 0.0

        # relevance_probability
        try:
            rp = max(0.0, min(1.0, float(item.get("relevance_probability", 0.0))))
            rel_probs[pid] = rp
        except (TypeError, ValueError):
            pass

        # relevance_level
        level = item.get("relevance_level", "")
        if isinstance(level, str) and level in ("high", "medium", "low", "irrelevant"):
            rel_levels[pid] = level

        # confidence
        try:
            conf = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
            rel_confs[pid] = conf
        except (TypeError, ValueError):
            pass

        # matched_aspects
        aspects = item.get("matched_aspects", [])
        if isinstance(aspects, list):
            matched[pid] = [str(a) for a in aspects]

        # should_include
        si = item.get("should_include")
        if isinstance(si, bool):
            should_inc[pid] = si

    if not scores:
        return None

    recommended_k = 10
    try:
        recommended_k = max(1, min(50, int(response.get("recommended_k", 10))))
    except (TypeError, ValueError):
        pass

    recommended_k_min = 1
    recommended_k_max = 20
    try:
        recommended_k_min = max(1, min(50, int(response.get("recommended_k_min", 1))))
    except (TypeError, ValueError):
        pass
    try:
        recommended_k_max = max(recommended_k_min, min(50, int(response.get("recommended_k_max", 20))))
    except (TypeError, ValueError):
        pass

    reason = response.get("overall_reason", "")

    return {
        "scores": scores,
        "recommended_k": recommended_k,
        "reason": reason,
        "relevance_probabilities": rel_probs,
        "relevance_levels": rel_levels,
        "relevance_confidences": rel_confs,
        "matched_aspects": matched,
        "should_include": should_inc,
        "recommended_k_min": recommended_k_min,
        "recommended_k_max": recommended_k_max,
    }


def llm_listwise_rerank(
    papers: list[Paper],
    selections: list[SelectionResult],
    plan: QueryPlan,
    llm_client: Any | None,
    original_query: str = "",
    config: Any | None = None,
    deadline: Any | None = None,
) -> ListwiseRerankResult | None:
    """Rank papers using LLM listwise reranking with progressive retry.

    Args:
        papers: Top candidates (high+medium from evidence selector), typically 30-40.
        selections: SelectionResults from evidence selector.
        plan: QueryPlan.
        llm_client: LLM client or None.
        original_query: Original user query string.
        config: AppConfig (for timeouts, max_tokens).
        deadline: Deadline object for time budget.

    Returns:
        ListwiseRerankResult with scores dict, or None if LLM unavailable/failed.
        When None, final reranker falls back to local scoring.
    """
    global _last_stats
    _last_stats = ListwiseStats()

    if not papers or llm_client is None:
        _last_stats.fallback = True
        return None

    # Get config values
    topk = 40
    fallback_topk = 20
    max_tokens = 8192
    timeout_override = None
    if config is not None:
        rank_cfg = getattr(config, "ranking", None)
        if rank_cfg:
            topk = getattr(rank_cfg, "listwise_topk", 40) or 40
            fallback_topk = getattr(rank_cfg, "listwise_fallback_topk", 20) or 20
        llm_cfg = getattr(config, "llm", None)
        if llm_cfg:
            max_tokens = getattr(llm_cfg, "max_tokens_listwise_rerank", 8192) or 8192
            timeout_override = getattr(llm_cfg, "timeout_listwise_rerank", None)

    # Check deadline
    if deadline is not None and deadline.expired():
        _last_stats.fallback = True
        LOGGER.info("Listwise reranker: deadline expired, skipping.")
        return None

    # Determine effective timeout
    effective_timeout = timeout_override
    if deadline is not None:
        rem = deadline.remaining()
        if effective_timeout is not None:
            effective_timeout = max(1.0, min(float(effective_timeout), rem))
        else:
            effective_timeout = max(1.0, rem)

    # Build selections map
    selections_map = {sel.paper_id: sel for sel in selections}

    # Sort papers by local_pre_rank_score for consistent input
    sorted_papers = sorted(
        papers,
        key=lambda p: p.metadata.get("local_pre_rank_score", 0.0),
        reverse=True,
    )

    # ── Attempt 1: Top {topk} papers, abstract[:300] ──
    top_papers = sorted_papers[:topk]
    _last_stats.papers_input = len(top_papers)
    _last_stats.attempts += 1
    _last_stats.strategy_used = f"top{topk}_abstract300"

    system_prompt, user_prompt = _build_listwise_prompt(
        plan, top_papers, selections_map, original_query, abstract_limit=300,
    )
    try:
        response = llm_client.complete_json(
            system_prompt, user_prompt,
            model_type="flash",
            timeout_seconds=effective_timeout,
            max_tokens=max_tokens,
        )
        # DEBUG: See what the LLM actually returned
        print(f"DEBUG LISTWISE: response type={type(response).__name__}, is None={response is None}")
        if isinstance(response, dict):
            print(f"DEBUG LISTWISE: keys={list(response.keys())}")
        parsed = _parse_listwise_response(response, {p.paper_id for p in top_papers})
        if parsed:
            scores = parsed["scores"]
            _min_scored = max(1, len(top_papers) // 2)
            if len(scores) >= _min_scored:
                _last_stats.successes += 1
                _last_stats.papers_scored = len(scores)
                LOGGER.info(
                    "Listwise reranker: SUCCESS with %d papers (top%d, abstract300). "
                    "Scored %d papers, recommended_k=%d.",
                    len(top_papers), topk, len(scores), parsed["recommended_k"],
                )
                return ListwiseRerankResult(
                    scores=scores,
                    recommended_k=parsed["recommended_k"],
                    success=True,
                    reason=parsed["reason"],
                    stats=_last_stats.to_dict(),
                    relevance_probabilities=parsed.get("relevance_probabilities", {}),
                    relevance_levels=parsed.get("relevance_levels", {}),
                    relevance_confidences=parsed.get("relevance_confidences", {}),
                    matched_aspects=parsed.get("matched_aspects", {}),
                    should_include=parsed.get("should_include", {}),
                    recommended_k_min=parsed.get("recommended_k_min", 1),
                    recommended_k_max=parsed.get("recommended_k_max", 20),
                )
            else:
                LOGGER.info(
                    "Listwise reranker attempt 1: partial parse (%d/%d papers, need >=%d). Trying next attempt.",
                    len(scores), len(top_papers), _min_scored,
                )
        _last_stats.parse_failures += 1
        LOGGER.debug("Listwise reranker attempt 1: parse failure")
    except Exception as exc:
        if _is_timeout_exception(exc):
            _last_stats.timeouts += 1
            LOGGER.debug("Listwise reranker attempt 1: timeout")
        else:
            _last_stats.errors += 1
            LOGGER.debug("Listwise reranker attempt 1: error: %s", exc)

    # ── Attempt 2: Top {fallback_topk} papers, abstract[:300] ──
    # Check deadline before retrying — don't waste time if deadline expired
    if deadline is not None and deadline.expired():
        _last_stats.fallback = True
        LOGGER.info(
            "Listwise reranker: deadline expired after attempt 1. Stats: %s. Using local fallback.",
            _last_stats.to_dict(),
        )
        return None

    # Recalculate effective timeout for remaining deadline
    if deadline is not None:
        rem = deadline.remaining()
        if timeout_override is not None:
            effective_timeout = max(1.0, min(float(timeout_override), rem))
        else:
            effective_timeout = max(1.0, rem)

    top_papers = sorted_papers[:fallback_topk]
    _last_stats.papers_input = len(top_papers)
    _last_stats.attempts += 1
    _last_stats.strategy_used = f"top{fallback_topk}_abstract300"

    system_prompt, user_prompt = _build_listwise_prompt(
        plan, top_papers, selections_map, original_query, abstract_limit=300,
    )
    try:
        response = llm_client.complete_json(
            system_prompt, user_prompt,
            model_type="flash",
            timeout_seconds=effective_timeout,
            max_tokens=max_tokens,
        )
        # DEBUG: See what the LLM actually returned
        print(f"DEBUG LISTWISE: response type={type(response).__name__}, is None={response is None}")
        if isinstance(response, dict):
            print(f"DEBUG LISTWISE: keys={list(response.keys())}")
        parsed = _parse_listwise_response(response, {p.paper_id for p in top_papers})
        if parsed:
            scores = parsed["scores"]
            _min_scored = max(1, len(top_papers) // 2)
            if len(scores) >= _min_scored:
                _last_stats.successes += 1
                _last_stats.papers_scored = len(scores)
                LOGGER.info(
                    "Listwise reranker: SUCCESS with %d papers (fallback top%d, abstract300). "
                    "Scored %d papers, recommended_k=%d.",
                    len(top_papers), fallback_topk, len(scores), parsed["recommended_k"],
                )
                return ListwiseRerankResult(
                    scores=scores,
                    recommended_k=parsed["recommended_k"],
                    success=True,
                    reason=parsed["reason"],
                    stats=_last_stats.to_dict(),
                    relevance_probabilities=parsed.get("relevance_probabilities", {}),
                    relevance_levels=parsed.get("relevance_levels", {}),
                    relevance_confidences=parsed.get("relevance_confidences", {}),
                    matched_aspects=parsed.get("matched_aspects", {}),
                    should_include=parsed.get("should_include", {}),
                    recommended_k_min=parsed.get("recommended_k_min", 1),
                    recommended_k_max=parsed.get("recommended_k_max", 20),
                )
            else:
                LOGGER.info(
                    "Listwise reranker attempt 2: partial parse (%d/%d papers, need >=%d). Trying next attempt.",
                    len(scores), len(top_papers), _min_scored,
                )
        _last_stats.parse_failures += 1
    except Exception as exc:
        if _is_timeout_exception(exc):
            _last_stats.timeouts += 1
        else:
            _last_stats.errors += 1

    # ── Attempt 3: Top {fallback_topk}, title only + local_features ──
    # Check deadline before final retry
    if deadline is not None and deadline.expired():
        _last_stats.fallback = True
        LOGGER.info(
            "Listwise reranker: deadline expired after attempt 2. Stats: %s. Using local fallback.",
            _last_stats.to_dict(),
        )
        return None

    # Recalculate effective timeout for remaining deadline
    if deadline is not None:
        rem = deadline.remaining()
        if timeout_override is not None:
            effective_timeout = max(1.0, min(float(timeout_override), rem))
        else:
            effective_timeout = max(1.0, rem)

    _last_stats.attempts += 1
    _last_stats.strategy_used = f"top{fallback_topk}_title_only"

    system_prompt, user_prompt = _build_listwise_prompt(
        plan, top_papers, selections_map, original_query,
        abstract_limit=0, include_local_features=True,
    )
    try:
        response = llm_client.complete_json(
            system_prompt, user_prompt,
            model_type="flash",
            timeout_seconds=effective_timeout,
            max_tokens=max_tokens,
        )
        # DEBUG: See what the LLM actually returned
        print(f"DEBUG LISTWISE: response type={type(response).__name__}, is None={response is None}")
        if isinstance(response, dict):
            print(f"DEBUG LISTWISE: keys={list(response.keys())}")
        parsed = _parse_listwise_response(response, {p.paper_id for p in top_papers})
        if parsed:
            scores = parsed["scores"]
            _min_scored = max(1, len(top_papers) // 2)
            if len(scores) >= _min_scored:
                _last_stats.successes += 1
                _last_stats.papers_scored = len(scores)
                LOGGER.info(
                    "Listwise reranker: SUCCESS with %d papers (title only). "
                    "Scored %d papers, recommended_k=%d.",
                    len(top_papers), len(scores), parsed["recommended_k"],
                )
                return ListwiseRerankResult(
                    scores=scores,
                    recommended_k=parsed["recommended_k"],
                    success=True,
                    reason=parsed["reason"],
                    stats=_last_stats.to_dict(),
                    relevance_probabilities=parsed.get("relevance_probabilities", {}),
                    relevance_levels=parsed.get("relevance_levels", {}),
                    relevance_confidences=parsed.get("relevance_confidences", {}),
                    matched_aspects=parsed.get("matched_aspects", {}),
                    should_include=parsed.get("should_include", {}),
                    recommended_k_min=parsed.get("recommended_k_min", 1),
                    recommended_k_max=parsed.get("recommended_k_max", 20),
                )
            else:
                LOGGER.info(
                    "Listwise reranker attempt 3: partial parse (%d/%d papers, need >=%d). Using local fallback.",
                    len(scores), len(top_papers), _min_scored,
                )
        _last_stats.parse_failures += 1
    except Exception as exc:
        if _is_timeout_exception(exc):
            _last_stats.timeouts += 1
        else:
            _last_stats.errors += 1

    # All attempts failed
    _last_stats.fallback = True
    LOGGER.info(
        "Listwise reranker: all attempts failed. Stats: %s. Using local fallback.",
        _last_stats.to_dict(),
    )
    return None
