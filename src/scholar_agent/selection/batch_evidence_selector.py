from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any

from scholar_agent.models.schemas import EvidenceItem, Paper, QueryPlan, SelectionResult
from scholar_agent.selection.evidence_selector import (
    _create_fallback_selection,
    _fallback_numeric_scores,
    _safe_score,
)

LOGGER = logging.getLogger(__name__)


# ─── Stats tracking (for ablation evaluation) ──────────────────────────────

@dataclass
class LLMEvidenceStats:
    """Track LLM call statistics for ablation evaluation."""
    total_batches: int = 0
    llm_successes: int = 0
    llm_timeouts: int = 0
    llm_parse_failures: int = 0
    llm_errors: int = 0
    fallback_local: int = 0
    papers_reviewed_by_llm: int = 0
    papers_fallback_local: int = 0

    @property
    def success_rate(self) -> float:
        return self.llm_successes / max(self.total_batches, 1)

    @property
    def timeout_rate(self) -> float:
        return self.llm_timeouts / max(self.total_batches, 1)

    @property
    def parse_failure_rate(self) -> float:
        return self.llm_parse_failures / max(self.total_batches, 1)

    def to_dict(self) -> dict:
        return {
            "total_batches": self.total_batches,
            "llm_successes": self.llm_successes,
            "llm_timeouts": self.llm_timeouts,
            "llm_parse_failures": self.llm_parse_failures,
            "llm_errors": self.llm_errors,
            "fallback_local": self.fallback_local,
            "papers_reviewed_by_llm": self.papers_reviewed_by_llm,
            "papers_fallback_local": self.papers_fallback_local,
            "success_rate": round(self.success_rate, 4),
            "timeout_rate": round(self.timeout_rate, 4),
            "parse_failure_rate": round(self.parse_failure_rate, 4),
        }


_last_stats: LLMEvidenceStats = LLMEvidenceStats()


def get_last_stats() -> dict:
    """Return stats from the last batch_select_and_extract_evidence call."""
    return _last_stats.to_dict()


def reset_stats() -> None:
    global _last_stats
    _last_stats = LLMEvidenceStats()


# ─── Local scoring helpers (unchanged) ──────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Split text into lowercased word tokens (length >= 2)."""
    if not text:
        return set()
    return set(re.findall(r"\b\w{2,}\b", text.lower()))


def _text_constraints(plan: QueryPlan) -> list[str]:
    """Text-matchable QueryPlan constraints used by local fallback scoring."""
    constraints: list[str] = []
    constraints.extend(plan.methods or [])
    constraints.extend(plan.datasets or [])
    constraints.extend(getattr(plan, "entities", []) or [])
    constraints.extend(getattr(plan, "must_have_constraints", []) or [])

    result: list[str] = []
    seen: set[str] = set()
    for constraint in constraints:
        if not constraint:
            continue
        value = str(constraint).strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered.startswith("year") or re.search(r"(?:^|\s)(?:year|20\d{2})\s*[<>=]", lowered):
            continue
        if lowered not in seen:
            seen.add(lowered)
            result.append(value)
    return result


def _local_score_paper(
    paper: Paper,
    plan: QueryPlan,
    original_query_terms: set[str],
) -> tuple[float, dict[str, float]]:
    """Comprehensive local scoring for a candidate paper.

    Returns (total_score, component_dict).
    Components: title_overlap, abstract_overlap, constraint_coverage,
    route_confidence, provider_confidence, recency, authority.
    """
    components: dict[str, float] = {}

    title_tokens = _tokenize(paper.title)
    abstract_tokens = _tokenize(paper.abstract or "")
    all_tokens = title_tokens | abstract_tokens

    # 1. Title overlap with original query (0-0.25)
    if original_query_terms and title_tokens:
        overlap = len(original_query_terms & title_tokens)
        components["title_overlap"] = min(overlap / max(len(original_query_terms), 1), 1.0) * 0.25
    else:
        components["title_overlap"] = 0.0

    # 2. Abstract overlap (0-0.20)
    if original_query_terms and abstract_tokens:
        overlap = len(original_query_terms & abstract_tokens)
        components["abstract_overlap"] = min(overlap / max(len(original_query_terms), 1), 1.0) * 0.20
    else:
        components["abstract_overlap"] = 0.0

    # 3. Constraint coverage: methods, datasets, entities, text must-haves (0-0.30)
    matched_constraints = []
    constraint_score = 0.0
    for constraint in _text_constraints(plan):
        if not constraint:
            continue
        c_lower = constraint.lower()
        c_tokens = _tokenize(c_lower)
        if c_tokens and c_tokens.issubset(all_tokens):
            matched_constraints.append(constraint)
            constraint_score += 0.15
        elif c_lower in (paper.title + " " + (paper.abstract or "")).lower():
            matched_constraints.append(constraint)
            constraint_score += 0.10
    components["constraint_coverage"] = min(constraint_score, 0.30)

    # 4. Route confidence (up to 0.60 for exact title search)
    route_score = 0.0
    for path in (paper.retrieval_path or []):
        if "title_exact" in path:
            route_score = max(route_score, 0.60)
        elif "title_like" in path:
            route_score = max(route_score, 0.40)
        elif "core_topic" in path:
            route_score = max(route_score, 0.15)
        elif any(r in path for r in ("method_task", "entity_dataset", "dataset")):
            route_score = max(route_score, 0.12)
        elif any(r in path for r in ("evolved", "hybrid", "query2doc", "hyde")):
            route_score = max(route_score, 0.10)
        else:
            route_score = max(route_score, 0.05)
    components["route_confidence"] = route_score

    # 5. Provider confidence: multi-provider agreement (0-0.10)
    providers = set()
    for path in (paper.retrieval_path or []):
        if path.startswith("provider:"):
            providers.add(path.replace("provider:", ""))
        elif ":" in path:
            parts = path.split(":")
            if parts[0] in ("provider", "prov"):
                providers.add(parts[1])
            elif len(parts) > 1 and parts[0] in ("openalex", "semantic_scholar", "arxiv", "pubmed", "pasa_local"):
                providers.add(parts[0])
    # Also check paper.source
    if paper.source:
        providers.add(paper.source)
    components["provider_confidence"] = min(len(providers) / 3.0, 1.0) * 0.10

    # 6. Recency (0-0.05)
    if paper.year:
        if paper.year >= 2023:
            components["recency"] = 0.05
        elif paper.year >= 2021:
            components["recency"] = 0.04
        elif paper.year >= 2019:
            components["recency"] = 0.03
        elif paper.year >= 2017:
            components["recency"] = 0.02
        else:
            components["recency"] = 0.01
    else:
        components["recency"] = 0.0

    # 7. Authority: citation count (0-0.05)
    if paper.citation_count:
        components["authority"] = min(math.log1p(max(0, paper.citation_count)) * 0.008, 0.05)
    else:
        components["authority"] = 0.0

    total = sum(components.values())
    return min(total, 1.0), components


def _local_score_to_selection(
    paper: Paper,
    plan: QueryPlan,
    score: float,
    components: dict[str, float],
) -> SelectionResult:
    """Convert a local score into a SelectionResult."""
    if score >= 0.50:
        relevance_level = "high"
    elif score >= 0.20:
        relevance_level = "medium"
    elif score >= 0.08:
        relevance_level = "low"
    else:
        relevance_level = "irrelevant"

    fb_rel, fb_constraint, fb_evidence = _fallback_numeric_scores(relevance_level)

    # Determine matched/missing constraints
    matched = []
    missing = []
    all_tokens = _tokenize(paper.title) | _tokenize(paper.abstract or "")
    for constraint in _text_constraints(plan):
        if not constraint:
            continue
        c_tokens = _tokenize(constraint)
        if c_tokens and c_tokens.issubset(all_tokens):
            matched.append(constraint)
        elif constraint.lower() in (paper.title + " " + (paper.abstract or "")).lower():
            matched.append(constraint)
        else:
            missing.append(constraint)

    if matched or missing:
        constraint_ratio = len(matched) / max(len(matched) + len(missing), 1)
        constraint_numeric = max(fb_constraint, constraint_ratio)
    else:
        constraint_numeric = fb_constraint

    return SelectionResult(
        paper_id=paper.paper_id,
        relevance_level=relevance_level,
        matched_constraints=matched,
        missing_constraints=missing,
        evidence=[],
        reason=f"Local score: {score:.3f} (title={components.get('title_overlap',0):.2f}, "
               f"abstract={components.get('abstract_overlap',0):.2f}, "
               f"constraint={components.get('constraint_coverage',0):.2f}, "
               f"route={components.get('route_confidence',0):.2f}, "
               f"provider={components.get('provider_confidence',0):.2f})",
        confidence=min(score + 0.1, 0.9),
        is_validated=True,
        relevance_score=max(fb_rel, score),
        constraint_score=constraint_numeric,
        evidence_score=fb_evidence,
        uncertainty=[],
    )


# ─── LLM prompt building ────────────────────────────────────────────────────

def _build_evidence_prompt(
    plan: QueryPlan,
    papers: list[Paper],
    abstract_limit: int = 800,
    include_local_features: bool = False,
) -> tuple[str, str]:
    """Build system and user prompts for LLM evidence selection.

    Returns (system_prompt, user_prompt).
    """
    system_prompt = (
        "You are an expert scholarly paper Evidence Selector with a RECALL-FIRST strategy. "
        "Your PRIMARY goal is to NOT MISS any relevant paper. False negatives (missing gold papers) are "
        "much worse than false positives (including borderline papers). "
        "For each paper, classify into 'high', 'medium', 'low', or 'irrelevant'.\n"
        "Guidelines:\n"
        "- 'high': Paper directly addresses the query's core topic and at least one key method/dataset.\n"
        "- 'medium': Paper is related to the query topic, even if it only partially matches constraints. "
        "When uncertain between high and medium, assign medium. "
        "When uncertain between medium and low, assign MEDIUM (err on the side of inclusion).\n"
        "- 'low': Paper only has superficial keyword overlap with no substantive connection.\n"
        "- 'irrelevant': Paper is clearly off-topic and has no connection to the research question.\n"
        "IMPORTANT: Only use 'irrelevant' when you are CERTAIN the paper does not address the query. "
        "If in doubt, always assign 'low' or 'medium' rather than 'irrelevant'.\n"
        "Scoring: high->relevance_score 0.8-1.0, medium->0.4-0.8, low->0.1-0.4, irrelevant-><0.1.\n"
        "For each paper, extract 1-3 specific evidence sentences (exact text from title/abstract) that justify the relevance level. "
        "If you cannot find any concrete evidence sentence but the topic is still related, assign 'low'.\n"
        "Return VALID JSON ONLY."
    )

    paper_payload = []
    for p in papers:
        entry: dict[str, Any] = {
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": (p.abstract or "")[:abstract_limit],
            "year": p.year,
            "venue": p.venue,
            "citation_count": p.citation_count,
            "retrieval_path": p.retrieval_path,
        }
        if include_local_features:
            entry["local_pre_rank_score"] = p.metadata.get("local_pre_rank_score", 0.0)
            entry["local_pre_rank_subscores"] = p.metadata.get("local_pre_rank_subscores", {})
        paper_payload.append(entry)

    user_prompt = (
        'Output format (JSON only):\n'
        '{"selections": [{"paper_id": "string", "relevance_level": "high"|"medium"|"low"|"irrelevant", '
        '"relevance_score": 0.0-1.0, "constraint_score": 0.0-1.0, "evidence_score": 0.0-1.0, '
        '"matched_constraints": [...], "missing_constraints": [...], '
        '"evidence": [{"field": "title"|"abstract", "text": "..."}], '
        '"reason": "string", "confidence": 0.0-1.0}]}\n\n'
        f"QueryPlan: {plan.model_dump_json()}\n"
        f"Papers: {json.dumps(paper_payload, ensure_ascii=False)}"
    )

    return system_prompt, user_prompt


def _parse_llm_response(
    response: Any,
    paper_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Parse LLM response into a dict of paper_id -> selection payload."""
    if response is None:
        return {}
    payload = response.get("selections") if isinstance(response, dict) else response
    if not isinstance(payload, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in payload:
        if isinstance(item, dict) and item.get("paper_id"):
            pid = item["paper_id"]
            if pid in paper_ids:
                result[pid] = item
    return result


def _build_selection_from_llm(
    paper: Paper,
    item: dict[str, Any],
    local_sel: SelectionResult,
    plan: QueryPlan,
) -> SelectionResult:
    """Build a SelectionResult from LLM response item."""
    evidence_items: list[EvidenceItem] = []
    for ev in item.get("evidence", []):
        if isinstance(ev, dict) and "field" in ev and "text" in ev:
            evidence_items.append(EvidenceItem(field=ev["field"], text=ev["text"]))

    relevance = item.get("relevance_level")
    if relevance not in {"high", "medium", "low", "irrelevant"}:
        relevance = local_sel.relevance_level

    default_rel, default_constraint, default_evidence = _fallback_numeric_scores(relevance)

    return SelectionResult(
        paper_id=paper.paper_id,
        relevance_level=relevance,
        matched_constraints=item.get("matched_constraints", local_sel.matched_constraints),
        missing_constraints=item.get("missing_constraints", local_sel.missing_constraints),
        evidence=evidence_items,
        reason=item.get("reason", local_sel.reason),
        confidence=_safe_score(item.get("confidence"), local_sel.confidence),
        is_validated=True,
        relevance_score=_safe_score(item.get("relevance_score"), default_rel),
        constraint_score=_safe_score(item.get("constraint_score"), default_constraint),
        evidence_score=_safe_score(item.get("evidence_score"), default_evidence),
        uncertainty=item.get("uncertainty", []),
    )


# ─── LLM batch review with progressive retry (Task 5) ──────────────────────

def _is_timeout_exception(exc: Exception) -> bool:
    exc_str = str(exc).lower()
    return any(k in exc_str for k in ("timeout", "timed out", "read time"))


def _llm_review_batch(
    papers: list[Paper],
    plan: QueryPlan,
    llm_client: Any,
    config: Any | None = None,
    deadline: Any | None = None,
    stats: LLMEvidenceStats | None = None,
) -> dict[str, dict[str, Any]]:
    """Review a batch of papers with LLM, with progressive retry strategy.

    Retry strategy (Task 5):
    1. Full batch, abstract[:500], Pro model
    2. If fail: split batch in half, retry each half
    3. If fail: truncated abstract[:200]
    4. If fail: title + abstract_head[:100] + local_features
    5. All fail: return empty dict (local fallback)

    Returns dict of paper_id -> LLM selection payload.
    """
    if stats is None:
        stats = LLMEvidenceStats()

    if not papers:
        return {}

    paper_ids = {p.paper_id for p in papers}

    # Get config values
    max_tokens = 8192
    timeout_override = None
    if config is not None:
        llm_cfg = getattr(config, "llm", None)
        if llm_cfg:
            max_tokens = getattr(llm_cfg, "max_tokens_evidence_selection", 8192) or 8192
            timeout_override = getattr(llm_cfg, "timeout_evidence_selection", None)

    # Check deadline
    if deadline is not None and deadline.expired():
        stats.fallback_local += len(papers)
        return {}

    def _timeout_for_next_call() -> float | None:
        configured_timeout = timeout_override
        if deadline is None:
            return float(configured_timeout) if configured_timeout is not None else 30.0

        remaining = deadline.remaining()
        if remaining <= 1.0:
            return None
        if configured_timeout is None:
            return remaining
        return max(1.0, min(float(configured_timeout), remaining))

    stats.total_batches += 1

    # ── Attempt 1: Full batch, abstract[:800] ──
    call_timeout = _timeout_for_next_call()
    if call_timeout is None:
        stats.fallback_local += len(papers)
        return {}
    system_prompt, user_prompt = _build_evidence_prompt(plan, papers, abstract_limit=800)
    try:
        response = llm_client.complete_json(
            system_prompt, user_prompt,
            model_type="flash",
            timeout_seconds=call_timeout,
            max_tokens=max_tokens,
        )
        if response is not None:
            parsed = _parse_llm_response(response, paper_ids)
            if parsed:
                stats.llm_successes += 1
                stats.papers_reviewed_by_llm += len(parsed)
                return parsed
        stats.llm_parse_failures += 1
        LOGGER.debug("LLM batch attempt 1 (full): parse failure for %d papers", len(papers))
    except Exception as exc:
        if _is_timeout_exception(exc):
            stats.llm_timeouts += 1
            LOGGER.debug("LLM batch attempt 1 (full): timeout for %d papers", len(papers))
        else:
            stats.llm_errors += 1
            LOGGER.debug("LLM batch attempt 1 (full): error: %s", exc)

    # ── Attempt 2: Split batch in half ──
    if len(papers) > 2:
        mid = len(papers) // 2
        halves = [papers[:mid], papers[mid:]]
        result: dict[str, dict[str, Any]] = {}
        for half in halves:
            if not half:
                continue
            call_timeout = _timeout_for_next_call()
            if call_timeout is None:
                stats.fallback_local += len(half)
                continue
            half_ids = {p.paper_id for p in half}
            sp, up = _build_evidence_prompt(plan, half, abstract_limit=800)
            try:
                resp = llm_client.complete_json(
                    sp, up,
                    model_type="flash",
                    timeout_seconds=call_timeout,
                    max_tokens=max_tokens,
                )
                if resp is not None:
                    parsed = _parse_llm_response(resp, half_ids)
                    if parsed:
                        result.update(parsed)
            except Exception:
                pass
        if result:
            stats.llm_successes += 1
            stats.papers_reviewed_by_llm += len(result)
            LOGGER.debug("LLM batch attempt 2 (half): got %d/%d papers", len(result), len(papers))
            return result

    # ── Attempt 3: Truncated abstract[:200] ──
    call_timeout = _timeout_for_next_call()
    if call_timeout is None:
        stats.fallback_local += len(papers)
        return {}
    system_prompt, user_prompt = _build_evidence_prompt(plan, papers, abstract_limit=200)
    try:
        response = llm_client.complete_json(
            system_prompt, user_prompt,
            model_type="flash",
            timeout_seconds=call_timeout,
            max_tokens=max_tokens,
        )
        if response is not None:
            parsed = _parse_llm_response(response, paper_ids)
            if parsed:
                stats.llm_successes += 1
                stats.papers_reviewed_by_llm += len(parsed)
                LOGGER.debug("LLM batch attempt 3 (truncated): got %d/%d papers", len(parsed), len(papers))
                return parsed
        stats.llm_parse_failures += 1
    except Exception as exc:
        if _is_timeout_exception(exc):
            stats.llm_timeouts += 1
        else:
            stats.llm_errors += 1

    # ── Attempt 4: Title + abstract_head[:100] + local_features ──
    call_timeout = _timeout_for_next_call()
    if call_timeout is None:
        stats.fallback_local += len(papers)
        return {}
    system_prompt, user_prompt = _build_evidence_prompt(
        plan, papers, abstract_limit=100, include_local_features=True
    )
    try:
        response = llm_client.complete_json(
            system_prompt, user_prompt,
            model_type="flash",
            timeout_seconds=call_timeout,
            max_tokens=max_tokens,
        )
        if response is not None:
            parsed = _parse_llm_response(response, paper_ids)
            if parsed:
                stats.llm_successes += 1
                stats.papers_reviewed_by_llm += len(parsed)
                LOGGER.debug("LLM batch attempt 4 (minimal): got %d/%d papers", len(parsed), len(papers))
                return parsed
        stats.llm_parse_failures += 1
    except Exception as exc:
        if _is_timeout_exception(exc):
            stats.llm_timeouts += 1
        else:
            stats.llm_errors += 1

    # All attempts failed — local fallback
    stats.fallback_local += len(papers)
    LOGGER.debug("LLM batch: all attempts failed for %d papers, using local fallback", len(papers))
    return {}


# ─── Main entry point (Task 2) ──────────────────────────────────────────────

def batch_select_and_extract_evidence(
    papers: list[Paper],
    plan: QueryPlan,
    llm_client: object | None,
    original_query: str = "",
    config: Any | None = None,
    deadline: Any | None = None,
) -> list[SelectionResult]:
    """LLM-led Evidence Selector (effect-first mode).

    Strategy:
    1. Score ALL candidates locally for fallback
    2. Sort by local score descending
    3. Take top N (60-80) candidates for LLM review
    4. Batch in groups of batch_size (8), call LLM with Pro model
    5. Progressive retry: full batch -> half batch -> truncated abstract -> title+local_features
    6. LLM failure -> local scores used as fallback
    7. LLM results override local scores when LLM succeeds
    """
    global _last_stats
    _last_stats = LLMEvidenceStats()

    if not papers:
        return []

    # Get config values
    batch_size = 8
    max_llm_candidates = 80
    if config is not None:
        sel_cfg = getattr(config, "selection", None)
        if sel_cfg:
            batch_size = getattr(sel_cfg, "batch_size", 8) or 8
        budget_cfg = getattr(config, "budget", None)
        if budget_cfg:
            max_llm_candidates = getattr(budget_cfg, "max_llm_selection_papers", 80) or 80

    # Build original query terms
    original_query_terms = _tokenize(original_query) if original_query else set()
    if plan.research_topic:
        original_query_terms |= _tokenize(plan.research_topic)

    # Step 1: Local score ALL candidates
    scored: list[tuple[Paper, float, dict[str, float]]] = []
    for paper in papers:
        score, components = _local_score_paper(paper, plan, original_query_terms)
        scored.append((paper, score, components))

    # Step 2: Sort by local score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Create local SelectionResults for all (fallback)
    local_results: dict[str, SelectionResult] = {}
    for paper, score, components in scored:
        local_results[paper.paper_id] = _local_score_to_selection(paper, plan, score, components)

    # Step 4: Select top N candidates for LLM review
    llm_candidates = [p for p, s, c in scored[:max_llm_candidates]]

    if not llm_candidates or llm_client is None:
        LOGGER.info(
            "LLM-led selector: %d papers total. %d for LLM review. LLM %s.",
            len(papers), len(llm_candidates),
            "skipped (no candidates)" if not llm_candidates else "unavailable",
        )
        _last_stats.fallback_local = len(papers)
        _last_stats.papers_fallback_local = len(papers)
        return [local_results[p.paper_id] for p in papers]

    # Step 5: Batch and call LLM with progressive retry
    LOGGER.info(
        "LLM-led selector: %d papers total. Sending top %d to LLM (Pro) in batches of %d.",
        len(papers), len(llm_candidates), batch_size,
    )

    llm_payloads: dict[str, dict[str, Any]] = {}
    for i in range(0, len(llm_candidates), batch_size):
        batch = llm_candidates[i:i + batch_size]

        # Check deadline before each batch
        if deadline is not None and deadline.expired():
            LOGGER.info(
                "Deadline expired after %d batches. Using local scores for remaining %d papers.",
                i // batch_size, len(llm_candidates) - i,
            )
            _last_stats.fallback_local += len(batch)
            continue

        batch_payloads = _llm_review_batch(
            batch, plan, llm_client, config, deadline, _last_stats
        )
        llm_payloads.update(batch_payloads)

        # Brief pause between batches to avoid rate limiting
        if i + batch_size < len(llm_candidates):
            time.sleep(0.5)

    LOGGER.info(
        "LLM-led selector done: %d/%d papers reviewed by LLM. Stats: %s",
        _last_stats.papers_reviewed_by_llm, len(llm_candidates),
        _last_stats.to_dict(),
    )

    # Step 6: Merge LLM results with local results
    results: list[SelectionResult] = []
    for paper in papers:
        local_sel = local_results[paper.paper_id]

        if paper.paper_id in llm_payloads:
            item = llm_payloads[paper.paper_id]
            try:
                results.append(_build_selection_from_llm(paper, item, local_sel, plan))
                continue
            except Exception as exc:
                LOGGER.debug("Parse error for LLM selection of %s: %s", paper.paper_id, exc)

        # Use local result
        results.append(local_sel)

    _last_stats.papers_fallback_local = sum(1 for p in papers if p.paper_id not in llm_payloads)

    return results
