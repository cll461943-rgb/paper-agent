"""F1-aware Dynamic-K Controller V2.

V2 improvements over V1:
  1. Three-way g_hat fusion (scope/pool/visible) — no longer solely
     dependent on listwise top-24 probability sum.
  2. Low-confidence uniform guard: if top-10 mean_prob < 0.35 and
     std < 0.05, cap max_k <= 5.
  3. Marginal p_floor: each additional paper must have p >= p_floor
     (by query type) to be included.
  4. Precision-first tie-break: if multiple K within 0.02, pick smaller K.
  5. V2 probability calibration via probability_calibration module.

Core formula remains:
  E[F1@K] = 2 * E[TP@K] / (K + g_hat)
where E[TP@K] = sum(calibrated_probability[i] for i in 0..K-1)
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from scholar_agent.ranking.probability_calibration import (
    calibrate_paper_probabilities_v2,
)

LOGGER = logging.getLogger(__name__)

# ── Query type → (prior g_hat, p_floor, is_large_scope) ──
QUERY_TYPE_PROFILE: dict[str, dict[str, Any]] = {
    "exact_title":       {"prior": 1.0,  "p_floor": 0.50, "large_scope": False},
    "single_gold":       {"prior": 1.0,  "p_floor": 0.50, "large_scope": False},
    "specific_paper":    {"prior": 2.0,  "p_floor": 0.45, "large_scope": False},
    "dataset_constraint":{"prior": 4.5,  "p_floor": 0.22, "large_scope": False},
    "method_comparison": {"prior": 4.5,  "p_floor": 0.22, "large_scope": False},
    "latest_work":       {"prior": 6.0,  "p_floor": 0.18, "large_scope": True},
    "broad_topic":       {"prior": 8.0,  "p_floor": 0.18, "large_scope": True},
    "survey":            {"prior": 10.0, "p_floor": 0.18, "large_scope": True},
    "unknown":           {"prior": 4.0,  "p_floor": 0.22, "large_scope": False},
}

# ── Low-confidence uniform guard thresholds ──
LC_MEAN_THRESHOLD = 0.35
LC_STD_THRESHOLD = 0.05
LC_MAX_K_CAP = 5

# ── Precision-first tie-break ──
TIE_BREAK_THRESHOLD = 0.02


@dataclass
class KControllerResultV2:
    """Result of V2 K selection."""
    chosen_k: int = 5
    expected_f1_curve: dict[int, float] = field(default_factory=dict)
    g_hat: float = 0.0
    g_hat_scope: float = 0.0
    g_hat_pool: float = 0.0
    g_hat_visible: float = 0.0
    reason: str = ""
    probabilities_used: list[float] = field(default_factory=list)
    low_confidence_uniform: bool = False
    p_floor: float = 0.0
    tie_break_reason: str = ""
    second_pass_triggered: bool = False
    query_type: str = "unknown"


# ── V1 backward-compat shim ──
KControllerResult = KControllerResultV2


# ═══════════════════════════════════════════════════════════════════
# Task 1: Three-way g_hat estimation
# ═══════════════════════════════════════════════════════════════════

def _estimate_g_hat_scope(query_type: str, query_plan: Any | None = None) -> float:
    """g_hat_scope: prior from query type / query scope.

    Uses the query_type profile prior, potentially adjusted by
    constraint count and time range from query_plan.
    """
    profile = QUERY_TYPE_PROFILE.get(query_type, QUERY_TYPE_PROFILE["unknown"])
    prior = profile["prior"]

    # Adjust by constraint count: more constraints → fewer expected papers
    if query_plan is not None:
        must_have = getattr(query_plan, "must_have_constraints", []) or []
        if len(must_have) > 3:
            prior *= 0.7  # tight constraints → smaller scope
        elif len(must_have) == 0:
            prior *= 1.3  # no constraints → broader scope

    return max(1.0, prior)


def _estimate_g_hat_pool(
    ranked_papers: list[Any],
    all_candidates_count: int = 0,
) -> float:
    """g_hat_pool: estimate from candidate pool density.

    Uses the distribution of local_pre_rank_score / rrf_score in the
    full candidate pool (not just the top-24 seen by LLM). Papers with
    high local scores indicate density of relevant papers.
    """
    if not ranked_papers:
        return 1.0

    # Count papers with strong local signals
    high_local = 0
    med_local = 0
    for rp in ranked_papers:
        meta = rp.paper.metadata
        local = meta.get("local_pre_rank_score") or meta.get("rrf_score") or 0.0
        sel_level = rp.selection.relevance_level
        if local >= 0.50 or sel_level == "high":
            high_local += 1
        elif local >= 0.25 or sel_level == "medium":
            med_local += 1

    # Estimate: high_local contribute ~0.7, medium contribute ~0.35
    pool_est = high_local * 0.7 + med_local * 0.35

    # If we know the full candidate pool size, extrapolate
    if all_candidates_count > len(ranked_papers):
        # The ranked_papers are the top-N; the pool may have more relevant
        # papers we haven't seen. Scale up by ratio.
        ratio = all_candidates_count / max(1, len(ranked_papers))
        pool_est *= min(math.sqrt(ratio), 1.8)  # cap scaling at 1.8x

    return max(1.0, pool_est)


def _estimate_g_hat_visible(
    ranked_papers: list[Any],
    listwise_result: Any | None = None,
) -> float:
    """g_hat_visible: sum of calibrated probabilities from LLM listwise.

    This is the V1 approach — sum of top-N probabilities. But now it's
    only one of three signals, not the sole estimate.
    """
    prob_sum = 0.0
    has_probs = False
    for rp in ranked_papers:
        p = rp.paper.metadata.get("calibrated_probability")
        if p is not None:
            has_probs = True
            if p > 0.05:
                prob_sum += p

    if not has_probs and listwise_result is not None:
        # Fallback to raw LLM probabilities
        llm_probs = getattr(listwise_result, "relevance_probabilities", None) or {}
        for rp in ranked_papers:
            p = llm_probs.get(rp.paper.paper_id, 0.0)
            if p > 0.05:
                prob_sum += p

    return max(0.0, prob_sum)


def estimate_g_hat_v2(
    ranked_papers: list[Any],
    query_type: str = "unknown",
    listwise_result: Any | None = None,
    all_candidates_count: int = 0,
    query_plan: Any | None = None,
) -> tuple[float, float, float, float]:
    """Three-way g_hat fusion.

    Returns (g_hat_fused, g_hat_scope, g_hat_pool, g_hat_visible).
    """
    g_scope = _estimate_g_hat_scope(query_type, query_plan)
    g_pool = _estimate_g_hat_pool(ranked_papers, all_candidates_count)
    g_visible = _estimate_g_hat_visible(ranked_papers, listwise_result)

    # Fusion weights:
    # - g_visible is most precise but biased (only sees top-N)
    # - g_scope is a prior, unbiased but imprecise
    # - g_pool captures pool density, less biased than visible
    #
    # Weight assignment:
    #   If g_visible is very low (< 3) and g_scope is high (> 10),
    #   we're likely underestimating → trust scope/pool more.
    #   If g_visible is high (> 5), trust it more.
    if g_visible < 3.0 and g_scope > 10.0:
        # Likely LLM missed many relevant papers — upweight scope
        w_scope, w_pool, w_visible = 0.45, 0.35, 0.20
    elif g_visible > 8.0:
        # LLM found many — trust visible but keep scope as anchor
        w_scope, w_pool, w_visible = 0.25, 0.25, 0.50
    else:
        # Balanced
        w_scope, w_pool, w_visible = 0.35, 0.35, 0.30

    g_fused = w_scope * g_scope + w_pool * g_pool + w_visible * g_visible

    # Anchor: g_fused should not be below 0.5 * g_scope
    # (prevents underestimation when LLM misses papers)
    g_fused = max(g_fused, 0.5 * g_scope)

    # Clamp
    g_fused = max(1.0, min(60.0, g_fused))

    return g_fused, g_scope, g_pool, g_visible


# ═══════════════════════════════════════════════════════════════════
# V1 backward-compat: keep old function signature working
# ═══════════════════════════════════════════════════════════════════

# V1 level ranges (for backward compat with any external callers)
LEVEL_PROBABILITY_RANGES: dict[str, tuple[float, float]] = {
    "high": (0.80, 0.95),
    "medium": (0.40, 0.70),
    "low": (0.08, 0.25),
    "irrelevant": (0.0, 0.08),
}


def calibrate_relevance_probability(
    relevance_level: str,
    llm_probability: float | None = None,
    llm_confidence: float | None = None,
    local_score: float | None = None,
) -> float:
    """V1 backward-compat wrapper. Delegates to V2 calibration."""
    from scholar_agent.ranking.probability_calibration import calibrate_single
    _, cal = calibrate_single(
        llm_probability=llm_probability,
        relevance_level=relevance_level,
        llm_confidence=llm_confidence,
        local_score=local_score,
    )
    return cal


def calibrate_paper_probabilities(
    ranked_papers: list[Any],
    listwise_result: Any | None = None,
    query_plan: Any | None = None,
) -> None:
    """V2 backward-compat: calibrate all papers and write to metadata."""
    calibrate_paper_probabilities_v2(ranked_papers, listwise_result, query_plan)


def estimate_g_hat(
    ranked_papers: list[Any],
    query_type: str = "unknown",
    listwise_result: Any | None = None,
    all_candidates_count: int = 0,
    query_plan: Any | None = None,
) -> float:
    """V1 backward-compat: return fused g_hat."""
    g_fused, _, _, _ = estimate_g_hat_v2(
        ranked_papers, query_type, listwise_result, all_candidates_count, query_plan,
    )
    return g_fused


# ═══════════════════════════════════════════════════════════════════
# Task 3: Low-confidence uniform guard
# ═══════════════════════════════════════════════════════════════════

def _detect_low_confidence_uniform(probs: list[float], top_n: int = 10) -> bool:
    """Check if top-N probabilities are low and uniform.

    Returns True if mean < 0.35 and std < 0.05.
    """
    if len(probs) < 3:
        return False
    subset = probs[:min(top_n, len(probs))]
    if len(subset) < 3:
        return False
    mean_p = statistics.mean(subset)
    std_p = statistics.pstdev(subset) if len(subset) > 1 else 0.0
    return mean_p < LC_MEAN_THRESHOLD and std_p < LC_STD_THRESHOLD


# ═══════════════════════════════════════════════════════════════════
# Task 4: Marginal p_floor
# ═══════════════════════════════════════════════════════════════════

def _get_p_floor(query_type: str) -> float:
    """Get p_floor for query type: papers below this prob are not added."""
    profile = QUERY_TYPE_PROFILE.get(query_type, QUERY_TYPE_PROFILE["unknown"])
    return profile["p_floor"]


def _is_large_scope(query_type: str) -> bool:
    """Check if query type is large-scope (survey, broad, etc.)."""
    profile = QUERY_TYPE_PROFILE.get(query_type, QUERY_TYPE_PROFILE["unknown"])
    return profile.get("large_scope", False)


# ═══════════════════════════════════════════════════════════════════
# Task 6: Precision-first tie-break
# ═══════════════════════════════════════════════════════════════════

def _precision_first_tie_break(
    curve: dict[int, float],
    k_max: int,
    p_floor: float,
    probs: list[float],
    effective_k_max: int,
) -> tuple[int, str]:
    """Select K with precision-first tie-breaking.

    Rules:
    1. Among all K, find the maximum E[F1@K].
    2. Among all K within TIE_BREAK_THRESHOLD of the max, pick the smallest.
    3. Additionally, enforce p_floor: K cannot extend beyond the point
       where p_next < p_floor.
    """
    if not curve:
        return 0, "Empty curve."

    # Step 1: Find max F1
    max_f1 = max(curve.values())

    # Step 2: Find all K within threshold of max
    candidates_ks = [
        k for k, f1 in curve.items()
        if max_f1 - f1 <= TIE_BREAK_THRESHOLD and k <= effective_k_max
    ]

    if not candidates_ks:
        # Shouldn't happen, but fallback
        best_k = max(curve, key=lambda k: curve[k])
        return best_k, f"Fallback: max F1 at K={best_k}."

    # Step 3: Among candidates, apply p_floor constraint
    # Start from smallest K and expand only if p_next >= p_floor
    best_k = min(candidates_ks)
    for k in range(best_k + 1, max(candidates_ks) + 1):
        if k - 1 < len(probs):
            p_next = probs[k - 1]
            if p_next < p_floor:
                # Stop expanding — p too low
                break
            # Check if this K is also in candidates
            if k in candidates_ks:
                best_k = k

    # Tie-break reason
    if best_k == min(candidates_ks) and len(candidates_ks) > 1:
        reason = (
            f"Precision-first: {len(candidates_ks)} K values within "
            f"{TIE_BREAK_THRESHOLD} of max F1={max_f1:.4f}; "
            f"chose smallest K={best_k}."
        )
    elif best_k == min(candidates_ks):
        reason = f"K={best_k} has max E[F1]={max_f1:.4f}."
    else:
        reason = (
            f"Expanded to K={best_k} (p_floor={p_floor:.2f} satisfied, "
            f"E[F1]={curve[best_k]:.4f})."
        )

    return best_k, reason


# ═══════════════════════════════════════════════════════════════════
# Main V2 entry point
# ═══════════════════════════════════════════════════════════════════

def decide_k_via_expected_f1(
    ranked_papers: list[Any],
    query_type: str = "unknown",
    listwise_result: Any | None = None,
    k_max: int = 20,
    config: Any | None = None,
    all_candidates_count: int = 0,
    query_plan: Any | None = None,
) -> KControllerResultV2:
    """V2 K selection with three-way g_hat, LC guard, p_floor, precision tie-break.

    Args:
        ranked_papers: sorted list of RankedPaper (by final_score, descending)
        query_type: query intent type
        listwise_result: optional LLM listwise result with probabilities
        k_max: maximum K to consider
        config: AppConfig (for hard_max_output override)
        all_candidates_count: total candidates in pool (for g_hat_pool extrapolation)
        query_plan: QueryPlan (for scope estimation and constraint analysis)

    Returns:
        KControllerResultV2 with chosen_k, curve, g_hat breakdown, flags.
    """
    if not ranked_papers:
        return KControllerResultV2(chosen_k=0, reason="No candidates.")

    # Get hard_max_output from config
    hard_max = 20
    dk_cfg = getattr(config, "dynamic_k", None) if config else None
    if dk_cfg:
        hard_max = getattr(dk_cfg, "hard_max_output", 20)
    k_max = min(k_max, hard_max, len(ranked_papers))

    # Ensure V2 probabilities are calibrated
    has_calibrated = any(
        rp.paper.metadata.get("calibrated_probability") is not None
        for rp in ranked_papers
    )
    if not has_calibrated:
        calibrate_paper_probabilities_v2(ranked_papers, listwise_result, query_plan)

    # Get probabilities for top-k_max papers
    probs: list[float] = []
    for rp in ranked_papers[:k_max]:
        p = rp.paper.metadata.get("calibrated_probability")
        if p is None:
            p = max(0.0, min(1.0, rp.final_score or 0.0))
        probs.append(p)

    # ── Task 1: Three-way g_hat ──
    if all_candidates_count == 0:
        all_candidates_count = len(ranked_papers)
    g_hat, g_scope, g_pool, g_visible = estimate_g_hat_v2(
        ranked_papers, query_type, listwise_result,
        all_candidates_count, query_plan,
    )

    # ── Task 3: Low-confidence uniform guard ──
    lc_uniform = _detect_low_confidence_uniform(probs, top_n=10)
    if lc_uniform:
        cap = 10 if _is_large_scope(query_type) else LC_MAX_K_CAP
        effective_k_max = min(k_max, cap)
        lc_msg = (
            f"LC guard: top10 mean={statistics.mean(probs[:10]):.3f} "
            f"std={statistics.pstdev(probs[:10]) if len(probs) > 1 else 0:.3f} → "
            f"max_k capped to {cap}."
        )
        LOGGER.info(lc_msg)
    else:
        effective_k_max = k_max

    # ── Task 4: p_floor ──
    p_floor = _get_p_floor(query_type)

    # Compute expected F1 curve
    cumulative_tp = 0.0
    curve: dict[int, float] = {}
    for k in range(1, effective_k_max + 1):
        cumulative_tp += probs[k - 1]
        if k + g_hat > 0:
            exp_f1 = 2.0 * cumulative_tp / (k + g_hat)
        else:
            exp_f1 = 0.0
        curve[k] = round(exp_f1, 4)

    # ── Task 6: Precision-first tie-break ──
    best_k, tie_reason = _precision_first_tie_break(
        curve, k_max, p_floor, probs, effective_k_max,
    )

    # Build reason string
    top_probs_str = ", ".join(f"{p:.2f}" for p in probs[:min(5, len(probs))])
    reason_parts = [
        f"g_hat={g_hat:.1f} (scope={g_scope:.1f}, pool={g_pool:.1f}, visible={g_visible:.1f})",
        f"best_K={best_k} (E[F1@{best_k}]={curve.get(best_k, 0):.4f})",
        f"top_probs=[{top_probs_str}]",
        f"query_type={query_type}",
    ]
    if lc_uniform:
        reason_parts.append("LC_GUARD_ACTIVE")
    reason_parts.append(f"p_floor={p_floor:.2f}")

    reason = ", ".join(reason_parts)

    LOGGER.info(
        "F1-K-Controller-V2: %s. %s. Full curve: %s",
        reason, tie_reason,
        {k: f"{v:.4f}" for k, v in curve.items()},
    )

    # Check if second-pass should be triggered (Task 5 — decided by caller)
    num_confident = sum(1 for p in probs[:min(30, len(probs))] if p >= 0.50)
    second_pass_needed = (
        _is_large_scope(query_type)
        and num_confident <= 2
        and not lc_uniform  # if LC guard is active, second pass won't help much
    )

    return KControllerResultV2(
        chosen_k=best_k,
        expected_f1_curve=curve,
        g_hat=g_hat,
        g_hat_scope=g_scope,
        g_hat_pool=g_pool,
        g_hat_visible=g_visible,
        reason=reason,
        probabilities_used=probs,
        low_confidence_uniform=lc_uniform,
        p_floor=p_floor,
        tie_break_reason=tie_reason,
        second_pass_triggered=second_pass_needed,
        query_type=query_type,
    )
