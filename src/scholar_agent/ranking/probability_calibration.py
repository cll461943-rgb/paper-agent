"""Probability Calibration Module (K Controller V2, Task 2).

Multi-signal calibration of LLM relevance probabilities. Combines:
  - LLM relevance_probability + relevance_level + confidence
  - constraint_coverage (from SelectionResult)
  - title_similarity / abstract_overlap (local lexical signals)
  - rrf_score / source_agreement (retrieval-level signals)

Key improvements over V1:
  - Cooling: low/weak-medium probabilities are pushed down to avoid
    uniform 0.25-0.31 bands that cause K inflation.
  - Multi-signal fusion: not solely dependent on LLM probability.
  - Both raw_probability and calibrated_probability are stored.
"""
from __future__ import annotations

import logging
import math
from typing import Any

LOGGER = logging.getLogger(__name__)

# ── Level → (low, high) probability range ──
LEVEL_RANGES: dict[str, tuple[float, float]] = {
    "high": (0.80, 0.95),
    "medium": (0.40, 0.70),
    "low": (0.08, 0.25),
    "irrelevant": (0.0, 0.08),
}

# ── Cooling parameters ──
# When mean top-10 prob is low and uniform, we apply extra cooling
# to push weak-medium papers down further.
COOLING_THRESHOLD = 0.35  # if calibrated prob < this, apply cooling
COOLING_FACTOR = 0.6      # multiply by this factor


def _sigmoid(x: float, k: float = 6.0, midpoint: float = 0.5) -> float:
    """Sigmoid with adjustable steepness — used for smooth fusion."""
    return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def calibrate_single(
    llm_probability: float | None = None,
    relevance_level: str = "unknown",
    llm_confidence: float | None = None,
    constraint_coverage: float | None = None,
    title_similarity: float | None = None,
    abstract_overlap: float | None = None,
    rrf_score: float | None = None,
    source_agreement: float | None = None,
    local_score: float | None = None,
) -> tuple[float, float]:
    """Calibrate a single paper's probability from multiple signals.

    Returns (raw_probability, calibrated_probability).
    raw_probability is the pre-cooling fused value; calibrated_probability
    has cooling applied.
    """
    # ── Step 1: Level-based prior ──
    lo, hi = LEVEL_RANGES.get(relevance_level, (0.0, 0.1))
    level_mid = (lo + hi) / 2.0

    # ── Step 2: LLM probability signal ──
    llm_p = llm_probability if llm_probability is not None else level_mid

    # ── Step 3: Local signals (lexical + retrieval) ──
    local_signals: list[float] = []
    if constraint_coverage is not None:
        local_signals.append(constraint_coverage)
    if title_similarity is not None:
        local_signals.append(title_similarity)
    if abstract_overlap is not None:
        local_signals.append(abstract_overlap)
    if rrf_score is not None:
        local_signals.append(min(1.0, rrf_score))
    if source_agreement is not None:
        local_signals.append(source_agreement)
    if local_score is not None:
        local_signals.append(local_score)

    local_avg = sum(local_signals) / len(local_signals) if local_signals else level_mid

    # ── Step 4: Fuse LLM probability with local signals ──
    # Weight: LLM gets alpha, local gets (1-alpha)
    # alpha depends on confidence and number of local signals available
    alpha = 0.55
    if llm_confidence is not None:
        alpha = _clamp(0.3 + 0.4 * llm_confidence, 0.3, 0.8)
    n_local = len(local_signals)
    if n_local < 2:
        alpha = max(alpha, 0.65)  # trust LLM more if few local signals

    fused = alpha * llm_p + (1.0 - alpha) * local_avg

    # ── Step 5: Level-based clamping with margin ──
    margin = 0.05
    fused = _clamp(fused, max(0.0, lo - margin), min(1.0, hi + margin))

    raw_probability = round(fused, 4)

    # ── Step 6: Cooling — push down low/weak-medium probabilities ──
    # This is the key V2 improvement: if the LLM's raw probability is low
    # (< 0.35), apply aggressive cooling to prevent uniform low-confidence
    # bands from inflating K. The cooled value is capped near the LLM's
    # actual assessment, not the level midpoint.
    calibrated = fused
    cooling_trigger = llm_probability if llm_probability is not None else fused
    if cooling_trigger < COOLING_THRESHOLD and relevance_level in ("low", "medium", "unknown"):
        # Cap at: min(fused, llm_p * 1.15, 0.30)
        # This ensures cooled value stays below 0.35 (LC guard threshold)
        cap = min(fused, 0.30)
        if llm_probability is not None:
            cap = min(cap, llm_probability * 1.15)
        calibrated = _clamp(cap, 0.0, hi + margin)

    # Extra cooling for "low" level: cap at 0.15
    if relevance_level == "low":
        calibrated = min(calibrated, 0.15)
    # Extra cooling for "irrelevant": cap at 0.05
    if relevance_level == "irrelevant":
        calibrated = min(calibrated, 0.05)

    calibrated_probability = round(calibrated, 4)
    return raw_probability, calibrated_probability


def calibrate_paper_probabilities_v2(
    ranked_papers: list[Any],
    listwise_result: Any | None = None,
    query_plan: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Calibrate probabilities for all ranked papers using V2 multi-signal.

    Writes into paper.metadata:
      - raw_probability
      - calibrated_probability

    Returns a dict: paper_id -> {"raw": float, "calibrated": float}
    """
    # Build lookup from listwise result
    llm_probs: dict[str, float] = {}
    llm_levels: dict[str, str] = {}
    llm_confs: dict[str, float] = {}
    if listwise_result is not None:
        llm_probs = getattr(listwise_result, "relevance_probabilities", None) or {}
        llm_levels = getattr(listwise_result, "relevance_levels", None) or {}
        llm_confs = getattr(listwise_result, "relevance_confidences", None) or {}

    # Extract must_have constraints count for coverage normalization
    must_have = []
    if query_plan is not None:
        must_have = getattr(query_plan, "must_have_constraints", []) or []

    result: dict[str, dict[str, float]] = {}

    for rp in ranked_papers:
        paper = rp.paper
        pid = paper.paper_id
        meta = paper.metadata

        # LLM signals
        level = llm_levels.get(pid) or rp.selection.relevance_level or "unknown"
        llm_p = llm_probs.get(pid)
        llm_conf = llm_confs.get(pid)

        # Constraint coverage: matched / total must_have
        matched = rp.selection.matched_constraints or []
        if must_have and len(must_have) > 0:
            constraint_coverage = len(matched) / len(must_have)
        else:
            # Fallback: use selection constraint_score
            cs = rp.selection.constraint_score
            constraint_coverage = cs if cs is not None else None

        # Local signals from metadata
        title_sim = meta.get("title_similarity")
        abstract_overlap = meta.get("abstract_overlap")
        rrf_score = meta.get("rrf_score") or meta.get("local_pre_rank_score")
        source_agreement = meta.get("source_agreement")

        # Local pre-rank score as fallback
        local_score = meta.get("local_pre_rank_score")

        raw_p, cal_p = calibrate_single(
            llm_probability=llm_p,
            relevance_level=level,
            llm_confidence=llm_conf,
            constraint_coverage=constraint_coverage,
            title_similarity=title_sim,
            abstract_overlap=abstract_overlap,
            rrf_score=rrf_score,
            source_agreement=source_agreement,
            local_score=local_score,
        )

        meta["raw_probability"] = raw_p
        meta["calibrated_probability"] = cal_p
        result[pid] = {"raw": raw_p, "calibrated": cal_p}

    LOGGER.debug(
        "V2 calibration: %d papers, mean calibrated=%.3f",
        len(ranked_papers),
        sum(r["calibrated"] for r in result.values()) / max(1, len(result)),
    )
    return result
