from __future__ import annotations

from scholar_agent.evaluation.metrics import (
    score_papers_against_gold,
    calculate_hit_at_k,
    calculate_precision_recall_f1_at_k,
    compute_selector_stats,
    is_paper_match_gold,
)
from scholar_agent.evaluation.topk_sweep import evaluate_topk_sweep
from scholar_agent.evaluation.error_analysis import (
    compute_candidate_to_final_loss,
    classify_drop_reason,
)

__all__ = [
    "score_papers_against_gold",
    "calculate_hit_at_k",
    "calculate_precision_recall_f1_at_k",
    "compute_selector_stats",
    "is_paper_match_gold",
    "evaluate_topk_sweep",
    "compute_candidate_to_final_loss",
    "classify_drop_reason",
]
