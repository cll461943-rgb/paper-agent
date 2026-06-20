from __future__ import annotations

from typing import Any
from scholar_agent.evaluation.metrics import score_papers_against_gold, calculate_hit_at_k

def evaluate_topk_sweep(
    all_predictions: dict[str, list[Any]],
    all_golds: dict[str, list[dict[str, Any] | str]],
    ks: list[int] | None = None
) -> dict[str, dict[str, float]]:
    """
    对不同 K (默认 [1, 2, 3, 5, 10]) 进行 Sweep 扫描评估。
    返回一个网格，例如 {"f1@1": 0.25, "precision@1": 0.3, "recall@1": 0.2, ...}
    """
    if ks is None:
        ks = [1, 2, 3, 5, 10]
        
    sweep_results: dict[str, dict[str, float]] = {}
    
    for k in ks:
        total_f1_strict = 0.0
        total_precision_strict = 0.0
        total_recall_strict = 0.0
        
        total_f1_relaxed = 0.0
        total_precision_relaxed = 0.0
        total_recall_relaxed = 0.0
        
        total_hit = 0.0
        
        n_queries = len(all_predictions)
        if n_queries == 0:
            continue
            
        for qid, papers in all_predictions.items():
            gold = all_golds.get(qid, [])
            
            # 计算 strict
            strict_scores = score_papers_against_gold(papers[:k], gold, strict=True)
            total_f1_strict += strict_scores["f1"]
            total_precision_strict += strict_scores["precision"]
            total_recall_strict += strict_scores["recall"]
            
            # 计算 relaxed
            relaxed_scores = score_papers_against_gold(papers[:k], gold, strict=False)
            total_f1_relaxed += relaxed_scores["f1"]
            total_precision_relaxed += relaxed_scores["precision"]
            total_recall_relaxed += relaxed_scores["recall"]
            
            # 计算 Hit@K
            total_hit += calculate_hit_at_k(papers, gold, k)
            
        sweep_results[f"K={k}"] = {
            "strict_f1": round(total_f1_strict / n_queries, 4),
            "strict_precision": round(total_precision_strict / n_queries, 4),
            "strict_recall": round(total_recall_strict / n_queries, 4),
            
            "relaxed_f1": round(total_f1_relaxed / n_queries, 4),
            "relaxed_precision": round(total_precision_relaxed / n_queries, 4),
            "relaxed_recall": round(total_recall_relaxed / n_queries, 4),
            
            "hit_ratio": round(total_hit / n_queries, 4)
        }
        
    return sweep_results
