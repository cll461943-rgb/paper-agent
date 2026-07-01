from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

def decide_dynamic_k(query_contract: Any, ranked_papers: list[Any], config: Any) -> int:
    """
    自适应决定最终推荐论文数目 K。
    - 对精确论文查询、单金标查询限制 K = 1
    - 对数据集约束、方法对比查询限制 K = 5-8
    - 对综述、最新进展查询限制 K >= 10
    - 分数断崖截断仅对 exact_title/single_gold/specific_paper 生效
    - 分数门槛筛选 min_final_score = 0.40-0.45
    - hard_max_output = 12: 硬上限，无论查询类型都适用
    - recall guard: final_score>=0.65 或 relevance_level=high 的候选由 synthesis 层保留
    """
    if not ranked_papers:
        return 0

    # 1. 提取配置，建立默认值兜底
    dynamic_k_cfg = getattr(config, "dynamic_k", None)
    score_gap_top1 = 0.15
    exact_title_k = 1
    specific_query_k = 2
    method_comparison_k = 6
    survey_k = 10
    min_final_score = 0.40
    fallback_k = 5
    max_final_papers = 20
    hard_max_output = 12  # P2: Hard cap regardless of query type

    if dynamic_k_cfg:
        score_gap_top1 = getattr(dynamic_k_cfg, "score_gap_top1", 0.15)
        exact_title_k = getattr(dynamic_k_cfg, "exact_title_k", 1)
        specific_query_k = getattr(dynamic_k_cfg, "specific_query_k", 2)
        method_comparison_k = getattr(dynamic_k_cfg, "method_comparison_k", 6)
        survey_k = getattr(dynamic_k_cfg, "survey_k", 10)
        min_final_score = getattr(dynamic_k_cfg, "min_final_score", 0.40)
        fallback_k = getattr(dynamic_k_cfg, "fallback_k", 5)
        hard_max_output = getattr(dynamic_k_cfg, "hard_max_output", 12)

    ranking_cfg = getattr(config, "ranking", None)
    if ranking_cfg:
        max_final_papers = getattr(ranking_cfg, "max_final_papers", 20)

    # 2. 意图类别匹配（提前到 cliff cut-off 之前）
    query_type = getattr(query_contract, "query_type", "unknown")
    original_query = getattr(query_contract, "original_query", "").lower()
    if query_type == "unknown":
        if "survey" in original_query or "review" in original_query or "综述" in original_query:
            query_type = "survey"
        elif "comparison" in original_query or "compare" in original_query or "对比" in original_query or "比较" in original_query:
            query_type = "method_comparison"
        elif any(c in original_query for c in ("dataset", "bench", "eval", "数据集")):
            query_type = "dataset_constraint"
        elif any(c in original_query for c in ("latest", "recent", "newest", "最新")):
            query_type = "latest_work"

    k: int | None = None

    # 3. 分数断崖规则 (Cliff cut-off) — P0-5: 仅对精确查询生效
    if query_type in ("exact_title", "single_gold", "specific_paper") and len(ranked_papers) >= 2:
        score_diff = ranked_papers[0].final_score - ranked_papers[1].final_score
        if score_diff >= score_gap_top1:
            LOGGER.info("Dynamic-K Cliff Cut-off: score gap %.3f >= %.3f. K = 1.", score_diff, score_gap_top1)
            k = 1

    # 4. 意图类别匹配规则
    if k is None:
        if query_type in ("exact_title", "single_gold"):
            k = min(exact_title_k, len(ranked_papers))
            LOGGER.info("Dynamic-K: Exact title / Single gold. K = %d", k)
        elif query_type == "specific_paper":
            k = min(specific_query_k, len(ranked_papers))
            LOGGER.info("Dynamic-K: Specific paper. K = %d", k)
        elif query_type in ("dataset_constraint", "method_comparison"):
            k = min(method_comparison_k, len(ranked_papers))
            LOGGER.info("Dynamic-K: Method comparison / Dataset constraint. K = %d", k)
        elif query_type in ("survey", "broad_topic", "latest_work"):
            k = min(survey_k, len(ranked_papers))
            LOGGER.info("Dynamic-K: Survey / Broad topic / Latest. K = %d", k)
        else:
            # 5. 分数门槛筛选
            valid_papers = [
                p for p in ranked_papers
                if p.final_score is not None and p.final_score >= min_final_score
            ]
            if valid_papers:
                k = min(len(valid_papers), max_final_papers)
                LOGGER.info("Dynamic-K: Score threshold >= %.3f. K = %d", min_final_score, k)
            else:
                k = min(fallback_k, len(ranked_papers))
                LOGGER.info("Dynamic-K: Fallback. K = %d", k)

    # P2: Apply hard_max_output cap regardless of query type
    if k > hard_max_output:
        LOGGER.info("Dynamic-K: Hard cap applied: %d -> %d", k, hard_max_output)
        k = hard_max_output

    return k
