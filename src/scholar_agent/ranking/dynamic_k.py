from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

def decide_dynamic_k(query_contract: Any, ranked_papers: list[Any], config: Any) -> int:
    """
    自适应决定最终推荐论文数目 K。
    - 对精确论文查询、单金标查询限制 K = 1
    - 对数据集约束、方法对比查询限制 K = 3
    - 对综述、最新进展查询限制 K = 5
    - 支持分数断崖截断（第一名明显高于第二名直接切 K = 1）
    - 支持分数门槛筛选（保留得分高于 min_final_score 的文献）
    """
    if not ranked_papers:
        return 0

    # 1. 提取配置，建立默认值兜底
    dynamic_k_cfg = getattr(config, "dynamic_k", None)
    score_gap_top1 = 0.15
    exact_title_k = 1
    method_comparison_k = 5
    survey_k = 20  # Survey 类型需要进行大量检索，不应被硬性截断
    min_final_score = 0.45  # 降低门槛，减少开方漏掌的情况
    fallback_k = 5
    max_final_papers = 20

    if dynamic_k_cfg:
        score_gap_top1 = getattr(dynamic_k_cfg, "score_gap_top1", 0.15)
        exact_title_k = getattr(dynamic_k_cfg, "exact_title_k", 1)
        method_comparison_k = getattr(dynamic_k_cfg, "method_comparison_k", 3)
        survey_k = getattr(dynamic_k_cfg, "survey_k", 5)
        min_final_score = getattr(dynamic_k_cfg, "min_final_score", 0.55)
        fallback_k = getattr(dynamic_k_cfg, "fallback_k", 3)

    ranking_cfg = getattr(config, "ranking", None)
    if ranking_cfg:
        max_final_papers = getattr(ranking_cfg, "max_final_papers", 10)

    # 2. 分数断崖规则 (Cliff cut-off)
    if len(ranked_papers) >= 2:
        score_diff = ranked_papers[0].final_score - ranked_papers[1].final_score
        if score_diff >= score_gap_top1:
            LOGGER.info("Dynamic-K Cliff Cut-off triggered: Score difference between Rank 1 & 2 is %.3f >= %.3f. Setting K = 1.", score_diff, score_gap_top1)
            return 1

    # 3. 意图类别匹配规则
    # 从 query_contract 获取 query_type 属性，做柔性兼容
    query_type = getattr(query_contract, "query_type", "unknown")
    
    # 启发式兜底：若 query_type 为 unknown，但 original_query 中包含 survey 等，自动推断为相应类型
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

    if query_type in ("exact_title", "specific_paper", "single_gold"):
        k = min(exact_title_k, len(ranked_papers))
        LOGGER.info("Dynamic-K: Exact title / Specific paper intent matched. Setting K = %d", k)
        return k

    if query_type in ("dataset_constraint", "method_comparison"):
        k = min(method_comparison_k, len(ranked_papers))
        LOGGER.info("Dynamic-K: Method comparison / Dataset constraint intent matched. Setting K = %d", k)
        return k

    if query_type in ("survey", "broad_topic", "latest_work"):
        k = min(survey_k, len(ranked_papers))
        LOGGER.info("Dynamic-K: Survey / Broad topic intent matched. Setting K = %d", k)
        return k

    # 4. 分数门槛筛选
    valid_papers = [
        p for p in ranked_papers
        if p.final_score is not None and p.final_score >= min_final_score
    ]
    if valid_papers:
        k = min(len(valid_papers), max_final_papers)
        LOGGER.info("Dynamic-K: Filtered by score threshold >= %.3f. Setting K = %d", min_final_score, k)
        return k

    k = min(fallback_k, len(ranked_papers))
    LOGGER.info("Dynamic-K: Fallback triggered. Setting K = %d", k)
    return k
