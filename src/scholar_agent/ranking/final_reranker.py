from __future__ import annotations

import logging
from typing import Any

from scholar_agent.models.schemas import Paper, RankedPaper, SelectionResult
from scholar_agent.selection.selection_cutoff import (
    FrontierPartition,
    partition_ranked_papers,
    summarize_partition,
)

LOGGER = logging.getLogger(__name__)


def compute_paper_score(
    paper: Paper,
    selection: SelectionResult,
    other_paper_ids: set[str] | None = None
) -> tuple[float, dict[str, float]]:
    """依据 V2.0 多维度综合评分公式计算单篇论文的最终得分和各个维度的子项得分。"""
    subscores: dict[str, float] = {}

    # 1. Atomic Selector 相关性分
    relevance_map = {"high": 1.0, "medium": 0.55, "low": 0.15, "irrelevant": 0.0}
    llm_relevance = relevance_map.get(selection.relevance_level, 0.0)
    sel_label = getattr(selection, "label", None)
    if sel_label == "gold":
        llm_relevance = 1.0
    elif sel_label == "strong_related":
        llm_relevance = max(llm_relevance, 0.9)
    elif sel_label == "medium_core":
        llm_relevance = min(max(llm_relevance, 0.55), 0.70)
    elif sel_label == "bridge":
        llm_relevance = min(max(llm_relevance, 0.35), 0.50)
    elif sel_label == "weak":
        llm_relevance = min(llm_relevance, 0.20)
    subscores["LLM_Relevance"] = llm_relevance

    # 2. BGE 精排分。缺失时不再默认 0.5，避免无证据候选获得虚高基础分。
    # 优先从 metadata 中读取 BGE 精排分，比如 'bge_score'
    bge_score = paper.metadata.get("bge_score")
    if bge_score is None:
        bge_score = paper.metadata.get("vector_score", 0.0)
    else:
        try:
            bge_score = float(bge_score)
        except (ValueError, TypeError):
            bge_score = 0.0
    subscores["BGE_Reranker"] = bge_score

    # 3. 约束命中率 (权重 0.20)
    matched_count = len(selection.matched_constraints)
    missing_count = len(selection.missing_constraints)
    total_constraints = matched_count + missing_count
    if total_constraints > 0:
        constraint_coverage = matched_count / total_constraints
    else:
        # 如果没有明确的约束项，中/高相关性给 1.0，低/无关给 0.5
        constraint_coverage = 1.0 if selection.relevance_level in ("high", "medium") else 0.5
    subscores["Constraint_Coverage"] = constraint_coverage

    # 4. 证据完整性
    # 高中相关性且具有明确提取证据的得分 1.0，否则 0.0
    evidence_completeness = getattr(selection, "evidence_support", None)
    if evidence_completeness is None:
        evidence_completeness = selection.evidence_score
    if evidence_completeness is None or evidence_completeness <= 0.0:
        evidence_completeness = 1.0 if selection.relevance_level in ("high", "medium") and len(selection.evidence) > 0 else 0.0
    evidence_completeness = min(max(float(evidence_completeness), 0.0), 1.0)
    subscores["Evidence_Completeness"] = evidence_completeness

    # 5. 多源共识分/召回率 (权重 0.10)
    # 基于 retrieval_path 长度
    ret_paths = paper.retrieval_path or []
    if ret_paths:
        source_agreement = min(len(ret_paths) / 2.0, 1.0)
    else:
        # 默认中位
        source_agreement = 0.5
    subscores["Source_Agreement"] = source_agreement

    # 6. 年份新颖度 (权重 0.05)
    # 基准为 2026 年
    current_year = 2026
    if paper.year is not None:
        recency = max(0.0, 1.0 - (current_year - paper.year) * 0.1)
    else:
        recency = 0.5
    subscores["Recency"] = recency

    # 7. 权威度 (权重 0.03)
    # 基于 citation_count
    if paper.citation_count is not None:
        authority = min(paper.citation_count / 100.0, 1.0)
    else:
        authority = 0.0
    subscores["Authority"] = authority

    # 8. 图多样性先验 (权重 0.02)
    # 检查当前文章是否与其它候选文章有引文连接
    diversity_graph_prior = 0.0
    if other_paper_ids:
        # references 或 citations 包含其它候选文章 ID，或者其它文章在 metadata 里提到
        # 为简化，比对 paper_id 或者是 DOI、Title (这儿比对 paper_id 即可)
        ref_set = set(paper.references or [])
        cit_set = set(paper.citations or [])
        # 求交集
        if ref_set.intersection(other_paper_ids) or cit_set.intersection(other_paper_ids):
            diversity_graph_prior = 1.0
    subscores["Diversity_Graph_Prior"] = diversity_graph_prior

    # 9. 路径匹配偏置 (Path matching bonus)
    is_exact_title = any("title_exact" in p for p in ret_paths)
    is_like_title = any("title_like" in p for p in ret_paths)
    title_exact_bonus = 0.10 if is_exact_title else 0.0
    title_like_bonus = 0.03 if (is_like_title and not is_exact_title) else 0.0

    # 9b. LLM Listwise Score — the PRIMARY LLM signal from the listwise reranker.
    # This was previously written to metadata but NEVER READ by compute_paper_score,
    # causing the entire listwise reranking output to be silently dropped.
    # Now it's the dominant component (effect-first: LLM-led reranking).
    llm_listwise_score = paper.metadata.get("llm_listwise_score")
    if llm_listwise_score is not None:
        try:
            llm_listwise_score = float(llm_listwise_score)
        except (ValueError, TypeError):
            llm_listwise_score = None
    subscores["LLM_Listwise"] = llm_listwise_score if llm_listwise_score is not None else 0.0

    # 9c. relevance_probability — separate from rank_score.
    # Used by the Expected-Fβ K controller for cutoff decisions.
    rel_prob = paper.metadata.get("relevance_probability")
    if rel_prob is not None:
        try:
            rel_prob = float(rel_prob)
        except (ValueError, TypeError):
            rel_prob = None
    subscores["relevance_probability"] = rel_prob if rel_prob is not None else llm_relevance

    local_pre_rank = paper.metadata.get("constraint_aware_local_score")
    if local_pre_rank is None:
        local_pre_rank = paper.metadata.get("local_pre_rank_score")
    try:
        local_pre_rank = float(local_pre_rank) if local_pre_rank is not None else 0.0
    except (ValueError, TypeError):
        local_pre_rank = 0.0
    subscores["Local_PreRank"] = min(max(local_pre_rank, 0.0), 1.0)

    # Effect-first weight scheme: LLM listwise is the primary signal (0.50).
    # If listwise_score is missing (reranker failed/skipped), redistribute its
    # weight to LLM_Relevance (evidence selector) so the formula degrades gracefully.
    if llm_listwise_score is not None and llm_listwise_score > 0:
        # Normal path: listwise reranker ran successfully
        final_score = (
            subscores["LLM_Listwise"]          * 0.50 +
            subscores["LLM_Relevance"]         * 0.15 +
            subscores["Constraint_Coverage"]   * 0.12 +
            subscores["Evidence_Completeness"] * 0.08 +
            subscores["Source_Agreement"]      * 0.05 +
            subscores["BGE_Reranker"]          * 0.04 +
            subscores["Recency"]               * 0.02 +
            subscores["Authority"]             * 0.02 +
            subscores["Diversity_Graph_Prior"] * 0.02 +
            title_exact_bonus +
            title_like_bonus
        )
    else:
        # Fallback path: listwise reranker didn't run or returned 0
        # Constraint-aware local pre-rank is important when the LLM is unavailable.
        final_score = (
            subscores["LLM_Relevance"]         * 0.22 +
            subscores["Local_PreRank"]         * 0.18 +
            subscores["BGE_Reranker"]          * 0.14 +
            subscores["Constraint_Coverage"]   * 0.15 +
            subscores["Source_Agreement"]      * 0.10 +
            subscores["Evidence_Completeness"] * 0.09 +
            subscores["Recency"]               * 0.04 +
            subscores["Authority"]             * 0.03 +
            subscores["Diversity_Graph_Prior"] * 0.05 +
            title_exact_bonus +
            title_like_bonus
        )
    final_score = min(max(final_score, 0.0), 1.0)
    score_cap = getattr(selection, "score_cap", None)
    if score_cap is not None:
        final_score = min(final_score, max(0.0, min(float(score_cap), 1.0)))
        subscores["Score_Cap"] = float(score_cap)

    return final_score, subscores


def rerank_papers(
    papers: list[Paper],
    selections: list[SelectionResult],
    config: Any | None = None,
    original_query: str | None = None,
    listwise_scores: dict[str, float] | None = None,
    relevance_probabilities: dict[str, float] | None = None,
) -> list[RankedPaper]:
    """对候选论文列表进行重排，计算综合评分，赋予 Rank 序号。"""
    if not papers:
        return []

    # Write listwise scores into paper.metadata for compute_paper_score
    if listwise_scores:
        for paper in papers:
            score = listwise_scores.get(paper.paper_id)
            if score is not None:
                paper.metadata["llm_listwise_score"] = float(score)

    # Write relevance probabilities into paper.metadata for K controller
    if relevance_probabilities:
        for paper in papers:
            prob = relevance_probabilities.get(paper.paper_id)
            if prob is not None:
                paper.metadata["relevance_probability"] = float(prob)

    # Normalize vector_score to [0, 1] for BGE_Reranker
    # faiss_vector returns raw dot-product scores (often > 1.0), not normalized cosine similarity.
    # Without normalization, all papers get BGE > 1.0, providing zero relative differentiation.
    _vec_scores = []
    for paper in papers:
        if paper.metadata.get("bge_score") is None:
            vs = paper.metadata.get("vector_score")
            if vs is not None:
                try:
                    vs = float(vs)
                    if vs > 0:
                        _vec_scores.append(vs)
                except (ValueError, TypeError):
                    pass
    if _vec_scores:
        _max_vec = max(_vec_scores)
        if _max_vec > 0:
            for paper in papers:
                if paper.metadata.get("bge_score") is None:
                    vs = paper.metadata.get("vector_score")
                    if vs is not None:
                        try:
                            paper.metadata["bge_score"] = min(float(vs) / _max_vec, 1.0)
                        except (ValueError, TypeError):
                            pass

    # 建立 paper_id 映射
    selection_map = {sel.paper_id: sel for sel in selections}
    paper_ids = {p.paper_id for p in papers}

    ranked_list: list[RankedPaper] = []

    for paper in papers:
        selection = selection_map.get(paper.paper_id)
        if not selection:
            selection = SelectionResult(
                paper_id=paper.paper_id,
                relevance_level="low",
                reason="No selection metadata found."
            )
        
        # 计算多样性引用关系时，排除自己
        other_ids = paper_ids - {paper.paper_id}
        score, subscores = compute_paper_score(paper, selection, other_ids)
        
        ranked_list.append(
            RankedPaper(
                paper=paper,
                selection=selection,
                final_score=score,
                subscores=subscores,
                rank=0  # 占位，排序后再填充
            )
        )

    # 降序排列
    ranked_list.sort(key=lambda x: x.final_score, reverse=True)

    # 填充真正的 rank (从 1 开始)
    for idx, rp in enumerate(ranked_list, start=1):
        rp.rank = idx

    # DEBUG: Print top-10 papers' scores and subscores
    print("=== RERANK DEBUG: Top-10 papers ===")
    for rp in ranked_list[:10]:
        _s = rp.subscores
        _title = (rp.paper.title or "")[:50]
        _pid = rp.paper.paper_id[:30]
        _has_listwise = "Y" if rp.paper.metadata.get("llm_listwise_score") is not None else "N"
        _has_bge = "Y" if rp.paper.metadata.get("bge_score") is not None else "N"
        _has_vec = "Y" if rp.paper.metadata.get("vector_score") is not None else "N"
        _paths = rp.paper.retrieval_path or []
        _route = _paths[0].split(".")[-1] if _paths else "none"
        print(
            f"  #{rp.rank} score={rp.final_score:.4f} | {_pid} | rel={rp.selection.relevance_level} "
            f"listwise={_has_listwise} bge={_has_bge} vec={_has_vec} | "
            f"LLM_R={_s.get('LLM_Relevance', 0):.3f} LLM_L={_s.get('LLM_Listwise', 0):.3f} "
            f"BGE={_s.get('BGE_Reranker', 0):.3f} Src={_s.get('Source_Agreement', 0):.3f} "
            f"Cst={_s.get('Constraint_Coverage', 0):.3f} | route={_route} | {_title}"
        )
    print(f"=== RERANK DEBUG END (total {len(ranked_list)} papers) ===")

    return ranked_list


def rerank_papers_with_cutoff(
    papers: list[Paper],
    selections: list[SelectionResult],
    *,
    recall_beta: float = 1.5,
    min_high: int = 0,
    max_output: int | None = None,
    set_should_output: bool = True,
    calibration: Any = None,
    config: Any | None = None,
    original_query: str | None = None,
    listwise_scores: dict[str, float] | None = None,
) -> tuple[list[RankedPaper], FrontierPartition]:
    """重排 + F1 最优截断的一体化入口。"""
    full_ranked = rerank_papers(
        papers, selections, config, original_query, listwise_scores
    )
    partition = partition_ranked_papers(
        full_ranked,
        recall_beta=recall_beta,
        calibration=calibration,
        annotate=True,
        set_should_output=set_should_output,
        min_high=min_high,
        max_output=max_output,
    )
    return full_ranked, partition


def rerank_and_structure(
    papers: list[Paper],
    selections: list[SelectionResult],
    *,
    recall_beta: float = 1.5,
    max_output: int | None = None,
) -> dict:
    """一步产出可直接渲染的结构化结果。"""
    _, partition = rerank_papers_with_cutoff(
        papers, selections, recall_beta=recall_beta, max_output=max_output
    )
    return summarize_partition(partition)
