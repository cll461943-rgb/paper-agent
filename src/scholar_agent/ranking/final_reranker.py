from __future__ import annotations

import logging
from typing import Any

from scholar_agent.models.schemas import Paper, RankedPaper, SelectionResult

LOGGER = logging.getLogger(__name__)


def compute_paper_score(
    paper: Paper,
    selection: SelectionResult,
    other_paper_ids: set[str] | None = None
) -> tuple[float, dict[str, float]]:
    """依据 V2.0 多维度综合评分公式计算单篇论文的最终得分和各个维度的子项得分。"""
    subscores: dict[str, float] = {}

    # 1. LLM 相关性分 (权重 0.25)
    relevance_map = {"high": 1.0, "medium": 0.6, "low": 0.2, "irrelevant": 0.0}
    llm_relevance = relevance_map.get(selection.relevance_level, 0.0)
    subscores["LLM_Relevance"] = llm_relevance

    # 2. BGE 精排分 (权重 0.20)
    # 优先从 metadata 中读取 BGE 精排分，比如 'bge_score'
    bge_score = paper.metadata.get("bge_score")
    if bge_score is None:
        # 兜底
        bge_score = 0.5
    else:
        try:
            bge_score = float(bge_score)
        except (ValueError, TypeError):
            bge_score = 0.5
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

    # 4. 证据完整性 (权重 0.15)
    # 高中相关性且具有明确提取证据的得分 1.0，否则 0.0
    if selection.relevance_level in ("high", "medium") and len(selection.evidence) > 0:
        evidence_completeness = 1.0
    else:
        evidence_completeness = 0.0
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

    # 混合计算总分
    final_score = (
        subscores["LLM_Relevance"] * 0.25 +
        subscores["BGE_Reranker"] * 0.20 +
        subscores["Constraint_Coverage"] * 0.20 +
        subscores["Evidence_Completeness"] * 0.15 +
        subscores["Source_Agreement"] * 0.10 +
        subscores["Recency"] * 0.05 +
        subscores["Authority"] * 0.03 +
        subscores["Diversity_Graph_Prior"] * 0.02
    )

    return final_score, subscores


def rerank_papers(
    papers: list[Paper],
    selections: list[SelectionResult]
) -> list[RankedPaper]:
    """对候选论文列表进行重排，计算综合评分，赋予 Rank 序号。"""
    if not papers:
        return []

    # 建立 paper_id 映射
    selection_map = {sel.paper_id: sel for sel in selections}
    paper_ids = {p.paper_id for p in papers}

    ranked_list: list[RankedPaper] = []

    for paper in papers:
        selection = selection_map.get(paper.paper_id)
        if not selection:
            # 兜底生成一个空 SelectionResult
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

    return ranked_list
