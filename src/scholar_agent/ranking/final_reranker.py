from __future__ import annotations

import logging
from typing import Any

from scholar_agent.models.schemas import Paper, RankedPaper, SelectionResult

LOGGER = logging.getLogger(__name__)


def _calculate_title_similarity(title1: str | None, title2: str | None) -> float:
    if not title1 or not title2:
        return 0.0
    import re
    def tokenize(text):
        return set(re.findall(r"\w+", text.lower()))
    w1 = tokenize(title1)
    w2 = tokenize(title2)
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))


def compute_paper_score(
    paper: Paper,
    selection: SelectionResult,
    other_paper_ids: set[str] | None = None,
    config: Any | None = None,
    original_query: str | None = None
) -> tuple[float, dict[str, float]]:
    """依据 V2.0 多维度综合评分公式计算单篇论文的最终得分和各个维度的子项得分。"""
    weights = {
        "llm_relevance_score": 0.35,
        "bge_reranker_score": 0.0,
        "constraint_coverage_score": 0.30,
        "evidence_completeness_score": 0.20,
        "source_agreement_score": 0.10,
        "recency_score": 0.03,
        "authority_score": 0.01,
        "diversity_graph_prior_score": 0.01,
    }
    if config and hasattr(config, "ranking"):
        cfg_ranking = getattr(config, "ranking", None)
        if cfg_ranking:
            cfg_weights = getattr(cfg_ranking, "weights", None)
            if not cfg_weights and isinstance(cfg_ranking, dict):
                cfg_weights = cfg_ranking.get("weights")
            if cfg_weights:
                for k in weights.keys():
                    if isinstance(cfg_weights, dict) and k in cfg_weights:
                        weights[k] = float(cfg_weights[k])
                    elif hasattr(cfg_weights, k):
                        weights[k] = float(getattr(cfg_weights, k))
    subscores: dict[str, float] = {}

    # 1. LLM 相关性分 (权重 0.25)
    relevance_map = {"high": 1.0, "medium": 0.55, "low": 0.15, "irrelevant": 0.0}
    llm_relevance = relevance_map.get(selection.relevance_level, 0.0)
    subscores["LLM_Relevance"] = llm_relevance

    # 2. BGE 精排分 (权重 0.20)
    # 优先从 metadata 中读取 BGE 精排分，比如 'bge_score'
    bge_score = paper.metadata.get("bge_score")
    if bge_score is None:
        # 兜底：如果缺乏 BGE 分数，则用 LLM 相关性分替代，避免 20% 权重完全失效
        bge_score = llm_relevance
    else:
        try:
            bge_score = float(bge_score)
        except (ValueError, TypeError):
            bge_score = llm_relevance
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
    # 仅按不同的唯一检索源来判定多源共识，防止同一源的多条 route 重复计算
    ret_paths = paper.retrieval_path or []
    providers = set()
    for path in ret_paths:
        if path.startswith("provider:"):
            providers.add(path.replace("provider:", ""))
        elif ":" in path:
            parts = path.split(":")
            if parts[0] in ("provider", "prov"):
                providers.add(parts[1])
    if providers:
        source_agreement = min(len(providers) / 3.0, 1.0)
    else:
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
    title_exact_bonus = 0.15 if is_exact_title else 0.0
    
    title_like_bonus = 0.0
    if is_like_title and not is_exact_title:
        # 仅当 title_similarity > 0.85 时，给予降低为 0.02 的 bonus
        if original_query:
            similarity = _calculate_title_similarity(paper.title, original_query)
            if similarity > 0.85:
                title_like_bonus = 0.02
        else:
            title_like_bonus = 0.02

    # 混合计算总分，使用 config 配置或默认的 weights
    final_score = (
        subscores["LLM_Relevance"] * weights["llm_relevance_score"] +
        subscores["BGE_Reranker"] * weights["bge_reranker_score"] +
        subscores["Constraint_Coverage"] * weights["constraint_coverage_score"] +
        subscores["Evidence_Completeness"] * weights["evidence_completeness_score"] +
        subscores["Source_Agreement"] * weights["source_agreement_score"] +
        subscores["Recency"] * weights["recency_score"] +
        subscores["Authority"] * weights["authority_score"] +
        subscores["Diversity_Graph_Prior"] * weights["diversity_graph_prior_score"] +
        title_exact_bonus +
        title_like_bonus
    )
    final_score = min(max(final_score, 0.0), 1.0)

    return final_score, subscores


def rerank_papers(
    papers: list[Paper],
    selections: list[SelectionResult],
    config: Any | None = None,
    original_query: str | None = None
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
        score, subscores = compute_paper_score(paper, selection, other_ids, config, original_query)
        
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
