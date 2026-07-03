from __future__ import annotations

import logging
from typing import Any

from scholar_agent.models.schemas import (
    Paper,
    QueryPlan,
    RankedPaper,
    RunMetrics,
    SearchProcessRound,
    WorkflowResult,
)

LOGGER = logging.getLogger(__name__)


class SynthesisAgent:
    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    def _create_fallback_synthesis(
        self,
        highly_papers: list[RankedPaper],
        partially_papers: list[RankedPaper],
        query_plan: QueryPlan,
        search_rounds: list[SearchProcessRound],
        metrics: RunMetrics
    ) -> dict[str, Any]:
        """如果大模型调用不可用或挂掉，启发式地生成分类簇、时间线及反思报告。"""
        # 1. 启发式方法聚类
        method_clusters: list[dict[str, Any]] = []
        # 以 QueryPlan 中定义的 methods 作为类别名进行聚类，另外加一个 General 类别
        methods_to_track = list(query_plan.methods) if query_plan.methods else ["General Method"]
        
        cluster_map: dict[str, list[str]] = {m: [] for m in methods_to_track}
        all_recommended = highly_papers + partially_papers
        
        for rp in all_recommended:
            text = (rp.paper.title + " " + (rp.paper.abstract or "")).lower()
            matched_any = False
            for m in methods_to_track:
                if m.lower() in text:
                    cluster_map[m].append(rp.paper.paper_id)
                    matched_any = True
            if not matched_any:
                if "General Method" not in cluster_map:
                    cluster_map["General Method"] = []
                cluster_map["General Method"].append(rp.paper.paper_id)

        for name, ids in cluster_map.items():
            if ids:
                method_clusters.append({
                    "cluster_name": name,
                    "paper_ids": ids,
                    "summary": f"Papers matching method / constraint: {name}"
                })

        # 2. 启发式时间线 (按发表年份从新到旧排序)
        timeline: list[dict[str, Any]] = []
        sorted_by_year = sorted(all_recommended, key=lambda x: x.paper.year or 0, reverse=True)
        for rp in sorted_by_year:
            year_str = str(rp.paper.year) if rp.paper.year else "Unknown"
            timeline.append({
                "year": rp.paper.year or 0,
                "event": f"[{year_str}] Publication of '{rp.paper.title}' (Venue: {rp.paper.venue or 'N/A'}) - Relevance: {rp.selection.relevance_level.upper()}"
            })

        # 3. 启发式自我报告
        agent_self_report = {
            "strategy_summary": (
                f"Agent executed {len(search_rounds)} retrieval rounds using query expansion. "
                f"Obtained a candidate pool of {metrics.candidate_pool_size} papers, "
                f"resulting in {len(highly_papers)} highly relevant and {len(partially_papers)} partially relevant papers. "
                "The search prioritizes precision using localized validation checkpoints."
            ),
            "gaps_identified": [
                "Heuristic synthesis cannot query missing subfields autonomously. "
                "All constraints matched via heuristic rules."
            ],
            "refinement_history": [
                f"Round {r.round_index}: {r.search_goal} (Found {r.candidates_found} candidates)"
                for r in search_rounds
            ]
        }

        return {
            "method_clusters": method_clusters,
            "timeline": timeline,
            "agent_self_report": agent_self_report
        }

    def synthesize(
        self,
        original_query: str,
        query_plan: QueryPlan,
        search_rounds: list[SearchProcessRound],
        ranked_papers: list[RankedPaper],
        metrics: RunMetrics,
        config: Any | None = None,
        listwise_result: Any | None = None,
    ) -> WorkflowResult:
        """结构化归纳合成 Agent。将多轮检索重排结果归并、分类、产生时间线与引文图并输出最终报告。

        Task 4: Uses F1-aware K controller to choose optimal output count.
        Weak/background papers go to supporting_papers, NOT into eval list.
        """
        # 1. 使用 selection_cutoff.py 进行概率最优双截断分层
        from scholar_agent.selection.selection_cutoff import (
            partition_ranked_papers,
            find_score_gap_k,
            find_percentile_drop_k,
            logistic_calibration,
        )

        # 读取 config 配置参数
        _dk_cfg = getattr(config, "dynamic_k", None) if config else None
        _recall_beta = float(getattr(_dk_cfg, "recall_beta", 1.5)) if _dk_cfg else 1.5
        _max_output = int(getattr(_dk_cfg, "hard_max_output", 12)) if _dk_cfg else 12
        _min_high = int(getattr(_dk_cfg, "min_high", 1)) if _dk_cfg else 1

        # ── Score-gap 检测：利用得分断崖定位自然分界点 ──
        # 在 F_beta 截断之前，先检查 top-N 论文中是否存在显著的得分断崖。
        # 断崖存在 → 用断崖位置作为 max_output（自然分界比数学截断更准确）
        # 断崖不存在 → 回退到 F_beta + logistic 校准
        _scores = [float(rp.final_score) for rp in ranked_papers] if ranked_papers else []
        _gap_k = find_score_gap_k(
            _scores,
            gap_threshold=0.08,
            relative_threshold=0.10,
            max_k=_max_output,
            min_k=2,
        )
        if _gap_k is not None:
            _effective_max = min(_gap_k, _max_output)
            LOGGER.info(
                "Score-gap K selection: gap at position %d → effective max_output=%d "
                "(config max=%d)",
                _gap_k, _effective_max, _max_output,
            )
        else:
            # Score-gap 未触发（平滑分布）→ 用相对衰减截断选择 K
            _drop_ratio = float(getattr(_dk_cfg, "drop_ratio", 0.15)) if _dk_cfg else 0.15
            _pct_k = find_percentile_drop_k(
                _scores,
                drop_ratio=_drop_ratio,
                max_k=_max_output,
                min_k=_min_high,
            )
            _effective_max = _pct_k
            LOGGER.info(
                "Score-gap: no gap found → percentile-drop K=%d "
                "(drop_ratio=%.0f%%, config max=%d)",
                _pct_k, _drop_ratio * 100, _max_output,
            )

        def _positive_int(value: Any) -> int | None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed > 0 else None

        _listwise_cap_note = ""
        if listwise_result is not None and getattr(listwise_result, "success", False):
            _recommended_k = _positive_int(getattr(listwise_result, "recommended_k", None))
            if _recommended_k is not None:
                _recommended_min = _positive_int(getattr(listwise_result, "recommended_k_min", None))
                _recommended_max = _positive_int(getattr(listwise_result, "recommended_k_max", None))
                if _recommended_min is not None:
                    _recommended_k = max(_recommended_k, _recommended_min)
                if _recommended_max is not None:
                    _recommended_k = min(_recommended_k, _recommended_max)

                _previous_max = _effective_max
                _effective_max = max(_min_high, min(_effective_max, _recommended_k, len(ranked_papers)))
                if _effective_max != _previous_max:
                    _listwise_cap_note = f"Listwise recommended_k={_recommended_k} cap applied; "
                    LOGGER.info(
                        "Listwise recommended_k cap: effective max_output %d → %d",
                        _previous_max, _effective_max,
                    )

        # ── Expected-Fβ 双截断 ──
        # P0-5/P0-6: 区分 rank_score (final_score, 用于排序) 和 relevance_probability
        # (用于 Expected-Fβ 截断)。如果 listwise reranker 输出了 relevance_probability，
        # 直接使用它作为 p_i（listwise reranker 已校准）；否则回退到 logistic 校准的 final_score。
        def _get_relevance_prob(rp) -> float:
            """Extract relevance_probability from subscores/metadata, fall back to final_score."""
            # Try subscores first (set by compute_paper_score)
            sub = getattr(rp, "subscores", None)
            if isinstance(sub, dict) and "relevance_probability" in sub:
                p = float(sub["relevance_probability"])
                if p > 0:
                    return p
            # Try paper.metadata (set by rerank_papers)
            paper = getattr(rp, "paper", None)
            if paper is not None and paper.metadata:
                p = paper.metadata.get("relevance_probability")
                if p is not None:
                    try:
                        p = float(p)
                        if p > 0:
                            return p
                    except (ValueError, TypeError):
                        pass
            # Fall back to final_score
            return float(getattr(rp, "final_score", 0.0))

        # Check if any paper has relevance_probability > 0
        _has_rel_prob = any(_get_relevance_prob(rp) > 0 for rp in ranked_papers) if ranked_papers else False

        if _has_rel_prob:
            # Use relevance_probability directly (already calibrated by listwise reranker)
            _cal = None  # identity calibration — probabilities are already calibrated
            _score_getter = _get_relevance_prob
            LOGGER.info("Using LLM relevance_probability for Expected-Fβ cutoff (listwise-calibrated)")
        else:
            # Fallback: logistic calibration of final_score
            _cal = logistic_calibration(midpoint=0.5, temperature=0.12)
            _score_getter = None  # use default (final_score)
            LOGGER.info("Falling back to logistic-calibrated final_score for Expected-Fβ cutoff")

        partition = partition_ranked_papers(
            ranked_papers,
            recall_beta=_recall_beta,
            calibration=_cal,
            score_getter=_score_getter,
            set_should_output=True,
            min_high=_min_high,
            max_output=_effective_max,
        )

        highly_relevant = partition.high
        partially_relevant = partition.partial
        supporting_papers = partition.excluded

        dynamic_k = len(highly_relevant) + len(partially_relevant)
        
        # 统计曲线与置信度元数据以兼容 evaluate.py
        _expected_f1_curve = {k: f for k, f in enumerate(partition.core.f_curve, start=1)} if hasattr(partition.core, "f_curve") else None
        _g_hat = partition.core.r_hat if hasattr(partition.core, "r_hat") else None
        _p_floor = partition.core.threshold if hasattr(partition.core, "threshold") else None

        recommended_ids = {rp.paper.paper_id for rp in (highly_relevant + partially_relevant)}
        recommended_papers = [rp.paper for rp in (highly_relevant + partially_relevant)]

        # 2. 构建引文图 (citation_graph)
        nodes: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        added_links: set[tuple[str, str]] = set()

        for rp in (highly_relevant + partially_relevant):
            nodes.append({
                "id": rp.paper.paper_id,
                "title": rp.paper.title,
                "relevance": rp.selection.relevance_level,
                "score": round(rp.final_score, 3)
            })

            # 自交叉引用分析
            refs = rp.paper.references or []
            for ref_id in refs:
                if ref_id in recommended_ids and ref_id != rp.paper.paper_id:
                    link_key = (rp.paper.paper_id, ref_id)
                    if link_key not in added_links:
                        added_links.add(link_key)
                        links.append({
                            "source": rp.paper.paper_id,
                            "target": ref_id,
                            "type": "references"
                        })

            cits = rp.paper.citations or []
            for cit_id in cits:
                if cit_id in recommended_ids and cit_id != rp.paper.paper_id:
                    link_key = (cit_id, rp.paper.paper_id)
                    if link_key not in added_links:
                        added_links.add(link_key)
                        links.append({
                            "source": cit_id,
                            "target": rp.paper.paper_id,
                            "type": "citations"
                        })

        citation_graph = {"nodes": nodes, "links": links}

        # 3. 产生推荐说明 (recommendation_reasoning) — P2: per-paper reason + evidence
        recommendation_reasoning: list[dict[str, Any]] = []
        for rp in (highly_relevant + partially_relevant):
            evidence_score = rp.subscores.get("Evidence_Completeness", -1.0)
            constraint_cov = rp.subscores.get("Constraint_Coverage", -1.0)
            penalty_mult = rp.subscores.get("Penalty_Multiplier", 1.0)
            recommendation_reasoning.append({
                "paper_id": rp.paper.paper_id,
                "title": rp.paper.title,
                "relevance": rp.selection.relevance_level,
                "final_score": round(rp.final_score or 0.0, 4),
                "strength": rp.selection.reason,
                "evidence_score": round(evidence_score, 3) if evidence_score >= 0 else None,
                "constraint_coverage": round(constraint_cov, 3) if constraint_cov >= 0 else None,
                "penalty_multiplier": round(penalty_mult, 3),
                "validation_notes": rp.selection.validation_notes[:3] if rp.selection.validation_notes else [],
                "critique": (
                    f"Score={rp.final_score:.3f} (penalty x{penalty_mult:.2f}). "
                    f"Constraint coverage: {constraint_cov:.2%}. "
                    f"Evidence: {evidence_score:.2%}. "
                    f"Year: {rp.paper.year or 'N/A'}."
                )
            })

        # 4. 获取大模型辅助的方法分类聚类、学术脉络时间线和自我反思报告
        fallback_data = self._create_fallback_synthesis(
            highly_relevant, partially_relevant, query_plan, search_rounds, metrics
        )

        method_clusters = fallback_data["method_clusters"]
        timeline = fallback_data["timeline"]
        agent_self_report = fallback_data["agent_self_report"]

        if self.llm_client is not None and len(recommended_papers) > 1:
            # 拼装提示词调用大模型，使用更高级的 Pro 模型做逻辑推理
            system_prompt = (
                "You are an academic Synthesis Agent. "
                "Your task is to analyze a set of recommended papers, group them into logical 'method_clusters' (categories based on technical methodologies), "
                "build an chronological 'timeline' of key milestones, and generate an 'agent_self_report' reflecting on the search strategy effectiveness, gaps identified, and refinement history. "
                "Return a JSON object containing fields: 'method_clusters', 'timeline', and 'agent_self_report'. "
                "Return JSON only.\n\n"
                "CRITICAL CONSTRAINT: You must NOT add, delete, or rerank any papers. "
                "The paper set is final and frozen. You can only organize papers into clusters and describe them. "
                "Do not suggest removing or adding papers. Do not change the order of papers. "
                "Your method_clusters paper_ids must reference ONLY papers from the provided list."
            )

            papers_payload = [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "abstract": (p.abstract[:150] + "...") if p.abstract else None,
                    "year": p.year,
                    "venue": p.venue,
                    "relevance": "high" if p.paper_id in {hr.paper.paper_id for hr in highly_relevant} else "medium"
                }
                for p in recommended_papers
            ]

            user_prompt = (
                "Output format schema (JSON):\n"
                "{\n"
                "  \"method_clusters\": [\n"
                "    {\"cluster_name\": \"Category Name\", \"paper_ids\": [\"p1\", \"p2\"], \"summary\": \"brief category description\"}\n"
                "  ],\n"
                "  \"timeline\": [\n"
                "    {\"year\": 2025, \"event\": \"Milestone description incorporating papers of this year\"}\n"
                "  ],\n"
                "  \"agent_self_report\": {\n"
                "    \"strategy_summary\": \"strategy explanation\",\n"
                "    \"gaps_identified\": [\"gap detail\"],\n"
                "    \"refinement_history\": [\"round detail\"]\n"
                "  }\n"
                "}\n\n"
                f"QueryPlan Contract: {query_plan.model_dump_json()}\n"
                f"Search Rounds History: {[r.model_dump() for r in search_rounds]}\n"
                f"Recommended Papers: {papers_payload}"
            )

            try:
                response = getattr(self.llm_client, "complete_json", lambda *_: None)(
                    system_prompt, user_prompt, model_type="pro"
                )
                if isinstance(response, dict):
                    if "method_clusters" in response:
                        # Guard: filter out any paper_ids not in the recommended set
                        _valid_ids = recommended_ids
                        _mc = response["method_clusters"]
                        if isinstance(_mc, list):
                            for cluster in _mc:
                                if isinstance(cluster, dict) and "paper_ids" in cluster:
                                    cluster["paper_ids"] = [
                                        pid for pid in cluster["paper_ids"]
                                        if pid in _valid_ids
                                    ]
                        method_clusters = _mc
                    if "timeline" in response:
                        timeline = response["timeline"]
                    if "agent_self_report" in response:
                        agent_self_report = response["agent_self_report"]
            except Exception as exc:
                LOGGER.warning("Error calling LLM Synthesis Agent: %s. Falling back to heuristic synthesis.", exc)

        return WorkflowResult(
            original_query=original_query,
            query_plan=query_plan,
            search_process=search_rounds,
            highly_relevant_papers=highly_relevant,
            partially_relevant_papers=partially_relevant,
            supporting_papers=supporting_papers,
            method_clusters=method_clusters,
            timeline=timeline,
            citation_graph=citation_graph,
            recommendation_reasoning=recommendation_reasoning,
            agent_self_report=agent_self_report,
            run_metrics=metrics,
            dynamic_k_chosen=dynamic_k,
            expected_f1_curve=_expected_f1_curve,
            g_hat=_g_hat,
            g_hat_scope=_g_hat,
            g_hat_pool=_g_hat,
            g_hat_visible=_g_hat,
            low_confidence_uniform=False,
            p_floor=_p_floor,
            tie_break_reason=_listwise_cap_note + (
                f"Score-gap K={_gap_k} → effective_max={_effective_max}"
                if _gap_k is not None
                else f"Percentile-drop K={_effective_max} → cutoff (k1={len(highly_relevant)}, k2={dynamic_k})"
            ),
            second_pass_triggered=False,
        )
