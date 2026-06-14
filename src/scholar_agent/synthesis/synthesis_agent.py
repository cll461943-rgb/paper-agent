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
        metrics: RunMetrics
    ) -> WorkflowResult:
        """结构化归纳合成 Agent。将多轮检索重排结果归并、分类、产生时间线与引文图并输出最终报告。"""
        # 1. 筛选与分类
        highly_relevant: list[RankedPaper] = []
        partially_relevant: list[RankedPaper] = []

        for rp in ranked_papers:
            if rp.selection.relevance_level == "high":
                highly_relevant.append(rp)
            elif rp.selection.relevance_level == "medium":
                partially_relevant.append(rp)

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

        # 3. 产生推荐说明 (recommendation_reasoning)
        recommendation_reasoning: list[dict[str, Any]] = []
        for rp in (highly_relevant + partially_relevant):
            recommendation_reasoning.append({
                "paper_id": rp.paper.paper_id,
                "title": rp.paper.title,
                "relevance": rp.selection.relevance_level,
                "strength": rp.selection.reason,
                "critique": f"Constraint coverage: {rp.subscores.get('Constraint_Coverage', 1.0):.2%}. Year: {rp.paper.year or 'N/A'}."
            })

        # 4. 获取大模型辅助的方法分类聚类、学术脉络时间线和自我反思报告
        fallback_data = self._create_fallback_synthesis(
            highly_relevant, partially_relevant, query_plan, search_rounds, metrics
        )

        method_clusters = fallback_data["method_clusters"]
        timeline = fallback_data["timeline"]
        agent_self_report = fallback_data["agent_self_report"]

        if self.llm_client is not None:
            # 拼装提示词调用大模型，使用更高级的 Pro 模型做逻辑推理
            system_prompt = (
                "You are an academic Synthesis Agent. "
                "Your task is to analyze a set of recommended papers, group them into logical 'method_clusters' (categories based on technical methodologies), "
                "build an chronological 'timeline' of key milestones, and generate an 'agent_self_report' reflecting on the search strategy effectiveness, gaps identified, and refinement history. "
                "Return a JSON object containing fields: 'method_clusters', 'timeline', and 'agent_self_report'. "
                "Return JSON only."
            )

            papers_payload = [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "abstract": p.abstract,
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
                        method_clusters = response["method_clusters"]
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
            method_clusters=method_clusters,
            timeline=timeline,
            citation_graph=citation_graph,
            recommendation_reasoning=recommendation_reasoning,
            agent_self_report=agent_self_report,
            run_metrics=metrics
        )
