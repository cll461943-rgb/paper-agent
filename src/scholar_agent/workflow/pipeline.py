from __future__ import annotations

import logging
import time
from typing import Any

from scholar_agent.models.schemas import (
    Paper,
    QueryPlan,
    RankedPaper,
    RunMetrics,
    SearchProcessRound,
    SearchQuery,
    SelectionResult,
    WorkflowResult,
)
from scholar_agent.planning import (
    understand_query,
    heuristic_understand_query,
    generate_search_queries,
    heuristic_generate_search_queries,
    review_retrieval_results,
    optimize_search_strategy,
)
from scholar_agent.retrieval import MultiRouteRetriever, build_providers, expand_with_refchain
from scholar_agent.selection.evidence_selector import select_and_extract_evidence
from scholar_agent.selection.evidence_validator import validate_selections
from scholar_agent.ranking.final_reranker import rerank_papers
from scholar_agent.synthesis.synthesis_agent import SynthesisAgent
from scholar_agent.workflow.budget import BudgetManager

LOGGER = logging.getLogger(__name__)


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """去重候选论文池，保留元数据最全的记录，并合并检索路径。"""
    import re
    
    def _normalize_title(val: str | None) -> str:
        if not val:
            return ""
        # 移除非数字字母并转小写
        t = re.sub(r"[^a-z0-9]+", " ", val.lower()).strip()
        return re.sub(r"\s+", " ", t)

    def _normalize_id(val: str | None) -> str:
        if not val:
            return ""
        t = val.strip().lower()
        t = re.sub(r"^https?://(dx\.)?doi\.org/", "", t)
        t = re.sub(r"^doi:", "", t)
        t = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", t)
        t = re.sub(r"\.pdf$", "", t)
        t = re.sub(r"v\d+$", "", t)
        return t.strip()

    seen_by_id: dict[str, Paper] = {}
    seen_by_doi: dict[str, Paper] = {}
    seen_by_arxiv: dict[str, Paper] = {}
    seen_by_title: dict[str, Paper] = {}

    for p in papers:
        # 查找已有匹配
        existing: Paper | None = None
        
        # 1. 尝试以 paper_id 查找
        pid = p.paper_id
        if pid in seen_by_id:
            existing = seen_by_id[pid]
            
        # 2. 尝试以 DOI 查找
        p_doi = _normalize_id(p.doi)
        if not existing and p_doi and p_doi in seen_by_doi:
            existing = seen_by_doi[p_doi]
            
        # 3. 尝试以 arXiv ID 查找
        p_arxiv = _normalize_id(p.arxiv_id)
        if not existing and p_arxiv and p_arxiv in seen_by_arxiv:
            existing = seen_by_arxiv[p_arxiv]
            
        # 4. 尝试以规范化 Title 查找
        p_title = _normalize_title(p.title)
        if not existing and p_title and p_title in seen_by_title:
            existing = seen_by_title[p_title]

        if not existing:
            # 记录新的
            seen_by_id[pid] = p
            if p_doi:
                seen_by_doi[p_doi] = p
            if p_arxiv:
                seen_by_arxiv[p_arxiv] = p
            if p_title:
                seen_by_title[p_title] = p
        else:
            # 合并字段
            if not existing.abstract and p.abstract:
                existing.abstract = p.abstract
            if not existing.year and p.year:
                existing.year = p.year
            if not existing.venue and p.venue:
                existing.venue = p.venue
            if not existing.doi and p.doi:
                existing.doi = p.doi
            if not existing.arxiv_id and p.arxiv_id:
                existing.arxiv_id = p.arxiv_id
            if not existing.url and p.url:
                existing.url = p.url
            if existing.citation_count is None and p.citation_count is not None:
                existing.citation_count = p.citation_count
            elif existing.citation_count is not None and p.citation_count is not None:
                existing.citation_count = max(existing.citation_count, p.citation_count)
            
            # 合并检索路径
            for path in p.retrieval_path:
                if path not in existing.retrieval_path:
                    existing.retrieval_path.append(path)
            
            # 合并引文关系
            for ref in p.references:
                if ref not in existing.references:
                    existing.references.append(ref)
            for cit in p.citations:
                if cit not in existing.citations:
                    existing.citations.append(cit)

            # 更新缺少的 key 索引
            if p_doi and p_doi not in seen_by_doi:
                seen_by_doi[p_doi] = existing
            if p_arxiv and p_arxiv not in seen_by_arxiv:
                seen_by_arxiv[p_arxiv] = existing
            if p_title and p_title not in seen_by_title:
                seen_by_title[p_title] = existing

    # 去重后，所有保留的 Paper 对象都记录在以唯一 ID 为 Key 的 values 中，
    # 我们用 set 对对象本身进行去重保留唯一物理对象列表即可。
    unique_papers = []
    seen_objects = set()
    for p in seen_by_id.values():
        if id(p) not in seen_objects:
            seen_objects.add(id(p))
            unique_papers.append(p)
            
    return unique_papers


class PaperAgentPipeline:
    def __init__(
        self,
        config: Any,
        llm_client: Any | None = None,
        providers: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.providers = providers or build_providers(config)

    def run(self, original_query: str, retrieval_only: bool = False) -> WorkflowResult:
        """运行智能学术检索 Agent V2.0 完整流程。"""
        # 1. 初始化 BudgetManager
        budget = BudgetManager(self.config)
        # 为 LLM 客户端注入当前会话的 budget
        if self.llm_client is not None:
            self.llm_client.budget = budget

        LOGGER.info("Starting PaperAgentPipeline for query: %s", original_query)

        # 2. 意图理解与提取意图契约
        try:
            query_plan = understand_query(original_query, self.llm_client)
        except Exception as exc:
            LOGGER.warning("LLM Query understanding failed: %s. Falling back to heuristic.", exc)
            query_plan = heuristic_understand_query(original_query)

        # 3. 初始化多路检索器
        retriever = MultiRouteRetriever(
            self.providers,
            budget,
            parallel=bool(getattr(self.config.app, "parallel_retrieval", False)),
        )
        rounds_history: list[SearchProcessRound] = []

        # 4. 第一轮检索规划与执行
        try:
            subqueries = generate_search_queries(query_plan, budget, self.llm_client)
        except Exception as exc:
            LOGGER.warning("LLM query generation failed: %s. Falling back to heuristic.", exc)
            subqueries = heuristic_generate_search_queries(query_plan)
            budget.record_search_queries(len(subqueries))

        # 定义用于粗排评分的内部函数，以便多轮迭代复用
        def _get_title_match_bonus(title: str, query: str) -> float:
            import re
            stop_words = {"the", "a", "an", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from", "that", "this", "these", "those"}
            q_words = set(re.findall(r"\b\w{3,}\b", query.lower())) - stop_words
            t_words = set(re.findall(r"\b\w{3,}\b", title.lower())) - stop_words
            intersect = q_words.intersection(t_words)
            return min(len(intersect) * 0.1, 0.5)

        def _get_rough_score(p: Paper) -> float:
            import math
            bge = p.metadata.get("bge_score")
            try:
                bge_val = float(bge) if bge is not None else 0.0
            except (ValueError, TypeError):
                bge_val = 0.0
            
            citation_score = 0.0
            if p.citation_count is not None:
                citation_score = math.log1p(max(0, p.citation_count)) * 0.01
                
            route_bonus = 0.0
            for path in p.retrieval_path:
                if "title_exact" in path:
                    route_bonus += 0.25
                elif "title_like" in path:
                    route_bonus += 0.05
                if "reference_expansion" in path or "citation_expansion" in path:
                    route_bonus += 0.05
                    
            paths_val = len(p.retrieval_path) if p.retrieval_path else 1
            base_score = bge_val if bge_val > 0.0 else 0.5
            
            title_bonus = _get_title_match_bonus(p.title, original_query)
            
            return base_score + paths_val * 0.01 + citation_score + route_bonus + title_bonus

        # 执行第一轮检索
        LOGGER.info("Executing Retrieval Round 1 with %d queries", len(subqueries))
        _, candidate_pool = retriever.retrieve(subqueries, original_query=original_query, query_plan=query_plan)
        all_candidates = deduplicate_papers(candidate_pool)
        all_candidates.sort(key=_get_rough_score, reverse=True)
 
        round_1_record = SearchProcessRound(
            round_index=1,
            search_goal="构建包含核心主题的基础论文候选池",
            queries=[q.query for q in subqueries],
            candidates_found=len(all_candidates),
            review_conclusion="第一轮检索完成"
        )
        rounds_history.append(round_1_record)

        # 5. 主控多轮迭代检索环 (最多 config.app.max_retrieval_rounds 轮)
        if retrieval_only and (not self.llm_client or not self.llm_client.is_available()):
            max_rounds = 1
        else:
            max_rounds = getattr(self.config.budget, "max_retrieval_rounds", 3)
        
        for r in range(2, max_rounds + 1):
            # 5.1 结果审阅与差距分析
            LOGGER.info("Reviewing candidates for Round %d", r)
            review_res = review_retrieval_results(query_plan, all_candidates, r - 1, self.llm_client, self.config)
            # 更新上一轮的审阅结论
            rounds_history[-1].review_conclusion = review_res.get("reason", "审阅完成")

            # 5.2 决策是否提前终止
            if review_res.get("next_action") == "stop_search":
                LOGGER.info("Result Review Agent decided to STOP search in Round %d", r)
                break

            # 5.3 检索策略优化
            LOGGER.info("Optimizing search strategy for Round %d", r)
            opt_res = optimize_search_strategy(query_plan, review_res, all_candidates, r - 1, self.llm_client)

            new_subqueries_data = opt_res.get("new_subqueries", [])
            new_queries = []
            for item in new_subqueries_data:
                route = item.get("route") or "evolved"
                new_queries.append(
                    SearchQuery(
                        query=item.get("query", ""),
                        route=route,
                        intent=item.get("reason", "expanded search"),
                        required_terms=item.get("required_terms", []),
                        optional_terms=item.get("optional_terms", []),
                        filters=item.get("filters", {}),
                        sources=item.get("sources", ["openalex", "semantic_scholar"]),
                        priority=1,
                    )
                )

            new_candidates: list[Paper] = []
            if new_queries:
                budget.record_search_queries(len(new_queries))
                _, new_pool = retriever.retrieve(new_queries, original_query=original_query, query_plan=query_plan)
                new_candidates.extend(new_pool)

            # 5.4 引文网络扩展
            enable_refchain = False
            query_type = getattr(query_plan, "query_type", "unknown")
            if query_type in {"citation_tracking", "survey", "method_comparison"}:
                enable_refchain = True
            elif len(all_candidates) < 100:
                enable_refchain = True

            if enable_refchain and (review_res.get("need_citation_expansion") or opt_res.get("citation_expansion_seeds")):
                seed_ids = set(opt_res.get("citation_expansion_seeds", []))
                seed_papers = [p for p in all_candidates if p.paper_id in seed_ids]
                if not seed_papers:
                    seed_papers = all_candidates[:3]
                
                LOGGER.info("Expanding citation network for Round %d with %d seeds", r, len(seed_papers))
                expanded_pool = retriever.expand_refchain(seed_papers, self.providers, limit_per_seed=5)
                new_candidates.extend(expanded_pool)

            # 合并去重新候选
            if new_candidates:
                all_candidates.extend(new_candidates)
                all_candidates = deduplicate_papers(all_candidates)
                all_candidates.sort(key=_get_rough_score, reverse=True)

            # 记录当前轮
            round_record = SearchProcessRound(
                round_index=r,
                search_goal=opt_res.get("search_goal", "补充缺失主题与硬约束项"),
                queries=[q.query for q in new_queries],
                candidates_found=len(all_candidates),
                review_conclusion="检索完成"
            )
            rounds_history.append(round_record)

            if opt_res.get("stop_after_this_round", False):
                LOGGER.info("Strategy Optimizer Agent decided to STOP after Round %d", r)
                break

        # 最后一轮的最终审阅更新
        if rounds_history and rounds_history[-1].review_conclusion == "检索完成":
            final_review = review_retrieval_results(query_plan, all_candidates, len(rounds_history), self.llm_client, self.config)
            rounds_history[-1].review_conclusion = final_review.get("reason", "最终检索完成")

        all_candidates.sort(key=_get_rough_score, reverse=True)

        # 存储候选池供评估框架审计
        self.candidate_pool = all_candidates
        budget.candidate_pool_size = len(all_candidates)

        if retrieval_only:
            LOGGER.info("Retrieval only mode enabled. Bypassing evidence selection and synthesis.")
            return WorkflowResult(
                original_query=original_query,
                query_plan=query_plan,
                search_process=rounds_history,
                highly_relevant_papers=[],
                partially_relevant_papers=[],
                method_clusters=[],
                timeline=[],
                citation_graph={},
                recommendation_reasoning=[],
                agent_self_report={"message": "Recall only mode. Evidence selection and synthesis were skipped."},
                run_metrics=budget.get_metrics()
            )

        query_type = getattr(query_plan, "query_type", "unknown")
        if query_type == "exact_title":
            max_selection = 10
        elif query_type == "specific_paper":
            max_selection = 20
        elif query_type == "dataset_constraint":
            max_selection = 30
        elif query_type == "method_comparison":
            max_selection = 35
        elif query_type == "survey":
            max_selection = 40
        elif query_type == "broad_topic":
            max_selection = 40
        elif query_type == "latest_work":
            max_selection = 35
        else:
            max_selection = getattr(self.config.budget, "max_llm_selection_papers", 15)

        selection_candidates = []
        # 1. 粗排 topN
        selection_candidates.extend(all_candidates[:max_selection])

        def top_by_route(papers: list[Paper], route: str, k: int) -> list[Paper]:
            selected = []
            for p in papers:
                paths = p.retrieval_path or []
                if any(path == f"route:{route}" for path in paths):
                    selected.append(p)
                    if len(selected) >= k:
                        break
            return selected

        def top_by_provider(papers: list[Paper], provider: str, k: int) -> list[Paper]:
            selected = []
            for p in papers:
                paths = p.retrieval_path or []
                if any(path == f"provider:{provider}" for path in paths):
                    selected.append(p)
                    if len(selected) >= k:
                        break
            return selected

        # 2. 每个 route 至少保留若干篇
        for route in ["core_topic", "method_task", "dataset", "translated", "broad_synonym", "query2doc", "hyde", "citation_seed"]:
            selection_candidates.extend(top_by_route(all_candidates, route, k=3))

        # 3. 每个 provider 至少保留若干篇
        for provider in ["pasa_local", "openalex", "semantic_scholar", "pubmed", "arxiv"]:
            selection_candidates.extend(top_by_provider(all_candidates, provider, k=3))

        # 4. title_exact/title_like 命中的论文
        title_match_candidates = []
        for p in all_candidates:
            has_title_match = any(
                "title_exact" in path or "title_like" in path
                for path in (p.retrieval_path or [])
            )
            if has_title_match:
                title_match_candidates.append(p)
        selection_candidates.extend(title_match_candidates[:10])

        selection_candidates = deduplicate_papers(selection_candidates)[:40]

        LOGGER.info("Starting fine-grained evidence selection for %d papers (filtered from %d candidates)", 
                    len(selection_candidates), len(all_candidates))
        from scholar_agent.selection.batch_evidence_selector import batch_select_and_extract_evidence
        selections = batch_select_and_extract_evidence(selection_candidates, query_plan, self.llm_client)

        # 7. 本地规则硬核校验与降级
        LOGGER.info("Validating evidence and constraints locally")
        validated_selections = validate_selections(all_candidates, selections, query_plan)
        self.selections = validated_selections

        # 8. 多维度综合重排
        LOGGER.info("Reranking papers based on blending formula")
        ranked_papers = rerank_papers(all_candidates, validated_selections, self.config, original_query)

        # 存储候选池供评估框架审计
        self.candidate_pool = all_candidates

        # 9. 结构化归纳合成与输出
        LOGGER.info("Synthesizing final structured report")
        synthesis_agent = SynthesisAgent(self.llm_client)
        
        # 统计指标录入
        budget.candidate_pool_size = len(all_candidates)
        
        result = synthesis_agent.synthesize(
            original_query=original_query,
            query_plan=query_plan,
            search_rounds=rounds_history,
            ranked_papers=ranked_papers,
            metrics=budget.get_metrics(),
            config=self.config
        )

        LOGGER.info("Pipeline completed successfully. Found %d recommended papers.", 
                    len(result.highly_relevant_papers) + len(result.partially_relevant_papers))
        return result
