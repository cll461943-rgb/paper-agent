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
from scholar_agent.retrieval.query_expansion import QueryExpander
from scholar_agent.retrieval.source_health import SourceHealthManager
from scholar_agent.selection.evidence_selector import select_and_extract_evidence
from scholar_agent.selection.evidence_validator import validate_selections
from scholar_agent.ranking.final_reranker import rerank_papers
from scholar_agent.synthesis.synthesis_agent import SynthesisAgent
from scholar_agent.workflow.budget import BudgetManager
from scholar_agent.workflow.deadline import (
    Deadline,
    DEFAULT_CASE_DEADLINE,
    DEFAULT_QUERY_UNDERSTANDING,
    DEFAULT_RETRIEVAL,
    DEFAULT_EVIDENCE_SELECTION,
    DEFAULT_LISTWISE_RERANK,
    DEFAULT_SYNTHESIS,
    DEFAULT_RESULT_REVIEW,
    DEFAULT_STRATEGY_OPTIMIZATION,
)
from scholar_agent.workflow.llm_circuit_breaker import LLMCircuitBreaker

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


def _build_diversified_top80(
    all_candidates: list[Paper],
    validated_selections: list[SelectionResult],
) -> list[Paper]:
    """Build diversified top-80 candidates for second-pass listwise rerank.

    Stratified sampling from 5 sources:
      1. Top 30 by local_pre_rank_score (rough candidates)
      2. Top 10 per retrieval route (per route diversity)
      3. Top 10 per provider/source (per provider diversity)
      4. Top 15 by RRF score (high retrieval agreement)
      5. Top 15 by matched_constraints count (title/constraint candidates)
    Deduplicates and caps at 80.
    """
    sel_map = {s.paper_id: s for s in validated_selections}
    seen_ids: set[str] = set()
    result: list[Paper] = []

    def _add(paper: Paper) -> None:
        if paper.paper_id not in seen_ids and len(result) < 80:
            seen_ids.add(paper.paper_id)
            result.append(paper)

    # 1. Top 30 by local pre-rank score
    sorted_by_local = sorted(
        all_candidates,
        key=lambda p: p.metadata.get("local_pre_rank_score", 0.0),
        reverse=True,
    )
    for p in sorted_by_local[:30]:
        _add(p)

    # 2. Top 10 per retrieval route
    route_buckets: dict[str, list[Paper]] = {}
    for p in all_candidates:
        for route in p.retrieval_path:
            route_buckets.setdefault(route, []).append(p)
    for route, papers in route_buckets.items():
        papers.sort(key=lambda p: p.metadata.get("local_pre_rank_score", 0.0), reverse=True)
        for p in papers[:10]:
            _add(p)

    # 3. Top 10 per provider/source
    source_buckets: dict[str, list[Paper]] = {}
    for p in all_candidates:
        src = p.source or "unknown"
        source_buckets.setdefault(src, []).append(p)
    for src, papers in source_buckets.items():
        papers.sort(key=lambda p: p.metadata.get("local_pre_rank_score", 0.0), reverse=True)
        for p in papers[:10]:
            _add(p)

    # 4. Top 15 by RRF score
    rrf_sorted = sorted(
        all_candidates,
        key=lambda p: p.metadata.get("rrf_score", 0.0),
        reverse=True,
    )
    for p in rrf_sorted[:15]:
        _add(p)

    # 5. Top 15 by matched_constraints count
    constraint_sorted = sorted(
        [p for p in all_candidates if p.paper_id in sel_map],
        key=lambda p: len(sel_map[p.paper_id].matched_constraints or []),
        reverse=True,
    )
    for p in constraint_sorted[:15]:
        _add(p)

    return result


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

        # P2: Create case-level deadline and source health manager
        # Effect-first: read case_deadline from config (default 180s, effect-first uses 240s)
        _case_deadline_secs = getattr(self.config.budget, "case_deadline_seconds", DEFAULT_CASE_DEADLINE)
        case_deadline = Deadline(_case_deadline_secs, stage_name="case")
        health_manager = SourceHealthManager()
        self._health_manager = health_manager

        # P2: Create LLM circuit breaker — trips after max_timeouts
        # Effect-first: read threshold from config (default 2, effect-first uses 5)
        _cb_max_timeouts = getattr(self.config.llm, "circuit_breaker_max_timeouts", 2)
        llm_circuit_breaker = LLMCircuitBreaker(max_timeouts=_cb_max_timeouts)
        self._llm_circuit_breaker = llm_circuit_breaker
        if self.llm_client is not None and hasattr(self.llm_client, "set_circuit_breaker"):
            self.llm_client.set_circuit_breaker(llm_circuit_breaker)

        LOGGER.info("Starting PaperAgentPipeline for query: %s (deadline=%.0fs)", original_query, _case_deadline_secs)

        # 2. 意图理解与提取意图契约
        query_deadline = case_deadline.child(DEFAULT_QUERY_UNDERSTANDING, "query_understanding")
        use_llm_query = (
            self.llm_client is not None
            and not query_deadline.expired()
            and getattr(self.config.budget, "max_llm_calls", 8) > 0
        )
        if use_llm_query:
            # Dynamically set LLM timeout based on remaining deadline
            _configured_timeout = getattr(self.config.llm, "timeout_seconds", 15)
            _effective_timeout = query_deadline.timeout_for(_configured_timeout)
            if hasattr(self.llm_client, "timeout"):
                self.llm_client.timeout = _effective_timeout
            try:
                query_plan = understand_query(original_query, self.llm_client)
            except Exception as exc:
                LOGGER.warning("LLM Query understanding failed: %s. Falling back to heuristic.", exc)
                query_plan = heuristic_understand_query(original_query)
        else:
            LOGGER.info("Skipping LLM query understanding (deadline=%s, llm=%s)", query_deadline, bool(self.llm_client))
            query_plan = heuristic_understand_query(original_query)

        # 3. 初始化多路检索器
        retriever = MultiRouteRetriever(
            self.providers,
            budget,
            parallel=bool(getattr(self.config.app, "parallel_retrieval", False)),
            health_manager=health_manager,
        )
        retrieval_deadline = case_deadline.child(DEFAULT_RETRIEVAL, "retrieval")
        query_expander = QueryExpander()
        rounds_history: list[SearchProcessRound] = []

        # 4. 第一轮检索规划与执行
        if self.llm_client is not None and not retrieval_deadline.expired() and getattr(self.config.budget, "max_llm_calls", 8) > 0:
            try:
                subqueries = generate_search_queries(query_plan, budget, self.llm_client)
            except Exception as exc:
                LOGGER.warning("LLM query generation failed: %s. Falling back to heuristic.", exc)
                subqueries = heuristic_generate_search_queries(query_plan)
                budget.record_search_queries(len(subqueries))
        else:
            LOGGER.info("Using heuristic query generation (deadline=%s)", retrieval_deadline)
            subqueries = heuristic_generate_search_queries(query_plan)
            budget.record_search_queries(len(subqueries))

        # P2: Expand queries with acronym/synonym expansion from taxonomy
        if query_expander.is_available():
            subqueries = [query_expander.expand_query(q) for q in subqueries]
            LOGGER.info("Query expansion: expanded %d queries with acronyms/synonyms", len(subqueries))

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
                    route_bonus += 1.0
                elif "title_like" in path:
                    route_bonus += 0.2
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
            # P2: Check retrieval deadline before each round
            if retrieval_deadline.expired():
                LOGGER.warning(
                    "Retrieval deadline expired (%.1fs elapsed), stopping multi-round loop at round %d",
                    retrieval_deadline.elapsed(), r,
                )
                break
            # 5.1 结果审阅与差距分析
            LOGGER.info("Reviewing candidates for Round %d (retrieval deadline: %.1fs remaining)", r, retrieval_deadline.remaining())

            # P1.5 Task 6: Heuristic-first result review
            # Only use LLM review when pool is small or query_type needs deep analysis
            query_type_for_review = getattr(query_plan, "query_type", "unknown")
            pool_size = len(all_candidates)
            use_llm_review = (
                pool_size < 50
                or query_type_for_review in ("survey", "latest_work", "method_comparison", "broad_topic")
            ) and getattr(self.config.budget, "max_llm_calls", 8) > 0
            # P2: Don't use LLM review if retrieval deadline is close to expiring
            if use_llm_review and retrieval_deadline.remaining() < 10.0:
                use_llm_review = False
                LOGGER.info("Skipping LLM review — low retrieval deadline remaining (%.1fs)", retrieval_deadline.remaining())

            review_llm = self.llm_client if use_llm_review else None
            review_res = review_retrieval_results(query_plan, all_candidates, r - 1, review_llm, self.config)
            # 更新上一轮的审阅结论
            rounds_history[-1].review_conclusion = review_res.get("reason", "审阅完成")

            # 5.2 决策是否提前终止
            if review_res.get("next_action") == "stop_search":
                LOGGER.info("Result Review Agent decided to STOP search in Round %d", r)
                break

            # 5.3 检索策略优化
            LOGGER.info("Optimizing search strategy for Round %d", r)
            # P2: Use heuristic strategy optimization if deadline is low
            opt_llm = self.llm_client if retrieval_deadline.remaining() >= 10.0 else None
            if opt_llm is None and self.llm_client is not None:
                LOGGER.info("Using heuristic strategy optimization (deadline remaining: %.1fs)", retrieval_deadline.remaining())
            opt_res = optimize_search_strategy(query_plan, review_res, all_candidates, r - 1, opt_llm)

            new_subqueries_data = opt_res.get("new_subqueries", [])
            new_queries = []
            for item in new_subqueries_data:
                # P0-2: Don't hardcode route="hybrid" — use evolved route and preserve all fields
                sources = item.get("sources") or ["openalex", "semantic_scholar"]
                new_queries.append(
                    SearchQuery(
                        query=item.get("query", ""),
                        route=item.get("route", "evolved"),
                        intent=item.get("reason", "expanded search"),
                        priority=item.get("priority", 1),
                        required_terms=item.get("required_terms", []),
                        optional_terms=item.get("optional_terms", []),
                        filters=item.get("filters", {}),
                        sources=sources,
                    )
                )

            # P2: Add feedback-based expansion queries from candidate pool
            if query_expander.is_available() and all_candidates:
                feedback_queries = query_expander.feedback_expand(
                    new_queries if new_queries else subqueries,
                    all_candidates,
                    query_plan,
                    max_new_queries=3,
                )
                if feedback_queries:
                    new_queries.extend(feedback_queries)
                    LOGGER.info("Added %d feedback expansion queries for Round %d", len(feedback_queries), r)

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

        # 最后一轮的最终审阅更新 (P1.5: always heuristic, no LLM)
        if rounds_history and rounds_history[-1].review_conclusion == "检索完成":
            final_review = review_retrieval_results(query_plan, all_candidates, len(rounds_history), None, self.config)
            rounds_history[-1].review_conclusion = final_review.get("reason", "最终检索完成")

        all_candidates.sort(key=_get_rough_score, reverse=True)

        # Local pre-rank: compress the rough-sorted pool down to pre_rank_topk
        # (default 200) using cheap local signals before LLM selection.
        from scholar_agent.ranking.local_pre_ranker import local_pre_rank
        pre_rank_topk = getattr(self.config.ranking, "pre_rank_topk", 200)
        pre_ranked_candidates = local_pre_rank(
            all_candidates,
            query_plan,
            original_query,
            topk=pre_rank_topk,
        )
        LOGGER.info(
            "Local pre-rank: %d -> %d candidates (topk=%d)",
            len(all_candidates), len(pre_ranked_candidates), pre_rank_topk,
        )

        # 存储候选池供评估框架审计 (full pool, not pre-ranked subset)
        self.candidate_pool = all_candidates
        budget.candidate_pool_size = len(all_candidates)

        # P2: Log source health stats and LLM circuit breaker for this case
        LOGGER.info("Source health stats: %s", health_manager.get_stats())
        LOGGER.info("LLM circuit breaker: %s", llm_circuit_breaker.get_stats())

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
        # P0-4: Stratified sampling — don't just take top-N, ensure route/provider diversity
        if query_type in ("exact_title", "specific_paper", "single_gold"):
            selection_budget = 40
        elif query_type in ("dataset_constraint", "method_comparison"):
            selection_budget = 50
        elif query_type in ("survey", "broad_topic", "latest_work"):
            selection_budget = 65
        else:
            selection_budget = 45

        selection_candidates: list[Paper] = []
        seen_ids: set[str] = set()

        def _add_unique(paper: Paper) -> None:
            if paper.paper_id not in seen_ids:
                seen_ids.add(paper.paper_id)
                selection_candidates.append(paper)

        # 1. Title-matched papers first (title_exact / title_like)
        for p in pre_ranked_candidates:
            paths = getattr(p, "retrieval_path", [])
            if any("title_exact" in path or "title_like" in path for path in paths):
                _add_unique(p)
                if len(selection_candidates) >= 10:
                    break

        # 2. Route-based stratified round-robin fill
        remaining_after_title = [p for p in pre_ranked_candidates if p.paper_id not in seen_ids]
        route_buckets: dict[str, list[Paper]] = {}
        for p in remaining_after_title:
            paths = getattr(p, "retrieval_path", [])
            route = "unknown"
            for path in paths:
                route = path.split(".")[-1] if "." in path else path
                break
            route_buckets.setdefault(route, []).append(p)

        remaining_budget = selection_budget - len(selection_candidates)
        if remaining_budget > 0 and route_buckets:
            bucket_keys = list(route_buckets.keys())
            idx = 0
            while remaining_budget > 0 and any(route_buckets[k] for k in bucket_keys):
                key = bucket_keys[idx % len(bucket_keys)]
                if route_buckets[key]:
                    _add_unique(route_buckets[key].pop(0))
                    remaining_budget -= 1
                idx += 1

        # 3. Fill any remaining slots with top-scored papers
        for p in pre_ranked_candidates:
            if len(selection_candidates) >= selection_budget:
                break
            _add_unique(p)

        selection_candidates = selection_candidates[:selection_budget]

        LOGGER.info("Starting fine-grained evidence selection for %d papers (filtered from %d pre-ranked candidates)",
                    len(selection_candidates), len(pre_ranked_candidates))
        evidence_deadline = case_deadline.child(DEFAULT_EVIDENCE_SELECTION, "evidence_selection")
        from scholar_agent.selection.batch_evidence_selector import batch_select_and_extract_evidence
        # Effect-first: Set LLM timeout from config's evidence_selection timeout; local-only if expired
        if self.llm_client is not None and not evidence_deadline.expired():
            _ev_timeout = evidence_deadline.timeout_for(
                getattr(self.config.llm, "timeout_evidence_selection", None) or
                getattr(self.config.llm, "timeout_seconds", 15)
            )
            if hasattr(self.llm_client, "timeout"):
                self.llm_client.timeout = _ev_timeout
            evidence_llm = self.llm_client
        else:
            evidence_llm = None
            LOGGER.info("Evidence selection: local-only mode (deadline=%s)", evidence_deadline)
        selections = batch_select_and_extract_evidence(
            selection_candidates, query_plan, evidence_llm,
            original_query=original_query, config=self.config, deadline=evidence_deadline,
        )

        # 7. 本地规则硬核校验与降级
        # Use all_candidates (full pool) — pre_ranked is only for LLM selection input
        LOGGER.info("Validating evidence and constraints locally")
        validated_selections = validate_selections(all_candidates, selections, query_plan)
        self.selections = validated_selections

        # 7.5 Effect-first: LLM Listwise Reranking
        # Select top candidates (high+medium) for listwise reranking
        listwise_scores: dict[str, float] | None = None
        listwise_recommended_k: int | None = None
        listwise_result = None
        if self.llm_client is not None and not case_deadline.expired():
            from scholar_agent.ranking.llm_listwise_reranker import llm_listwise_rerank

            # Build top candidates: high + medium from validated selections
            _sel_map = {sel.paper_id: sel for sel in validated_selections}
            _high_medium_papers = [
                p for p in all_candidates
                if p.paper_id in _sel_map
                and _sel_map[p.paper_id].relevance_level in ("high", "medium")
            ]


            # Relevance gate: filter only extremely weak medium papers before listwise reranking.
            # RECALL-FIRST: we err on the side of inclusion — gold papers that fall through
            # LLM as medium (without evidence due to local fallback) must not be excluded.
            _confident_papers = []
            _filtered_count = 0
            for p in _high_medium_papers:
                sr = _sel_map[p.paper_id]
                if sr.relevance_level == "high":
                    _confident_papers.append(p)
                elif sr.relevance_level == "medium":
                    rel_score = getattr(sr, "relevance_score", 0.0) or 0.0
                    # Only filter out truly weak medium papers (very low relevance score)
                    if rel_score >= 0.25:
                        _confident_papers.append(p)
                    else:
                        _filtered_count += 1

            if _filtered_count > 0:
                LOGGER.info(
                    "Relevance gate: filtered %d weak-medium papers before listwise "
                    "(was %d, now %d)",
                    _filtered_count, len(_high_medium_papers), len(_confident_papers),
                )
            _high_medium_papers = _confident_papers

            # Sort by local pre-rank score for consistent input
            _high_medium_papers.sort(
                key=lambda p: p.metadata.get("local_pre_rank_score", 0.0),
                reverse=True,
            )

            if _high_medium_papers:
                # Reset circuit breaker between evidence selection and listwise reranking.
                # Evidence selection uses v4-flash (many calls); listwise uses v4-pro (1-3 calls).
                # A flash timeout shouldn't block the pro call — different phases, different models.
                if hasattr(self, '_llm_circuit_breaker') and self._llm_circuit_breaker is not None:
                    _pre_reset_stats = self._llm_circuit_breaker.get_stats()
                    self._llm_circuit_breaker.reset()
                    LOGGER.info(
                        "Reset LLM circuit breaker for listwise reranking phase (was: %s)",
                        _pre_reset_stats,
                    )
                # Create a child deadline for listwise reranking
                _listwise_deadline = case_deadline.child(
                    DEFAULT_LISTWISE_RERANK, "listwise_rerank"
                )
                _lr_timeout = _listwise_deadline.timeout_for(
                    getattr(self.config.llm, "timeout_listwise_rerank", None) or
                    getattr(self.config.llm, "timeout_seconds", 15)
                )
                if hasattr(self.llm_client, "timeout"):
                    self.llm_client.timeout = _lr_timeout

                LOGGER.info(
                    "LLM listwise reranking: %d high+medium candidates (deadline=%s)",
                    len(_high_medium_papers), _listwise_deadline,
                )
                listwise_result = llm_listwise_rerank(
                    _high_medium_papers,
                    validated_selections,
                    query_plan,
                    self.llm_client,
                    original_query=original_query,
                    config=self.config,
                    deadline=_listwise_deadline,
                )
                if listwise_result is not None and listwise_result.success:
                    listwise_scores = listwise_result.scores
                    listwise_recommended_k = listwise_result.recommended_k
                    LOGGER.info(
                        "Listwise reranker succeeded: %d papers scored, recommended_k=%d",
                        len(listwise_scores), listwise_recommended_k,
                    )
                else:
                    LOGGER.info("Listwise reranker failed or unavailable. Using local fallback scoring.")
            else:
                LOGGER.info("No high+medium candidates for listwise reranking.")
        else:
            LOGGER.info("Listwise reranking skipped (LLM unavailable or deadline expired).")

        # ── Task 5 (V2): Large-scope second-pass rerank ──
        # If query is large-scope and few confident papers from first pass,
        # trigger a second listwise rerank with diversified top-80 candidates.
        if (
            listwise_result is not None
            and listwise_result.success
            and self.llm_client is not None
            and not case_deadline.expired()
        ):
            _qt = getattr(query_plan, "query_type", "unknown")
            _is_large = _qt in ("survey", "broad_topic", "method_comparison", "latest_work")
            _llm_probs = getattr(listwise_result, "relevance_probabilities", {}) or {}
            _num_confident = sum(1 for v in _llm_probs.values() if v >= 0.50)
            if _is_large and _num_confident <= 2 and len(all_candidates) > 30:
                LOGGER.info(
                    "V2 Second-pass trigger: query_type=%s, confident=%d (<=2), "
                    "pool=%d. Building diversified top-80 for second-pass rerank.",
                    _qt, _num_confident, len(all_candidates),
                )
                _second_pass_papers = _build_diversified_top80(
                    all_candidates, validated_selections,
                )
                if _second_pass_papers and len(_second_pass_papers) > len(_high_medium_papers):
                    _sp_deadline = case_deadline.child(
                        DEFAULT_LISTWISE_RERANK, "second_pass_rerank"
                    )
                    _sp_timeout = _sp_deadline.timeout_for(
                        getattr(self.config.llm, "timeout_listwise_rerank", None) or 30
                    )
                    if hasattr(self.llm_client, "timeout"):
                        self.llm_client.timeout = _sp_timeout
                    _sp_result = llm_listwise_rerank(
                        _second_pass_papers,
                        validated_selections,
                        query_plan,
                        self.llm_client,
                        original_query=original_query,
                        config=self.config,
                        deadline=_sp_deadline,
                    )
                    if _sp_result is not None and _sp_result.success:
                        # Merge: second-pass results override first-pass for shared papers,
                        # and add new papers not in first-pass
                        _merged_scores = dict(listwise_scores or {})
                        _merged_probs = dict(getattr(listwise_result, "relevance_probabilities", {}) or {})
                        _merged_levels = dict(getattr(listwise_result, "relevance_levels", {}) or {})
                        _merged_confs = dict(getattr(listwise_result, "relevance_confidences", {}) or {})
                        _new_count = 0
                        for pid, sc in _sp_result.scores.items():
                            if pid not in _merged_scores:
                                _new_count += 1
                            _merged_scores[pid] = sc
                        for pid, p in (_sp_result.relevance_probabilities or {}).items():
                            _merged_probs[pid] = p
                        for pid, lv in (_sp_result.relevance_levels or {}).items():
                            _merged_levels[pid] = lv
                        for pid, cf in (_sp_result.relevance_confidences or {}).items():
                            _merged_confs[pid] = cf
                        listwise_scores = _merged_scores
                        listwise_result.relevance_probabilities = _merged_probs
                        listwise_result.relevance_levels = _merged_levels
                        listwise_result.relevance_confidences = _merged_confs
                        listwise_result.scores = _merged_scores
                        LOGGER.info(
                            "V2 Second-pass rerank: SUCCESS. Added %d new papers, "
                            "total scored=%d.",
                            _new_count, len(_merged_scores),
                        )
                    else:
                        LOGGER.info("V2 Second-pass rerank: failed or unavailable. Using first-pass results.")
                else:
                    LOGGER.info(
                        "V2 Second-pass: diversified pool not larger than first-pass (%d vs %d). Skipping.",
                        len(_second_pass_papers) if _second_pass_papers else 0,
                        len(_high_medium_papers) if _high_medium_papers else 0,
                    )

        # 8. Effect-first 多维度综合重排
        # Use all_candidates (full pool) so gold papers cut by pre_rank aren't lost
        LOGGER.info("Reranking papers based on effect-first fusion formula")
        ranked_papers = rerank_papers(
            all_candidates, validated_selections, self.config, original_query,
            listwise_scores=listwise_scores,
        )

        # P1.5 Task 3: Store intermediate results for candidate-to-final trace
        self.pre_ranked_candidates = pre_ranked_candidates
        self.selection_candidates = selection_candidates
        self.ranked_papers = ranked_papers

        # 9. 结构化归纳合成与输出
        synthesis_deadline = case_deadline.child(DEFAULT_SYNTHESIS, "synthesis")
        LOGGER.info("Synthesizing final structured report (deadline=%s)", synthesis_deadline)
        # P2: Use LLM synthesis only if deadline allows; otherwise local-only
        if self.llm_client is not None and not synthesis_deadline.expired():
            _syn_timeout = synthesis_deadline.timeout_for(getattr(self.config.llm, "timeout_seconds", 15))
            if hasattr(self.llm_client, "timeout"):
                self.llm_client.timeout = _syn_timeout
            synthesis_llm = self.llm_client
        else:
            synthesis_llm = None
            LOGGER.info("Synthesis: local-only mode (deadline=%s)", synthesis_deadline)
        synthesis_agent = SynthesisAgent(synthesis_llm)

        result = synthesis_agent.synthesize(
            original_query=original_query,
            query_plan=query_plan,
            search_rounds=rounds_history,
            ranked_papers=ranked_papers,
            metrics=budget.get_metrics(),
            config=self.config,
            listwise_result=listwise_result,
        )

        LOGGER.info("Pipeline completed successfully. Found %d recommended papers.", 
                    len(result.highly_relevant_papers) + len(result.partially_relevant_papers))
        return result
