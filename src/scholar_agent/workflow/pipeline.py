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
from scholar_agent.retrieval.embedding_service import EmbeddingService
from scholar_agent.retrieval.semantic_bridge import SemanticBridge
from scholar_agent.selection.evidence_selector import select_and_extract_evidence
from scholar_agent.selection.evidence_validator import validate_selections
from scholar_agent.ranking.final_reranker import rerank_papers
from scholar_agent.ranking.local_pre_ranker import local_pre_rank_score
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


def _selection_budget_for_query(
    query_type: str,
    *,
    llm_off_fallback: bool,
    pool_size: int | None = None,
) -> int:
    """Choose selector depth.

    When LLM review is unavailable, local evidence scoring is cheap and recall
    should dominate. SPAR Case30 showed gold papers at ranks 103-198 that were
    in the pool but cut before selector input.
    """
    if query_type in ("exact_title", "specific_paper", "single_gold"):
        selection_budget = 40
    elif query_type in ("dataset_constraint", "method_comparison"):
        selection_budget = 50
    elif query_type in ("survey", "broad_topic", "latest_work"):
        selection_budget = 65
    else:
        selection_budget = 45

    if llm_off_fallback:
        selection_budget = max(selection_budget, 200)

    if pool_size is not None:
        selection_budget = min(selection_budget, max(pool_size, 0))
    return selection_budget


def _metadata_float(paper: Paper, key: str, default: float = 0.0) -> float:
    try:
        return float(paper.metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def _has_retrieval_route(paper: Paper, needle: str) -> bool:
    return any(needle in path for path in (paper.retrieval_path or []))


def _is_listwise_recall_bridge_candidate(
    paper: Paper,
    selection: SelectionResult,
) -> bool:
    """Allow strong low-level retrieval hits into listwise without relabeling them."""
    if selection.relevance_level != "low":
        return False

    notes = " ".join(selection.validation_notes or []).lower()
    if "year constraint violated" in notes:
        return False

    local_score = _metadata_float(paper, "local_pre_rank_score")
    matched_count = len(selection.matched_constraints or [])
    route_count = len(set(paper.retrieval_path or []))
    has_title_exact = _has_retrieval_route(paper, "title_exact")
    has_title_like = _has_retrieval_route(paper, "title_like")

    if selection.is_validated is False and not (has_title_exact or has_title_like):
        return False

    if has_title_exact:
        return local_score >= 0.30 or matched_count > 0 or route_count >= 2
    if has_title_like:
        return local_score >= 0.55 or (local_score >= 0.45 and matched_count > 0)

    return local_score >= 0.70 and (matched_count > 0 or route_count >= 3)


def _select_listwise_input_candidates(
    all_candidates: list[Paper],
    validated_selections: list[SelectionResult],
    *,
    max_low_bridge: int = 12,
) -> tuple[list[Paper], int, int]:
    """Select recall-first listwise inputs from validated evidence results.

    High and medium candidates keep the existing behavior. A small number of
    low-labeled candidates can be bridged into listwise when retrieval evidence
    is strong enough that the listwise reranker should adjudicate them.
    """
    sel_map = {sel.paper_id: sel for sel in validated_selections}
    selected: list[Paper] = []
    bridge_candidates: list[Paper] = []
    filtered_medium_count = 0

    for paper in all_candidates:
        selection = sel_map.get(paper.paper_id)
        if selection is None:
            continue

        if selection.relevance_level == "high":
            selected.append(paper)
        elif selection.relevance_level == "medium":
            rel_score = getattr(selection, "relevance_score", 0.0) or 0.0
            if rel_score >= 0.25:
                selected.append(paper)
            else:
                filtered_medium_count += 1
        elif _is_listwise_recall_bridge_candidate(paper, selection):
            bridge_candidates.append(paper)

    selected_ids = {paper.paper_id for paper in selected}
    bridge_candidates.sort(
        key=lambda paper: _metadata_float(paper, "local_pre_rank_score"),
        reverse=True,
    )
    bridged_count = 0
    for paper in bridge_candidates:
        if paper.paper_id not in selected_ids and len(selected) < 40:
            selected.append(paper)
            selected_ids.add(paper.paper_id)
            bridged_count += 1
        if bridged_count >= max_low_bridge:
            break

    selected.sort(
        key=lambda paper: _metadata_float(paper, "local_pre_rank_score"),
        reverse=True,
    )
    return selected, filtered_medium_count, bridged_count


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

        # P0-3: Log all prompt contracts at startup for auditing
        try:
            from scholar_agent.prompts import PromptRegistry
            LOGGER.info("=== Prompt Contracts ===")
            for summary in PromptRegistry.summaries():
                LOGGER.info("  %s", summary)
        except Exception:
            pass  # prompts module is optional, pipeline still works without it

        LOGGER.info("Starting PaperAgentPipeline for query: %s (deadline=%.0fs)", original_query, _case_deadline_secs)
        _listwise_reserve = min(
            DEFAULT_LISTWISE_RERANK + DEFAULT_SYNTHESIS,
            max(45.0, float(_case_deadline_secs) * 0.32),
        )

        # 2. 初始化多路检索器
        retriever = MultiRouteRetriever(
            self.providers,
            budget,
            parallel=bool(getattr(self.config.app, "parallel_retrieval", False)),
            health_manager=health_manager,
        )
        query_expander = QueryExpander()
        rounds_history: list[SearchProcessRound] = []

        # 导入新模块：动态路由 + LLM 池审阅
        from scholar_agent.retrieval.dynamic_router import get_routing_config
        from scholar_agent.selection.llm_pool_reviewer import llm_review_pool

        # 粗排评分函数（用于内部排序，不决定去留）
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
            plan_bonus = 0.0
            constraint_bonus = 0.0
            if query_plan is not None:
                local_score, local_subscores = local_pre_rank_score(p, query_plan, original_query)
                p.metadata["local_pre_rank_subscores"] = local_subscores
                p.metadata["constraint_aware_local_score"] = local_score
                plan_bonus = local_score * 0.65
                constraint_bonus = max(0.0, local_subscores.get("constraint_coverage", 0.0)) * 0.35

            return base_score + paths_val * 0.01 + citation_score + route_bonus + title_bonus + plan_bonus + constraint_bonus

        # ── 闭环迭代架构 ──
        # LLM-first 理解 → 动态路由 → 检索 → LLM 审阅粗排 → 重新理解（带 feedback）→ 收敛
        all_candidates: list[Paper] = []
        query_plan = None
        prev_plan = None
        last_review_feedback = None
        max_rounds = getattr(self.config.budget, "max_retrieval_rounds", 3)
        # retrieval_only 模式也要跑 LLM 审阅（评测才能反映新架构）
        if not self.llm_client or not self.llm_client.is_available():
            max_rounds = 1

        pool_review_deadline = case_deadline.child(DEFAULT_RETRIEVAL, "pool_review")

        for round_idx in range(1, max_rounds + 1):
            if case_deadline.remaining() <= _listwise_reserve:
                LOGGER.warning(
                    "Stopping retrieval loop before round %d to preserve listwise reserve (remaining=%.1fs, reserve=%.1fs)",
                    round_idx, case_deadline.remaining(), _listwise_reserve,
                )
                break

            # ── Step A: LLM-first query understanding（首轮独立，后续带 feedback） ──
            query_deadline = case_deadline.child_with_reserve(
                DEFAULT_QUERY_UNDERSTANDING,
                reserve_seconds=_listwise_reserve,
                stage_name=f"query_understanding_r{round_idx}",
            )
            use_llm_query = (
                self.llm_client is not None
                and not query_deadline.expired()
                and getattr(self.config.budget, "max_llm_calls", 8) > 0
            )
            if use_llm_query:
                _configured_timeout = getattr(self.config.llm, "timeout_seconds", 15)
                _effective_timeout = query_deadline.timeout_for(_configured_timeout)
                if hasattr(self.llm_client, "timeout"):
                    self.llm_client.timeout = _effective_timeout
                try:
                    query_plan = understand_query(
                        original_query, self.llm_client,
                        feedback=last_review_feedback,
                        prev_plan=prev_plan,
                    )
                except Exception as exc:
                    LOGGER.warning("LLM Query understanding failed in round %d: %s. Falling back.", round_idx, exc)
                    query_plan = heuristic_understand_query(original_query)
            else:
                LOGGER.info("Skipping LLM query understanding in round %d", round_idx)
                query_plan = heuristic_understand_query(original_query)

            prev_plan = query_plan
            LOGGER.info("Round %d: query_type=%s, methods=%s", round_idx, query_plan.query_type, query_plan.methods[:3])

            # ── Step B: 动态路由配置 ──
            review_feedback_for_routing = None
            if last_review_feedback:
                review_feedback_for_routing = {
                    "missing_aspects": last_review_feedback.get("missing_aspects", []),
                    "noise_patterns": last_review_feedback.get("noise_patterns", []),
                    "kept_count": len(all_candidates),
                    "target_size": 150,
                }
            routing_config = get_routing_config(query_plan, review_feedback_for_routing)
            LOGGER.info("Round %d routing: %s", round_idx, routing_config.reason)

            # ── Step C: 查询生成（首轮 LLM，后续轮可复用或重新生成） ──
            retrieval_deadline = case_deadline.child_with_reserve(
                DEFAULT_RETRIEVAL,
                reserve_seconds=_listwise_reserve,
                stage_name=f"retrieval_r{round_idx}",
            )
            if round_idx == 1 or last_review_feedback:
                if self.llm_client is not None and not retrieval_deadline.expired():
                    try:
                        subqueries = generate_search_queries(query_plan, budget, self.llm_client)
                    except Exception as exc:
                        LOGGER.warning("LLM query generation failed round %d: %s. Falling back.", round_idx, exc)
                        subqueries = heuristic_generate_search_queries(query_plan)
                        budget.record_search_queries(len(subqueries))
                else:
                    subqueries = heuristic_generate_search_queries(query_plan)
                    budget.record_search_queries(len(subqueries))

                if query_expander.is_available():
                    subqueries = [query_expander.expand_query(q) for q in subqueries]

            # ── Step D: 多源检索（用动态路由配置） ──
            LOGGER.info("Round %d: retrieving with %d queries", round_idx, len(subqueries))
            _, new_pool = retriever.retrieve(
                subqueries,
                original_query=original_query,
                query_plan=query_plan,
                routing_config=routing_config,
                deadline=retrieval_deadline,
            )

            # 合并去重
            all_candidates.extend(new_pool)
            all_candidates = deduplicate_papers(all_candidates)
            all_candidates.sort(key=_get_rough_score, reverse=True)

            # ── Step E: 引文网络扩展（所有 query_type 均启用） ──
            # faiss_vector 向量检索已覆盖大部分语义召回缺口，refchain 保持轻量
            if round_idx <= 2:
                seed_papers = all_candidates[:5]
                try:
                    expanded_pool = retriever.expand_refchain(
                        seed_papers,
                        self.providers,
                        limit_per_seed=10,
                        deadline=retrieval_deadline,
                    )
                    all_candidates.extend(expanded_pool)
                    all_candidates = deduplicate_papers(all_candidates)
                    all_candidates.sort(key=_get_rough_score, reverse=True)
                    LOGGER.info("Refchain expansion: +%d papers (seeds=%d, total=%d)",
                                len(expanded_pool), len(seed_papers), len(all_candidates))
                except Exception as exc:
                    LOGGER.warning("Refchain expansion failed round %d: %s", round_idx, exc)

            # ── Step F: LLM 审阅粗排（替代 local_pre_rank） ──
            pool_review_deadline = case_deadline.child_with_reserve(
                DEFAULT_RETRIEVAL,
                reserve_seconds=_listwise_reserve,
                stage_name=f"pool_review_r{round_idx}",
            )
            pool_review_llm = self.llm_client if (
                self.llm_client is not None
                and not pool_review_deadline.expired()
                and getattr(self.config.budget, "max_llm_calls", 8) > 0
            ) else None

            if pool_review_llm and hasattr(self.llm_client, "timeout"):
                _pr_timeout = pool_review_deadline.timeout_for(
                    getattr(self.config.llm, "timeout_pool_review", None) or 60
                )
                self.llm_client.timeout = _pr_timeout

            review_result = llm_review_pool(
                all_candidates,
                query_plan,
                pool_review_llm,
                original_query,
                config=self.config,
                deadline=pool_review_deadline,
                batch_size=getattr(getattr(self.config, "selection", None), "pool_review_batch_size", 10) or 10,
                max_review_papers=getattr(getattr(self.config, "selection", None), "pool_review_max_papers", 200) or 200,
            )

            # 更新 all_candidates 为 LLM 审阅后的保留集
            all_candidates = review_result["kept_papers"]
            # Set local_pre_rank_score so downstream sorting (Stage 6 sampling, listwise input) works
            for p in all_candidates:
                p.metadata["local_pre_rank_score"] = _get_rough_score(p)
            all_candidates.sort(key=_get_rough_score, reverse=True)

            # 记录本轮
            round_record = SearchProcessRound(
                round_index=round_idx,
                search_goal=f"Round {round_idx}: 动态路由+LLM审阅 (kept={len(all_candidates)}, missing={len(review_result.get('missing_aspects', []))})",
                queries=[q.query for q in subqueries],
                candidates_found=len(all_candidates),
                review_conclusion=f"coverage={review_result.get('coverage_score', 0):.2f}, converged={review_result.get('converged', True)}",
            )
            rounds_history.append(round_record)

            # ── Step G: 收敛判定 ──
            missing_aspects = review_result.get("missing_aspects", [])
            converged = review_result.get("converged", True)
            if converged and not missing_aspects:
                LOGGER.info("Round %d converged (no missing aspects, coverage=%.2f). Stopping loop.",
                            round_idx, review_result.get("coverage_score", 0))
                break
            if round_idx >= max_rounds:
                LOGGER.info("Round %d reached max_rounds. Stopping loop.", round_idx)
                break

            # 准备下一轮的 feedback
            last_review_feedback = review_result
            LOGGER.info("Round %d: %d missing aspects, will re-understand in round %d",
                        round_idx, len(missing_aspects), round_idx + 1)

        # 最终排序
        for p in all_candidates:
            if "local_pre_rank_score" not in p.metadata:
                p.metadata["local_pre_rank_score"] = _get_rough_score(p)
        all_candidates.sort(key=_get_rough_score, reverse=True)

        # pre_ranked_candidates = LLM 审阅后的池子（替代旧的 local_pre_rank 输出）
        pre_ranked_candidates = all_candidates
        LOGGER.info(
            "Stage 1 final: %d candidates after LLM pool review (closed-loop, %d rounds)",
            len(pre_ranked_candidates), len(rounds_history),
        )

        # Semantic bridge: BGE-M3 vector re-ranking + RRF fusion
        # Bridges vocabulary mismatch that keyword search cannot handle
        enable_semantic = getattr(self.config, "enable_semantic_bridge", False)
        if enable_semantic and all_candidates:
            try:
                embedding_service = EmbeddingService(
                    model_name=getattr(self.config, "embedding_model_name", "BAAI/bge-m3"),
                    device=getattr(self.config, "embedding_device", "cuda"),
                    cache_dir=getattr(self.config, "embedding_cache_dir", "data/cache/embeddings"),
                )
                bridge = SemanticBridge(
                    embedding_service,
                    keyword_weight=getattr(self.config, "rrf_keyword_weight", 1.0),
                    vector_weight=getattr(self.config, "rrf_vector_weight", 1.5),
                    rrf_k=getattr(self.config, "rrf_k", 60),
                )
                all_candidates = bridge.apply(
                    candidate_pool=all_candidates,
                    original_query=original_query,
                    llm_client=self.llm_client,
                    query_plan=query_plan,
                    enable_hyde=False,  # Skip HyDE LLM call to save 20-30s per case
                )
                pre_ranked_candidates = all_candidates
                LOGGER.info(
                    "Semantic bridge applied: %d candidates after RRF fusion",
                    len(all_candidates),
                )
            except Exception as e:
                LOGGER.warning("Semantic bridge failed (falling back to keyword-only): %s", e)

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
        llm_off_fallback = (
            self.llm_client is None
            or not self.llm_client.is_available()
            or getattr(llm_circuit_breaker, "is_tripped", False)
        )
        # P0-4: Stratified sampling — don't just take top-N, ensure route/provider diversity.
        # When LLM is unavailable, local selector is cheap; use a deeper recall-first budget.
        selection_budget = _selection_budget_for_query(
            query_type,
            llm_off_fallback=llm_off_fallback,
            pool_size=len(pre_ranked_candidates),
        )
        if llm_off_fallback:
            LOGGER.info("LLM-off fallback: expanded selection budget to %d", selection_budget)

        selection_candidates: list[Paper] = []
        seen_ids: set[str] = set()

        def _add_unique(paper: Paper) -> None:
            if paper.paper_id not in seen_ids:
                seen_ids.add(paper.paper_id)
                selection_candidates.append(paper)

        # 1. Title-matched papers first (title_exact / title_like)
        # LLM-off fallback relies heavily on title-like routes; keep enough
        # depth for multi-route title hits before route diversity fills the cap.
        title_match_budget = min(20, max(10, selection_budget // 2))
        for p in pre_ranked_candidates:
            paths = getattr(p, "retrieval_path", [])
            if any("title_exact" in path or "title_like" in path for path in paths):
                _add_unique(p)
                if len(selection_candidates) >= title_match_budget:
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

        # 3a. Dense-only high-score papers (BGE-M3 vector search results not yet selected)
        # These papers were found by faiss_vector but missed by route-based sampling.
        # They may contain gold papers that keyword search couldn't find.
        _dense_only = [
            p for p in pre_ranked_candidates
            if p.paper_id not in seen_ids
            and any("faiss_vector" in path or "semantic" in path for path in (p.retrieval_path or []))
            and float(p.metadata.get("vector_score", 0) or p.metadata.get("bge_score", 0) or 0) > 0.3
        ]
        _dense_only.sort(
            key=lambda p: float(p.metadata.get("vector_score", 0) or p.metadata.get("bge_score", 0) or 0),
            reverse=True,
        )
        _dense_budget = min(len(_dense_only), max(5, selection_budget // 8))
        for p in _dense_only[:_dense_budget]:
            if len(selection_candidates) >= selection_budget:
                break
            _add_unique(p)
        if _dense_only:
            LOGGER.info("Dense-only sampling: %d candidates (vector_score>0.3), added %d",
                        len(_dense_only), _dense_budget)

        # 3b. Soft-dropped high-score papers (pool review said soft_drop but score is high)
        # Non-destructive pool review tags papers as soft_drop/noise instead of deleting.
        # Give high-score soft_dropped papers a second chance at evidence selection.
        _soft_dropped = [
            p for p in pre_ranked_candidates
            if p.paper_id not in seen_ids
            and p.metadata.get("pool_review_action") in ("soft_drop", "noise")
        ]
        _soft_dropped.sort(key=_get_rough_score, reverse=True)
        _soft_budget = min(len(_soft_dropped), max(3, selection_budget // 12))
        for p in _soft_dropped[:_soft_budget]:
            if len(selection_candidates) >= selection_budget:
                break
            _add_unique(p)
        if _soft_dropped:
            LOGGER.info("Soft-dropped sampling: %d candidates (pool_review=soft_drop/noise), added %d",
                        len(_soft_dropped), _soft_budget)

        # 4. Fill any remaining slots with top-scored papers
        for p in pre_ranked_candidates:
            if len(selection_candidates) >= selection_budget:
                break
            _add_unique(p)

        selection_candidates = selection_candidates[:selection_budget]

        LOGGER.info("Starting fine-grained evidence selection for %d papers (filtered from %d pre-ranked candidates)",
                    len(selection_candidates), len(pre_ranked_candidates))
        evidence_deadline = case_deadline.child_with_reserve(
            DEFAULT_EVIDENCE_SELECTION,
            reserve_seconds=_listwise_reserve,
            stage_name="evidence_selection",
        )
        from scholar_agent.selection.batch_evidence_selector import batch_select_and_extract_evidence
        # Effect-first: Set LLM timeout from config's evidence_selection timeout; local-only if expired
        if llm_off_fallback:
            evidence_llm = None
            LOGGER.info("Evidence selection: local-only mode (LLM unavailable or circuit breaker open)")
        elif self.llm_client is not None and not evidence_deadline.expired():
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
        if self.llm_client is not None and not llm_off_fallback and not case_deadline.expired():
            from scholar_agent.ranking.llm_listwise_reranker import llm_listwise_rerank

            _listwise_input_papers, _filtered_count, _bridged_count = (
                _select_listwise_input_candidates(all_candidates, validated_selections)
            )

            if _filtered_count > 0:
                LOGGER.info(
                    "Relevance gate: filtered %d weak-medium papers before listwise "
                    "(was %d, now %d)",
                    _filtered_count,
                    len(_listwise_input_papers) + _filtered_count,
                    len(_listwise_input_papers),
                )
            if _bridged_count > 0:
                LOGGER.info(
                    "Listwise recall bridge: added %d strong low-level candidates.",
                    _bridged_count,
                )

            if _listwise_input_papers:
                # Reset circuit breaker between evidence selection and listwise reranking.
                # Evidence selection uses v4-flash (many calls); listwise uses v4-pro (1-3 calls).
                # A flash timeout shouldn't block the pro call — different phases, different models.
                if hasattr(self, '_llm_circuit_breaker') and self._llm_circuit_breaker is not None:
                    _pre_reset_stats = self._llm_circuit_breaker.get_stats()
                    if hasattr(self._llm_circuit_breaker, "reset_transient"):
                        self._llm_circuit_breaker.reset_transient()
                    else:
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
                    "LLM listwise reranking: %d candidates (deadline=%s)",
                    len(_listwise_input_papers), _listwise_deadline,
                )
                listwise_result = llm_listwise_rerank(
                    _listwise_input_papers,
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
                LOGGER.info("No eligible candidates for listwise reranking.")
        else:
            LOGGER.info("Listwise reranking skipped (LLM unavailable, circuit breaker open, or deadline expired).")

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
                if _second_pass_papers and len(_second_pass_papers) > len(_listwise_input_papers):
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
                        len(_listwise_input_papers) if _listwise_input_papers else 0,
                    )

        # 8. Effect-first 多维度综合重排
        # Use all_candidates (full pool) so gold papers cut by pre_rank aren't lost
        LOGGER.info("Reranking papers based on effect-first fusion formula")
        _rel_probs = {}
        if listwise_result is not None and listwise_result.success:
            _rel_probs = getattr(listwise_result, "relevance_probabilities", {}) or {}
        ranked_papers = rerank_papers(
            all_candidates, validated_selections, self.config, original_query,
            listwise_scores=listwise_scores,
            relevance_probabilities=_rel_probs,
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
