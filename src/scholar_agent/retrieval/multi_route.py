from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import time

LOGGER = logging.getLogger(__name__)

from scholar_agent.models.schemas import Paper, RetrievalResult, SearchQuery
from scholar_agent.retrieval.base import PaperProvider
from scholar_agent.retrieval.refchain import annotate_refchain_candidate, refchain_operations
from scholar_agent.retrieval.source_health import SourceHealthManager
from scholar_agent.utils.title_query import looks_like_paper_title
from scholar_agent.workflow.budget import BudgetManager

LOCAL_ZERO_API_PROVIDERS = {"mock", "pasa_local", "faiss_vector"}


class MultiRouteRetriever:
    def __init__(
        self,
        providers: list[PaperProvider],
        budget: BudgetManager,
        parallel: bool = False,
        health_manager: SourceHealthManager | None = None,
    ) -> None:
        self.providers = providers
        self.budget = budget
        self.parallel = parallel
        self._health = health_manager
        self._provider_error_counts: dict[str, int] = {}
        self._provider_block_reasons: dict[str, str] = {}

    @staticmethod
    def _has_query_cache_hit(provider: PaperProvider, query: SearchQuery, limit: int) -> bool:
        cache = getattr(provider, "cache", None)
        if cache is None:
            return False
        try:
            return cache.get_query(provider.query_cache_key(query, limit)) is not None
        except Exception:
            return False

    @staticmethod
    def _search_api_calls(provider: PaperProvider, cache_hit: bool) -> int:
        if provider.name in LOCAL_ZERO_API_PROVIDERS or cache_hit:
            return 0
        return 1

    @staticmethod
    def _is_rate_limit_error(error: str) -> bool:
        lowered = error.lower()
        return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered

    def _mark_provider_error(self, provider: PaperProvider, error: str) -> None:
        error_count = self._provider_error_counts.get(provider.name, 0) + 1
        self._provider_error_counts[provider.name] = error_count
        if self._is_rate_limit_error(error):
            # Allow up to 3 rate-limit errors before blocking (was 1)
            if error_count >= 3:
                self._provider_block_reasons[provider.name] = "rate_limited"
        elif error_count >= 5:
            self._provider_block_reasons[provider.name] = "repeated_errors"

    def _is_provider_blocked(self, provider_name: str) -> str | None:
        """Check if a provider is blocked. Delegates to health_manager when available."""
        if self._health is not None:
            if not self._health.before_call(provider_name):
                state = self._health.get_state(provider_name)
                return state.value
            return None
        # Legacy fallback
        return self._provider_block_reasons.get(provider_name)

    def _record_provider_error(self, provider: PaperProvider, error: str) -> None:
        """Record a provider error. Delegates to health_manager when available."""
        if self._health is not None:
            self._health.record_error(provider.name, error)
        # Also track in legacy system for backward-compatible block_reasons
        self._mark_provider_error(provider, error)

    def _record_provider_success(
        self, provider: PaperProvider, elapsed: float, items: int
    ) -> None:
        """Record a successful provider call. Only health_manager tracks this."""
        if self._health is not None:
            self._health.record_success(provider.name, elapsed, items)

    @staticmethod
    def _overrides(method_name: str, provider: PaperProvider) -> bool:
        return getattr(type(provider), method_name) is not getattr(PaperProvider, method_name)

    @staticmethod
    def _reference_api_calls(provider: PaperProvider, paper: Paper, limit: int, route: str) -> int:
        route_method = {
            "reference_expansion": "get_references",
            "citation_expansion": "get_citations",
            "related_work_expansion": "get_related_works",
        }.get(route)
        if provider.name in LOCAL_ZERO_API_PROVIDERS or route_method is None or not MultiRouteRetriever._overrides(route_method, provider):
            return 0
        if provider.name == "openalex" and route == "reference_expansion":
            return min(len(paper.references), limit)
        if provider.name == "openalex" and route == "related_work_expansion":
            return min(len(paper.metadata.get("related_works", [])), limit)
        if provider.name == "openalex" and route == "citation_expansion":
            return 1 if paper.metadata.get("cited_by_api_url") else 0
        return 1

    def _search_one(self, provider: PaperProvider, query: SearchQuery) -> RetrievalResult:
        limit = self.budget.config.max_results_per_query
        component = f"retrieval.{provider.name}.{query.route}"
        cache_hit = self._has_query_cache_hit(provider, query, limit)
        api_calls_delta = self._search_api_calls(provider, cache_hit)
        api_calls_reserved = 0
        cache_hits_delta = 1 if cache_hit else 0
        errors_delta = 0
        papers: list[Paper] = []
        started_at = time.perf_counter()
        block_reason = self._is_provider_blocked(provider.name)
        if block_reason and not cache_hit:
            self.budget.record_component_cost(
                component,
                time.perf_counter() - started_at,
                items_delta=0,
            )
            return RetrievalResult(
                provider=provider.name,
                route=query.route,
                search_query=query,
                papers=[],
                error=f"skipped_after_{block_reason}",
            )
        try:
            if api_calls_delta:
                self.budget.reserve_api_call(api_calls_delta)
                api_calls_reserved = api_calls_delta
            papers = provider.search(query, limit=limit)
            _elapsed = time.perf_counter() - started_at
            provider_error = getattr(provider, "last_error", None)
            if provider_error:
                self._record_provider_error(provider, provider_error)
                self.budget.record_error(f"{provider.name}.{query.route}: {provider_error}")
                errors_delta = 1
            else:
                self._record_provider_success(provider, _elapsed, len(papers))
            if cache_hits_delta:
                self.budget.record_cache_hit(cache_hits_delta)
            return RetrievalResult(
                provider=provider.name,
                route=query.route,
                search_query=query,
                papers=papers[:limit],
                truncated=len(papers) > limit,
                error=provider_error,
            )
        except Exception as exc:
            errors_delta = 1
            self._record_provider_error(provider, str(exc))
            if cache_hits_delta:
                self.budget.record_cache_hit(cache_hits_delta)
            self.budget.record_error(f"{provider.name} search failed for route={query.route}: {exc}")
            return RetrievalResult(
                provider=provider.name,
                route=query.route,
                search_query=query,
                papers=[],
                error=str(exc),
            )
        finally:
            self.budget.record_component_cost(
                component,
                time.perf_counter() - started_at,
                api_calls_delta=api_calls_reserved,
                cache_hits_delta=cache_hits_delta,
                errors_delta=errors_delta,
                items_delta=len(papers),
            )

    @staticmethod
    def _title_exact_api_calls(provider: PaperProvider) -> int:
        if provider.name in LOCAL_ZERO_API_PROVIDERS:
            return 0
        return 1

    @staticmethod
    def _title_exact_query(title: str) -> SearchQuery:
        return SearchQuery(
            query=title,
            route="title_exact",
            intent="title_exact_recall",
            required_terms=[],
            optional_terms=[],
            filters={},
            priority=0,
        )

    @staticmethod
    def _looks_like_title_query(text: str) -> bool:
        return looks_like_paper_title(text)

    def _domain_scores(self, text: str) -> dict[str, float]:
        text_l = text.lower()

        keyword_groups = {
            "biomedical": [
                "antibody", "protein", "gene", "dna", "rna", "medical",
                "clinical", "disease", "cancer", "vaccine", "biomedical",
                "drug", "biological", "cell", "receptor", "virus",
                "pathogen", "diagnosis", "protein design", "antibody design",
            ],
            "cs_ai": [
                "llm", "large language model", "transformer", "diffusion",
                "neural network", "deep learning", "nlp", "computer vision",
                "video generation", "image", "dataset", "reinforcement learning",
                "fine-tuning", "finetuning", "agent", "dpo", "language model",
                "autoregressive", "scaling law", "vision-language",
                "vision language", "watermarking", "test time training",
                "mixture of experts", "moe", "multimodal",
            ],
            "physics_math": [
                "quantum", "physics", "math", "equation", "monte carlo",
                "differential", "algebra", "topology", "geometry",
                "statistical", "thermodynamics", "optics",
            ],
        }

        scores = {}
        for domain, kws in keyword_groups.items():
            hit = 0
            for kw in kws:
                if kw in text_l:
                    hit += 1
            scores[domain] = hit / max(len(kws), 1)

        return scores

    def _determine_providers_by_keywords(
        self,
        query,
        original_query: str = "",
        query_plan=None,
    ) -> set[str]:
        # 1. 永远保留核心库
        allowed = {"pasa_local", "openalex", "semantic_scholar"}

        # 2. 聚合上下文语义特征
        parts = [
            original_query or "",
            getattr(query, "query", "") or "",
            " ".join(getattr(query, "required_terms", []) or []),
            " ".join(getattr(query, "optional_terms", []) or []),
        ]

        if query_plan is not None:
            parts.extend(getattr(query_plan, "methods", []) or [])
            parts.extend(getattr(query_plan, "datasets", []) or [])
            parts.extend(getattr(query_plan, "entities", []) or [])
            parts.append(getattr(query_plan, "research_topic", "") or "")

        routing_text = " ".join(parts)
        scores = self._domain_scores(routing_text)

        route = getattr(query, "route", "")

        # 3. 生医领域追加 PubMed
        if scores["biomedical"] > 0 or any(
            x in routing_text.lower()
            for x in ["protein", "antibody", "clinical", "medical", "biomedical", "drug"]
        ):
            allowed.add("pubmed")

        # 4. AI/CS/latest/title-like 追加 arXiv
        if (
            scores["cs_ai"] > 0
            or route in {"title_like", "title_exact", "latest", "translated"}
            or any(x in routing_text.lower() for x in ["arxiv", "preprint", "latest", "recent"])
        ):
            allowed.add("arxiv")

        # 5. 物理/数学追加 arXiv
        if scores["physics_math"] > 0:
            allowed.add("arxiv")

        # 6. 自带 sources 取并集
        explicit_sources = set(getattr(query, "sources", []) or [])
        if explicit_sources:
            allowed |= explicit_sources

        return allowed

    def _title_exact_results(
        self,
        queries: list[SearchQuery],
        providers: list[PaperProvider],
        deadline=None,
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        if deadline is not None and deadline.expired():
            return results
        title_like_queries = [
            query
            for query in queries
            if query.route == "title_like" and self._looks_like_title_query(query.query)
        ]
        title_like_queries = title_like_queries[:5]

        def _fetch(provider: PaperProvider, query: SearchQuery) -> RetrievalResult | None:
            component = f"retrieval.{provider.name}.title_exact"
            started_at = time.perf_counter()
            papers: list[Paper] = []
            limit = min(10, self.budget.config.max_results_per_query)
            exact_query = self._title_exact_query(query.query)
            cache_hit = self._has_query_cache_hit(provider, exact_query, limit)
            block_reason = self._is_provider_blocked(provider.name)
            if block_reason and not cache_hit:
                self.budget.record_component_cost(
                    component,
                    time.perf_counter() - started_at,
                    items_delta=0,
                )
                return RetrievalResult(
                    provider=provider.name,
                    route="title_exact",
                    search_query=query,
                    papers=[],
                    error=f"skipped_after_{block_reason}",
                )
            api_calls_delta = 0 if cache_hit else self._title_exact_api_calls(provider)
            api_calls_reserved = 0
            cache_hits_delta = 1 if cache_hit else 0
            errors_delta = 0
            try:
                if api_calls_delta:
                    self.budget.reserve_api_call(api_calls_delta)
                    api_calls_reserved = api_calls_delta
                papers = provider.search_title_exact(
                    query.query,
                    limit=limit,
                )
                self._record_provider_success(provider, time.perf_counter() - started_at, len(papers))
                if cache_hits_delta:
                    self.budget.record_cache_hit(cache_hits_delta)
            except Exception as exc:
                errors_delta = 1
                self._record_provider_error(provider, str(exc))
                self.budget.record_error(f"{provider.name} title_exact failed: {exc}")
            finally:
                self.budget.record_component_cost(
                    component,
                    time.perf_counter() - started_at,
                    api_calls_delta=api_calls_reserved,
                    cache_hits_delta=cache_hits_delta,
                    errors_delta=errors_delta,
                    items_delta=len(papers),
                )
            if not papers:
                return None
            return RetrievalResult(
                provider=provider.name,
                route="title_exact",
                search_query=query,
                papers=papers,
                truncated=False,
            )

        tasks = []
        for provider in providers:
            if not self._overrides("search_title_exact", provider):
                continue
            for query in title_like_queries:
                tasks.append((provider, query))

        if not tasks:
            return results

        if self.parallel and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
                future_map = [executor.submit(_fetch, p, q) for p, q in tasks]
                for future in as_completed(future_map):
                    res = future.result()
                    if res is not None:
                        results.append(res)
        else:
            for p, q in tasks:
                if deadline is not None and deadline.expired():
                    break
                res = _fetch(p, q)
                if res is not None:
                    results.append(res)
                    
        return results

    def _search_provider_routes(
        self,
        provider: PaperProvider,
        queries: list[SearchQuery],
        deadline=None,
    ) -> list[RetrievalResult]:
        results = []
        started_at = time.perf_counter()
        provider_time_budget = self._health.get_time_budget(provider.name) if self._health else 60.0
        if deadline is not None:
            provider_time_budget = min(provider_time_budget, max(0.0, deadline.remaining()))
        for query in queries:
            if deadline is not None and deadline.expired():
                self.budget.record_error(f"{provider.name} skipped remaining queries due to retrieval_deadline")
                break
            if time.perf_counter() - started_at > provider_time_budget:
                self.budget.record_error(f"{provider.name} skipped remaining queries due to provider_time_budget ({provider_time_budget}s)")
                break
            results.append(self._search_one(provider, query))
        return results

    def _record_unavailable_provider(self, provider: PaperProvider, queries: list[SearchQuery]) -> RetrievalResult | None:
        message = getattr(provider, "last_error", None) or "provider unavailable"
        self.budget.record_error(f"{provider.name}.unavailable: {message}")
        self.budget.record_component_cost(
            f"retrieval.{provider.name}.unavailable",
            0.0,
            errors_delta=1,
            items_delta=0,
        )
        if not queries:
            return None
        return RetrievalResult(
            provider=provider.name,
            route="unavailable",
            search_query=queries[0],
            papers=[],
            error=message,
        )

    def expand_refchain(
        self,
        seed_papers: list[Paper],
        providers: list[PaperProvider],
        limit_per_seed: int,
        deadline=None,
    ) -> list[Paper]:
        expanded: list[Paper] = []
        if limit_per_seed <= 0:
            return expanded
        if deadline is not None and deadline.expired():
            return expanded
        for provider in providers:
            if deadline is not None and deadline.expired():
                break
            if self._is_provider_blocked(provider.name):
                LOGGER.info("Skipping refchain for blocked provider %s", provider.name)
                continue
            operations = refchain_operations(provider)
            for seed_index, paper in enumerate(seed_papers, start=1):
                # Each route gets its own independent budget (not shared across routes)
                for route, source, operation in operations:
                    if deadline is not None and deadline.expired():
                        return expanded
                    component = f"retrieval.{provider.name}.{route}"
                    operation_limit = limit_per_seed
                    api_calls_delta = self._reference_api_calls(provider, paper, operation_limit, route)
                    api_calls_reserved = 0
                    errors_delta = 0
                    papers: list[Paper] = []
                    started_at = time.perf_counter()
                    try:
                        if api_calls_delta:
                            self.budget.reserve_api_call(api_calls_delta)
                            api_calls_reserved = api_calls_delta
                        papers = operation(paper, limit=operation_limit)[:operation_limit]
                        self._record_provider_success(provider, time.perf_counter() - started_at, len(papers))
                    except Exception as exc:
                        errors_delta = 1
                        self._record_provider_error(provider, str(exc))
                        self.budget.record_error(f"{provider.name} {route} failed: {exc}")
                    finally:
                        self.budget.record_component_cost(
                            component,
                            time.perf_counter() - started_at,
                            api_calls_delta=api_calls_reserved,
                            errors_delta=errors_delta,
                            items_delta=len(papers),
                        )
                    expanded.extend(
                        annotate_refchain_candidate(
                            candidate,
                            provider_name=provider.name,
                            route=route,
                            source=source,
                            seed_paper=paper,
                            seed_index=seed_index,
                            limit_per_seed=limit_per_seed,
                        )
                        for candidate in papers
                    )
        return expanded

    def _need_broad_safety_search(self, candidate_pool: list[Paper], results: list[RetrievalResult], query_plan, routing_config=None) -> bool:
        # 动态阈值：优先用 routing_config，否则默认 120
        if routing_config is not None and hasattr(routing_config, "broad_safety_threshold"):
            threshold = routing_config.broad_safety_threshold
        else:
            threshold = 120

        if len(candidate_pool) < threshold:
            return True

        providers_used = {r.provider for r in results if not r.error}
        if "semantic_scholar" not in providers_used:
            return True

        query_type = getattr(query_plan, "query_type", "unknown")
        if query_type in {"survey", "broad_topic", "method_comparison"} and len(candidate_pool) < max(threshold, 250):
            return True

        return False

    def retrieve(
        self,
        queries: list[SearchQuery],
        include_title_exact: bool = True,
        original_query: str = "",
        query_plan = None,
        routing_config = None,
        deadline=None,
    ) -> tuple[list[RetrievalResult], list[Paper]]:
        if deadline is not None and deadline.expired():
            self.budget.record_error("retrieval skipped due to retrieval_deadline")
            return [], []

        self.budget.reserve_retrieval_round()
        providers: list[PaperProvider] = []
        results: list[RetrievalResult] = []
        for provider in self.providers:
            if provider.is_available() and not self._is_provider_blocked(provider.name):
                providers.append(provider)
                continue
            unavailable_result = self._record_unavailable_provider(provider, queries)
            if unavailable_result is not None:
                results.append(unavailable_result)

        def _get_queries_for_provider(prov: PaperProvider) -> list[SearchQuery]:
            if prov.name in LOCAL_ZERO_API_PROVIDERS or prov.name == "pasa_local":
                return queries

            # 动态路由：优先用 routing_config，否则 fallback 到硬编码表
            if routing_config is not None:
                allowed_routes = routing_config.routes_per_provider.get(prov.name)
                if allowed_routes is None:
                    allowed_routes = None  # None = 全路由
                else:
                    allowed_routes = set(allowed_routes)
                max_cap = routing_config.caps_per_provider.get(prov.name, 4)
            else:
                # P2 Task 5: 静态硬编码 fallback（向后兼容）
                route_priority = {
                    "openalex": {"title_like", "title_exact", "core_topic", "method_task", "broad_synonym", "entity_dataset", "dataset", "translated", "query2doc", "hyde", "evolved", "hybrid", "feedback_expansion"},
                    "arxiv": {"latest", "title_like", "title_exact", "core_topic", "broad_synonym", "feedback_expansion"},
                    "pubmed": {"biomedical", "dataset", "method_task", "feedback_expansion"},
                    "semantic_scholar": {"title_like", "title_exact", "core_topic", "broad_synonym", "method_task", "citation_seed", "translated", "feedback_expansion"},
                }
                allowed_routes = route_priority.get(prov.name, set())
                max_by_provider = {
                    "openalex": 8,
                    "semantic_scholar": 5,
                    "arxiv": 3,
                    "pubmed": 2,
                }
                max_cap = max_by_provider.get(prov.name, 4)

            selected = []
            for q in queries:
                # 动态关键词文献库过滤匹配
                allowed_provs = self._determine_providers_by_keywords(
                    q,
                    original_query=original_query,
                    query_plan=query_plan,
                )
                if prov.name not in allowed_provs:
                    continue

                if getattr(q, "sources", None) and prov.name in q.sources:
                    selected.append(q)
                elif allowed_routes is None or q.route in allowed_routes:
                    selected.append(q)

            return selected[:max_cap]

        if self.parallel and len(providers) > 1:
            with ThreadPoolExecutor(max_workers=len(providers)) as executor:
                future_map = [
                    executor.submit(self._search_provider_routes, provider, _get_queries_for_provider(provider), deadline)
                    for provider in providers
                ]
                for future in as_completed(future_map):
                    results.extend(future.result())
        else:
            for provider in providers:
                if deadline is not None and deadline.expired():
                    break
                prov_queries = _get_queries_for_provider(provider)
                results.extend(self._search_provider_routes(provider, prov_queries, deadline))

        if include_title_exact and not (deadline is not None and deadline.expired()):
            results.extend(self._title_exact_results(queries, providers, deadline))

        candidate_pool: list[Paper] = []
        for result in results:
            for paper in result.papers:
                candidate_pool.append(paper)

        # 5. 安全兜底检索逻辑
        if self._need_broad_safety_search(candidate_pool, results, query_plan, routing_config):
            safety_queries = [
                q for q in queries
                if q.route in {"core_topic", "method_task", "broad_synonym", "translated", "title_like"}
            ][:3]

            safety_providers = [
                p for p in providers
                if p.name in {"openalex", "semantic_scholar", "pasa_local", "arxiv"}
            ]

            for provider in safety_providers:
                for q in safety_queries:
                    if deadline is not None and deadline.expired():
                        return results, candidate_pool
                    already_run = any(
                        r.provider == provider.name and r.search_query.query == q.query
                        for r in results
                        if not r.error
                    )
                    if not already_run:
                        LOGGER.info("Triggered safety fallback search for query: '%s' on %s", q.query, provider.name)
                        fallback_res = self._search_one(provider, q)
                        results.append(fallback_res)
                        for paper in fallback_res.papers:
                            candidate_pool.append(paper)

        return results, candidate_pool
