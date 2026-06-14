from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

from scholar_agent.models.schemas import Paper, RetrievalResult, SearchQuery
from scholar_agent.retrieval.base import PaperProvider
from scholar_agent.retrieval.refchain import annotate_refchain_candidate, refchain_operations
from scholar_agent.workflow.budget import BudgetManager

LOCAL_ZERO_API_PROVIDERS = {"mock", "pasa_local"}


class MultiRouteRetriever:
    def __init__(
        self,
        providers: list[PaperProvider],
        budget: BudgetManager,
        parallel: bool = False,
    ) -> None:
        self.providers = providers
        self.budget = budget
        self.parallel = parallel
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
            self._provider_block_reasons[provider.name] = "rate_limited"
        elif error_count >= 2:
            self._provider_block_reasons[provider.name] = "repeated_errors"

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
        cache_hits_delta = 1 if cache_hit else 0
        errors_delta = 0
        papers: list[Paper] = []
        started_at = time.perf_counter()
        block_reason = self._provider_block_reasons.get(provider.name)
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
            papers = provider.search(query, limit=limit)
            provider_error = getattr(provider, "last_error", None)
            if provider_error:
                self._mark_provider_error(provider, provider_error)
                self.budget.record_error(f"{provider.name}.{query.route}: {provider_error}")
                errors_delta = 1
            if api_calls_delta:
                self.budget.record_api_call(api_calls_delta)
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
            self._mark_provider_error(provider, str(exc))
            if api_calls_delta:
                self.budget.record_api_call(api_calls_delta)
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
                api_calls_delta=api_calls_delta,
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
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text or "")
        if len(words) < 3:
            return False
        lowered = " ".join(words).lower()
        bad_prefixes = [
            "are there", "can you", "do you", "find papers", "give me",
            "i am looking", "list all", "papers about", "papers on",
            "papers that", "provide", "research papers", "show me",
        ]
        if any(lowered.startswith(prefix) for prefix in bad_prefixes):
            return False
        titleish_words = sum(1 for word in words if word[:1].isupper() or word.isupper())
        return titleish_words >= 3

    def _title_exact_results(self, queries: list[SearchQuery], providers: list[PaperProvider]) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        title_like_queries = [
            query
            for query in queries
            if query.route == "title_like" and self._looks_like_title_query(query.query)
        ]
        for provider in providers:
            if not self._overrides("search_title_exact", provider):
                continue
            for query in title_like_queries:
                component = f"retrieval.{provider.name}.title_exact"
                started_at = time.perf_counter()
                papers: list[Paper] = []
                limit = min(10, self.budget.config.max_results_per_query)
                exact_query = self._title_exact_query(query.query)
                cache_hit = self._has_query_cache_hit(provider, exact_query, limit)
                block_reason = self._provider_block_reasons.get(provider.name)
                if block_reason and not cache_hit:
                    self.budget.record_component_cost(
                        component,
                        time.perf_counter() - started_at,
                        items_delta=0,
                    )
                    results.append(
                        RetrievalResult(
                            provider=provider.name,
                            route="title_exact",
                            search_query=query,
                            papers=[],
                            error=f"skipped_after_{block_reason}",
                        )
                    )
                    continue
                api_calls_delta = 0 if cache_hit else self._title_exact_api_calls(provider)
                cache_hits_delta = 1 if cache_hit else 0
                try:
                    papers = provider.search_title_exact(
                        query.query,
                        limit=limit,
                    )
                    if api_calls_delta:
                        self.budget.record_api_call(api_calls_delta)
                    if cache_hits_delta:
                        self.budget.record_cache_hit(cache_hits_delta)
                finally:
                    self.budget.record_component_cost(
                        component,
                        time.perf_counter() - started_at,
                        api_calls_delta=api_calls_delta,
                        cache_hits_delta=cache_hits_delta,
                        items_delta=len(papers),
                    )
                if not papers:
                    continue
                results.append(
                    RetrievalResult(
                        provider=provider.name,
                        route="title_exact",
                        search_query=query,
                        papers=papers,
                        truncated=False,
                    )
                )
        return results

    def _search_provider_routes(self, provider: PaperProvider, queries: list[SearchQuery]) -> list[RetrievalResult]:
        return [self._search_one(provider, query) for query in queries]

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

    def expand_refchain(self, seed_papers: list[Paper], providers: list[PaperProvider], limit_per_seed: int) -> list[Paper]:
        expanded: list[Paper] = []
        if limit_per_seed <= 0:
            return expanded
        for provider in providers:
            operations = refchain_operations(provider)
            for seed_index, paper in enumerate(seed_papers, start=1):
                remaining = limit_per_seed
                for route, source, operation in operations:
                    if remaining <= 0:
                        break
                    component = f"retrieval.{provider.name}.{route}"
                    operation_limit = remaining
                    api_calls_delta = self._reference_api_calls(provider, paper, operation_limit, route)
                    papers: list[Paper] = []
                    started_at = time.perf_counter()
                    try:
                        papers = operation(paper, limit=operation_limit)[:operation_limit]
                        if api_calls_delta:
                            self.budget.record_api_call(api_calls_delta)
                    finally:
                        self.budget.record_component_cost(
                            component,
                            time.perf_counter() - started_at,
                            api_calls_delta=api_calls_delta,
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
                    remaining -= len(papers)
        return expanded

    def retrieve(self, queries: list[SearchQuery], include_title_exact: bool = True) -> tuple[list[RetrievalResult], list[Paper]]:
        self.budget.reserve_retrieval_round()
        providers: list[PaperProvider] = []
        results: list[RetrievalResult] = []
        for provider in self.providers:
            if provider.is_available():
                providers.append(provider)
                continue
            unavailable_result = self._record_unavailable_provider(provider, queries)
            if unavailable_result is not None:
                results.append(unavailable_result)

        def _get_queries_for_provider(prov: PaperProvider) -> list[SearchQuery]:
            if prov.name in LOCAL_ZERO_API_PROVIDERS or prov.name == "pasa_local":
                return queries
            return queries[:4]

        if self.parallel and len(providers) > 1:
            with ThreadPoolExecutor(max_workers=len(providers)) as executor:
                future_map = [
                    executor.submit(self._search_provider_routes, provider, _get_queries_for_provider(provider))
                    for provider in providers
                ]
                for future in as_completed(future_map):
                    results.extend(future.result())
        else:
            for provider in providers:
                prov_queries = _get_queries_for_provider(provider)
                for query in prov_queries:
                    results.append(self._search_one(provider, query))

        if include_title_exact:
            results.extend(self._title_exact_results(queries, providers))

        candidate_pool: list[Paper] = []
        for result in results:
            for paper in result.papers:
                candidate_pool.append(paper)

        return results, candidate_pool
