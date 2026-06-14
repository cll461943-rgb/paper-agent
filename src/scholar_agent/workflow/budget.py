from __future__ import annotations

import logging
import time
from typing import Any

from scholar_agent.models.schemas import ComponentMetric, RunMetrics

LOGGER = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """当检索或大模型调用超出预算设定时抛出。"""
    pass


class BudgetManager:
    def __init__(self, config: Any) -> None:
        self.config = config.budget  # 对应 budget 相关的限制
        self.raw_config = config
        self.start_time = time.perf_counter()
        
        self.llm_calls_used = 0
        self.llm_elapsed_seconds = 0.0
        self.api_calls_used = 0
        self.token_estimate = 0
        self.cache_hits = 0
        self.retrieval_rounds_used = 0
        self.search_queries_used = 0
        self.candidate_pool_size = 0
        self.final_papers = 0
        self.evidence_summaries = 0
        
        self.errors: list[str] = []
        self.component_metrics_map: dict[str, ComponentMetric] = {}

    def reserve_llm_call(self) -> None:
        max_llm = getattr(self.config, "max_llm_calls", 30)
        if self.llm_calls_used >= max_llm:
            raise BudgetExceededError(f"LLM call budget exceeded: current={self.llm_calls_used}, max={max_llm}")
        self.llm_calls_used += 1

    def reserve_retrieval_round(self) -> None:
        max_rounds = getattr(self.config, "max_retrieval_rounds", 3)
        if self.retrieval_rounds_used >= max_rounds:
            raise BudgetExceededError(f"Retrieval round budget exceeded: current={self.retrieval_rounds_used}, max={max_rounds}")
        self.retrieval_rounds_used += 1

    def record_llm_elapsed(self, elapsed: float) -> None:
        self.llm_elapsed_seconds += elapsed

    def record_token_estimate(self, count: int) -> None:
        self.token_estimate += count

    def record_error(self, message: str) -> None:
        self.errors.append(message)

    def record_api_call(self, count: int = 1) -> None:
        self.api_calls_used += count

    def record_cache_hit(self, count: int = 1) -> None:
        self.cache_hits += count

    def record_search_queries(self, count: int = 1) -> None:
        self.search_queries_used += count

    def reserve_search_queries(self, count: int) -> None:
        max_queries = getattr(self.config, "max_search_queries", 15)
        if self.search_queries_used + count > max_queries:
            raise BudgetExceededError(
                f"Search queries budget exceeded: current={self.search_queries_used}, requested={count}, max={max_queries}"
            )
        self.search_queries_used += count

    def record_component_cost(
        self,
        component: str,
        elapsed_seconds: float,
        items_delta: int = 0,
        api_calls_delta: int = 0,
        llm_calls_delta: int = 0,
        token_estimate_delta: int = 0,
        cache_hits_delta: int = 0,
        errors_delta: int = 0,
        search_queries_delta: int = 0,
        retrieval_rounds_delta: int = 0,
    ) -> None:
        metric = self.component_metrics_map.get(component)
        if metric is None:
            metric = ComponentMetric(component=component)
            self.component_metrics_map[component] = metric

        metric.elapsed_seconds += elapsed_seconds
        metric.items_delta += items_delta
        metric.api_calls_delta += api_calls_delta
        metric.api_calls_total += api_calls_delta
        metric.llm_calls_delta += llm_calls_delta
        metric.llm_calls_total += llm_calls_delta
        metric.token_estimate_delta += token_estimate_delta
        metric.token_estimate_total += token_estimate_delta
        metric.cache_hits_delta += cache_hits_delta
        metric.cache_hits_total += cache_hits_delta
        metric.errors_delta += errors_delta
        metric.errors_total += errors_delta
        metric.search_queries_delta += search_queries_delta
        metric.search_queries_total += search_queries_delta
        metric.retrieval_rounds_delta += retrieval_rounds_delta
        metric.retrieval_rounds_total += retrieval_rounds_delta

    def get_metrics(self) -> RunMetrics:
        elapsed_total = time.perf_counter() - self.start_time
        return RunMetrics(
            search_queries_used=self.search_queries_used,
            retrieval_rounds_used=self.retrieval_rounds_used,
            candidate_pool_size=self.candidate_pool_size,
            final_papers=self.final_papers,
            evidence_summaries=self.evidence_summaries,
            llm_calls_used=self.llm_calls_used,
            llm_elapsed_seconds=round(self.llm_elapsed_seconds, 3),
            api_calls_used=self.api_calls_used,
            token_estimate=self.token_estimate,
            elapsed_seconds=round(elapsed_total, 3),
            cache_hits=self.cache_hits,
            errors=self.errors,
            component_metrics=list(self.component_metrics_map.values()),
        )
