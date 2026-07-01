from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    original_query: str
    language: str = "zh"
    query_type: str = "unknown"
    research_topic: str | None = None
    task: str | None = None
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    time_range: dict[str, Any] | None = None
    venues: list[str] = Field(default_factory=list)
    must_have_constraints: list[str] = Field(default_factory=list)
    nice_to_have_constraints: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    expected_output: str = "top_papers"
    uncertainty: list[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    query: str
    route: str
    intent: str
    required_terms: list[str] = Field(default_factory=list)
    optional_terms: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    priority: int = 1
    sources: list[str] = Field(default_factory=list)
    max_results: int | None = None


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    citation_count: int | None = None
    source: str = "mock"
    retrieval_path: list[str] = Field(default_factory=list)  # 用于多源、多轮路径记录
    references: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    provider: str
    route: str
    search_query: SearchQuery
    papers: list[Paper] = Field(default_factory=list)
    truncated: bool = False
    error: str | None = None


class EvidenceItem(BaseModel):
    field: str  # 'title', 'abstract', 'metadata', 等
    text: str


class SelectionResult(BaseModel):
    paper_id: str
    relevance_level: Literal["high", "medium", "low", "irrelevant"]
    matched_constraints: list[str] = Field(default_factory=list)
    missing_constraints: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    reason: str
    confidence: float = 1.0
    # 证据校验判定属性
    is_validated: bool = True
    validation_notes: list[str] = Field(default_factory=list)
    # 数值化分数（供 final_reranker 使用）
    relevance_score: float | None = None
    constraint_score: float | None = None
    evidence_score: float | None = None
    uncertainty: list[str] = Field(default_factory=list)


class RankedPaper(BaseModel):
    paper: Paper
    selection: SelectionResult
    final_score: float
    subscores: dict[str, float] = Field(default_factory=dict)
    rank: int


class EvidenceSummary(BaseModel):
    paper_id: str
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class ResearchMap(BaseModel):
    topic: str
    method_groups: dict[str, list[str]] = Field(default_factory=dict)
    dataset_distribution: dict[str, int] = Field(default_factory=dict)
    time_trend: dict[str, int] = Field(default_factory=dict)
    citation_edges: list[dict[str, str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ComponentMetric(BaseModel):
    component: str
    elapsed_seconds: float = 0.0
    items_delta: int = 0
    api_calls_delta: int = 0
    api_calls_total: int = 0
    llm_calls_delta: int = 0
    llm_calls_total: int = 0
    token_estimate_delta: int = 0
    token_estimate_total: int = 0
    cache_hits_delta: int = 0
    cache_hits_total: int = 0
    errors_delta: int = 0
    errors_total: int = 0
    search_queries_delta: int = 0
    search_queries_total: int = 0
    retrieval_rounds_delta: int = 0
    retrieval_rounds_total: int = 0


class SearchProcessRound(BaseModel):
    round_index: int
    search_goal: str
    queries: list[str] = Field(default_factory=list)
    candidates_found: int = 0
    review_conclusion: str = ""


class RunMetrics(BaseModel):
    search_queries_used: int = 0
    retrieval_rounds_used: int = 0
    candidate_pool_size: int = 0
    final_papers: int = 0
    evidence_summaries: int = 0
    llm_calls_used: int = 0
    llm_elapsed_seconds: float = 0.0
    api_calls_used: int = 0
    token_estimate: int = 0
    elapsed_seconds: float = 0.0
    cache_hits: int = 0
    errors: list[str] = Field(default_factory=list)
    component_metrics: list[ComponentMetric] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    original_query: str
    query_plan: QueryPlan
    search_process: list[SearchProcessRound] = Field(default_factory=list)
    highly_relevant_papers: list[RankedPaper] = Field(default_factory=list)
    partially_relevant_papers: list[RankedPaper] = Field(default_factory=list)
    supporting_papers: list[RankedPaper] = Field(default_factory=list)  # Task 4: weak/background, not in eval
    method_clusters: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    citation_graph: dict[str, Any] = Field(default_factory=dict)
    recommendation_reasoning: list[dict[str, Any]] = Field(default_factory=list)
    agent_self_report: dict[str, Any] = Field(default_factory=dict)
    run_metrics: RunMetrics
    # Task 5: K controller metadata for evaluation
    dynamic_k_chosen: int | None = None
    expected_f1_curve: dict[int, float] | None = None
    g_hat: float | None = None
    # V2 K controller metadata
    g_hat_scope: float | None = None
    g_hat_pool: float | None = None
    g_hat_visible: float | None = None
    low_confidence_uniform: bool = False
    p_floor: float | None = None
    tie_break_reason: str = ""
    second_pass_triggered: bool = False
