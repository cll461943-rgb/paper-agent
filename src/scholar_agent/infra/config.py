from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AppSection(BaseModel):
    name: str = "scholar-agent"
    mode: str = "mock"
    output_format: str = "json"
    providers: list[str] = Field(default_factory=lambda: ["mock"])
    cache_dir: str = "data/cache"
    parallel_retrieval: bool = False


class ProviderConfig(BaseModel):
    enabled: bool = True
    base_url: str
    timeout_seconds: int = 20
    retry_times: int = 2
    min_interval_seconds: float = 0.0
    max_concurrency: int = 1
    backoff_base_seconds: float = 1.0
    mailto: str = ""
    api_key_env: str = ""
    model_name: str = ""
    device: str = ""
    batch_size: int = 16
    max_length: int = 512
    fp16: bool = False
    normalize_embeddings: bool = True
    embedding_cache_dir: str = "data/cache/embeddings"
    enable_fts: bool = False


class ProviderSection(BaseModel):
    pasa_local: ProviderConfig = ProviderConfig(
        enabled=True,
        base_url="external/pasa/data/paper_database",
    )
    # P0-6: OpenAlex timeout=8 retry=1 min_interval=1.0 backoff=2.0
    openalex: ProviderConfig = ProviderConfig(
        base_url="https://api.openalex.org/works",
        timeout_seconds=8,
        retry_times=1,
        min_interval_seconds=1.0,
        backoff_base_seconds=2.0,
    )
    # P0-6: arXiv default disabled, timeout=5 retry=0
    arxiv: ProviderConfig = ProviderConfig(
        enabled=False,
        base_url="https://export.arxiv.org/api/query",
        timeout_seconds=5,
        retry_times=0,
    )
    # P0-6: Semantic Scholar enabled, timeout=8 retry=1
    semantic_scholar: ProviderConfig = ProviderConfig(
        enabled=True,
        base_url="https://api.semanticscholar.org/graph/v1/paper/search",
        api_key_env="SEMANTIC_SCHOLAR_API_KEY",
        timeout_seconds=8,
        retry_times=1,
    )
    # P0-6: PubMed timeout=8 retry=1
    pubmed: ProviderConfig = ProviderConfig(
        enabled=True,
        base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        timeout_seconds=8,
        retry_times=1,
    )


class LLMConfig(BaseModel):
    enabled: bool = True
    mode: str = "auto"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url_env: str = "DEEPSEEK_BASE_URL"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    model_flash: str = "deepseek-v4-flash"
    model_pro: str = "deepseek-v4-pro"
    timeout_seconds: int = 30  # P0-6: 90→30
    max_tokens: int = 2048
    temperature: float = 0.2
    trust_env: bool = False
    max_retries: int = 1  # P0-6: 3→1
    # Effect-first: per-task max_tokens (overrides global max_tokens when >0)
    max_tokens_query_understanding: int = 4096
    max_tokens_evidence_selection: int = 8192
    max_tokens_listwise_rerank: int = 8192
    max_tokens_synthesis: int = 8192
    # Effect-first: per-stage timeouts (overrides global timeout_seconds when >0)
    timeout_evidence_selection: int = 35
    timeout_listwise_rerank: int = 90
    # Effect-first: circuit breaker threshold (reasoning models need more tolerance)
    circuit_breaker_max_timeouts: int = 2


class BudgetConfig(BaseModel):
    max_search_queries: int = 20
    max_retrieval_rounds: int = 3
    max_results_per_query: int = 40
    max_candidate_pool_size: int = 600
    max_seed_papers: int = 10
    max_refchain_depth: int = 1
    max_llm_selection_papers: int = 50
    max_final_papers: int = 20
    max_evidence_summary_papers: int = 10
    max_llm_calls: int = 30
    max_api_calls: int = 1000
    llm_timeout_seconds: int = 30  # P0-6: 90→30
    api_timeout_seconds: int = 20
    retry_times: int = 2
    known_title_seed_only: bool = False
    enable_query_expansion: bool = True
    adaptive_provider_fallback: bool = True
    source_auto_route: bool = True  # 按 query_type 自动启用源
    enable_translated_route: bool = True  # 自动生成英文翻译检索式
    # Effect-first: configurable case deadline (default 180s, effect-first uses 240s)
    case_deadline_seconds: int = 180


class DynamicKConfig(BaseModel):
    exact_title_k: int = 1
    specific_query_k: int = 2
    dataset_constraint_k: int = 3
    method_comparison_k: int = 3
    survey_k: int = 10  # P0-5: 5→10
    broad_topic_k: int = 10  # P0-5: 5→10
    score_gap_top1: float = 0.15
    score_gap_top3: float = 0.10
    min_final_score: float = 0.45  # P0-5: 0.55→0.45
    fallback_k: int = 5  # P0-5: 3→5
    # F1-aware controller ceiling: max K the controller is allowed to pick.
    # Raise (e.g. 40) for datasets with many gold papers (real50 avg_gold=15.8).
    k_max: int = 20
    # Hard cap on total output (chosen K + recall guard). Raise for high-recall
    # datasets so the guard can actually add high-confidence papers beyond K.
    hard_max_output: int = 12


class RankingConfig(BaseModel):
    max_final_papers: int = 20
    pre_rank_topk: int = 200
    # P1: Default weights matching final_reranker.py defaults
    weights: dict[str, float] = Field(default_factory=lambda: {
        "llm_relevance": 0.25,
        "bge": 0.18,
        "constraint": 0.20,
        "evidence": 0.15,
        "source_agreement": 0.08,
        "recency": 0.05,
        "authority": 0.05,
        "graph_prior": 0.02,
        "title_bonus": 0.02,
    })
    # Effect-first: LLM listwise reranker params
    listwise_topk: int = 40
    listwise_fallback_topk: int = 20
    listwise_score_weight: float = 0.55
    llm_evidence_weight: float = 0.20


class KnownTitleCluesConfig(BaseModel):
    enabled: bool = True


class SelectionConfig(BaseModel):
    exact_title_selection_topk: int = 8
    constraint_selection_topk: int = 15
    broad_selection_topk: int = 20
    default_selection_topk: int = 12
    batch_size: int = 5


class AppConfig(BaseModel):
    app: AppSection = AppSection()
    providers: ProviderSection = ProviderSection()
    llm: LLMConfig = LLMConfig()
    budget: BudgetConfig = BudgetConfig()
    ranking: RankingConfig = RankingConfig()
    dynamic_k: DynamicKConfig = DynamicKConfig()
    known_title_clues: KnownTitleCluesConfig = KnownTitleCluesConfig()
    selection: SelectionConfig = SelectionConfig()
    # Semantic bridge: BGE-M3 vector re-ranking + RRF fusion
    enable_semantic_bridge: bool = False
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_cache_dir: str = "data/cache/embeddings"
    rrf_k: int = 60
    rrf_keyword_weight: float = 1.0
    rrf_vector_weight: float = 1.5


def _load_dotenv_if_present(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


def load_config(path: str | Path | None = None) -> AppConfig:
    _load_dotenv_if_present()
    config_path = Path(path or "configs/default.yaml")
    if not config_path.exists():
        # Fallback to local default configs inside project
        config_path = Path(__file__).parents[2] / "configs" / "default.yaml"
    raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw_data)
