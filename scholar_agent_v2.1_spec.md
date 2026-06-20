Scholar Agent V2.1 Implementation.md
====================================

0. 文件说明
-----------

本文档定义 Scholar Agent V2.1 的工程实现方案，目标是在 V2.0
已经具备高召回候选池的基础上，重点提升最终
Precision、F1、运行效率和赛题要求中的 LLM 迭代搜索能力。

V2.1 不推翻 V2.0 架构，而是在以下位置做关键修正：

1.  将固定 Top3 输出改为 Dynamic-K 自适应输出。

2.  将 Known Title Clues 从默认强规则改为 debug-only 或弱加分。

3.  将 title\_exact/title\_like 从“强制改判”改为“证据约束加分”。

4.  将 Evidence Selector 从逐篇串行调用改为 batch JSON 调用。

5.  将 arXiv 从默认在线源改为按需启用源。

6.  增加 Result Review Agent，实现赛题要求的 LLM 自主搜索策略迭代优化。

7.  增加 TopK sweep、selector false positive、candidate-to-final loss
    等评测指标。

8.  为后续学习式 reranker 预留特征导出接口。

1. V2.1 总体目标
----------------

### 1.1 效果目标

V2.1
的核心目标不是继续扩大候选池，而是把已经召回的正确论文稳定推到最终输出列表。

建议目标如下：

    avg_f1_final          >= 0.45
    avg_precision_final   >= 0.50
    avg_recall_final      >= 0.55
    candidate_recall@300  >= 0.75
    candidate_recall@500  >= 0.75
    selector_false_negative <= 2
    selector_false_positive 可统计、可解释

其中，F1 和 Precision 优先级高于 Recall 的继续提升。V2.0
已经证明候选召回能力较强，当前瓶颈主要在最终选择和排序。

### 1.2 效率目标

    live-lite 模式平均 wall_time <= 30s
    live-full 模式平均 wall_time <= 45s
    平均 LLM calls/query <= 4
    平均 token/query <= 10000
    over_budget query 数显著下降
    errors = 0 或接近 0

### 1.3 赛题对齐目标

V2.1 必须显式覆盖赛题的四个核心能力：

    查询理解与分解
    基于 LLM 的自主搜索策略迭代优化
    论文综合排序
    搜索结果归纳整理

因此，V2.1 不应只是检索流水线，而应是：

    LLM Agent 规划
    → 工具检索
    → LLM 审阅候选池
    → LLM 生成下一轮策略
    → 迭代检索
    → 证据筛选
    → 综合排序
    → 结构化输出

2. 当前 V2.0 问题诊断
---------------------

### 2.1 指标层面问题

V2.0 的主要现象：

    候选池 Recall@300 提升
    最终 Recall 提升
    Precision 明显下降
    F1 明显下降
    wall_time 明显上升
    Evidence Selector 成为最大耗时组件

这说明 V2.0 已经解决了“找不到”的一部分问题，但没有解决“选得准”的问题。

### 2.2 工程层面问题

当前主要风险点如下：

  问题                             表现                              V2.1 处理策略
  -------------------------------- --------------------------------- -------------------------------------
  固定 Top3 输出                   单金标任务中 Precision 天然受限   改为 Dynamic-K
  Known Title Clues                有测试集过拟合风险                official 模式禁用，debug 模式保留
  title\_exact 强制 high           可能把错误标题提升为 high         改为条件加分
  title\_like 保底 medium          可能引入大量 false positive       改为弱加分
  Evidence Selector 串行逐篇调用   耗时和 token 高                   改为 batch JSON
  arXiv 默认请求                   慢、易 timeout、易 429            改为按 query 类型启用
  缺少 LLM Review                  不满足自主搜索策略迭代优化要求    增加 Result Review Agent
  人工权重 rerank                  对不同 query 类型不稳定           先做权重校准，再预留学习式 reranker

3. V2.1 总体架构
----------------

    User Query
      ↓
    Query Understanding Agent
      ↓
    Query Contract
      ↓
    Initial Search Planning Agent
      ↓
    Round 1 Search Plan
      ↓
    Retrieval Tool Layer
      ├── OpenAlex
      ├── Semantic Scholar
      ├── PubMed
      ├── arXiv Router
      ├── BM25
      ├── BGE-M3
      ├── SPECTER2
      └── Local Cache
      ↓
    Candidate Pool Round 1
      ↓
    Candidate Processor
      ├── normalize
      ├── deduplicate
      ├── merge metadata
      └── rough rank
      ↓
    Result Review Agent
      ├── coverage analysis
      ├── missing aspect detection
      ├── noise pattern detection
      └── next search strategy
      ↓
    Strategy Optimization Agent
      ├── continue search
      ├── refine query
      ├── add negative keywords
      ├── enable citation expansion
      └── stop search
      ↓
    Iterative Search Loop
      ↓
    Final Candidate Pool
      ↓
    Local Reranker
      ↓
    Batch Evidence Selector
      ↓
    Evidence Validator
      ↓
    Final Reasoning Reranker
      ↓
    Dynamic-K Controller
      ↓
    Synthesis Agent
      ↓
    Structured Report

4. 推荐项目目录结构
-------------------

    paper-agent/
    ├── docs/
    │   ├── implementation.md
    │   ├── evaluation_protocol.md
    │   ├── ablation_plan.md
    │   └── prompt_templates.md
    │
    ├── src/scholar_agent/
    │   ├── app.py
    │   ├── cli.py
    │   │
    │   ├── models/
    │   │   ├── schemas.py
    │   │   ├── enums.py
    │   │   └── metrics_schema.py
    │   │
    │   ├── workflow/
    │   │   ├── pipeline.py
    │   │   ├── agent_loop.py
    │   │   ├── state.py
    │   │   ├── budget.py
    │   │   └── run_metrics.py
    │   │
    │   ├── planning/
    │   │   ├── query_understanding.py
    │   │   ├── search_planning.py
    │   │   ├── result_review.py
    │   │   ├── strategy_optimization.py
    │   │   └── query_generation.py
    │   │
    │   ├── retrieval/
    │   │   ├── base.py
    │   │   ├── multi_route.py
    │   │   ├── source_router.py
    │   │   ├── openalex.py
    │   │   ├── semantic_scholar.py
    │   │   ├── arxiv.py
    │   │   ├── pubmed.py
    │   │   ├── pasa_local.py
    │   │   ├── bm25_retriever.py
    │   │   ├── dense_retriever.py
    │   │   ├── specter_retriever.py
    │   │   └── citation_expander.py
    │   │
    │   ├── processing/
    │   │   ├── normalization.py
    │   │   ├── dedup.py
    │   │   ├── candidate_quality.py
    │   │   ├── known_title_clues.py
    │   │   └── title_matcher.py
    │   │
    │   ├── ranking/
    │   │   ├── rough_ranker.py
    │   │   ├── bge_reranker.py
    │   │   ├── final_reranker.py
    │   │   ├── dynamic_k.py
    │   │   ├── feature_exporter.py
    │   │   └── learning_to_rank.py
    │   │
    │   ├── selection/
    │   │   ├── evidence_selector.py
    │   │   ├── batch_evidence_selector.py
    │   │   ├── evidence_validator.py
    │   │   └── selector_calibration.py
    │   │
    │   ├── synthesis/
    │   │   ├── synthesis_agent.py
    │   │   ├── evidence_card.py
    │   │   ├── research_map.py
    │   │   └── citation_graph.py
    │   │
    │   ├── evaluation/
    │   │   ├── metrics.py
    │   │   ├── evaluator.py
    │   │   ├── topk_sweep.py
    │   │   ├── error_analysis.py
    │   │   └── ablation_runner.py
    │   │
    │   └── utils/
    │       ├── json_utils.py
    │       ├── text_utils.py
    │       ├── retry.py
    │       ├── cache.py
    │       └── logging.py
    │
    ├── configs/
    │   ├── default.yaml
    │   ├── live_lite.yaml
    │   ├── live_full.yaml
    │   ├── debug.yaml
    │   └── official.yaml
    │
    ├── data/
    │   ├── cache/
    │   ├── eval/
    │   ├── benchmarks/
    │   └── logs/
    │
    ├── scripts/
    │   ├── run_eval.py
    │   ├── run_ablation.py
    │   ├── run_topk_sweep.py
    │   ├── export_errors.py
    │   └── reproduce_v21.sh
    │
    └── tests/
        ├── test_query_understanding.py
        ├── test_source_router.py
        ├── test_dynamic_k.py
        ├── test_evidence_validator.py
        ├── test_batch_selector.py
        └── test_metrics.py

5. 配置文件设计
---------------

### 5.1 official.yaml

`official.yaml`
是最终正式评测模式，禁止使用可能过拟合当前开发集的人工金标映射。

    mode: official

    agent:
      max_rounds: 2
      max_subqueries_per_round: 5
      enable_result_review: true
      enable_strategy_optimization: true

    retrieval:
      max_candidates_total: 500
      max_candidates_for_rerank: 300
      max_results_per_query: 30
      enable_bm25: true
      enable_bge_m3: true
      enable_specter2: true
      enable_citation_expansion: true
      citation_expansion_depth: 1
      citation_expansion_seeds: 5
      citation_expansion_per_seed: 5

    source_router:
      enable_openalex: true
      enable_semantic_scholar: true
      enable_pubmed: true
      enable_arxiv: auto
      arxiv_timeout_seconds: 5
      arxiv_max_calls_per_query: 1
      disable_arxiv_for_general_queries: true

    known_title_clues:
      enabled: false
      mode: disabled

    title_matching:
      enable_generic_title_fuzzy: true
      title_exact_force_high: false
      title_like_force_medium: false
      title_exact_bonus: 0.15
      title_like_bonus: 0.05
      title_similarity_high_threshold: 0.92

    selection:
      enable_batch_evidence_selector: true
      max_papers_for_selector: 10
      batch_size: 5
      max_tokens_per_batch: 6000
      selector_temperature: 0.0
      enable_evidence_validator: true

    ranking:
      enable_bge_reranker: true
      bge_topk: 50
      enable_final_reranker: true
      enable_dynamic_k: true
      fallback_k: 3
      max_final_papers: 10

    dynamic_k:
      exact_title_k: 1
      specific_query_k: 2
      dataset_constraint_k: 3
      method_comparison_k: 3
      survey_k: 5
      broad_topic_k: 5
      score_gap_top1: 0.15
      score_gap_top3: 0.10
      min_final_score: 0.55

    budget:
      max_llm_calls_per_query: 4
      max_tokens_per_query: 10000
      max_wall_time_seconds: 45
      enable_cache: true

    logging:
      save_agent_trace: true
      save_retrieval_trace: true
      save_selector_trace: true
      save_ranking_features: true
      save_error_analysis: true

### 5.2 debug.yaml

`debug.yaml` 用于开发和错误定位，允许开启 Known Title Clues。

    mode: debug

    known_title_clues:
      enabled: true
      mode: weak_bonus
      allow_force_title_route: true

    title_matching:
      title_exact_force_high: false
      title_like_force_medium: false
      title_exact_bonus: 0.20
      title_like_bonus: 0.08

    selection:
      max_papers_for_selector: 20

    budget:
      max_llm_calls_per_query: 10
      max_tokens_per_query: 20000
      max_wall_time_seconds: 90

6. 核心数据结构
---------------

### 6.1 QueryContract

    class QueryContract(BaseModel):
        query_id: str | None = None
        original_query: str
        language: Literal["zh", "en", "mixed", "unknown"]

        research_intent: str | None = None
        query_type: Literal[
            "exact_title",
            "specific_paper",
            "single_gold",
            "multi_gold",
            "survey",
            "method_comparison",
            "dataset_constraint",
            "latest_work",
            "citation_tracking",
            "broad_topic",
            "unknown",
        ] = "unknown"

        topic: str | None = None
        domain: str | None = None
        methods: list[str] = []
        datasets: list[str] = []
        entities: list[str] = []
        venues: list[str] = []
        time_range: dict[str, int] | None = None

        must_have_constraints: list[str] = []
        nice_to_have_constraints: list[str] = []
        negative_constraints: list[str] = []
        ambiguity: list[str] = []

        normalized_query_en: str | None = None
        normalized_query_zh: str | None = None

### 6.2 SearchPlan

    class SearchPlan(BaseModel):
        round_id: int
        search_goal: str
        subqueries: list["SearchQuery"]
        source_policy: dict[str, Any]
        expected_evidence: list[str] = []
        negative_filters: list[str] = []
        citation_expansion_seeds: list[str] = []
        should_stop_after_round: bool = False
        reason: str | None = None

### 6.3 SearchQuery

    class SearchQuery(BaseModel):
        query: str
        route: Literal[
            "raw",
            "translated",
            "task",
            "method",
            "dataset",
            "entity",
            "venue",
            "latest",
            "title_fuzzy",
            "citation_seed",
            "evolved",
        ]
        purpose: str
        sources: list[str]
        retrievers: list[str]
        required_terms: list[str] = []
        optional_terms: list[str] = []
        filters: dict[str, Any] = {}
        priority: int = 1

### 6.4 PaperCandidate

    class PaperCandidate(BaseModel):
        paper_id: str
        title: str
        abstract: str | None = None
        year: int | None = None
        venue: str | None = None
        authors: list[str] = []
        doi: str | None = None
        arxiv_id: str | None = None
        url: str | None = None

        citation_count: int | None = None
        references: list[str] = []
        citations: list[str] = []

        sources: list[str] = []
        retrieval_paths: list[str] = []
        route_hits: list[str] = []

        rough_score: float | None = None
        bge_score: float | None = None
        specter_score: float | None = None
        title_score: float | None = None
        constraint_score: float | None = None
        final_score: float | None = None

        metadata: dict[str, Any] = {}

### 6.5 EvidenceSelection

    class EvidenceSelection(BaseModel):
        paper_id: str
        relevance_level: Literal["high", "medium", "low", "irrelevant"]
        confidence: float

        matched_constraints: list[str] = []
        missing_constraints: list[str] = []
        evidence: list[dict[str, str]] = []
        uncertainty: list[str] = []

        llm_reason: str | None = None
        validator_action: Literal[
            "keep",
            "upgrade",
            "downgrade",
            "remove_evidence",
            "reject",
        ] = "keep"

        selector_score: float | None = None

### 6.6 AgentState

    class AgentState(BaseModel):
        query_contract: QueryContract
        search_plans: list[SearchPlan] = []
        retrieval_rounds: list[dict[str, Any]] = []
        candidate_pool: list[PaperCandidate] = []
        review_reports: list[dict[str, Any]] = []
        selections: list[EvidenceSelection] = []
        final_papers: list[PaperCandidate] = []

        current_round: int = 0
        stop_reason: str | None = None
        errors: list[str] = []
        run_metrics: dict[str, Any] = {}

7. Agent 主流程实现
-------------------

### 7.1 run\_agent 主函数

    def run_agent(query: str, config: AppConfig) -> WorkflowResult:
        budget = BudgetManager(config.budget)
        state = AgentState(
            query_contract=QueryContract(original_query=query, language="unknown")
        )

        # 1. Query Understanding
        state.query_contract = understand_query(query, config, budget)

        # 2. Initial Search Planning
        plan = create_initial_search_plan(state.query_contract, config, budget)
        state.search_plans.append(plan)

        # 3. Iterative Search Loop
        for round_id in range(config.agent.max_rounds):
            state.current_round = round_id

            current_plan = state.search_plans[-1]
            candidates = execute_search_plan(current_plan, state.query_contract, config, budget)

            state.candidate_pool = merge_candidates(state.candidate_pool, candidates)
            state.candidate_pool = deduplicate_papers(state.candidate_pool)
            state.candidate_pool = rough_rank_candidates(state.candidate_pool, state.query_contract, config)

            if not config.agent.enable_result_review:
                break

            review = review_search_results(
                query_contract=state.query_contract,
                search_plan=current_plan,
                candidates=state.candidate_pool[:50],
                config=config,
                budget=budget,
            )
            state.review_reports.append(review)

            if should_stop_search(review, state, config, budget):
                state.stop_reason = review.get("stop_reason", "coverage_sufficient")
                break

            next_plan = optimize_search_strategy(
                query_contract=state.query_contract,
                review_report=review,
                previous_plans=state.search_plans,
                config=config,
                budget=budget,
            )
            state.search_plans.append(next_plan)

        # 4. Final Candidate Preparation
        final_candidates = prepare_final_candidates(state.candidate_pool, state.query_contract, config)

        # 5. Local Rerank before LLM selection
        rerank_candidates = local_rerank_before_selector(
            final_candidates,
            state.query_contract,
            topk=config.selection.max_papers_for_selector,
            config=config,
        )

        # 6. Batch Evidence Selection
        selections = batch_select_and_extract_evidence(
            query_contract=state.query_contract,
            candidates=rerank_candidates,
            config=config,
            budget=budget,
        )

        # 7. Evidence Validation
        validated_selections = validate_selections(
            selections=selections,
            candidates=rerank_candidates,
            query_contract=state.query_contract,
            config=config,
        )
        state.selections = validated_selections

        # 8. Final Reranking
        ranked_papers = final_rerank(
            candidates=final_candidates,
            selections=validated_selections,
            query_contract=state.query_contract,
            config=config,
        )

        # 9. Dynamic-K Output
        output_k = decide_dynamic_k(
            query_contract=state.query_contract,
            ranked_papers=ranked_papers,
            config=config,
        )
        state.final_papers = ranked_papers[:output_k]

        # 10. Structured Synthesis
        report = synthesize_report(
            query_contract=state.query_contract,
            final_papers=state.final_papers,
            selections=validated_selections,
            search_trace=state.search_plans,
            review_reports=state.review_reports,
            config=config,
            budget=budget,
        )

        # 11. Metrics
        state.run_metrics = budget.finalize()

        return WorkflowResult(
            query_contract=state.query_contract,
            final_papers=state.final_papers,
            report=report,
            state=state,
            run_metrics=state.run_metrics,
        )

8. Query Understanding Agent
----------------------------

### 8.1 功能目标

Query Understanding Agent 负责将用户自然语言查询转化为结构化
QueryContract。

必须抽取：

    研究意图
    查询类型
    主题
    领域
    方法
    数据集
    实体
    时间
    venue
    强约束
    弱约束
    排除条件
    中英文规范化 query

### 8.2 Prompt 模板

    你是科研论文检索系统的 Query Understanding Agent。

    你的任务是把用户自然语言查询转成严格 JSON。
    禁止编造用户没有表达的硬约束。
    不确定的内容放入 ambiguity。
    如果用户查询是中文，需要补充英文规范检索式。
    如果用户查询明显是综述型、方法比较型、数据集约束型、最新进展型、精确论文标题型，请给出 query_type。

    用户查询：
    {query}

    请输出 JSON：
    {
      "language": "...",
      "research_intent": "...",
      "query_type": "...",
      "topic": "...",
      "domain": "...",
      "methods": [],
      "datasets": [],
      "entities": [],
      "venues": [],
      "time_range": {"from": 2020, "to": 2026},
      "must_have_constraints": [],
      "nice_to_have_constraints": [],
      "negative_constraints": [],
      "ambiguity": [],
      "normalized_query_en": "...",
      "normalized_query_zh": "..."
    }

### 8.3 失败降级

若 LLM 解析失败，使用 `heuristic_understand_query()`：

    def heuristic_understand_query(query: str) -> QueryContract:
        contract = QueryContract(original_query=query, language=detect_language(query))

        contract.methods = extract_methods_by_regex(query)
        contract.datasets = extract_datasets_by_regex(query)
        contract.time_range = extract_time_range(query)
        contract.venues = extract_venues(query)
        contract.entities = extract_entities(query)
        contract.query_type = infer_query_type(query)

        contract.normalized_query_en = translate_or_normalize(query)
        return contract

9. Search Planning Agent
------------------------

### 9.1 功能目标

根据 QueryContract 生成第一轮 SearchPlan。

第一轮 SearchPlan 应包括：

    raw route
    translated route
    task route
    method route
    dataset route
    entity route
    latest route
    title_fuzzy route

### 9.2 实现规则

    def create_initial_search_plan(contract, config, budget):
        subqueries = []

        subqueries.append(raw_route(contract))
        subqueries.append(translated_route(contract))

        if contract.topic:
            subqueries.append(topic_route(contract))

        if contract.methods:
            subqueries.extend(method_routes(contract))

        if contract.datasets:
            subqueries.extend(dataset_routes(contract))

        if contract.entities:
            subqueries.extend(entity_routes(contract))

        if contract.query_type in ["latest_work", "survey"]:
            subqueries.append(latest_route(contract))

        if looks_like_title_query(contract.original_query):
            subqueries.append(generic_title_fuzzy_route(contract))

        subqueries = rank_and_trim_subqueries(
            subqueries,
            max_n=config.agent.max_subqueries_per_round,
        )

        return SearchPlan(
            round_id=1,
            search_goal="build initial high-recall candidate pool",
            subqueries=subqueries,
            source_policy=route_sources(contract, config),
        )

10. Source Router
-----------------

### 10.1 路由原则

不要所有 query 都默认调用所有 API。

    def route_sources(contract: QueryContract, config: AppConfig) -> dict:
        sources = {
            "openalex": True,
            "semantic_scholar": True,
            "pubmed": False,
            "arxiv": False,
            "pasa_local": config.retrieval.enable_pasa_local,
        }

        if contract.domain and any(x in contract.domain.lower() for x in ["medicine", "biomedical", "clinical"]):
            sources["pubmed"] = True

        if contract.query_type in ["latest_work", "survey"] and is_cs_or_ai_query(contract):
            sources["arxiv"] = True

        if "arxiv" in contract.entities or "preprint" in contract.original_query.lower():
            sources["arxiv"] = True

        return sources

### 10.2 arXiv 限制

    ARXIV_TIMEOUT_SECONDS = 5
    ARXIV_MAX_CALLS_PER_QUERY = 1
    ARXIV_RETRY = 0

如果 arXiv timeout，不阻塞主流程：

    try:
        results = arxiv_provider.search(query, timeout=5)
    except TimeoutError:
        log_warning("arxiv timeout, skip")
        results = []
    except RateLimitError:
        log_warning("arxiv 429, skip")
        results = []

11. Known Title Clues 改造
--------------------------

### 11.1 设计原则

Known Title Clues 不允许在 official 模式中作为金标重定向工具。

    debug 模式：允许用于错误定位
    official 模式：禁用

### 11.2 实现方式

    def apply_known_title_clues(query: str, config: AppConfig) -> list[SearchQuery]:
        if not config.known_title_clues.enabled:
            return []

        if config.mode == "official":
            return []

        clues = match_known_title_clues(query)

        search_queries = []
        for clue in clues:
            search_queries.append(
                SearchQuery(
                    query=clue.title,
                    route="title_fuzzy",
                    purpose="debug clue based title search",
                    sources=["openalex", "semantic_scholar"],
                    retrievers=["bm25"],
                    priority=1,
                )
            )

        return search_queries

### 11.3 禁止行为

以下行为在 official 模式禁止：

    命中 clue 后直接重定向到 gold title
    命中 clue 后强制 relevance=high
    命中 clue 后绕过 reranker
    命中 clue 后绕过 evidence validator

12. Candidate Processor
-----------------------

### 12.1 去重逻辑

    def deduplicate_papers(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
        buckets = {}

        for paper in candidates:
            keys = [
                normalize_doi(paper.doi),
                normalize_arxiv_id(paper.arxiv_id),
                normalize_paper_id(paper.paper_id),
                normalize_title(paper.title),
            ]

            key = first_valid_key(keys)

            if key not in buckets:
                buckets[key] = paper
            else:
                buckets[key] = merge_paper_metadata(buckets[key], paper)

        return list(buckets.values())

### 12.2 合并字段

合并时保留：

    sources
    retrieval_paths
    route_hits
    abstract
    year
    venue
    citation_count
    references
    citations
    url
    doi

优先级：

    DOI / arXiv ID / S2 ID / OpenAlex ID > title hash
    有摘要 > 无摘要
    有 DOI > 无 DOI
    有引用数 > 无引用数
    多源命中 > 单源命中

13. Rough Ranker
----------------

### 13.1 V2.1 粗排分

V2.1 不再使用过强的 title\_exact +1.0 偏置。

    rough_score =
        0.30 * lexical_score
      + 0.30 * dense_score
      + 0.15 * title_match_score
      + 0.15 * constraint_coverage_score
      + 0.05 * source_agreement_score
      + 0.03 * citation_score
      + 0.02 * recency_score

### 13.2 缺失分数处理

禁止将缺失 BGE 分数默认设置为 0.5。

改为：

    if bge_score is None:
        dense_score = None
        reweight_available_features()

或简单处理：

    dense_score = 0.2

推荐使用特征重归一化：

    def weighted_sum_with_missing(features, weights):
        valid = {k: v for k, v in features.items() if v is not None}
        total_weight = sum(weights[k] for k in valid)
        if total_weight == 0:
            return 0.0

        return sum(valid[k] * weights[k] / total_weight for k in valid)

14. Result Review Agent
-----------------------

### 14.1 功能目标

Result Review Agent 是 V2.1
的核心新增模块，用于满足赛题要求中的“基于大模型的自主搜索策略迭代优化”。

它在每一轮检索后审阅候选池，判断：

    当前搜索覆盖了哪些方面
    缺少哪些方向
    哪些结果是噪声
    是否需要继续搜索
    下一轮应该搜索哪些关键词
    是否需要引用扩展
    是否需要收缩范围

### 14.2 输入

    QueryContract
    当前 SearchPlan
    Top 30-50 候选论文
    候选池统计信息
    预算信息

### 14.3 输出

    {
      "covered_aspects": [],
      "missing_aspects": [],
      "noise_patterns": [],
      "candidate_quality": {
        "enough_candidates": true,
        "hard_constraint_coverage": "high",
        "precision_risk": "medium",
        "recall_risk": "low"
      },
      "next_action": "stop | continue_search | citation_expand | narrow_search",
      "suggested_new_keywords": [],
      "suggested_negative_keywords": [],
      "suggested_sources": [],
      "need_citation_expansion": false,
      "stop_reason": "",
      "reason": ""
    }

### 14.4 Prompt 模板

    你是 Scholar Agent 的 Result Review Agent。

    你的任务不是最终推荐论文，而是审阅当前搜索结果是否足够覆盖用户查询，并决定下一轮检索策略。

    用户查询契约：
    {query_contract}

    当前搜索计划：
    {search_plan}

    当前候选论文 Top {n}：
    {candidate_summaries}

    候选池统计：
    {pool_stats}

    预算状态：
    {budget_status}

    请判断：
    1. 当前结果覆盖了用户哪些需求？
    2. 还缺少哪些研究方向、方法、数据集、年份或代表论文？
    3. 当前候选中有什么噪声模式？
    4. 是否需要继续搜索？
    5. 下一轮应搜索哪些关键词？
    6. 是否需要引用扩展？
    7. 是否需要加入排除词？

    输出严格 JSON，不要输出 Markdown。

15. Strategy Optimization Agent
-------------------------------

### 15.1 策略生成

    def optimize_search_strategy(contract, review_report, previous_plans, config, budget):
        if review_report["next_action"] == "stop":
            return None

        subqueries = []

        for kw in review_report.get("suggested_new_keywords", []):
            subqueries.append(
                SearchQuery(
                    query=kw,
                    route="evolved",
                    purpose="cover missing aspect from result review",
                    sources=review_report.get("suggested_sources", ["openalex", "semantic_scholar"]),
                    retrievers=["bm25", "bge_m3"],
                    priority=1,
                )
            )

        if review_report.get("need_citation_expansion"):
            seeds = select_citation_seeds_from_current_candidates()
            subqueries.extend(create_citation_seed_queries(seeds))

        subqueries = remove_duplicate_or_seen_queries(subqueries, previous_plans)

        return SearchPlan(
            round_id=len(previous_plans) + 1,
            search_goal="optimize search based on previous result review",
            subqueries=subqueries[:config.agent.max_subqueries_per_round],
            source_policy=route_sources(contract, config),
            negative_filters=review_report.get("suggested_negative_keywords", []),
            reason=review_report.get("reason"),
        )

### 15.2 停止条件

    def should_stop_search(review, state, config, budget):
        if budget.exceed_soft_limit():
            return True

        if state.current_round + 1 >= config.agent.max_rounds:
            return True

        if review["next_action"] == "stop":
            return True

        if review["candidate_quality"]["hard_constraint_coverage"] == "high" \
           and review["candidate_quality"]["recall_risk"] == "low":
            return True

        return False

16. Batch Evidence Selector
---------------------------

### 16.1 设计目标

将 V2.0 的逐篇 LLM 调用改为 batch JSON 调用，降低调用次数和 wall time。

V2.0：

    top20 papers → 20 次 LLM 调用

V2.1：

    top10 papers → batch_size=5 → 2 次 LLM 调用

### 16.2 输入候选控制

    def select_candidates_for_evidence_selector(candidates, config):
        # 先用本地排序器压缩到 top10 或 top12
        return candidates[:config.selection.max_papers_for_selector]

### 16.3 Batch Prompt

    你是严格的学术论文相关性判断器。

    只能基于用户查询、QueryContract、论文 title、abstract、year、venue 和 metadata 判断。
    禁止推测论文全文中可能存在但当前不可见的内容。
    证据必须来自 title 或 abstract 的原文片段。
    如果证据不足，请写入 uncertainty，不要强行判 high。

    QueryContract:
    {query_contract}

    Candidate Papers:
    {papers_json}

    请对每篇论文输出：
    paper_id
    relevance_level: high / medium / low / irrelevant
    confidence: 0-1
    matched_constraints
    missing_constraints
    evidence: [{"field": "title|abstract|metadata", "text": "..."}]
    uncertainty
    reason

    输出严格 JSON 数组。

### 16.4 Batch 调用实现

    def batch_select_and_extract_evidence(query_contract, candidates, config, budget):
        batches = chunk(candidates, config.selection.batch_size)
        all_results = []

        for batch in batches:
            prompt = build_batch_selector_prompt(query_contract, batch)
            response = llm_json_call(
                prompt=prompt,
                model=config.llm.selector_model,
                temperature=0.0,
                max_tokens=config.selection.max_tokens_per_batch,
                budget=budget,
            )

            parsed = parse_selector_json(response)
            repaired = repair_selector_outputs(parsed, batch)
            all_results.extend(repaired)

        return all_results

### 16.5 并发可选

batch 后再做并发：

    async def batch_select_async(query_contract, candidates, config, budget):
        batches = chunk(candidates, config.selection.batch_size)
        tasks = [
            async_llm_json_call(build_batch_selector_prompt(query_contract, batch))
            for batch in batches
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return merge_batch_responses(responses)

17. Evidence Validator
----------------------

### 17.1 核心原则

LLM 给出的 high 判断必须有可验证证据支撑。

### 17.2 校验规则

    def validate_selection(selection, paper, contract, config):
        # 1. 年份校验
        if contract.time_range and paper.year:
            if not in_time_range(paper.year, contract.time_range):
                downgrade(selection, reason="year_out_of_range")

        # 2. 证据片段校验
        valid_evidence = []
        for ev in selection.evidence:
            if evidence_text_exists(ev["text"], paper.title, paper.abstract):
                valid_evidence.append(ev)

        selection.evidence = valid_evidence

        if selection.relevance_level == "high" and not valid_evidence:
            downgrade(selection, reason="unsupported_high_relevance")

        # 3. must constraints 校验
        supported_constraints = count_supported_constraints(
            contract.must_have_constraints,
            paper,
            selection.evidence,
        )

        if contract.must_have_constraints:
            coverage = supported_constraints / len(contract.must_have_constraints)
            if selection.relevance_level == "high" and coverage < 0.5:
                downgrade(selection, reason="low_must_constraint_coverage")

        # 4. title path 校准不再强制改判，只加 metadata
        selection = apply_path_bonus_metadata(selection, paper, config)

        return selection

### 17.3 禁止强改判

V2.1 official 模式禁止：

    title_exact → 强制 high
    title_like → 强制 medium
    Known Clue → 强制 high

允许：

    title_exact_bonus
    title_like_bonus
    source_agreement_bonus

18. Final Reranker
------------------

### 18.1 V2.1 最终得分公式

    FinalScore =
    0.25 * LLM_RelevanceScore
    + 0.20 * BGE_RerankerScore
    + 0.20 * ConstraintCoverage
    + 0.15 * EvidenceSupport
    + 0.08 * SourceAgreement
    + 0.05 * TitleMatch
    + 0.03 * Recency
    + 0.02 * Authority
    + 0.02 * GraphPrior

### 18.2 EvidenceSupport 计算

V2.0 的 Evidence\_Completeness 过粗，V2.1 改为：

    def evidence_support_score(contract, selection, paper):
        if not contract.must_have_constraints:
            return 1.0 if selection.evidence else 0.5

        supported = 0
        for constraint in contract.must_have_constraints:
            if constraint_supported_by_evidence(constraint, selection.evidence, paper):
                supported += 1

        return supported / len(contract.must_have_constraints)

### 18.3 LLM\_RelevanceScore

    RELEVANCE_SCORE = {
        "high": 1.0,
        "medium": 0.55,
        "low": 0.15,
        "irrelevant": 0.0,
    }

建议将 medium 从 0.6 降到 0.55，减少 medium 噪声挤入最终列表。

### 18.4 TitleMatch Score

    def title_match_score(paper, contract, config):
        score = compute_title_similarity(paper.title, contract.original_query)

        if score > 0.92:
            return 1.0
        elif score > 0.80:
            return 0.6
        elif score > 0.65:
            return 0.3
        else:
            return 0.0

19. Dynamic-K Controller
------------------------

### 19.1 为什么替代 Top3

固定 Top3 对不同 query 类型不公平：

    精确论文查询：Top1 更合适
    单金标查询：Top1 或 Top2 更合适
    方法比较查询：Top3 更合适
    综述型查询：Top5-10 更合适

### 19.2 实现

    def decide_dynamic_k(query_contract, ranked_papers, config):
        query_type = query_contract.query_type

        if not ranked_papers:
            return 0

        # 分数断崖：第一名明显高于第二名
        if len(ranked_papers) >= 2:
            if ranked_papers[0].final_score - ranked_papers[1].final_score >= config.dynamic_k.score_gap_top1:
                return 1

        # query 类型规则
        if query_type in ["exact_title", "specific_paper", "single_gold"]:
            return min(config.dynamic_k.exact_title_k, len(ranked_papers))

        if query_type in ["dataset_constraint", "method_comparison"]:
            return min(config.dynamic_k.method_comparison_k, len(ranked_papers))

        if query_type in ["survey", "broad_topic", "latest_work"]:
            return min(config.dynamic_k.survey_k, len(ranked_papers))

        # 分数阈值规则
        valid = [
            p for p in ranked_papers
            if p.final_score is not None and p.final_score >= config.dynamic_k.min_final_score
        ]

        if len(valid) > 0:
            return min(len(valid), config.ranking.max_final_papers)

        return min(config.dynamic_k.fallback_k, len(ranked_papers))

### 19.3 输出分层

    def split_high_and_partial(final_papers, selections):
        high = []
        partial = []

        for paper in final_papers:
            sel = find_selection(paper.paper_id, selections)
            if sel and sel.relevance_level == "high":
                high.append(paper)
            else:
                partial.append(paper)

        return high, partial

20. Synthesis Agent
-------------------

### 20.1 输出内容

最终输出必须包含：

    query_understanding
    agent_search_process
    highly_relevant_papers
    partially_relevant_papers
    method_clusters
    timeline
    citation_graph
    recommendation_reasons
    run_metrics

### 20.2 Synthesis 不参与重新选择论文

Synthesis Agent 只负责组织表达，不允许改变 final\_papers。

禁止：

    LLM 在 synthesis 阶段新增论文
    LLM 在 synthesis 阶段删除论文
    LLM 在 synthesis 阶段改 final_score

允许：

    聚类
    时间线
    推荐理由润色
    结构化摘要

### 20.3 输出 JSON

    {
      "query_understanding": {},
      "agent_search_process": [
        {
          "round": 1,
          "goal": "",
          "queries": [],
          "found_candidates": 0,
          "review_summary": ""
        }
      ],
      "highly_relevant_papers": [],
      "partially_relevant_papers": [],
      "method_clusters": [],
      "timeline": [],
      "citation_graph": {
        "nodes": [],
        "edges": []
      },
      "recommendation_reasons": [],
      "run_metrics": {}
    }

21. Evaluation 指标升级
-----------------------

### 21.1 必须新增指标

    F1@1
    F1@2
    F1@3
    F1@5
    F1@10

    Precision@1
    Precision@2
    Precision@3
    Precision@5
    Precision@10

    Recall@1
    Recall@2
    Recall@3
    Recall@5
    Recall@10

    Hit@1
    Hit@3
    Hit@5

    strict_f1
    relaxed_f1

    selector_false_positive
    selector_false_negative

    candidate_recall@100
    candidate_recall@300
    candidate_recall@500

    candidate_to_final_loss
    ranking_drop
    source_missing

    avg_llm_calls
    avg_tokens
    avg_wall_time
    over_budget
    errors

### 21.2 TopK Sweep

    def evaluate_topk_sweep(pred_rankings, golds, ks=[1,2,3,5,10]):
        results = {}
        for k in ks:
            metrics = evaluate_predictions(
                predictions={qid: ranking[:k] for qid, ranking in pred_rankings.items()},
                golds=golds,
            )
            results[f"f1@{k}"] = metrics["avg_f1"]
            results[f"precision@{k}"] = metrics["avg_precision"]
            results[f"recall@{k}"] = metrics["avg_recall"]
        return results

### 21.3 Candidate-to-Final Loss

    def compute_candidate_to_final_loss(gold_set, candidate_set, final_set):
        in_candidate = gold_set & candidate_set
        in_final = gold_set & final_set

        return {
            "candidate_gold_count": len(in_candidate),
            "final_gold_count": len(in_final),
            "lost_after_candidate": len(in_candidate - in_final),
            "lost_papers": list(in_candidate - in_final),
        }

22. Error Analysis 导出
-----------------------

### 22.1 drop cases 表

每次评测后导出：

    query_id,
    query_text,
    gold_paper_id,
    gold_title,
    in_candidate,
    candidate_rank,
    rough_score,
    bge_score,
    llm_relevance,
    evidence_valid,
    final_score,
    final_rank,
    in_final,
    drop_reason

### 22.2 drop\_reason 分类

    def classify_drop_reason(row):
        if not row["in_candidate"]:
            return "source_missing"

        if row["candidate_rank"] > 300:
            return "candidate_rank_too_low"

        if row["llm_relevance"] in ["low", "irrelevant"]:
            return "llm_selector_misjudge"

        if not row["evidence_valid"]:
            return "evidence_validator_downgrade"

        if row["final_rank"] > row["output_k"]:
            return "dynamic_k_cutoff"

        return "unknown"

23. Ablation 实验
-----------------

### 23.1 V2.1 必跑消融

  实验编号   设置                              目的
  ---------- --------------------------------- ---------------------
  A0         V2.0 原始版本                     对照
  A1         V2.0 + TopK sweep                 判断 Top3 是否合理
  A2         Dynamic-K                         验证自适应输出
  A3         关闭 Known Title Clues            检查泛化风险
  A4         title\_exact 只加分不强改判       检查 Precision
  A5         title\_like 只加分不保底 medium   检查 false positive
  A6         Evidence Selector top20 → top10   降本
  A7         Evidence Selector batch JSON      降低调用次数
  A8         arXiv auto router                 降低 wall time
  A9         增加 Result Review Agent          验证迭代搜索收益
  A10        V2.1 full                         最终版本

### 23.2 评价标准

每个实验必须记录：

    avg_f1_final
    avg_precision_final
    avg_recall_final
    candidate_recall@300
    selector_false_positive
    selector_false_negative
    ranking_drop
    source_missing
    avg_wall_time
    avg_llm_calls
    avg_tokens
    errors

24. Learning-to-Rank 预留方案
-----------------------------

V2.1 可以先保留人工权重，但必须导出训练特征，为 V2.2 做学习式 reranker
准备。

### 24.1 特征导出

    features = {
        "llm_relevance_score": ...,
        "bge_score": ...,
        "constraint_coverage": ...,
        "evidence_support": ...,
        "source_agreement": ...,
        "title_match_score": ...,
        "recency": ...,
        "authority": ...,
        "graph_prior": ...,
        "has_title_exact": ...,
        "has_title_like": ...,
        "candidate_rank": ...,
        "route_hit_count": ...,
        "source_count": ...,
    }

### 24.2 标签

    label = 1 if paper_id in gold_set else 0

如果有 graded label：

    label = 2 for highly_relevant
    label = 1 for partially_relevant
    label = 0 for irrelevant

### 24.3 模型

第一版推荐：

    Logistic Regression
    LightGBM
    LambdaMART

先不使用复杂深度模型。

25. RunMetrics
--------------

### 25.1 必须记录

    {
      "query_id": "",
      "agent_rounds": 2,
      "search_queries_generated": 8,
      "candidate_pool_size": 320,
      "final_output_k": 3,
      "api_calls": {
        "openalex": 4,
        "semantic_scholar": 3,
        "pubmed": 0,
        "arxiv": 1
      },
      "llm_calls": {
        "query_understanding": 1,
        "search_planning": 1,
        "result_review": 1,
        "strategy_optimization": 1,
        "evidence_selection": 2,
        "synthesis": 1
      },
      "tokens": {
        "query_understanding": 1000,
        "result_review": 2000,
        "evidence_selection": 5000,
        "synthesis": 1200
      },
      "latency": {
        "query_understanding": 2.1,
        "retrieval": 8.3,
        "result_review": 3.2,
        "evidence_selection": 6.5,
        "reranking": 1.0,
        "synthesis": 2.0,
        "total": 23.1
      },
      "cache_hits": 5,
      "stop_reason": "coverage_sufficient",
      "errors": []
    }

26. 测试计划
------------

### 26.1 单元测试

    test_query_understanding_json_schema
    test_source_router_arxiv_auto
    test_known_title_clues_disabled_in_official
    test_title_exact_not_force_high
    test_title_like_not_force_medium
    test_batch_selector_json_parse
    test_evidence_validator_downgrade
    test_dynamic_k_exact_title
    test_dynamic_k_survey
    test_topk_sweep_metrics

### 26.2 集成测试

    test_full_pipeline_live_lite
    test_full_pipeline_official_no_known_clues
    test_agent_review_generates_next_strategy
    test_arxiv_timeout_does_not_crash
    test_batch_selector_budget_control

### 26.3 回归测试

每次大改后固定跑：

    python scripts/run_eval.py --config configs/official.yaml --split dev
    python scripts/run_topk_sweep.py --config configs/official.yaml --split dev
    python scripts/run_ablation.py --config configs/ablation_v21.yaml
    python scripts/export_errors.py --run_id latest

27. 一周实现计划
----------------

### Day 1：评测脚本升级

完成：

    TopK sweep
    Hit@K
    strict_f1
    relaxed_f1
    selector_false_positive
    candidate_to_final_loss
    drop_reason export

交付：

    evaluation/metrics.py
    evaluation/topk_sweep.py
    evaluation/error_analysis.py

### Day 2：Dynamic-K 替换 Top3 Cap

完成：

    ranking/dynamic_k.py
    synthesis_agent 不再硬性 Top3
    query_type 到 K 的映射
    score gap 判断

交付：

    F1@1/2/3/5/10 对比表
    Dynamic-K 初版结果

### Day 3：Known Title Clues 与 Path Calibration 降风险

完成：

    official 模式禁用 Known Title Clues
    title_exact 不再强制 high
    title_like 不再保底 medium
    改为 title bonus

交付：

    with/without known clues 消融表
    path calibration 消融表

### Day 4：Batch Evidence Selector

完成：

    batch_evidence_selector.py
    batch prompt
    JSON list parser
    repair parser
    batch-level retry

交付：

    LLM calls/query 降低
    tokens/query 降低
    wall_time 降低

### Day 5：Source Router 优化

完成：

    arXiv auto routing
    arXiv timeout=5s
    arXiv max_calls_per_query=1
    timeout skip

交付：

    live-lite wall_time 对比
    over_budget 对比

### Day 6：Result Review Agent

完成：

    planning/result_review.py
    planning/strategy_optimization.py
    agent_loop.py 接入一轮 review

交付：

    Agent search trace
    第 1 轮审阅报告
    第 2 轮搜索策略

### Day 7：V2.1 全量评测与报告

完成：

    V2.1 full run
    A0-A10 ablation
    error cases export
    implementation.md 更新

交付：

    evaluation_report_v21.md
    ablation_report_v21.md
    error_analysis_v21.csv

28. V2.1 验收标准
-----------------

### 28.1 必须达成

    official 模式无 Known Title Clues 强依赖
    Top3 Cap 替换为 Dynamic-K
    title_exact 不再无条件强制 high
    title_like 不再无条件保底 medium
    Evidence Selector 支持 batch JSON
    arXiv 支持 auto routing
    输出 Agent 搜索过程报告
    评测支持 TopK sweep
    支持 selector_false_positive 统计
    支持 candidate_to_final_loss 导出

### 28.2 性能验收

    avg_f1_final >= V2.0
    avg_precision_final 明显高于 V2.0
    candidate_recall@300 不明显下降
    avg_wall_time 明显低于 V2.0
    LLM calls/query 明显低于 V2.0
    errors 不增加

### 28.3 答辩验收

系统必须能展示：

    用户 query 被如何理解
    第一轮搜索计划是什么
    第一轮候选池发现了什么缺口
    LLM 如何生成第二轮搜索策略
    最终为什么停止搜索
    为什么推荐这些论文
    为什么区分 high 和 partial
    消耗了多少 API 和 token

29. 最终实现原则
----------------

V2.1 的核心原则：

    先选准，再搜多。
    先本地排序，再 LLM 判断。
    先批量判断，再并发优化。
    先动态输出，再固定截断。
    先通用泛化，再开发集补丁。
    先可解释日志，再答辩包装。

一句话总结：

    Scholar Agent V2.1 的目标，是把 V2.0 的高召回候选池转化为高 Precision、高 F1、低成本、可解释、符合赛题要求的 LLM-driven Scholarly Search Agent。
