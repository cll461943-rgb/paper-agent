#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scholar Agent V2.0 - 验证集全量在线评估脚本
用户可在 IDE 中直接运行。
评估指标与基准指标直接对齐，统计 F1、Recall、Precision、以及各组件开销和预算限制。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# 确保把 src 目录加到 Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.infra import (
    MockLLMClient,
    OpenAICompatibleLLMClient,
    load_config,
    setup_logging,
)
from scholar_agent.retrieval import build_providers
from scholar_agent.workflow.pipeline import PaperAgentPipeline


# ==========================================
# 1. 移植自 metrics.py 的核心匹配与评估算法 (确保无包导入冲突)
# ==========================================

def _normalize_identifier(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:", "", text)
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text.strip() or None


def _normalize_title(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", text) or None


def _normalize_corpus_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^corpusid:", "", text, flags=re.IGNORECASE)
    return f"corpus:{text.lower()}"


def paper_match_keys(paper: Any) -> set[str]:
    keys = {
        _normalize_identifier(paper.paper_id),
        _normalize_identifier(paper.doi),
        _normalize_identifier(paper.arxiv_id),
        _normalize_title(paper.title),
        _normalize_corpus_id(paper.metadata.get("corpus_id")),
        _normalize_corpus_id(paper.metadata.get("corpusId")),
    }
    return {key for key in keys if key}


def gold_match_keys(item: dict[str, Any] | str) -> set[str]:
    if isinstance(item, str):
        keys = {_normalize_identifier(item), _normalize_title(item)}
    else:
        keys = {
            _normalize_identifier(item.get("paper_id")),
            _normalize_identifier(item.get("doi")),
            _normalize_identifier(item.get("arxiv_id")),
            _normalize_title(item.get("title")),
            _normalize_corpus_id(item.get("corpus_id")),
            _normalize_corpus_id(item.get("corpusid")),
            _normalize_corpus_id(item.get("semantic_scholar_corpus_id")),
        }
    return {key for key in keys if key}


def _merge_gold_key_sets(gold_key_sets: list[set[str]], paper_key_sets: list[set[str]]) -> list[set[str]]:
    parents = list(range(len(gold_key_sets)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_keys in enumerate(gold_key_sets):
        for right_index in range(left_index + 1, len(gold_key_sets)):
            if left_keys & gold_key_sets[right_index]:
                union(left_index, right_index)

    for paper_keys in paper_key_sets:
        matched_indexes = [index for index, gold_keys in enumerate(gold_key_sets) if paper_keys & gold_keys]
        for index in matched_indexes[1:]:
            union(matched_indexes[0], index)

    merged: dict[int, set[str]] = {}
    for index, gold_keys in enumerate(gold_key_sets):
        root = find(index)
        merged.setdefault(root, set()).update(gold_keys)
    return list(merged.values())


def score_papers_against_gold(papers: list[Any], gold_items: list[dict[str, Any] | str]) -> dict[str, float | int]:
    gold_key_sets = []
    for item in gold_items:
        keys = gold_match_keys(item)
        if keys:
            gold_key_sets.append(keys)
    paper_key_sets = [paper_match_keys(paper) for paper in papers]
    gold_key_sets = _merge_gold_key_sets(gold_key_sets, paper_key_sets)
    matched_gold_indexes: set[int] = set()

    for paper_keys in paper_key_sets:
        for index, gold_keys in enumerate(gold_key_sets):
            if index in matched_gold_indexes:
                continue
            if paper_keys & gold_keys:
                matched_gold_indexes.add(index)
                break

    tp = len(matched_gold_indexes)
    predicted_count = len(papers)
    gold_count = len(gold_key_sets)

    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "true_positive": tp,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# Task 5: TopK Sweep evaluation
TOPK_VALUES = [1, 2, 3, 5, 8, 10, 15, 20]


def compute_topk_sweep(
    ranked_papers: list[Any],
    gold_items: list[dict[str, Any] | str],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Compute F1@K for multiple K values and find oracle_best_k.

    Args:
        ranked_papers: list of RankedPaper (sorted by final_score descending)
        gold_items: gold standard papers
        k_values: list of K values to evaluate (default TOPK_VALUES)

    Returns:
        dict with:
          f1_at_k: {K: F1} for each K
          precision_at_k: {K: precision}
          recall_at_k: {K: recall}
          oracle_best_k: K with highest F1
          oracle_f1: F1 at oracle_best_k
    """
    if k_values is None:
        k_values = TOPK_VALUES

    f1_at_k: dict[int, float] = {}
    prec_at_k: dict[int, float] = {}
    rec_at_k: dict[int, float] = {}
    oracle_best_k = 0
    oracle_f1 = 0.0

    for k in k_values:
        top_k_papers = [rp.paper for rp in ranked_papers[:k]]
        scores = score_papers_against_gold(top_k_papers, gold_items)
        f1_at_k[k] = scores["f1"]
        prec_at_k[k] = scores["precision"]
        rec_at_k[k] = scores["recall"]
        if scores["f1"] > oracle_f1:
            oracle_f1 = scores["f1"]
            oracle_best_k = k

    return {
        "f1_at_k": f1_at_k,
        "precision_at_k": prec_at_k,
        "recall_at_k": rec_at_k,
        "oracle_best_k": oracle_best_k,
        "oracle_f1": round(oracle_f1, 4),
    }


# Task 7: K predictor case classification
def classify_k_prediction(
    oracle_k: int,
    dynamic_k: int,
    gap: float,
    g_hat: float | None = None,
    gold_count: int | None = None,
    lc_uniform: bool = False,
    mean_prob: float | None = None,
    std_prob: float | None = None,
) -> str:
    """Classify a K prediction case into a diagnostic category.

    Categories:
      - good_match: gap < 0.02 and dynamic_k close to oracle_k
      - under_output: dynamic_k < oracle_k - 2 (K too small)
      - over_output: dynamic_k > oracle_k + 2 (K too large)
      - ranking_failure: oracle_k=0 or no gold in topN
      - probability_calibration_failure: low mean/low std but K too large
      - g_hat_underestimate_failure: g_hat << gold and K too small
    """
    # Ranking failure: no gold found
    if oracle_k == 0 or (gold_count is not None and gold_count == 0):
        return "ranking_failure"

    # Good match: gap is small
    if abs(gap) < 0.02 and abs(dynamic_k - oracle_k) <= 2:
        return "good_match"

    # Probability calibration failure: low confidence uniform but K large
    if (
        lc_uniform
        and mean_prob is not None
        and std_prob is not None
        and mean_prob < 0.35
        and std_prob < 0.05
        and dynamic_k > 5
    ):
        return "probability_calibration_failure"

    # g_hat underestimate: g_hat much smaller than gold and K too small
    if (
        g_hat is not None
        and gold_count is not None
        and g_hat < gold_count * 0.5
        and dynamic_k < oracle_k - 2
    ):
        return "g_hat_underestimate_failure"

    # Under/over output
    if dynamic_k < oracle_k - 2:
        return "under_output"
    if dynamic_k > oracle_k + 2:
        return "over_output"

    return "good_match"


# ==========================================
# 2. 统计与大模型代理包装类
# ==========================================

class StatsLLMClient:
    """装饰器类，用来精确统计 LLM 在大模型提供商 / 不同模型级别上的调用指标"""
    def __init__(self, base_client: Any, stats_dict: dict[str, dict[str, Any]]) -> None:
        object.__setattr__(self, "base_client", base_client)
        object.__setattr__(self, "stats_dict", stats_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "budget":
            setattr(self.base_client, "budget", value)
        else:
            setattr(self.base_client, name, value)

    def complete_json(self, system_prompt: str, user_prompt: str, model_type: str = "flash", timeout_seconds: float | None = None, max_tokens: int | None = None) -> Any | None:
        if isinstance(self.base_client, MockLLMClient):
            model_name = "mock-pro" if model_type == "pro" else "mock-flash"
        else:
            import os
            model_name = (
                os.getenv("DEEPSEEK_MODEL_PRO", self.base_client.config.model_pro)
                if model_type == "pro"
                else os.getenv("DEEPSEEK_MODEL_FLASH", self.base_client.config.model_flash)
            )

        started_at = time.perf_counter()
        
        curr_budget = getattr(self.base_client, "budget", None)
        prev_tokens = curr_budget.token_estimate if curr_budget else 0

        # 调用底层客户端
        result = self.base_client.complete_json(system_prompt, user_prompt, model_type, timeout_seconds=timeout_seconds, max_tokens=max_tokens)

        elapsed = time.perf_counter() - started_at
        tokens_used = (curr_budget.token_estimate - prev_tokens) if curr_budget else 0

        # 录入模型级别统计
        m_stats = self.stats_dict.setdefault(model_name, {"calls": 0, "tokens": 0, "elapsed": 0.0})
        m_stats["calls"] += 1
        m_stats["tokens"] += tokens_used
        m_stats["elapsed"] += elapsed

        # 录入组件开销
        if curr_budget:
            lowered = f"{system_prompt}\n{user_prompt}".lower()
            if "highly_relevant_papers" in lowered or "structuredsynthesis" in lowered:
                comp_name = "synthesis"
            elif "relevance_level" in lowered or "evidence" in lowered:
                comp_name = "evidence_selector"
            elif "coverage_analysis" in lowered or "resultreview" in lowered or "审阅" in user_prompt:
                comp_name = "result_review"
            elif "subqueries" in lowered or "search_goal" in lowered or "round" in lowered:
                comp_name = "search_planning"
            elif "queryplan" in lowered:
                comp_name = "query_understanding"
            else:
                comp_name = "other_llm_task"
            
            curr_budget.record_component_cost(
                component=comp_name,
                elapsed_seconds=elapsed,
                llm_calls_delta=1,
                token_estimate_delta=tokens_used
            )

        return result



# ==========================================
# 3. 评估逻辑与报表展示
# ==========================================

# 用户提供的 Baseline 指标值 (用于对比)
BASELINE_METRICS = {
    "avg_f1_final": 0.4445,
    "avg_precision_final": 0.5300,
    "avg_recall_final": 0.4452,
    "avg_candidate_recall_300": 0.7335,
    "avg_candidate_recall_500": 0.7430,
    "avg_wall_time": 24.7200,
    "queries_over_time_budget": 0,
    "queries_with_errors": 0,
}


def parse_cases_arg(cases_str: str, max_cases: int) -> list[int]:
    """解析用例索引参数，如 '1,2,5-7'"""
    selected = set()
    parts = cases_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                for i in range(start, end + 1):
                    if 1 <= i <= max_cases:
                        selected.add(i - 1)
            except ValueError:
                pass
        else:
            try:
                i = int(part)
                if 1 <= i <= max_cases:
                    selected.add(i - 1)
            except ValueError:
                pass
    return sorted(list(selected))


def select_case_indices(
    max_cases: int,
    *,
    cases_filter: str | None = None,
    simple: bool = False,
    limit: int | None = None,
    random_seed: int = 123,
) -> list[int]:
    """Select evaluation case indexes deterministically.

    ``limit`` intentionally samples rather than taking the first N cases, but
    the sample must be reproducible so benchmark deltas are attributable to
    code/config changes rather than a new random case set.
    """
    indices = list(range(max_cases))
    if cases_filter:
        return parse_cases_arg(cases_filter, max_cases)
    if simple:
        return [0]
    if limit is not None and limit > 0 and limit < len(indices):
        rng = random.Random(random_seed)
        return sorted(rng.sample(indices, limit))
    return indices


def apply_time_budget_to_config(config: Any, time_budget: float | None) -> float:
    """Keep evaluation timeout accounting and pipeline deadline aligned."""
    budget_config = getattr(config, "budget", None)
    configured_deadline = getattr(budget_config, "case_deadline_seconds", 180.0)
    effective_budget = float(time_budget) if time_budget is not None else float(configured_deadline)

    if budget_config is not None:
        budget_config.case_deadline_seconds = effective_budget

    return effective_budget


def format_compare(label: str, current: float, baseline: float, is_time: bool = False) -> str:
    diff = current - baseline
    if is_time:
        if abs(diff) < 1e-5:
            change_str = "持平"
        else:
            change_str = f"仅 +{diff:.4f}s" if diff > 0 else f"{diff:.4f}s"
        return f"• {label} : {baseline:.2f} -> {current:.4f} , {change_str}"
    else:
        if abs(diff) < 1e-5:
            change_str = "持平"
        else:
            change_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        return f"• {label} : {baseline:.4f} -> {current:.4f} , {change_str}"


def _print_gold_trace(
    candidate_pool: list[Any],
    selection_candidates: list[Any],
    ranked_papers: list[Any],
    result_papers: list[Any],
    gold: list[Any],
    case_idx: int = 0,
) -> None:
    """P2: Print candidate-to-final trace for gold papers in any case.
    For each gold paper, show: candidate_rank, retrieval_path, in_selection_candidates,
    selection_level, final_score, final_rank, drop_reason."""
    gold_key_sets = [gold_match_keys(item) for item in gold if gold_match_keys(item)]
    pool_key_sets = [paper_match_keys(p) for p in candidate_pool]
    selection_ids = {p.paper_id for p in selection_candidates}
    result_ids = {p.paper_id for p in result_papers}

    # Build ranked_papers lookup: paper_id -> (RankedPaper, rank)
    ranked_map: dict[str, tuple[Any, int]] = {}
    for i, rp in enumerate(ranked_papers):
        ranked_map[rp.paper.paper_id] = (rp, i + 1)

    print("  " + "=" * 68)
    print(f"  🔍 CASE #{case_idx} GOLD TRACE (candidate-to-final diagnosis)")
    print("  " + "=" * 68)

    for gi, gold_keys in enumerate(gold_key_sets):
        # Find in candidate pool
        found_in_pool = None
        pool_rank = None
        for pi, pkeys in enumerate(pool_key_sets):
            if pkeys & gold_keys:
                found_in_pool = candidate_pool[pi]
                pool_rank = pi + 1
                break

        if not found_in_pool:
            print(f"  Gold#{gi}: NOT IN POOL (retrieval miss)")
            continue

        title = (found_in_pool.title or "")[:60]
        in_sel = found_in_pool.paper_id in selection_ids
        in_result = found_in_pool.paper_id in result_ids

        # Find selection level and final_score
        sel_level = "not_selected"
        final_score = None
        final_rank = None
        for rp in ranked_papers:
            if rp.paper.paper_id == found_in_pool.paper_id:
                sel_level = rp.selection.relevance_level
                final_score = rp.final_score
                break
        if found_in_pool.paper_id in ranked_map:
            rp, final_rank = ranked_map[found_in_pool.paper_id]

        # Determine drop reason
        if not in_sel:
            drop_reason = "cut_by_pre_ranker_or_stratified_sampling"
        elif final_rank is None:
            drop_reason = "cut_by_reranker"
        elif not in_result:
            drop_reason = "cut_by_synthesis_threshold_or_dynamic_k"
        else:
            drop_reason = "IN OUTPUT"

        print(f"  Gold#{gi}: [{title}]")
        print(f"    candidate_rank={pool_rank} | retrieval_path={found_in_pool.retrieval_path}")
        print(f"    in_selection={in_sel} | selection_level={sel_level} | final_score={final_score}")
        print(f"    final_rank={final_rank} | in_output={in_result} | drop_reason={drop_reason}")
    print("  " + "=" * 68)


def run_evaluation(
    mode: str = "live",
    limit: int | None = None,
    cases_filter: str | None = None,
    simple: bool = False,
    output_path: str | None = None,
    time_budget: float | None = None,
    config_path: str | None = None,
    trace_gold: bool = False,
    dataset_file: str | None = None,
    random_seed: int = 123,
) -> None:
    # 加载配置
    config = load_config(config_path)
    time_budget = apply_time_budget_to_config(config, time_budget)
    config.app.mode = mode

    # P1.5: Respect YAML provider settings — don't force-enable all providers
    # Only set providers from config if not already set
    if mode == "live":
        # Build providers list from config's enabled providers
        enabled_providers = []
        for prov_name in ("pasa_local", "openalex", "arxiv", "semantic_scholar", "pubmed"):
            prov_cfg = getattr(config.providers, prov_name, None)
            if prov_cfg and getattr(prov_cfg, "enabled", False):
                enabled_providers.append(prov_name)
        if enabled_providers:
            config.app.providers = enabled_providers
        else:
            config.app.providers = ["pasa_local", "openalex", "semantic_scholar"]
    else:
        config.app.providers = ["mock"]

    print("=" * 70)
    print(f"📊 Starting Scholar Agent V2.0 Evaluation in '{mode}' mode")
    print("=" * 70)

    # 全量验证集路径寻址
    # If a dataset file is explicitly provided via -f, use it directly.
    if dataset_file:
        dataset_path = Path(dataset_file)
        if not dataset_path.exists():
            print(f"❌ Error: Specified dataset file not found: {dataset_path}", file=sys.stderr)
            sys.exit(1)
    else:
        candidate_paths = [
            Path("data/RealScholarQuery_test.jsonl"),
            Path("data/开发+测试集/CNScholarQuery_ZH_real50_polished.jsonl"),
            Path("data/开发+测试集/CNScholarQuery_ZH_test_250_polished.jsonl"),
            Path("C:/Users/33316/Desktop/claude-code-src-main/agengt-code/data/benchmarks/litsearch_dev_20.json"),
            Path("data/benchmarks/litsearch_dev_20.json"),
            Path("../agengt-code/data/benchmarks/litsearch_dev_20.json"),
            Path("data/benchmarks/litsearch_dev_5.json"),
            Path("tests/litsearch_smoke.json"),
        ]

        dataset_path = None
        for path in candidate_paths:
            if path.exists():
                dataset_path = path
                break

    if not dataset_path:
        print(f"❌ Error: Evaluation dataset not found in searched paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {dataset_path.resolve()}")
    all_cases = []
    try:
        if dataset_path.suffix == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw = json.loads(line)
                        all_cases.append({
                            "name": raw.get("qid") or f"case-{len(all_cases)+1}",
                            "query": raw.get("question") or raw.get("query") or "",
                            "gold": raw.get("answer") or raw.get("answers") or [],
                            "k": 500,
                        })
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_cases = data.get("cases", [])
    except Exception as exc:
        print(f"❌ Error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    if not all_cases:
        print("❌ Error: No test cases found in the dataset.", file=sys.stderr)
        sys.exit(1)

    indices = select_case_indices(
        len(all_cases),
        cases_filter=cases_filter,
        simple=simple,
        limit=limit,
        random_seed=random_seed,
    )
    if cases_filter and not indices:
        print(f"❌ Error: filter '{cases_filter}' did not match any test cases.", file=sys.stderr)
        sys.exit(1)

    eval_cases = [all_cases[i] for i in indices]
    seed_note = f", seed={random_seed}" if limit is not None and limit > 0 and not cases_filter and not simple else ""
    print(f"Total cases in dataset: {len(all_cases)}. Selected {len(eval_cases)} cases for evaluation (indices: {[i+1 for i in indices]}{seed_note}).")

    # 构建统一 providers
    providers = build_providers(config)

    # 统计数据汇总
    total_f1_final = 0.0
    total_precision_final = 0.0
    total_recall_final = 0.0
    total_candidate_recall_50 = 0.0
    total_candidate_recall_100 = 0.0
    total_candidate_recall_300 = 0.0
    total_candidate_recall_500 = 0.0
    total_selector_recall = 0.0
    total_final_transfer_rate = 0.0
    total_wall_time = 0.0
    queries_over_time_budget = 0
    queries_with_errors = 0

    # Task 5: TopK Sweep accumulators
    total_f1_at_k: dict[int, float] = {k: 0.0 for k in TOPK_VALUES}
    total_precision_at_k: dict[int, float] = {k: 0.0 for k in TOPK_VALUES}
    total_recall_at_k: dict[int, float] = {k: 0.0 for k in TOPK_VALUES}
    total_oracle_f1 = 0.0
    total_dynamic_k_f1 = 0.0
    total_oracle_best_k = 0
    total_dynamic_k_chosen = 0
    total_output_k = 0
    topk_case_count = 0

    # Task 7: K predictor case classification accumulators
    k_prediction_categories = {
        "good_match": 0,
        "under_output": 0,
        "over_output": 0,
        "ranking_failure": 0,
        "probability_calibration_failure": 0,
        "g_hat_underestimate_failure": 0,
    }
    k_prediction_details: list[dict[str, Any]] = []

    # 预算开销与 LLM 细分统计
    total_llm_calls = 0
    total_api_calls = 0
    total_token_estimate = 0
    total_cache_hits = 0
    total_errors_logged = 0

    llm_model_stats: dict[str, dict[str, Any]] = {}
    component_costs_accum: dict[str, dict[str, float | int]] = {}
    results_log = []

    for idx, case in enumerate(eval_cases, start=1):
        name = case.get("name", f"case-{idx}")
        query = case["query"]
        gold = case["gold"]
        k = int(case.get("k", 20))

        # 映射回数据集中的 1-indexed 原始用例编号
        original_idx = all_cases.index(case) + 1
        print(f"\n[{idx}/{len(eval_cases)}] Evaluating Case #{original_idx}: {name}")
        print(f"  Query: \"{query[:75]}...\"")

        # 对每一次运行重新生成一个独立的 budget 隔离以记录各 case 指标
        from scholar_agent.workflow.budget import BudgetManager
        budget = BudgetManager(config)

        if mode == "mock":
            raw_llm_client = MockLLMClient(budget)
        else:
            raw_llm_client = OpenAICompatibleLLMClient(config.llm, budget)

        # 包装 LLMClient 以统计不同大模型的用量与时间
        llm_client = StatsLLMClient(raw_llm_client, llm_model_stats)
        pipeline = PaperAgentPipeline(config, llm_client, providers)

        start_time = time.perf_counter()
        has_error = False
        res = None
        result_papers = []
        candidate_pool = []
        err_list = []

        try:
            res = pipeline.run(query)

            # 汇聚推荐论文
            result_papers = [
                rp.paper for rp in (res.highly_relevant_papers + res.partially_relevant_papers)
            ]
            
            # 从 pipeline.candidate_pool 中取得完整的候选池，若不存在，以 result_papers 为兜底
            candidate_pool = getattr(pipeline, "candidate_pool", result_papers)
            err_list = res.run_metrics.errors

            # 累加运行指标
            total_llm_calls += res.run_metrics.llm_calls_used
            total_api_calls += res.run_metrics.api_calls_used
            total_token_estimate += res.run_metrics.token_estimate
            total_cache_hits += res.run_metrics.cache_hits
            total_errors_logged += len(res.run_metrics.errors)
            if res.run_metrics.errors:
                print(f"  ⚠️ Logged {len(res.run_metrics.errors)} errors in Case #{original_idx}:")
                for err in res.run_metrics.errors[:5]:
                    print(f"    - {err}")

            # 汇总各组件开销
            for comp_metric in res.run_metrics.component_metrics:
                cname = comp_metric.component
                c_data = component_costs_accum.setdefault(cname, {"elapsed": 0.0, "calls": 0, "tokens": 0})
                c_data["elapsed"] += comp_metric.elapsed_seconds
                c_data["calls"] += comp_metric.api_calls_delta + comp_metric.llm_calls_delta
                c_data["tokens"] += comp_metric.token_estimate_delta

        except Exception as exc:
            has_error = True
            queries_with_errors += 1
            err_list = [str(exc)]
            total_errors_logged += 1
            print(f"  ❌ Case failed with exception: {exc}")

        elapsed = time.perf_counter() - start_time
        total_wall_time += elapsed
        if elapsed > time_budget:
            queries_over_time_budget += 1

        case_f1, case_prec, case_rec = 0.0, 0.0, 0.0
        cand_rec_50, cand_rec_100, cand_rec_300, cand_rec_500 = 0.0, 0.0, 0.0, 0.0
        selector_recall, final_transfer_rate = 0.0, 0.0

        # 计算得分
        if not has_error:
            # 最终推荐得分 (限制前 k)
            final_scores = score_papers_against_gold(result_papers[:k], gold)
            case_f1 = final_scores["f1"]
            case_prec = final_scores["precision"]
            case_rec = final_scores["recall"]

            # 候选池 @50/@100/@300/@500 得分
            cand_scores_50 = score_papers_against_gold(candidate_pool[:50], gold)
            cand_scores_100 = score_papers_against_gold(candidate_pool[:100], gold)
            cand_scores_300 = score_papers_against_gold(candidate_pool[:300], gold)
            cand_scores_500 = score_papers_against_gold(candidate_pool[:500], gold)
            cand_rec_50 = cand_scores_50["recall"]
            cand_rec_100 = cand_scores_100["recall"]
            cand_rec_300 = cand_scores_300["recall"]
            cand_rec_500 = cand_scores_500["recall"]

            # P1.5 Task 5: selector_recall = gold in selection_candidates / gold in candidate_pool
            selection_candidates = getattr(pipeline, "selection_candidates", [])
            sel_gold = score_papers_against_gold(selection_candidates, gold)
            pool_gold = score_papers_against_gold(candidate_pool, gold)
            if pool_gold["gold_count"] > 0:
                selector_recall = sel_gold["true_positive"] / pool_gold["gold_count"]
            # final_transfer_rate = final_gold / candidate_gold
            if pool_gold["true_positive"] > 0:
                final_transfer_rate = final_scores["true_positive"] / pool_gold["true_positive"]

            total_f1_final += case_f1
            total_precision_final += case_prec
            total_recall_final += case_rec
            total_candidate_recall_50 += cand_rec_50
            total_candidate_recall_100 += cand_rec_100
            total_candidate_recall_300 += cand_rec_300
            total_candidate_recall_500 += cand_rec_500
            total_selector_recall += selector_recall
            total_final_transfer_rate += final_transfer_rate

            print(f"  -> Prec: {case_prec:.4f} | Recall: {case_rec:.4f} | F1: {case_f1:.4f}")
            print(f"  -> CandRec@50: {cand_rec_50:.4f} | @100: {cand_rec_100:.4f} | @300: {cand_rec_300:.4f} | @500: {cand_rec_500:.4f}")
            print(f"  -> SelectorRecall: {selector_recall:.4f} | FinalTransfer: {final_transfer_rate:.4f}")
            print(f"  -> Elapsed: {elapsed:.2f}s | LLM calls: {res.run_metrics.llm_calls_used} | API calls: {res.run_metrics.api_calls_used}")

            # Task 5: TopK Sweep evaluation
            _ranked_papers = getattr(pipeline, "ranked_papers", [])
            if _ranked_papers:
                topk_sweep = compute_topk_sweep(_ranked_papers, gold)
                dynamic_k = getattr(res, "dynamic_k_chosen", None) or len(result_papers)
                # Compute F1 at exact dynamic_k (may not be in TOPK_VALUES)
                _dyn_k_papers = [rp.paper for rp in _ranked_papers[:dynamic_k]]
                _dyn_k_scores = score_papers_against_gold(_dyn_k_papers, gold)
                dynamic_k_f1 = _dyn_k_scores["f1"]
                oracle_best_k = topk_sweep["oracle_best_k"]
                oracle_f1 = topk_sweep["oracle_f1"]
                gap = oracle_f1 - dynamic_k_f1

                topk_case_count += 1
                for _k in TOPK_VALUES:
                    total_f1_at_k[_k] += topk_sweep["f1_at_k"].get(_k, 0.0)
                    total_precision_at_k[_k] += topk_sweep["precision_at_k"].get(_k, 0.0)
                    total_recall_at_k[_k] += topk_sweep["recall_at_k"].get(_k, 0.0)
                total_oracle_f1 += oracle_f1
                total_dynamic_k_f1 += dynamic_k_f1
                total_oracle_best_k += oracle_best_k
                total_dynamic_k_chosen += dynamic_k
                total_output_k += len(result_papers)

                # Task 7: K predictor case classification
                _g_hat = getattr(res, "g_hat", None)
                _lc_uniform = getattr(res, "low_confidence_uniform", False)
                _gold_count = len(gold) if gold else 0
                # Compute mean/std of top-10 calibrated probabilities
                _probs = [
                    rp.paper.metadata.get("calibrated_probability", 0.0)
                    for rp in _ranked_papers[:min(10, len(_ranked_papers))]
                ]
                _mean_prob = sum(_probs) / len(_probs) if _probs else 0.0
                _std_prob = 0.0
                if len(_probs) > 1:
                    _std_prob = (sum((p - _mean_prob) ** 2 for p in _probs) / len(_probs)) ** 0.5
                _category = classify_k_prediction(
                    oracle_k=oracle_best_k,
                    dynamic_k=dynamic_k,
                    gap=gap,
                    g_hat=_g_hat,
                    gold_count=_gold_count,
                    lc_uniform=_lc_uniform,
                    mean_prob=_mean_prob,
                    std_prob=_std_prob,
                )
                k_prediction_categories[_category] = k_prediction_categories.get(_category, 0) + 1
                k_prediction_details.append({
                    "case": f"Case #{original_idx}",
                    "category": _category,
                    "oracle_k": oracle_best_k,
                    "dynamic_k": dynamic_k,
                    "gap": round(gap, 4),
                    "g_hat": round(_g_hat, 2) if _g_hat else None,
                    "gold_count": _gold_count,
                    "lc_uniform": _lc_uniform,
                    "mean_prob": round(_mean_prob, 4),
                    "std_prob": round(_std_prob, 4),
                })

                f1_str = " | ".join(f"F1@{k}={topk_sweep['f1_at_k'].get(k, 0.0):.4f}" for k in TOPK_VALUES)
                print(f"  -> TopK Sweep: {f1_str}")
                print(f"  -> Oracle: best_k={oracle_best_k} f1={oracle_f1:.4f} | Dynamic: k={dynamic_k} f1={dynamic_k_f1:.4f} | Gap={gap:.4f}")
                _g_hat_str = f"{_g_hat:.1f}" if _g_hat else "?"
                print(f"  -> K-Category: {_category} (g_hat={_g_hat_str}, gold={_gold_count}, mean_p={_mean_prob:.3f}, std_p={_std_prob:.3f})")

            # P2: Generic gold trace — no per-QID patches, controlled by --trace-gold flag
            if trace_gold:
                _print_gold_trace(candidate_pool, selection_candidates,
                                  getattr(pipeline, "ranked_papers", []),
                                  result_papers, gold, case_idx=original_idx)
        else:
            print(f"  -> Case failed. Skipping scores.")

        # P1.5 Task 5: Use res.run_metrics (pipeline's internal budget) instead of local budget
        # The pipeline creates its OWN BudgetManager internally, so the local `budget` is disconnected.
        case_llm_calls = 0
        case_api_calls = 0
        if not has_error and res is not None:
            case_llm_calls = res.run_metrics.llm_calls_used
            case_api_calls = res.run_metrics.api_calls_used

        # 保存用例结果日志
        _topk_data = {}
        _dyn_k = None
        _oracle_k = None
        _oracle_f1 = None
        if not has_error and _ranked_papers:
            _topk_data = topk_sweep
            _dyn_k = dynamic_k
            _oracle_k = oracle_best_k
            _oracle_f1 = oracle_f1

        results_log.append({
            "case_index": original_idx,
            "name": name,
            "query": query,
            "success": not has_error,
            "elapsed_seconds": round(elapsed, 4),
            "precision": case_prec,
            "recall": case_rec,
            "f1": case_f1,
            "candidate_recall_50": cand_rec_50,
            "candidate_recall_100": cand_rec_100,
            "candidate_recall_300": cand_rec_300,
            "candidate_recall_500": cand_rec_500,
            "selector_recall": selector_recall,
            "final_transfer_rate": final_transfer_rate,
            "llm_calls": case_llm_calls,
            "api_calls": case_api_calls,
            "dynamic_k_chosen": _dyn_k,
            "oracle_best_k": _oracle_k,
            "oracle_f1": _oracle_f1,
            "topk_sweep": _topk_data,
            "errors": err_list
        })

    n_cases = len(eval_cases)
    avg_f1 = total_f1_final / n_cases
    avg_prec = total_precision_final / n_cases
    avg_rec = total_recall_final / n_cases
    avg_cand_rec_50 = total_candidate_recall_50 / n_cases
    avg_cand_rec_100 = total_candidate_recall_100 / n_cases
    avg_cand_rec_300 = total_candidate_recall_300 / n_cases
    avg_cand_rec_500 = total_candidate_recall_500 / n_cases
    avg_selector_recall = total_selector_recall / n_cases
    avg_final_transfer_rate = total_final_transfer_rate / n_cases
    avg_time = total_wall_time / n_cases

    # 3. 输出汇总模型性能报表 (与 baseline 对齐并展示提升)
    print("\n" + "=" * 70)
    print("和 baseline 比:")
    print(format_compare("avg_f1_final", avg_f1, BASELINE_METRICS["avg_f1_final"]))
    print(format_compare("avg_precision_final", avg_prec, BASELINE_METRICS["avg_precision_final"]))
    print(format_compare("avg_recall_final", avg_rec, BASELINE_METRICS["avg_recall_final"]))
    print(f"• avg_candidate_recall_50  : {avg_cand_rec_50:.4f}")
    print(f"• avg_candidate_recall_100 : {avg_cand_rec_100:.4f}")
    print(format_compare("avg_candidate_recall_300", avg_cand_rec_300, BASELINE_METRICS["avg_candidate_recall_300"]))
    print(format_compare("avg_candidate_recall_500", avg_cand_rec_500, BASELINE_METRICS["avg_candidate_recall_500"]))
    print(f"• avg_selector_recall      : {avg_selector_recall:.4f}")
    print(f"• avg_final_transfer_rate  : {avg_final_transfer_rate:.4f}")
    print(format_compare("avg_wall_time", avg_time, BASELINE_METRICS["avg_wall_time"], is_time=True))
    print(f"• queries_over_time_budget : {BASELINE_METRICS['queries_over_time_budget']} -> {queries_over_time_budget}")
    print(f"• queries_with_errors : {BASELINE_METRICS['queries_with_errors']} -> {queries_with_errors}")
    print("=" * 70)

    # Task 5: TopK Sweep summary
    if topk_case_count > 0:
        avg_oracle_f1 = total_oracle_f1 / topk_case_count
        avg_dynamic_k_f1 = total_dynamic_k_f1 / topk_case_count
        avg_oracle_best_k = total_oracle_best_k / topk_case_count
        avg_dynamic_k_chosen = total_dynamic_k_chosen / topk_case_count
        avg_output_k = total_output_k / topk_case_count
        oracle_gap = avg_oracle_f1 - avg_dynamic_k_f1

        print("\n" + "=" * 70)
        print("📊 F1-AWARE DYNAMIC-K CONTROLLER — TopK Sweep Summary:")
        print("=" * 70)

        # F1@K curve
        print("  F1@K curve:")
        for k in TOPK_VALUES:
            avg_f1_k = total_f1_at_k[k] / topk_case_count
            avg_p_k = total_precision_at_k[k] / topk_case_count
            avg_r_k = total_recall_at_k[k] / topk_case_count
            print(f"    K={k:<3} : F1={avg_f1_k:.4f} | P={avg_p_k:.4f} | R={avg_r_k:.4f}")

        # Fixed-K comparison
        print(f"\n  Fixed-K F1 comparison:")
        for k in (5, 10, 20):
            if k in total_f1_at_k:
                avg_f1_k = total_f1_at_k[k] / topk_case_count
                print(f"    Fixed-K={k:<3} : avg_F1={avg_f1_k:.4f}")

        # Dynamic-K vs Oracle
        print(f"\n  Dynamic-K vs Oracle:")
        print(f"    avg_oracle_best_k    = {avg_oracle_best_k:.2f}")
        print(f"    avg_dynamic_k_chosen = {avg_dynamic_k_chosen:.2f}")
        print(f"    avg_output_k         = {avg_output_k:.2f}")
        print(f"    avg_oracle_f1        = {avg_oracle_f1:.4f}")
        print(f"    avg_dynamic_k_f1     = {avg_dynamic_k_f1:.4f}")
        print(f"    gap (oracle - dynamic) = {oracle_gap:.4f}")
        print("=" * 70)

        # Task 7: K predictor case classification summary
        print("\n" + "=" * 70)
        print("📊 K PREDICTOR CASE CLASSIFICATION (Task 7):")
        print("=" * 70)
        for cat, count in k_prediction_categories.items():
            pct = count / topk_case_count * 100 if topk_case_count > 0 else 0
            print(f"  {cat:<40} : {count}/{topk_case_count} ({pct:.1f}%)")
        print("-" * 70)
        print("  Per-case details:")
        for d in k_prediction_details:
            print(
                f"    {d['category']:<35} | {d['case']:<30} | "
                f"oracle_k={d['oracle_k']} dyn_k={d['dynamic_k']} "
                f"gap={d['gap']:.4f} g_hat={d['g_hat']} gold={d['gold_count']} "
                f"mean_p={d['mean_prob']:.3f} std_p={d['std_prob']:.3f}"
            )
        print("=" * 70)

    # 4. 输出各组件预算参数
    print("\n📊 COMPONENT BUDGET AND RESOURCE STATS (Averages per query):")
    print(f"  Avg LLM Calls: {total_llm_calls / n_cases:.1f} (Limit: {config.budget.max_llm_calls})")
    print(f"  Avg API Calls: {total_api_calls / n_cases:.1f} (Limit: {config.budget.max_search_queries})")
    print(f"  Avg Token Estimate: {total_token_estimate / n_cases:.1f}")
    print(f"  Avg Cache Hits: {total_cache_hits / n_cases:.1f}")
    print(f"  Total Errors Logged: {total_errors_logged}")

    # 大模型提供商统计
    if llm_model_stats:
        print("\n🤖 LLM Models and Providers Usage breakdown:")
        print(f"  {'Model / Provider Name':<35} | {'Calls / Requests':<16} | {'Tokens Used':<12} | {'Total Time (s)':<12}")
        print("  " + "-" * 82)
        for m_name, m_data in sorted(llm_model_stats.items(), key=lambda x: x[1]["elapsed"], reverse=True):
            print(f"  {m_name:<35} | {m_data['calls']:<16} | {m_data['tokens']:<12} | {m_data['elapsed']:<12.2f}")

    # 学术检索服务商统计 (把 retrieval.xxx 按 xxx 合并分组)
    retrieval_providers_stats: dict[str, dict[str, Any]] = {}
    other_components_stats: dict[str, dict[str, Any]] = {}
    for cname, c_data in component_costs_accum.items():
        if cname.startswith("retrieval."):
            parts = cname.split(".")
            provider_name = parts[1] if len(parts) > 1 else "unknown"
            p_stats = retrieval_providers_stats.setdefault(provider_name, {"elapsed": 0.0, "calls": 0, "items": 0})
            p_stats["elapsed"] += c_data["elapsed"]
            # 检索服务的 calls 记录的是真实的网络请求次数而非 component 请求
            p_stats["calls"] += c_data["calls"]
        else:
            other_components_stats[cname] = c_data

    if retrieval_providers_stats:
        print("\n🌐 Academic Retrieval Services Providers stats:")
        print(f"  {'Service Provider':<35} | {'Elapsed Time (s)':<16} | {'Total Requests':<14}")
        print("  " + "-" * 72)
        for p_name, p_data in sorted(retrieval_providers_stats.items(), key=lambda x: x[1]["elapsed"], reverse=True):
            print(f"  {p_name:<35} | {p_data['elapsed']:<16.2f} | {p_data['calls']:<14}")

    # 各核心组件统计明细
    print("\n⏱️ Core Component Pipelines breakdown:")
    print(f"  {'Component Name':<35} | {'Elapsed Time (s)':<16} | {'Calls / Requests':<14} | {'Tokens':<10}")
    print("  " + "-" * 82)
    for cname, c_data in sorted(other_components_stats.items(), key=lambda x: x[1]["elapsed"], reverse=True):
        print(f"  {cname:<35} | {c_data['elapsed']:<16.2f} | {c_data['calls']:<14} | {c_data['tokens']:<10}")
    print("=" * 70)

    # 5. 输出 JSON 日志
    if output_path:
        output_data = {
            "summary": {
                "mode": mode,
                "n_cases": n_cases,
                "avg_f1_final": round(avg_f1, 4),
                "avg_precision_final": round(avg_prec, 4),
                "avg_recall_final": round(avg_rec, 4),
                "avg_candidate_recall_50": round(avg_cand_rec_50, 4),
                "avg_candidate_recall_100": round(avg_cand_rec_100, 4),
                "avg_candidate_recall_300": round(avg_cand_rec_300, 4),
                "avg_candidate_recall_500": round(avg_cand_rec_500, 4),
                "avg_selector_recall": round(avg_selector_recall, 4),
                "avg_final_transfer_rate": round(avg_final_transfer_rate, 4),
                "avg_wall_time": round(avg_time, 4),
                "queries_over_time_budget": queries_over_time_budget,
                "queries_with_errors": queries_with_errors,
                "total_llm_calls": total_llm_calls,
                "total_api_calls": total_api_calls,
                "total_token_estimate": total_token_estimate,
                "total_cache_hits": total_cache_hits,
            },
            "topk_sweep_summary": {
                "topk_case_count": topk_case_count,
                "avg_f1_at_k": {str(k): round(total_f1_at_k[k] / topk_case_count, 4) for k in TOPK_VALUES} if topk_case_count else {},
                "avg_precision_at_k": {str(k): round(total_precision_at_k[k] / topk_case_count, 4) for k in TOPK_VALUES} if topk_case_count else {},
                "avg_recall_at_k": {str(k): round(total_recall_at_k[k] / topk_case_count, 4) for k in TOPK_VALUES} if topk_case_count else {},
                "avg_oracle_f1": round(total_oracle_f1 / topk_case_count, 4) if topk_case_count else 0.0,
                "avg_dynamic_k_f1": round(total_dynamic_k_f1 / topk_case_count, 4) if topk_case_count else 0.0,
                "avg_oracle_best_k": round(total_oracle_best_k / topk_case_count, 2) if topk_case_count else 0.0,
                "avg_dynamic_k_chosen": round(total_dynamic_k_chosen / topk_case_count, 2) if topk_case_count else 0.0,
                "avg_output_k": round(total_output_k / topk_case_count, 2) if topk_case_count else 0.0,
            } if topk_case_count > 0 else {},
            "k_predictor_classification": {
                "categories": k_prediction_categories,
                "per_case": k_prediction_details,
            } if topk_case_count > 0 else {},
            "cases": results_log,
            "llm_stats": llm_model_stats,
            "retrieval_stats": retrieval_providers_stats,
            "component_stats": component_costs_accum
        }
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Detailed evaluation report saved to {Path(output_path).resolve()}")
        except Exception as exc:
            print(f"❌ Failed to save output file: {exc}", file=sys.stderr)


if __name__ == "__main__":
    # 强制设置 sys.stdout 编码为 utf-8，解决 Windows 环境下打印 emoji 时 UnicodeEncodeError 问题
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    setup_logging(logging.INFO)
    
    parser = argparse.ArgumentParser(description="Scholar Agent V2.0 Evaluation Runner")
    parser.add_argument(
        "-m", "--mode",
        choices=["live", "mock"],
        default=None,
        help="运行模式 (若未指定且存在 DEEPSEEK_API_KEY，默认 live；否则默认 mock)"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="限制运行的评估案例数量"
    )
    parser.add_argument(
        "-c", "--cases",
        type=str,
        default=None,
        help="指定仅运行的部分案例编号（如：'1,2,5-7'），不指定则默认全量"
    )
    parser.add_argument(
        "-s", "--simple",
        action="store_true",
        help="只运行1条简单测试（即第1个案例）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="指定保存评估结果 JSON 文件的路径"
    )
    parser.add_argument(
        "-t", "--time-budget",
        type=float,
        default=None,
        help="单个 Query 的最大耗时预算（秒），同时作为 pipeline deadline；未指定则使用配置 case_deadline_seconds"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="指定 YAML 配置文件路径（如 configs/live_fast.yaml），默认 configs/default.yaml"
    )
    parser.add_argument(
        "--trace-gold",
        action="store_true",
        default=False,
        help="P2: 为每个 case 打印 gold paper candidate-to-final 诊断 trace"
    )
    parser.add_argument(
        "-f", "--dataset-file",
        type=str,
        default=None,
        help="指定评测数据集文件路径（如 data/开发+测试集/CNScholarQuery_ZH_dev_1000.jsonl），覆盖默认搜索路径"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="limit 随机抽样的固定种子，默认 123；更换 seed 可得到另一组可复现样本"
    )

    args = parser.parse_args()

    # 智能探测默认运行模式
    run_mode = args.mode
    if run_mode is None:
        # 探测 API key，看是否配置
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        # 如果从环境获取不到，尝试手动看下 .env 文件内容
        if not api_key and Path(".env").exists():
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    content = f.read()
                    match = re.search(r"DEEPSEEK_API_KEY\s*=\s*([^\s#]+)", content)
                    if match:
                        api_key = match.group(1).strip("'\"")
            except Exception:
                pass
        
        if api_key and "mock" not in api_key.lower():
            run_mode = "live"
        else:
            run_mode = "mock"
            print("💡 No active DEEPSEEK_API_KEY found, running in 'mock' mode by default.")

    run_evaluation(
        mode=run_mode,
        limit=args.limit,
        cases_filter=args.cases,
        simple=args.simple,
        output_path=args.output,
        time_budget=args.time_budget,
        config_path=args.config,
        trace_gold=args.trace_gold,
        dataset_file=args.dataset_file,
        random_seed=args.seed,
    )
