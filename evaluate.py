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

    def complete_json(self, system_prompt: str, user_prompt: str, model_type: str = "flash", timeout_seconds: float | None = None) -> Any | None:
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
        result = self.base_client.complete_json(system_prompt, user_prompt, model_type, timeout_seconds=timeout_seconds)

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


def run_evaluation(
    mode: str = "live",
    limit: int | None = None,
    cases_filter: str | None = None,
    simple: bool = False,
    output_path: str | None = None,
    time_budget: float = 30.0,
) -> None:
    # 加载配置
    config = load_config()
    config.app.mode = mode

    # 强制在 live 模式下激活主要的在线检索提供商
    if mode == "live":
        config.app.providers = ["pasa_local", "openalex", "arxiv", "semantic_scholar", "pubmed"]
        config.providers.pasa_local.enabled = True
        config.providers.openalex.enabled = True
        config.providers.arxiv.enabled = True
        config.providers.semantic_scholar.enabled = True
        config.providers.pubmed.enabled = True
    else:
        config.app.providers = ["mock"]

    print("=" * 70)
    print(f"📊 Starting Scholar Agent V2.0 Evaluation in '{mode}' mode")
    print("=" * 70)

    # 全量验证集路径寻址
    candidate_paths = [
        Path("data/RealScholarQuery_test.jsonl"),
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

    # 用例筛选
    import random as _random
    indices = list(range(len(all_cases)))
    if cases_filter:
        indices = parse_cases_arg(cases_filter, len(all_cases))
        if not indices:
            print(f"❌ Error: filter '{cases_filter}' did not match any test cases.", file=sys.stderr)
            sys.exit(1)
    elif simple:
        indices = [0]
    elif limit is not None and limit > 0:
        # 随机抽样，而非顺序取前 N 个
        if limit < len(indices):
            indices = sorted(_random.sample(indices, limit))
        # 若 limit >= 总数，则直接使用全量（不截断）

    eval_cases = [all_cases[i] for i in indices]
    print(f"Total cases in dataset: {len(all_cases)}. Selected {len(eval_cases)} cases for evaluation (indices: {[i+1 for i in indices]}).")

    # 构建统一 providers
    providers = build_providers(config)

    # 统计数据汇总
    total_f1_final = 0.0
    total_precision_final = 0.0
    total_recall_final = 0.0
    total_candidate_recall_300 = 0.0
    total_candidate_recall_500 = 0.0
    total_wall_time = 0.0
    queries_over_time_budget = 0
    queries_with_errors = 0

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
        cand_rec_300, cand_rec_500 = 0.0, 0.0

        # 计算得分
        if not has_error:
            # 最终推荐得分 (限制前 k)
            final_scores = score_papers_against_gold(result_papers[:k], gold)
            case_f1 = final_scores["f1"]
            case_prec = final_scores["precision"]
            case_rec = final_scores["recall"]

            # 候选池 @300 & @500 得分
            cand_scores_300 = score_papers_against_gold(candidate_pool[:300], gold)
            cand_scores_500 = score_papers_against_gold(candidate_pool[:500], gold)
            cand_rec_300 = cand_scores_300["recall"]
            cand_rec_500 = cand_scores_500["recall"]

            total_f1_final += case_f1
            total_precision_final += case_prec
            total_recall_final += case_rec
            total_candidate_recall_300 += cand_rec_300
            total_candidate_recall_500 += cand_rec_500

            print(f"  -> Prec: {case_prec:.4f} | Recall: {case_rec:.4f} | F1: {case_f1:.4f}")
            print(f"  -> Candidate Recall@300: {cand_rec_300:.4f} | Recall@500: {cand_rec_500:.4f}")
            print(f"  -> Elapsed: {elapsed:.2f}s | LLM calls: {budget.llm_calls_used}")
        else:
            print(f"  -> Case failed. Skipping scores.")

        # 保存用例结果日志
        results_log.append({
            "case_index": original_idx,
            "name": name,
            "query": query,
            "success": not has_error,
            "elapsed_seconds": round(elapsed, 4),
            "precision": case_prec,
            "recall": case_rec,
            "f1": case_f1,
            "candidate_recall_300": cand_rec_300,
            "candidate_recall_500": cand_rec_500,
            "llm_calls": budget.llm_calls_used,
            "api_calls": budget.api_calls_used,
            "errors": err_list
        })

    n_cases = len(eval_cases)
    avg_f1 = total_f1_final / n_cases
    avg_prec = total_precision_final / n_cases
    avg_rec = total_recall_final / n_cases
    avg_cand_rec_300 = total_candidate_recall_300 / n_cases
    avg_cand_rec_500 = total_candidate_recall_500 / n_cases
    avg_time = total_wall_time / n_cases

    # 3. 输出汇总模型性能报表 (与 baseline 对齐并展示提升)
    print("\n" + "=" * 70)
    print("和 baseline 比:")
    print(format_compare("avg_f1_final", avg_f1, BASELINE_METRICS["avg_f1_final"]))
    print(format_compare("avg_precision_final", avg_prec, BASELINE_METRICS["avg_precision_final"]))
    print(format_compare("avg_recall_final", avg_rec, BASELINE_METRICS["avg_recall_final"]))
    print(format_compare("avg_candidate_recall_300", avg_cand_rec_300, BASELINE_METRICS["avg_candidate_recall_300"]))
    print(format_compare("avg_candidate_recall_500", avg_cand_rec_500, BASELINE_METRICS["avg_candidate_recall_500"]))
    print(format_compare("avg_wall_time", avg_time, BASELINE_METRICS["avg_wall_time"], is_time=True))
    print(f"• queries_over_time_budget : {BASELINE_METRICS['queries_over_time_budget']} -> {queries_over_time_budget}")
    print(f"• queries_with_errors : {BASELINE_METRICS['queries_with_errors']} -> {queries_with_errors}")
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
                "avg_candidate_recall_300": round(avg_cand_rec_300, 4),
                "avg_candidate_recall_500": round(avg_cand_rec_500, 4),
                "avg_wall_time": round(avg_time, 4),
                "queries_over_time_budget": queries_over_time_budget,
                "queries_with_errors": queries_with_errors,
                "total_llm_calls": total_llm_calls,
                "total_api_calls": total_api_calls,
                "total_token_estimate": total_token_estimate,
                "total_cache_hits": total_cache_hits,
            },
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
        default=30.0,
        help="单个 Query 的最大耗时预算（秒），超过此值算为超时，默认 30.0"
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
        time_budget=args.time_budget
    )
