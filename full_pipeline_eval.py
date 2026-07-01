#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Pipeline 评估脚本 — 跑完整三阶段（检索→证据选择→合成），输出最终模型性能。
针对指定 QID 的 case，输出：
  - Stage 1: 候选池召回率 (Pool Recall)
  - Stage 2: 证据选择结果（选中论文数、验证后论文数）
  - Stage 3: 最终输出论文 + Precision/Recall/F1
  - 各组件损耗（时间、LLM调用、API调用、Token）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# 确保 src 目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.infra import (
    OpenAICompatibleLLMClient,
    load_config,
    setup_logging,
)
from scholar_agent.retrieval import build_providers
from scholar_agent.workflow.pipeline import PaperAgentPipeline
from scholar_agent.workflow.budget import BudgetManager


# ---------- 匹配算法（与 eval_pool_recall_temp.py 保持一致）----------

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


def count_hits(papers: list[Any], gold_items: list[dict[str, Any] | str]) -> tuple[int, int]:
    """返回 (命中数, gold总数)"""
    gold_key_sets = []
    for item in gold_items:
        keys = gold_match_keys(item)
        if keys:
            gold_key_sets.append(keys)
    paper_key_sets = [paper_match_keys(paper) for paper in papers]
    matched_gold_indexes: set[int] = set()

    # DEBUG: Print gold keys
    print(f"DEBUG MATCH: {len(gold_key_sets)} gold items, {len(paper_key_sets)} papers")
    for i, gks in enumerate(gold_key_sets):
        print(f"  gold[{i}] keys: {gks}")

    for paper_idx, paper_keys in enumerate(paper_key_sets):
        for index, gold_keys in enumerate(gold_key_sets):
            if index in matched_gold_indexes:
                continue
            if paper_keys & gold_keys:
                matched_gold_indexes.add(index)
                break
        # DEBUG: Print paper keys for first 5 papers
        if paper_idx < 5:
            _title = getattr(papers[paper_idx], 'title', '')[:40]
            _intersect = paper_keys & gold_key_sets[0] if gold_key_sets else set()
            print(f"  paper[{paper_idx}] keys: {paper_keys}")
            print(f"    title: {_title}, intersect with gold[0]: {_intersect}")

    print(f"DEBUG MATCH: matched {len(matched_gold_indexes)}/{len(gold_key_sets)}")

    return len(matched_gold_indexes), len(gold_key_sets)


def compute_precision_recall_f1(output_papers: list[Any], gold_items: list[dict[str, Any] | str]) -> dict:
    hits, gold_total = count_hits(output_papers, gold_items)
    output_total = len(output_papers)
    precision = hits / output_total if output_total > 0 else 0.0
    recall = hits / gold_total if gold_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "hits": hits,
        "gold_total": gold_total,
        "output_total": output_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------- 主逻辑 ----------

def run_full_pipeline_eval(
    qids: list[int],
    config_path: str = "configs/effect_first.yaml",
    dataset: str = "data/RealScholarQuery_test.jsonl",
) -> None:
    """对指定 QID 跑完整三阶段 pipeline，输出最终模型性能。"""

    # 1. 加载配置
    config = load_config(config_path)
    config.app.mode = "live"

    # 尊重 YAML provider 设置
    enabled_providers = []
    for prov_name in ("pasa_local", "openalex", "arxiv", "semantic_scholar", "pubmed"):
        prov_cfg = getattr(config.providers, prov_name, None)
        if prov_cfg and getattr(prov_cfg, "enabled", False):
            enabled_providers.append(prov_name)
    if enabled_providers:
        enabled_providers.append("faiss_vector")
        config.app.providers = enabled_providers
    else:
        config.app.providers = ["pasa_local", "openalex", "semantic_scholar", "faiss_vector"]

    # 2. 加载数据集
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        print(f"❌ 找不到数据集: {dataset_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    all_cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                answers = raw.get("answer") or raw.get("answers") or []
                arxiv_ids = raw.get("answer_arxiv_id") or raw.get("answer_arxiv_ids") or []
                gold = []
                for i, ans in enumerate(answers):
                    if isinstance(ans, dict):
                        gold.append(ans)
                    else:
                        g = {"title": ans}
                        if i < len(arxiv_ids) and arxiv_ids[i]:
                            g["arxiv_id"] = arxiv_ids[i]
                        gold.append(g)
                all_cases.append({
                    "name": raw.get("qid") or f"case-{len(all_cases)+1}",
                    "query": raw.get("question") or raw.get("query") or "",
                    "gold": gold,
                })

    print("=" * 90)
    print("📊 Full Pipeline 评估 — 完整三阶段（检索→证据选择→合成）")
    print(f"   数据集: {dataset_path.name} (共 {len(all_cases)} 条)")
    print(f"   目标 QID: {qids}")
    print(f"   配置: {config_path}")
    print(f"   启用检索源: {config.app.providers}")
    print("=" * 90)

    # 3. 构建 providers（只构建一次）
    providers = build_providers(config, provider_names=config.app.providers)
    provider_names = [p.name for p in providers]
    print(f"   实际加载检索源: {provider_names}")

    # 4. 逐条评估
    all_results = []

    for qid in qids:
        case_idx = qid - 1  # 0-indexed
        if case_idx < 0 or case_idx >= len(all_cases):
            print(f"\n⚠️ QID={qid} 超出数据集范围 (1-{len(all_cases)})，跳过")
            continue

        case = all_cases[case_idx]
        query = case["query"]
        gold = case["gold"]
        gold_count = len(gold)

        print(f"\n{'='*80}")
        print(f"QID={qid} | Gold={gold_count} | Query: \"{query[:100]}...\"")
        print(f"{'='*80}")

        # 每条 query 独立 budget 和 LLM client
        budget = BudgetManager(config)
        llm_client = OpenAICompatibleLLMClient(config.llm, budget)
        pipeline = PaperAgentPipeline(config, llm_client, providers)

        start_t = time.perf_counter()
        has_error = False
        error_msg = ""

        try:
            # 跑完整三阶段
            result = pipeline.run(query, retrieval_only=False)
        except Exception as exc:
            has_error = True
            error_msg = str(exc)
            print(f"  ❌ Pipeline 异常: {exc}")
            result = None

        elapsed = time.perf_counter() - start_t

        if result is None:
            print(f"  ❌ Pipeline 失败，跳过结果分析")
            all_results.append({
                "qid": qid,
                "query": query,
                "gold_count": gold_count,
                "error": error_msg,
                "elapsed_seconds": round(elapsed, 2),
            })
            continue

        # ---- Stage 1: 候选池召回率 ----
        candidate_pool = getattr(pipeline, "candidate_pool", [])
        pool_size = len(candidate_pool)
        pool_hits, pool_gold_total = count_hits(candidate_pool, gold)
        pool_recall = pool_hits / pool_gold_total if pool_gold_total > 0 else 0.0

        print(f"\n--- Stage 1: 检索阶段 ---")
        print(f"  候选池大小: {pool_size}")
        print(f"  Pool Recall: {pool_hits}/{pool_gold_total} = {pool_recall:.4f}")

        # ---- Stage 2: 证据选择 ----
        selection_candidates = getattr(pipeline, "selection_candidates", [])
        selections = getattr(pipeline, "selections", [])
        ranked_papers = getattr(pipeline, "ranked_papers", [])

        print(f"\n--- Stage 2: 证据选择与精排 ---")
        print(f"  精排输入候选: {len(selection_candidates)}")
        print(f"  证据选择结果: {len(selections)}")
        print(f"  重排后论文数: {len(ranked_papers)}")

        # ---- Stage 3: 最终合成输出 ----
        highly_relevant = result.highly_relevant_papers
        partially_relevant = result.partially_relevant_papers
        supporting = result.supporting_papers

        # 提取 Paper 对象用于匹配
        final_papers = []
        for rp in highly_relevant:
            final_papers.append(rp.paper)
        for rp in partially_relevant:
            final_papers.append(rp.paper)

        # 计算最终 Precision/Recall/F1
        metrics = compute_precision_recall_f1(final_papers, gold)

        print(f"\n--- Stage 3: 合成输出 ---")
        print(f"  Highly Relevant: {len(highly_relevant)}")
        print(f"  Partially Relevant: {len(partially_relevant)}")
        print(f"  Supporting: {len(supporting)}")
        print(f"  最终输出论文数: {metrics['output_total']}")
        print(f"  Gold 命中: {metrics['hits']}/{metrics['gold_total']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")

        # ---- 各组件损耗 ----
        run_metrics = result.run_metrics
        print(f"\n--- 组件损耗 ---")
        print(f"  总耗时: {elapsed:.1f}s (pipeline elapsed: {run_metrics.elapsed_seconds:.1f}s)")
        print(f"  LLM 调用次数: {run_metrics.llm_calls_used}")
        print(f"  LLM 耗时: {run_metrics.llm_elapsed_seconds:.1f}s")
        print(f"  API 调用次数: {run_metrics.api_calls_used}")
        print(f"  Token 估算: {run_metrics.token_estimate}")
        print(f"  搜索查询数: {run_metrics.search_queries_used}")
        print(f"  检索轮数: {run_metrics.retrieval_rounds_used}")
        print(f"  缓存命中: {run_metrics.cache_hits}")
        print(f"  证据摘要数: {run_metrics.evidence_summaries}")

        if run_metrics.component_metrics:
            print(f"\n  各组件明细:")
            for cm in run_metrics.component_metrics:
                print(f"    {cm.component}: elapsed={cm.elapsed_seconds:.1f}s, "
                      f"llm_calls={cm.llm_calls_delta}, api_calls={cm.api_calls_delta}, "
                      f"tokens={cm.token_estimate_delta}, queries={cm.search_queries_delta}")

        # 动态 K 控制器信息
        dk = result.dynamic_k_chosen
        g_hat = result.g_hat
        print(f"\n--- K 控制器 ---")
        print(f"  dynamic_k_chosen: {dk}")
        print(f"  g_hat: {g_hat}")
        print(f"  g_hat_scope: {result.g_hat_scope}")
        print(f"  g_hat_pool: {result.g_hat_pool}")
        print(f"  g_hat_visible: {result.g_hat_visible}")

        # 输出最终论文列表
        print(f"\n--- 最终输出论文 ---")
        for i, rp in enumerate(highly_relevant, 1):
            p = rp.paper
            print(f"  [H{i}] score={rp.final_score:.3f} | {p.title[:80]} | arxiv={p.arxiv_id}")
        for i, rp in enumerate(partially_relevant, 1):
            p = rp.paper
            print(f"  [P{i}] score={rp.final_score:.3f} | {p.title[:80]} | arxiv={p.arxiv_id}")

        all_results.append({
            "qid": qid,
            "query": query,
            "gold_count": gold_count,
            "pool_size": pool_size,
            "pool_recall": round(pool_recall, 4),
            "pool_hits": pool_hits,
            "selection_candidates": len(selection_candidates),
            "selections": len(selections),
            "ranked_papers": len(ranked_papers),
            "highly_relevant": len(highly_relevant),
            "partially_relevant": len(partially_relevant),
            "final_output": metrics["output_total"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "hits": metrics["hits"],
            "elapsed_seconds": round(elapsed, 2),
            "llm_calls": run_metrics.llm_calls_used,
            "llm_elapsed": round(run_metrics.llm_elapsed_seconds, 1),
            "api_calls": run_metrics.api_calls_used,
            "token_estimate": run_metrics.token_estimate,
            "search_queries": run_metrics.search_queries_used,
            "retrieval_rounds": run_metrics.retrieval_rounds_used,
            "dynamic_k": dk,
            "g_hat": g_hat,
            "error": error_msg if has_error else None,
        })

    # 5. 汇总表格
    print(f"\n\n{'='*100}")
    print("📊 Full Pipeline 汇总")
    print(f"{'='*100}")
    print(f"{'QID':<6}{'Gold':<6}{'Pool':<6}{'PoolR':<8}{'Sel':<6}{'Rank':<6}"
          f"{'High':<6}{'Part':<6}{'Out':<6}{'Hits':<6}"
          f"{'P':<8}{'R':<8}{'F1':<8}{'Time':<8}{'LLM':<6}{'API':<6}")
    print("-" * 100)
    for r in all_results:
        if "error" in r and r.get("error"):
            print(f"{r['qid']:<6}{r['gold_count']:<6}{'ERR':<6}{'-':<8}{'-':<6}{'-':<6}"
                  f"{'-':<6}{'-':<6}{'-':<6}{'-':<6}"
                  f"{'-':<8}{'-':<8}{'-':<8}{r['elapsed_seconds']:<8.1f}{'-':<6}{'-':<6}")
        else:
            print(f"{r['qid']:<6}{r['gold_count']:<6}{r['pool_size']:<6}{r['pool_recall']:<8.4f}"
                  f"{r['selection_candidates']:<6}{r['ranked_papers']:<6}"
                  f"{r['highly_relevant']:<6}{r['partially_relevant']:<6}"
                  f"{r['final_output']:<6}{r['hits']:<6}"
                  f"{r['precision']:<8.4f}{r['recall']:<8.4f}{r['f1']:<8.4f}"
                  f"{r['elapsed_seconds']:<8.1f}{r['llm_calls']:<6}{r['api_calls']:<6}")

    # 微观平均
    valid = [r for r in all_results if not r.get("error")]
    if valid:
        total_hits = sum(r["hits"] for r in valid)
        total_gold = sum(r["gold_count"] for r in valid)
        total_output = sum(r["final_output"] for r in valid)
        micro_p = total_hits / total_output if total_output > 0 else 0.0
        micro_r = total_hits / total_gold if total_gold > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        avg_time = sum(r["elapsed_seconds"] for r in valid) / len(valid)
        avg_llm = sum(r["llm_calls"] for r in valid) / len(valid)
        avg_api = sum(r["api_calls"] for r in valid) / len(valid)

        print(f"\n整体统计 ({len(valid)} cases):")
        print(f"  Micro Precision: {micro_p:.4f}")
        print(f"  Micro Recall:    {micro_r:.4f}")
        print(f"  Micro F1:        {micro_f1:.4f}")
        print(f"  平均耗时: {avg_time:.1f}s")
        print(f"  平均 LLM 调用: {avg_llm:.1f}")
        print(f"  平均 API 调用: {avg_api:.1f}")

    # 保存 JSON 结果
    output_path = Path("outputs/full_pipeline_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Pipeline 评估")
    parser.add_argument("--qids", type=int, nargs="+", default=[26, 115],
                        help="要评估的 QID 列表（1-indexed）")
    parser.add_argument("--config", type=str, default="configs/effect_first.yaml",
                        help="配置文件路径")
    parser.add_argument("--dataset", type=str, default="data/RealScholarQuery_test.jsonl",
                        help="数据集路径")
    args = parser.parse_args()

    run_full_pipeline_eval(
        qids=args.qids,
        config_path=args.config,
        dataset=args.dataset,
    )
