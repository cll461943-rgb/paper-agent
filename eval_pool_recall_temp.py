#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pool Recall 评估脚本 — 接入 Semantic Scholar API 后的候选池召回率测试
输出格式对齐用户截图中的表格：
  QID | Gold | 命中 | Pool Recall | R@100 | R@300 | R@500 | Pool | LLM领域扩展 | 硬编码领域词
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
from datetime import datetime
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


# ---------- 匹配算法（与 evaluate.py 保持一致）----------

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


def count_pool_hits(papers: list[Any], gold_items: list[dict[str, Any] | str]) -> tuple[int, int]:
    """返回 (命中数, gold总数)"""
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

    return len(matched_gold_indexes), len(gold_key_sets)


def recall_at_k(papers: list[Any], gold_items: list[dict[str, Any] | str], k: int) -> tuple[float, int]:
    """计算 Recall@K，返回 (recall, 命中数)"""
    hits, total = count_pool_hits(papers[:k], gold_items)
    if total == 0:
        return 0.0, 0
    return hits / total, hits


def count_pool_hits_relaxed(papers: list[Any], gold_items: list[dict[str, Any] | str]) -> tuple[int, int]:
    """使用 is_relaxed_match 来计算宽松匹配的命中数（模块缺失时降级为 strict）。"""
    try:
        from scholar_agent.evaluation.diagnose_recall_loss import is_relaxed_match
    except ImportError:
        return count_pool_hits(papers, gold_items)
    hits = 0
    for item in gold_items:
        gold_dict = item if isinstance(item, dict) else {"title": item, "paper_id": item}
        matched = False
        for p in papers:
            if is_relaxed_match(gold_dict, p):
                matched = True
                break
        if matched:
            hits += 1
    return hits, len(gold_items)


def recall_at_k_relaxed(papers: list[Any], gold_items: list[dict[str, Any] | str], k: int) -> tuple[float, int]:
    hits, total = count_pool_hits_relaxed(papers[:k], gold_items)
    if total == 0:
        return 0.0, 0
    return hits / total, hits


# ---------- 主评估逻辑 ----------

def run_pool_recall_eval(
    n_samples: int = 10,
    timeout_per_query: float = 300.0,
    dataset: str = "data/RealScholarQuery_test.jsonl",
    seed: int | None = None,
    config_path: str | None = None,
) -> None:
    """随机抽样 n_samples 个 case，仅跑 retrieval_only 获得候选池，计算 Pool Recall 表格。"""

    # 1. 加载配置
    config = load_config(config_path)
    config.app.mode = "live"

    if config_path:
        # P1.5: Respect YAML provider settings when a custom config is provided
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
    else:
        # Default behavior: force-enable all providers with extended timeouts
        config.app.providers = ["pasa_local", "openalex", "arxiv", "semantic_scholar", "faiss_vector", "pubmed"]
        config.providers.pasa_local.enabled = True
        config.providers.openalex.enabled = True
        config.providers.arxiv.enabled = True
        config.providers.semantic_scholar.enabled = True
        config.providers.pubmed.enabled = True

        # 提高超时上限，确保每条 query 都能跑完
        config.providers.openalex.timeout_seconds = 60
        config.providers.arxiv.timeout_seconds = 60
        config.providers.semantic_scholar.timeout_seconds = 60
        config.providers.pubmed.timeout_seconds = 60
        config.llm.timeout_seconds = 120
        config.budget.max_retrieval_rounds = 3
        config.budget.max_search_queries = 30
        config.budget.max_results_per_query = 50
        config.budget.max_candidate_pool_size = 800

    # 2. 加载数据集
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        fallback_path = Path("data/benchmarks/litsearch_dev_20.json")
        if fallback_path.exists():
            dataset_path = fallback_path

    if not dataset_path.exists():
        print(f"❌ 找不到数据集: {dataset_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    all_cases = []
    if dataset_path.suffix == ".jsonl":
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
                        "k": 500,
                    })
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_cases = data.get("cases", [])

    # 3. 随机抽样
    use_baseline_qids = os.environ.get("USE_BASELINE_QIDS", "0") == "1"
    use_zero_recall_qids = os.environ.get("USE_ZERO_RECALL_QIDS", "0") == "1"
    if use_zero_recall_qids:
        indices = [142, 250, 654, 754, 759]  # 5 zero-recall cases from n=10 seed=42
    elif use_baseline_qids:
        indices = [1, 4, 5, 8, 11, 12, 13, 16, 17, 18]
    elif n_samples >= len(all_cases):
        indices = list(range(len(all_cases)))
    else:
        if seed is not None:
            random.seed(seed)
        indices = sorted(random.sample(range(len(all_cases)), n_samples))


    eval_cases = [all_cases[i] for i in indices]
    original_ids = [i + 1 for i in indices]  # 1-indexed QID

    print("=" * 90)
    print(f"📊 Pool Recall 评估 — Semantic Scholar API 接入后")
    print(f"   样本数: {n_samples} | 数据集: {dataset_path.name} (共 {len(all_cases)} 条)")
    print(f"   选中 QID: {original_ids}")
    print(f"   单条超时上限: {timeout_per_query}s")
    print("=" * 90)

    # 4. 构建 providers（只构建一次）
    # P1.5: Use config.app.providers which respects YAML settings
    providers = build_providers(config, provider_names=config.app.providers)
    provider_names = [p.name for p in providers]
    print(f"   启用的检索源: {provider_names}")

    # 5. 逐条评估
    results = []
    total_hits = 0
    total_gold = 0
    total_hits_rel = 0
    total_gold_rel = 0

    diagnosis_csv = Path("diagnose_recall_loss.csv")
    if diagnosis_csv.exists():
        try:
            diagnosis_csv.unlink()
        except Exception:
            pass

    for idx, (case, qid) in enumerate(zip(eval_cases, original_ids), 1):
        query = case["query"]
        gold = case["gold"]
        gold_count = len(gold)

        print(f"\n[{idx}/{n_samples}] QID={qid} | Gold={gold_count} | Query: \"{query[:70]}...\"")

        # 每条 query 独立 budget 和 LLM client
        budget = BudgetManager(config)
        llm_client = OpenAICompatibleLLMClient(config.llm, budget)
        pipeline = PaperAgentPipeline(config, llm_client, providers)

        start_t = time.perf_counter()
        candidate_pool = []
        raw_results = []
        has_error = False
        error_msg = ""

        try:
            # 仅做检索，不做 LLM 精排/合成
            res = pipeline.run(query, retrieval_only=True)
            candidate_pool = getattr(pipeline, "candidate_pool", [])
            raw_results = getattr(res, "raw_results", [])
        except Exception as exc:
            has_error = True
            error_msg = str(exc)
            print(f"  ❌ 异常: {exc}")
            # 尝试使用已有的候选池
            candidate_pool = getattr(pipeline, "candidate_pool", [])

        elapsed = time.perf_counter() - start_t
        pool_size = len(candidate_pool)

        # 整理 gold 为 dict 列表以供诊断
        gold_dicts = []
        for g in gold:
            if isinstance(g, dict):
                gold_dicts.append(g)
            else:
                gold_dicts.append({"title": g, "paper_id": g})

        # 计算各级别召回率 (Strict)
        hits_all, gold_total = count_pool_hits(candidate_pool, gold)
        pool_recall = hits_all / gold_total if gold_total > 0 else 0.0

        r100, hits100 = recall_at_k(candidate_pool, gold, 100)
        r300, hits300 = recall_at_k(candidate_pool, gold, 300)
        r500, hits500 = recall_at_k(candidate_pool, gold, 500)

        # 计算各级别召回率 (Relaxed)
        hits_all_rel, gold_total_rel = count_pool_hits_relaxed(candidate_pool, gold)
        pool_recall_rel = hits_all_rel / gold_total_rel if gold_total_rel > 0 else 0.0

        r100_rel, hits100_rel = recall_at_k_relaxed(candidate_pool, gold, 100)
        r300_rel, hits300_rel = recall_at_k_relaxed(candidate_pool, gold, 300)
        r500_rel, hits500_rel = recall_at_k_relaxed(candidate_pool, gold, 500)

        total_hits += hits_all
        total_gold += gold_total
        total_hits_rel += hits_all_rel
        total_gold_rel += gold_total_rel

        # 导出丢失诊断
        try:
            from scholar_agent.evaluation.diagnose_recall_loss import export_gold_diagnosis
            export_gold_diagnosis(
                qid=str(qid),
                query=query,
                gold_items=gold_dicts,
                raw_results=raw_results or [],
                candidate_pool=candidate_pool or [],
                ranked_papers=[],
                output_k=100,
                path="diagnose_recall_loss.csv"
            )
        except Exception as e:
            print(f"  ⚠️ 导出诊断报告失败: {e}")

        # LLM领域扩展 = retrieval rounds used (检索轮数)
        llm_domain_expand = budget.llm_calls_used
        # 硬编码领域词 = 0（当前无硬编码逻辑）
        hardcoded_domain = 0

        case_result = {
            "qid": qid,
            "name": case.get("name", ""),
            "query": query,
            "gold_count": gold_total,
            "hits": hits_all,
            "pool_recall": round(pool_recall, 4),
            "recall_at_100": round(r100, 4),
            "recall_at_300": round(r300, 4),
            "recall_at_500": round(r500, 4),
            "hits_relaxed": hits_all_rel,
            "pool_recall_relaxed": round(pool_recall_rel, 4),
            "recall_at_100_relaxed": round(r100_rel, 4),
            "recall_at_300_relaxed": round(r300_rel, 4),
            "recall_at_500_relaxed": round(r500_rel, 4),
            "pool_size": pool_size,
            "llm_domain_expand": llm_domain_expand,
            "hardcoded_domain": hardcoded_domain,
            "elapsed_seconds": round(elapsed, 2),
            "error": error_msg if has_error else None,
        }
        results.append(case_result)

        print(f"  -> Pool={pool_size} | Strict命中={hits_all}/{gold_total} | "
              f"Strict Pool Recall={pool_recall:.4f} | R@100={r100:.4f} | R@300={r300:.4f} | R@500={r500:.4f} | "
              f"耗时={elapsed:.1f}s")
        print(f"  -> Relaxed命中={hits_all_rel}/{gold_total_rel} | "
              f"Relaxed Pool Recall={pool_recall_rel:.4f} | R@100={r100_rel:.4f} | R@300={r300_rel:.4f} | R@500={r500_rel:.4f}")

    # 6. 输出汇总表格
    print("\n" + "=" * 100)
    print("📊 [Strict Match] Evaluation Table:")
    print("QID  Gold  命中  Pool Recall  R@100   R@300   R@500   Pool  LLM领域扩展  硬编码领域词  耗时(s)")
    print("-" * 100)
    for r in results:
        print(f"{r['qid']:<5}{r['gold_count']:<6}{r['hits']:<6}"
              f"{r['pool_recall']:<13.4f}"
              f"{r['recall_at_100']:<8.4f}"
              f"{r['recall_at_300']:<8.4f}"
              f"{r['recall_at_500']:<8.4f}"
              f"{r['pool_size']:<6}"
              f"{r['llm_domain_expand']:<12}"
              f"{r['hardcoded_domain']:<13}"
              f"{r['elapsed_seconds']:.1f}")

    print("\n" + "=" * 100)
    print("📊 [Relaxed Match] Evaluation Table:")
    print("QID  Gold  命中  Pool Recall  R@100   R@300   R@500   Pool  LLM领域扩展  硬编码领域词  耗时(s)")
    print("-" * 100)
    for r in results:
        print(f"{r['qid']:<5}{r['gold_count']:<6}{r['hits_relaxed']:<6}"
              f"{r['pool_recall_relaxed']:<13.4f}"
              f"{r['recall_at_100_relaxed']:<8.4f}"
              f"{r['recall_at_300_relaxed']:<8.4f}"
              f"{r['recall_at_500_relaxed']:<8.4f}"
              f"{r['pool_size']:<6}"
              f"{r['llm_domain_expand']:<12}"
              f"{r['hardcoded_domain']:<13}"
              f"{r['elapsed_seconds']:.1f}")

    # 7. 整体统计
    n = len(results)
    micro_pool_recall = total_hits / total_gold if total_gold > 0 else 0.0
    avg_pool_recall = sum(r["pool_recall"] for r in results) / n if n > 0 else 0.0
    avg_r100 = sum(r["recall_at_100"] for r in results) / n if n > 0 else 0.0
    avg_r300 = sum(r["recall_at_300"] for r in results) / n if n > 0 else 0.0
    avg_r500 = sum(r["recall_at_500"] for r in results) / n if n > 0 else 0.0

    micro_pool_recall_rel = total_hits_rel / total_gold_rel if total_gold_rel > 0 else 0.0
    avg_pool_recall_rel = sum(r["pool_recall_relaxed"] for r in results) / n if n > 0 else 0.0
    avg_r100_rel = sum(r["recall_at_100_relaxed"] for r in results) / n if n > 0 else 0.0
    avg_r300_rel = sum(r["recall_at_300_relaxed"] for r in results) / n if n > 0 else 0.0
    avg_r500_rel = sum(r["recall_at_500_relaxed"] for r in results) / n if n > 0 else 0.0

    # 平均耗时 = 所有完成 case 的 elapsed_seconds 之和 / case 数
    avg_latency = sum(r["elapsed_seconds"] for r in results) / n if n > 0 else 0.0

    print("\n整体 [Strict Match]:")
    print(f"  总命中: {total_hits} / {total_gold}")
    print(f"  strict_micro_pool_recall: {micro_pool_recall:.4f}")
    print(f"  strict_avg_pool_recall: {avg_pool_recall:.4f}")
    print(f"  strict_avg_recall_at_100: {avg_r100:.4f}")
    print(f"  strict_avg_recall_at_300: {avg_r300:.4f}")
    print(f"  strict_avg_recall_at_500: {avg_r500:.4f}")
    print(f"  strict_avg_latency_s: {avg_latency:.2f}")

    print("\n整体 [Relaxed Match]:")
    print(f"  总命中: {total_hits_rel} / {total_gold_rel}")
    print(f"  relaxed_micro_pool_recall: {micro_pool_recall_rel:.4f}")
    print(f"  relaxed_avg_pool_recall: {avg_pool_recall_rel:.4f}")
    print(f"  relaxed_avg_recall_at_100: {avg_r100_rel:.4f}")
    print(f"  relaxed_avg_recall_at_300: {avg_r300_rel:.4f}")
    print(f"  relaxed_avg_recall_at_500: {avg_r500_rel:.4f}")
    print(f"  relaxed_avg_latency_s: {avg_latency:.2f}")
    print("=" * 100)

    # 8. 保存 JSON 结果（文件名含时间和样本数）
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"pool_recall_{now_str}_n{n_samples}.json"
    output_path = Path(output_filename)

    output_data = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "dataset": str(dataset_path),
            "selected_qids": original_ids,
            "providers": provider_names,
            "timeout_per_query": timeout_per_query,
        },
        "summary": {
            "total_hits": total_hits,
            "total_gold": total_gold,
            "micro_pool_recall": round(micro_pool_recall, 4),
            "avg_pool_recall": round(avg_pool_recall, 4),
            "avg_recall_at_100": round(avg_r100, 4),
            "avg_recall_at_300": round(avg_r300, 4),
            "avg_recall_at_500": round(avg_r500, 4),
            "avg_latency_s": round(avg_latency, 2),
        },
        "cases": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 结果已保存到: {output_path.resolve()}")


if __name__ == "__main__":
    # Windows UTF-8 支持
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    setup_logging(logging.INFO)

    parser = argparse.ArgumentParser(description="Pool Recall 评估 (Semantic Scholar API)")
    parser.add_argument("-n", "--samples", type=int, default=10,
                        help="随机抽样数量 (默认 10)")
    parser.add_argument("-t", "--timeout", type=float, default=300.0,
                        help="单条 query 超时上限 (秒, 默认 300)")
    parser.add_argument("-d", "--dataset", type=str, default="data/RealScholarQuery_test.jsonl",
                        help="数据集路径 (默认 data/RealScholarQuery_test.jsonl)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (固定 QID 选择，便于对比)")
    parser.add_argument("--config", type=str, default=None,
                        help="指定 YAML 配置文件路径（如 configs/live_fast.yaml），默认使用内置参数")
    args = parser.parse_args()

    run_pool_recall_eval(n_samples=args.samples, timeout_per_query=args.timeout, dataset=args.dataset, seed=args.seed, config_path=args.config)
