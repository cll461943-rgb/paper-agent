#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2 Ablation Runner — 运行 5 个消融级别并汇总对比表。

用法:
  # 在 dev 集上跑全部 5 个消级别
  python run_ablation.py --split dev

  # 在 test 集上只跑 local_only 和 full
  python run_ablation.py --split test --modes local_only,full

  # 指定超时
  python run_ablation.py --split dev --timeout 120

消融级别:
  1. local_only     — 仅 pasa_local, 无 LLM, 无查询扩展
  2. openalex       — pasa_local + openalex, 无 LLM
  3. s2_health      — + semantic_scholar + SourceHealthManager
  4. llm_verifier   — + LLM 验证 (无查询扩展)
  5. full           — 全部 P2 特性
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

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.infra import (
    MockLLMClient,
    OpenAICompatibleLLMClient,
    load_config,
    setup_logging,
)
from scholar_agent.retrieval import build_providers
from scholar_agent.workflow.pipeline import PaperAgentPipeline

# 复用 evaluate.py 的匹配算法
from evaluate import score_papers_against_gold, parse_cases_arg

ABLATION_MODES = {
    "local_only": "configs/ablation_local_only.yaml",
    "openalex": "configs/ablation_openalex.yaml",
    "s2_health": "configs/ablation_s2_health.yaml",
    "llm_verifier": "configs/ablation_llm_verifier.yaml",
    "full": "configs/ablation_full.yaml",
}


def load_split() -> dict:
    split_path = Path("configs/eval_split.json")
    if not split_path.exists():
        print(f"❌ Split file not found: {split_path}", file=sys.stderr)
        sys.exit(1)
    with open(split_path) as f:
        return json.load(f)


def load_cases(qid_list: list[int]) -> list[dict]:
    dataset_path = Path("data/RealScholarQuery_test.jsonl")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    all_cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                all_cases.append({
                    "name": raw.get("qid") or f"case-{len(all_cases)+1}",
                    "query": raw.get("question") or raw.get("query") or "",
                    "gold": raw.get("answer") or raw.get("answers") or [],
                    "k": 20,
                })
    # qid_list is 1-indexed
    return [all_cases[qid - 1] for qid in qid_list if 1 <= qid <= len(all_cases)]


def run_single_ablation(
    mode_name: str,
    config_path: str,
    eval_cases: list[dict],
    qid_list: list[int],
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Run a single ablation mode and return summary metrics."""
    print(f"\n{'='*70}")
    print(f"🔬 Ablation: {mode_name} (config: {config_path})")
    print(f"{'='*70}")

    config = load_config(config_path)
    config.app.mode = "live"

    # Respect YAML provider settings
    enabled_providers = []
    for prov_name in ("pasa_local", "openalex", "arxiv", "semantic_scholar", "pubmed"):
        prov_cfg = getattr(config.providers, prov_name, None)
        if prov_cfg and getattr(prov_cfg, "enabled", False):
            enabled_providers.append(prov_name)
    if enabled_providers:
        config.app.providers = enabled_providers
    else:
        config.app.providers = ["pasa_local"]

    providers = build_providers(config)
    provider_names = [p.name for p in providers]
    print(f"  Providers: {provider_names}")
    print(f"  LLM enabled: {config.llm.enabled}")
    print(f"  Query expansion: {getattr(config.budget, 'enable_query_expansion', False)}")

    total_f1 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    total_cand_rec = 0.0
    total_time = 0.0
    errors = 0
    case_results = []

    for idx, (case, qid) in enumerate(zip(eval_cases, qid_list), 1):
        query = case["query"]
        gold = case["gold"]
        k = int(case.get("k", 20))

        print(f"\n  [{idx}/{len(eval_cases)}] QID={qid} | Gold={len(gold)} | Q: \"{query[:60]}...\"")

        from scholar_agent.workflow.budget import BudgetManager
        budget = BudgetManager(config)

        if config.llm.enabled and config.app.mode == "live":
            llm_client = OpenAICompatibleLLMClient(config.llm, budget)
        else:
            llm_client = MockLLMClient(budget)

        pipeline = PaperAgentPipeline(config, llm_client, providers)

        start_t = time.perf_counter()
        has_error = False
        try:
            res = pipeline.run(query)
            result_papers = [rp.paper for rp in (res.highly_relevant_papers + res.partially_relevant_papers)]
            candidate_pool = getattr(pipeline, "candidate_pool", result_papers)
        except Exception as exc:
            has_error = True
            errors += 1
            print(f"    ❌ Error: {exc}")
            result_papers = []
            candidate_pool = getattr(pipeline, "candidate_pool", [])
            res = None

        elapsed = time.perf_counter() - start_t
        total_time += elapsed

        if not has_error:
            final_scores = score_papers_against_gold(result_papers[:k], gold)
            cand_scores = score_papers_against_gold(candidate_pool, gold)
            total_f1 += final_scores["f1"]
            total_prec += final_scores["precision"]
            total_rec += final_scores["recall"]
            total_cand_rec += cand_scores["recall"]
            print(f"    -> P={final_scores['precision']:.4f} R={final_scores['recall']:.4f} F1={final_scores['f1']:.4f} | PoolRec={cand_scores['recall']:.4f} | {elapsed:.1f}s")
            case_results.append({
                "qid": qid,
                "precision": final_scores["precision"],
                "recall": final_scores["recall"],
                "f1": final_scores["f1"],
                "pool_recall": cand_scores["recall"],
                "elapsed": round(elapsed, 2),
            })
        else:
            case_results.append({"qid": qid, "precision": 0, "recall": 0, "f1": 0, "pool_recall": 0, "elapsed": round(elapsed, 2), "error": True})

    n = len(eval_cases)
    summary = {
        "mode": mode_name,
        "config": config_path,
        "providers": provider_names,
        "n_cases": n,
        "avg_f1": round(total_f1 / n, 4) if n else 0,
        "avg_precision": round(total_prec / n, 4) if n else 0,
        "avg_recall": round(total_rec / n, 4) if n else 0,
        "avg_pool_recall": round(total_cand_rec / n, 4) if n else 0,
        "avg_time": round(total_time / n, 2) if n else 0,
        "errors": errors,
        "cases": case_results,
    }

    print(f"\n  📊 {mode_name} Summary: F1={summary['avg_f1']:.4f} P={summary['avg_precision']:.4f} R={summary['avg_recall']:.4f} PoolRec={summary['avg_pool_recall']:.4f} Time={summary['avg_time']:.1f}s Errors={errors}")
    return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_logging(logging.WARNING)

    parser = argparse.ArgumentParser(description="P2 Ablation Runner")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Use dev or test split")
    parser.add_argument("--modes", type=str, default=None, help="Comma-separated ablation modes (default: all)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-query timeout (seconds)")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    split = load_split()
    qid_list = split[f"{args.split}_qids"]
    print(f"Split: {args.split} | QIDs: {qid_list} | Count: {len(qid_list)}")

    eval_cases = load_cases(qid_list)
    if not eval_cases:
        print("❌ No cases loaded.", file=sys.stderr)
        sys.exit(1)

    if args.modes:
        mode_names = [m.strip() for m in args.modes.split(",") if m.strip() in ABLATION_MODES]
    else:
        mode_names = list(ABLATION_MODES.keys())

    all_results = []
    for mode_name in mode_names:
        config_path = ABLATION_MODES[mode_name]
        if not Path(config_path).exists():
            print(f"⚠️ Config not found: {config_path}, skipping {mode_name}")
            continue
        result = run_single_ablation(mode_name, config_path, eval_cases, qid_list, args.timeout)
        all_results.append(result)

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"📊 Ablation Comparison Table ({args.split} split, {len(qid_list)} cases)")
    print(f"{'='*90}")
    print(f"{'Mode':<18} {'F1':>8} {'Prec':>8} {'Rec':>8} {'PoolRec':>8} {'Time':>8} {'Errors':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['mode']:<18} {r['avg_f1']:>8.4f} {r['avg_precision']:>8.4f} {r['avg_recall']:>8.4f} {r['avg_pool_recall']:>8.4f} {r['avg_time']:>7.1f}s {r['errors']:>8}")
    print("=" * 90)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "split": args.split,
                "qids": qid_list,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
