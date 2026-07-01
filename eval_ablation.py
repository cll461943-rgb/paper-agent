#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect-First Ablation Evaluation Script (Task 6)

Runs A0-A5 ablation levels on the 5-case benchmark (QID 2, 8, 18, 41, 48).
Outputs: avg_f1, avg_precision, avg_recall, candidate_recall@500,
         selector_recall, final_transfer_rate, avg_wall_time,
         LLM success rate, LLM timeout rate, parse failure rate.

Ablation levels:
  A0: local-only selector + local rerank (no LLM)
  A1: LLM pointwise selector + local rerank (LLM evidence selection, no listwise)
  A2: local selector + LLM listwise rerank (no LLM evidence selection)
  A3: LLM selector + LLM listwise rerank (full effect-first)
  A4: A3 + local validator (validation always on → A4=A3)
  A5: A4 + LLM result review + strategy optimization (max_llm_calls=60)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
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
from scholar_agent.workflow.budget import BudgetManager

# Reuse scoring functions from evaluate.py
from evaluate import (
    score_papers_against_gold,
    paper_match_keys,
    gold_match_keys,
    StatsLLMClient,
)

DEFAULT_CASES = "2,8,18,41,48"
DEFAULT_CONFIG = "configs/effect_first.yaml"


def _get_ablation_config(base_config: Any, level: str) -> Any:
    """Create a config copy modified for the given ablation level."""
    cfg = deepcopy(base_config)

    if level == "A0":
        # No LLM at all
        cfg.llm.enabled = False
    elif level == "A1":
        # LLM evidence selection only, no listwise
        cfg.ranking.listwise_topk = 0
        cfg.ranking.listwise_fallback_topk = 0
    elif level == "A2":
        # LLM listwise only, local evidence selection
        cfg.budget.max_llm_selection_papers = 0
    elif level == "A3":
        # Full effect-first (both LLM stages)
        pass
    elif level == "A4":
        # A3 + local validator (validation always on → same as A3)
        pass
    elif level == "A5":
        # A4 + LLM result review + strategy optimization
        cfg.budget.max_llm_calls = 60
    else:
        raise ValueError(f"Unknown ablation level: {level}")

    return cfg


def run_ablation(
    level: str,
    config_path: str,
    cases_str: str,
    mode: str = "live",
) -> dict[str, Any]:
    """Run a single ablation level and return metrics."""
    print(f"\n{'='*70}")
    print(f"  Ablation {level} — config={config_path}, mode={mode}")
    print(f"{'='*70}")

    base_config = load_config(config_path)
    config = _get_ablation_config(base_config, level)
    config.app.mode = mode

    # Build providers list from config
    if mode == "live":
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

    # Load dataset
    dataset_path = Path("data/RealScholarQuery_test.jsonl")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
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
                })

    # Parse case indices
    from evaluate import parse_cases_arg
    indices = parse_cases_arg(cases_str, len(all_cases))
    eval_cases = [all_cases[i] for i in indices]
    print(f"Cases: {[i+1 for i in indices]} ({len(eval_cases)} total)")

    # Build providers
    providers = build_providers(config)

    # Stats accumulators
    total_f1 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    total_cand_rec_500 = 0.0
    total_selector_recall = 0.0
    total_transfer_rate = 0.0
    total_wall_time = 0.0
    total_llm_success_rate = 0.0
    total_llm_timeout_rate = 0.0
    total_llm_parse_failure_rate = 0.0
    n_cases = len(eval_cases)
    has_llm_stats = 0  # count cases where LLM stats were available

    case_results = []

    for idx, case in enumerate(eval_cases, start=1):
        query = case["query"]
        gold = case["gold"]
        original_idx = all_cases.index(case) + 1
        print(f"\n  [{idx}/{n_cases}] Case #{original_idx}: {query[:70]}...")

        budget = BudgetManager(config)
        if mode == "mock":
            raw_llm_client = MockLLMClient(budget)
        else:
            raw_llm_client = OpenAICompatibleLLMClient(config.llm, budget)

        llm_client = StatsLLMClient(raw_llm_client, {})
        pipeline = PaperAgentPipeline(config, llm_client, providers)

        start_time = time.perf_counter()
        has_error = False
        try:
            res = pipeline.run(query)
            result_papers = [
                rp.paper for rp in (res.highly_relevant_papers + res.partially_relevant_papers)
            ]
            candidate_pool = getattr(pipeline, "candidate_pool", result_papers)
        except Exception as exc:
            has_error = True
            print(f"    ERROR: {exc}")
            res = None
            result_papers = []
            candidate_pool = []

        elapsed = time.perf_counter() - start_time
        total_wall_time += elapsed

        if not has_error:
            final_scores = score_papers_against_gold(result_papers[:500], gold)
            cand_scores_500 = score_papers_against_gold(candidate_pool[:500], gold)

            selection_candidates = getattr(pipeline, "selection_candidates", [])
            sel_gold = score_papers_against_gold(selection_candidates, gold)
            pool_gold = score_papers_against_gold(candidate_pool, gold)

            selector_recall = 0.0
            if pool_gold["gold_count"] > 0:
                selector_recall = sel_gold["true_positive"] / pool_gold["gold_count"]
            transfer_rate = 0.0
            if pool_gold["true_positive"] > 0:
                transfer_rate = final_scores["true_positive"] / pool_gold["true_positive"]

            total_f1 += final_scores["f1"]
            total_prec += final_scores["precision"]
            total_rec += final_scores["recall"]
            total_cand_rec_500 += cand_scores_500["recall"]
            total_selector_recall += selector_recall
            total_transfer_rate += transfer_rate

            print(f"    F1={final_scores['f1']:.4f} Prec={final_scores['precision']:.4f} "
                  f"Rec={final_scores['recall']:.4f} | CandRec@500={cand_scores_500['recall']:.4f} "
                  f"| SelRec={selector_recall:.4f} | Transfer={transfer_rate:.4f} "
                  f"| {elapsed:.1f}s")

            # Collect LLM stats from evidence selector and listwise reranker
            from scholar_agent.selection.batch_evidence_selector import get_last_stats as get_ev_stats
            from scholar_agent.ranking.llm_listwise_reranker import get_last_stats as get_lr_stats

            ev_stats = get_ev_stats()
            lr_stats = get_lr_stats()

            # Merge stats
            if ev_stats.get("total_batches", 0) > 0:
                total_llm_success_rate += ev_stats["success_rate"]
                total_llm_timeout_rate += ev_stats["timeout_rate"]
                total_llm_parse_failure_rate += ev_stats["parse_failure_rate"]
                has_llm_stats += 1
            elif lr_stats.get("attempts", 0) > 0:
                # Use listwise stats if evidence selector had no batches
                success = lr_stats.get("successes", 0)
                attempts = lr_stats.get("attempts", 1)
                total_llm_success_rate += success / attempts
                total_llm_timeout_rate += lr_stats.get("timeouts", 0) / attempts
                total_llm_parse_failure_rate += lr_stats.get("parse_failures", 0) / attempts
                has_llm_stats += 1
            else:
                # No LLM stats for this case (A0 or all LLM skipped)
                pass

            case_results.append({
                "case": original_idx,
                "f1": final_scores["f1"],
                "precision": final_scores["precision"],
                "recall": final_scores["recall"],
                "cand_rec_500": cand_scores_500["recall"],
                "selector_recall": selector_recall,
                "transfer_rate": transfer_rate,
                "wall_time": round(elapsed, 2),
                "ev_stats": ev_stats,
                "lr_stats": lr_stats,
            })
        else:
            case_results.append({"case": original_idx, "error": True, "wall_time": round(elapsed, 2)})

    n_success = max(n_cases, 1)
    n_llm = max(has_llm_stats, 1)

    summary = {
        "level": level,
        "avg_f1": round(total_f1 / n_success, 4),
        "avg_precision": round(total_prec / n_success, 4),
        "avg_recall": round(total_rec / n_success, 4),
        "candidate_recall_500": round(total_cand_rec_500 / n_success, 4),
        "selector_recall": round(total_selector_recall / n_success, 4),
        "final_transfer_rate": round(total_transfer_rate / n_success, 4),
        "avg_wall_time": round(total_wall_time / n_success, 2),
        "llm_success_rate": round(total_llm_success_rate / n_llm, 4) if has_llm_stats > 0 else "N/A",
        "llm_timeout_rate": round(total_llm_timeout_rate / n_llm, 4) if has_llm_stats > 0 else "N/A",
        "llm_parse_failure_rate": round(total_llm_parse_failure_rate / n_llm, 4) if has_llm_stats > 0 else "N/A",
        "cases": case_results,
    }

    return summary


def print_comparison_table(results: list[dict]) -> None:
    """Print a comparison table of all ablation levels."""
    print(f"\n{'='*100}")
    print("  ABLATION COMPARISON TABLE (Effect-First)")
    print(f"{'='*100}")
    header = f"  {'Level':<6} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'CandR@500':<10} | {'SelRec':<8} | {'Transfer':<8} | {'Time(s)':<8} | {'LLM Succ':<9} | {'Timeout':<8} | {'ParseFl':<8}"
    print(header)
    print(f"  {'-'*96}")
    for r in results:
        llm_s = r.get("llm_success_rate", "N/A")
        llm_t = r.get("llm_timeout_rate", "N/A")
        llm_p = r.get("llm_parse_failure_rate", "N/A")
        llm_s_str = f"{llm_s:.4f}" if isinstance(llm_s, float) else llm_s
        llm_t_str = f"{llm_t:.4f}" if isinstance(llm_t, float) else llm_t
        llm_p_str = f"{llm_p:.4f}" if isinstance(llm_p, float) else llm_p
        print(f"  {r['level']:<6} | {r['avg_f1']:<8.4f} | {r['avg_precision']:<8.4f} | {r['avg_recall']:<8.4f} | "
              f"{r['candidate_recall_500']:<10.4f} | {r['selector_recall']:<8.4f} | {r['final_transfer_rate']:<8.4f} | "
              f"{r['avg_wall_time']:<8.2f} | {llm_s_str:<9} | {llm_t_str:<8} | {llm_p_str:<8}")
    print(f"{'='*100}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    setup_logging(logging.WARNING)  # Reduce noise during ablation

    parser = argparse.ArgumentParser(description="Effect-First Ablation Evaluation")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
    parser.add_argument("-c", "--cases", default=DEFAULT_CASES, help="Case IDs (default: 2,8,18,41,48)")
    parser.add_argument("-m", "--mode", default=None, help="Run mode (live/mock)")
    parser.add_argument("--levels", default="A0,A1,A2,A3,A4,A5", help="Ablation levels to run")
    parser.add_argument("-o", "--output", default=None, help="Save results JSON to path")

    args = parser.parse_args()

    # Detect mode
    run_mode = args.mode
    if run_mode is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key and Path(".env").exists():
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    import re
                    match = re.search(r"DEEPSEEK_API_KEY\s*=\s*([^\s#]+)", f.read())
                    if match:
                        api_key = match.group(1).strip("'\"")
            except Exception:
                pass
        run_mode = "live" if api_key and "mock" not in api_key.lower() else "mock"

    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    all_results = []

    for level in levels:
        result = run_ablation(level, args.config, args.cases, mode=run_mode)
        all_results.append(result)

    print_comparison_table(all_results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")
