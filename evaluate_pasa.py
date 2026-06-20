#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.evaluation.pasa import (  # noqa: E402
    cutoff_from_source_meta,
    filter_by_cutoff,
    gold_coverage,
    load_pasa_jsonl,
    macro_average,
    recall_at_k,
    score_pasa,
)
from scholar_agent.infra import MockLLMClient, OpenAICompatibleLLMClient, load_config, setup_logging  # noqa: E402
from scholar_agent.retrieval import build_providers  # noqa: E402
from scholar_agent.workflow.pipeline import PaperAgentPipeline  # noqa: E402
from scholar_agent.workflow.budget import BudgetManager  # noqa: E402


def _select_mode(explicit: str | None) -> str:
    if explicit:
        return explicit
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key and Path(".env").exists():
        match = re.search(r"DEEPSEEK_API_KEY\s*=\s*([^\s#]+)", Path(".env").read_text(encoding="utf-8"))
        api_key = match.group(1).strip("'\"") if match else ""
    return "live" if api_key and "mock" not in api_key.lower() else "mock"


def _build_llm_client(mode: str, config: Any) -> Any:
    budget = BudgetManager(config)
    if mode == "mock":
        return MockLLMClient(budget)
    return OpenAICompatibleLLMClient(config.llm, budget)


def _build_optional_llm_client(mode: str, config: Any, disable_llm: bool) -> Any:
    if disable_llm:
        return None
    return _build_llm_client(mode, config)


def _apply_budget_overrides(
    config: Any,
    *,
    max_llm_calls: int | None = None,
    max_api_calls: int | None = None,
    max_search_queries: int | None = None,
    max_retrieval_rounds: int | None = None,
    max_results_per_query: int | None = None,
    max_seed_papers: int | None = None,
) -> None:
    overrides = {
        "max_llm_calls": max_llm_calls,
        "max_api_calls": max_api_calls,
        "max_search_queries": max_search_queries,
        "max_retrieval_rounds": max_retrieval_rounds,
        "max_results_per_query": max_results_per_query,
        "max_seed_papers": max_seed_papers,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config.budget, name, value)


def _prepare_config(args: argparse.Namespace) -> tuple[Any, str]:
    config = load_config(args.config)
    mode = _select_mode(args.mode)
    config.app.mode = mode
    if args.parallel_retrieval:
        config.app.parallel_retrieval = True
    config.known_title_clues.enabled = bool(args.enable_known_title_clues)
    if args.enable_pasa_local_fts:
        config.providers.pasa_local.enable_fts = True
    if args.providers:
        config.app.providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    elif mode == "live":
        config.app.providers = ["pasa_local", "openalex", "arxiv", "pubmed"]
    else:
        config.app.providers = ["mock"]
    _apply_budget_overrides(
        config,
        max_llm_calls=args.max_llm_calls,
        max_api_calls=args.max_api_calls,
        max_search_queries=args.max_search_queries,
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_results_per_query=args.max_results_per_query,
        max_seed_papers=args.max_seed_papers,
    )
    return config, mode


def _select_case_groups(
    cases: Sequence[Any],
    *,
    sample_size: int | None = None,
    sample_runs: int = 1,
    random_seed: int | None = None,
) -> list[list[int]]:
    indexes = list(range(len(cases)))
    if sample_size is None:
        return [indexes]
    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")
    if sample_size > len(indexes):
        raise ValueError(f"--sample-size={sample_size} exceeds dataset size {len(indexes)}")
    if sample_runs <= 0:
        raise ValueError("--sample-runs must be greater than 0")
    rng = random.Random(random_seed)
    return [rng.sample(indexes, sample_size) for _ in range(sample_runs)]


def _case_cutoff(case: Any, args: argparse.Namespace) -> Any:
    if args.no_time_cutoff:
        return None
    return cutoff_from_source_meta(case.raw.get("source_meta") or {}, days=args.cutoff_days)


def _empty_case_log(
    case: Any,
    *,
    dataset_index: int,
    run_index: int,
    case_index: int,
    args: argparse.Namespace,
    error: str,
    timed_out: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    cutoff = _case_cutoff(case, args)
    candidate_score = score_pasa([], case.answer)
    return {
        "run_index": run_index,
        "case_index": case_index,
        "dataset_index": dataset_index + 1,
        "question": case.question,
        "success": False,
        "timed_out": timed_out,
        "error": error,
        "candidate_count": 0,
        "cutoff_date": cutoff.isoformat() if cutoff else None,
        "candidate_pool": candidate_score,
        "gold_coverage": gold_coverage([], case.answer),
        "recall_at_20": 0.0,
        "recall_at_50": 0.0,
        "recall_at_100": 0.0,
        "recall_at_300": 0.0,
        "recall_at_500": 0.0,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "run_metrics": None,
        "search_queries": [],
    }


def _result_search_queries(result: Any | None) -> list[str]:
    if result is None:
        return []
    queries: list[str] = []
    for search_round in getattr(result, "search_process", []) or []:
        for query in getattr(search_round, "queries", []) or []:
            if query:
                queries.append(str(query))
    return queries


def _run_single_case(
    case: Any,
    *,
    dataset_index: int,
    run_index: int,
    case_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, mode = _prepare_config(args)
    providers = build_providers(config, mode=mode, provider_names=config.app.providers)
    llm_client = _build_optional_llm_client(mode, config, args.disable_llm)
    pipeline = PaperAgentPipeline(config, llm_client, providers)
    error = None
    result = None
    candidate_pool = []
    try:
        result = pipeline.run(case.question, retrieval_only=True)
        candidate_pool = getattr(pipeline, "candidate_pool", [])
    except Exception as exc:
        error = str(exc)
        candidate_pool = getattr(pipeline, "candidate_pool", [])

    cutoff = _case_cutoff(case, args)
    candidate_pool = filter_by_cutoff(candidate_pool, cutoff)
    candidate_score = score_pasa(candidate_pool, case.answer)
    run_metrics = result.run_metrics.model_dump(mode="json") if result is not None else None
    return {
        "run_index": run_index,
        "case_index": case_index,
        "dataset_index": dataset_index + 1,
        "question": case.question,
        "success": error is None,
        "timed_out": False,
        "error": error,
        "candidate_count": len(candidate_pool),
        "cutoff_date": cutoff.isoformat() if cutoff else None,
        "candidate_pool": candidate_score,
        "gold_coverage": gold_coverage(candidate_pool, case.answer),
        "recall_at_20": round(recall_at_k(candidate_pool, case.answer, 20), 4),
        "recall_at_50": round(recall_at_k(candidate_pool, case.answer, 50), 4),
        "recall_at_100": round(recall_at_k(candidate_pool, case.answer, 100), 4),
        "recall_at_300": round(recall_at_k(candidate_pool, case.answer, 300), 4),
        "recall_at_500": round(recall_at_k(candidate_pool, case.answer, 500), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "run_metrics": run_metrics,
        "search_queries": _result_search_queries(result),
    }


def _case_worker(
    case: Any,
    dataset_index: int,
    run_index: int,
    case_index: int,
    args: argparse.Namespace,
    result_queue: Any,
) -> None:
    try:
        result_queue.put(
            {
                "case_log": _run_single_case(
                    case,
                    dataset_index=dataset_index,
                    run_index=run_index,
                    case_index=case_index,
                    args=args,
                )
            }
        )
    except BaseException as exc:
        result_queue.put({"worker_error": str(exc), "traceback": traceback.format_exc()})


def _run_case_with_timeout(
    case: Any,
    *,
    dataset_index: int,
    run_index: int,
    case_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.case_timeout_seconds:
        return _run_single_case(
            case,
            dataset_index=dataset_index,
            run_index=run_index,
            case_index=case_index,
            args=args,
        )

    started = time.perf_counter()
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_case_worker,
        args=(case, dataset_index, run_index, case_index, args, result_queue),
    )
    process.start()
    process.join(args.case_timeout_seconds)
    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join(5)
        return _empty_case_log(
            case,
            dataset_index=dataset_index,
            run_index=run_index,
            case_index=case_index,
            args=args,
            error=f"case timed out after {args.case_timeout_seconds}s",
            timed_out=True,
            elapsed_seconds=elapsed,
        )

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        return _empty_case_log(
            case,
            dataset_index=dataset_index,
            run_index=run_index,
            case_index=case_index,
            args=args,
            error=f"case worker exited without result (exitcode={process.exitcode})",
            timed_out=False,
            elapsed_seconds=elapsed,
        )
    if "worker_error" in payload:
        return _empty_case_log(
            case,
            dataset_index=dataset_index,
            run_index=run_index,
            case_index=case_index,
            args=args,
            error=payload["worker_error"],
            timed_out=False,
            elapsed_seconds=elapsed,
        )
    return payload["case_log"]


def _case_summary(case_logs: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_scores = [item["candidate_pool"] for item in case_logs]
    recall_at_20 = [float(item["recall_at_20"]) for item in case_logs]
    recall_at_50 = [float(item["recall_at_50"]) for item in case_logs]
    recall_at_100 = [float(item["recall_at_100"]) for item in case_logs]
    recall_at_300 = [float(item["recall_at_300"]) for item in case_logs]
    recall_at_500 = [float(item["recall_at_500"]) for item in case_logs]
    return {
        "n_cases": len(case_logs),
        "n_success": sum(1 for item in case_logs if item["success"]),
        "n_timeouts": sum(1 for item in case_logs if item.get("timed_out")),
        "candidate_pool_macro": macro_average(candidate_scores),
        "recall_at_20": round(sum(recall_at_20) / len(recall_at_20), 4) if recall_at_20 else 0.0,
        "recall_at_50": round(sum(recall_at_50) / len(recall_at_50), 4) if recall_at_50 else 0.0,
        "recall_at_100": round(sum(recall_at_100) / len(recall_at_100), 4) if recall_at_100 else 0.0,
        "recall_at_300": round(sum(recall_at_300) / len(recall_at_300), 4) if recall_at_300 else 0.0,
        "recall_at_500": round(sum(recall_at_500) / len(recall_at_500), 4) if recall_at_500 else 0.0,
        "avg_elapsed_seconds": round(
            sum(float(item["elapsed_seconds"]) for item in case_logs) / len(case_logs), 4
        )
        if case_logs
        else 0.0,
    }


def _average_run_summaries(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_summaries:
        return {
            "candidate_pool_macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "recall_at_100": 0.0,
            "recall_at_300": 0.0,
            "recall_at_500": 0.0,
        }

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "candidate_pool_macro": {
            "precision": mean([float(item["candidate_pool_macro"]["precision"]) for item in run_summaries]),
            "recall": mean([float(item["candidate_pool_macro"]["recall"]) for item in run_summaries]),
            "f1": mean([float(item["candidate_pool_macro"]["f1"]) for item in run_summaries]),
        },
        "recall_at_20": mean([float(item["recall_at_20"]) for item in run_summaries]),
        "recall_at_50": mean([float(item["recall_at_50"]) for item in run_summaries]),
        "recall_at_100": mean([float(item["recall_at_100"]) for item in run_summaries]),
        "recall_at_300": mean([float(item["recall_at_300"]) for item in run_summaries]),
        "recall_at_500": mean([float(item["recall_at_500"]) for item in run_summaries]),
    }


def run_pasa_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    config, mode = _prepare_config(args)
    cases = load_pasa_jsonl(args.dataset, limit=args.limit)
    case_groups = _select_case_groups(
        cases,
        sample_size=args.sample_size,
        sample_runs=args.sample_runs,
        random_seed=args.random_seed,
    )

    all_case_logs: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for run_index, group in enumerate(case_groups, start=1):
        run_case_logs: list[dict[str, Any]] = []
        for case_index, dataset_index in enumerate(group, start=1):
            case = cases[dataset_index]
            case_log = _run_case_with_timeout(
                case,
                dataset_index=dataset_index,
                run_index=run_index,
                case_index=case_index,
                args=args,
            )
            run_case_logs.append(case_log)
            all_case_logs.append(case_log)
            status = "timeout" if case_log.get("timed_out") else "ok" if case_log["success"] else "error"
            print(
                f"[run {run_index}/{len(case_groups)} case {case_index}/{len(group)} dataset#{dataset_index + 1}] "
                f"status={status} cand_recall={case_log['candidate_pool']['recall']:.4f} "
                f"r@20={case_log['recall_at_20']:.4f} r@50={case_log['recall_at_50']:.4f} "
                f"r@100={case_log['recall_at_100']:.4f} r@300={case_log['recall_at_300']:.4f} "
                f"r@500={case_log['recall_at_500']:.4f} candidates={case_log['candidate_count']} "
                f"elapsed={case_log['elapsed_seconds']:.1f}s"
            )
        run_summary = _case_summary(run_case_logs)
        run_summary["run_index"] = run_index
        run_summary["sampled_dataset_indexes"] = [index + 1 for index in group]
        run_summaries.append(run_summary)

    averaged = _average_run_summaries(run_summaries)
    all_case_summary = _case_summary(all_case_logs)
    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "mode": mode,
        "providers": config.app.providers,
        "parallel_retrieval": config.app.parallel_retrieval,
        "known_title_clues_enabled": config.known_title_clues.enabled,
        "pasa_local_fts_enabled": config.providers.pasa_local.enable_fts,
        "llm_enabled": not args.disable_llm,
        "time_cutoff_days": None if args.no_time_cutoff else args.cutoff_days,
        "aggregation": "mean_of_sample_run_macros",
        "sample_size": args.sample_size,
        "sample_runs": len(case_groups),
        "random_seed": args.random_seed,
        "case_timeout_seconds": args.case_timeout_seconds,
        "request_limits": {
            "max_llm_calls": config.budget.max_llm_calls,
            "max_api_calls": config.budget.max_api_calls,
            "max_search_queries": config.budget.max_search_queries,
            "max_retrieval_rounds": config.budget.max_retrieval_rounds,
            "max_results_per_query": config.budget.max_results_per_query,
            "max_seed_papers": config.budget.max_seed_papers,
        },
        "n_cases": all_case_summary["n_cases"],
        "n_success": all_case_summary["n_success"],
        "n_timeouts": all_case_summary["n_timeouts"],
        "candidate_pool_macro": averaged["candidate_pool_macro"],
        "recall_at_20": averaged["recall_at_20"],
        "recall_at_50": averaged["recall_at_50"],
        "recall_at_100": averaged["recall_at_100"],
        "recall_at_300": averaged["recall_at_300"],
        "recall_at_500": averaged["recall_at_500"],
        "avg_elapsed_seconds": all_case_summary["avg_elapsed_seconds"],
        "all_case_macro": {
            "candidate_pool_macro": all_case_summary["candidate_pool_macro"],
            "recall_at_20": all_case_summary["recall_at_20"],
            "recall_at_50": all_case_summary["recall_at_50"],
            "recall_at_100": all_case_summary["recall_at_100"],
            "recall_at_300": all_case_summary["recall_at_300"],
            "recall_at_500": all_case_summary["recall_at_500"],
        },
    }
    output = {"summary": summary, "runs": run_summaries, "cases": all_case_logs}
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="PaSa-style retrieval-only evaluator.")
    parser.add_argument("--dataset", required=True, help="Path to AutoScholarQuery/RealScholarQuery jsonl file.")
    parser.add_argument("--config", default=None, help="Config yaml path. Defaults to configs/default.yaml.")
    parser.add_argument("--mode", choices=["mock", "live"], default=None)
    parser.add_argument("--providers", default=None, help="Comma-separated provider names.")
    parser.add_argument("--parallel-retrieval", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None, help="Random cases per sampled run.")
    parser.add_argument("--sample-runs", type=int, default=1, help="Number of random sampled runs.")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for reproducible random sampling.")
    parser.add_argument("--case-timeout-seconds", type=float, default=None, help="Wall-clock timeout per case.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--cutoff-days", type=int, default=7)
    parser.add_argument("--no-time-cutoff", action="store_true")
    parser.add_argument("--enable-known-title-clues", action="store_true")
    parser.add_argument("--enable-pasa-local-fts", action="store_true")
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-api-calls", type=int, default=None)
    parser.add_argument("--max-search-queries", type=int, default=None)
    parser.add_argument("--max-retrieval-rounds", type=int, default=None)
    parser.add_argument("--max-results-per-query", type=int, default=None)
    parser.add_argument("--max-seed-papers", type=int, default=None)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    output = run_pasa_evaluation(args)
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
