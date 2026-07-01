#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20例测试集最终评估脚本 — 逐阶段指标 + budget分析 +增量保存
输出:
  - outputs/eval_20cases.json   (结构化数据)
  - outputs/eval_20cases_report.md  (可读报告)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.infra import OpenAICompatibleLLMClient, load_config, setup_logging
from scholar_agent.retrieval import build_providers
from scholar_agent.workflow.pipeline import PaperAgentPipeline
from scholar_agent.workflow.budget import BudgetManager


# ---------- 匹配算法 ----------

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
    gold_key_sets = []
    for item in gold_items:
        keys = gold_match_keys(item)
        if keys:
            gold_key_sets.append(keys)
    paper_key_sets = [paper_match_keys(paper) for paper in papers]
    matched_gold_indexes: set[int] = set()
    for paper_keys in paper_key_sets:
        for index, gold_keys in enumerate(gold_key_sets):
            if index in matched_gold_indexes:
                continue
            if paper_keys & gold_keys:
                matched_gold_indexes.add(index)
                break
    return len(matched_gold_indexes), len(gold_key_sets)

def compute_prf(hits: int, output_total: int, gold_total: int) -> dict:
    p = hits / output_total if output_total > 0 else 0.0
    r = hits / gold_total if gold_total > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "hits": hits, "output_total": output_total, "gold_total": gold_total}


# ---------- 逐阶段 budget 聚合 ----------

def aggregate_stage_budget(component_metrics: list, stage: str) -> dict:
    """按阶段聚合 component_metrics。
    stage: 'retrieval' | 'evidence' | 'synthesis' | 'other'
    """
    total_elapsed = 0.0
    total_llm = 0
    total_api = 0
    total_tokens = 0
    total_queries = 0
    components = []

    for cm in component_metrics:
        name = cm.component.lower()
        if stage == "retrieval" and name.startswith("retrieval."):
            pass
        elif stage == "evidence" and ("evidence" in name or "rerank" in name or "batch" in name):
            pass
        elif stage == "synthesis" and ("synthesis" in name or "llm_pool" in name):
            pass
        elif stage == "other":
            if not (name.startswith("retrieval.") or "evidence" in name or "rerank" in name
                    or "batch" in name or "synthesis" in name or "llm_pool" in name):
                pass
            else:
                continue
        else:
            continue

        total_elapsed += cm.elapsed_seconds
        total_llm += cm.llm_calls_delta
        total_api += cm.api_calls_delta
        total_tokens += cm.token_estimate_delta
        total_queries += cm.search_queries_delta
        components.append({
            "name": cm.component,
            "elapsed": round(cm.elapsed_seconds, 1),
            "llm_calls": cm.llm_calls_delta,
            "api_calls": cm.api_calls_delta,
            "tokens": cm.token_estimate_delta,
        })

    return {
        "elapsed_seconds": round(total_elapsed, 1),
        "llm_calls": total_llm,
        "api_calls": total_api,
        "tokens": total_tokens,
        "search_queries": total_queries,
        "component_count": len(components),
    }


# ---------- 增量保存 ----------

def save_incremental(results: list, path: str):
    """每跑完一条就保存，防止崩溃丢数据。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ---------- 主逻辑 ----------

def run_eval(
    qids: list[int],
    config_path: str = "configs/effect_first.yaml",
    dataset: str = "data/benchmarks/AutoScholarQuery_dev.jsonl",
    output_json: str = "outputs/eval_20cases.json",
) -> list[dict]:
    """对指定 QID 列表跑完整三阶段 pipeline，返回结构化结果。"""

    config = load_config(config_path)
    config.app.mode = "live"

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

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        print(f"数据集不存在: {dataset_path}", file=sys.stderr)
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
                    "qid": raw.get("qid") or len(all_cases) + 1,
                    "query": raw.get("question") or raw.get("query") or "",
                    "gold": gold,
                })

    print("=" * 90)
    print(f"20例最终评估 | 数据集: {dataset_path.name} | QIDs: {qids}")
    print(f"LLM Provider: {os.environ.get('ACTIVE_LLM_PROVIDER', 'deepseek')}")
    print("=" * 90)

    providers = build_providers(config, provider_names=config.app.providers)
    print(f"检索源: {[p.name for p in providers]}")

    # 加载已有结果（支持断点续跑）
    results = []
    if Path(output_json).exists():
        with open(output_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_qids = {r["qid"] for r in results if not r.get("error")}
        print(f"已有 {len(results)} 条结果，已完成 QIDs: {sorted(done_qids)}")
        remaining = [q for q in qids if q not in done_qids]
    else:
        remaining = list(qids)

    for qid in remaining:
        case_idx = qid - 1
        if case_idx < 0 or case_idx >= len(all_cases):
            print(f"QID={qid} 超出范围，跳过")
            continue

        case = all_cases[case_idx]
        query = case["query"]
        gold = case["gold"]
        gold_count = len(gold)

        print(f"\n{'='*80}")
        print(f"QID={qid} | Gold={gold_count} | {query[:100]}")
        print(f"{'='*80}")

        budget = BudgetManager(config)
        llm_client = OpenAICompatibleLLMClient(config.llm, budget)
        pipeline = PaperAgentPipeline(config, llm_client, providers)

        start_t = time.perf_counter()
        error_msg = ""
        try:
            result = pipeline.run(query, retrieval_only=False)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            print(f"  Pipeline异常: {error_msg}")
            traceback.print_exc()
            result = None

        elapsed = time.perf_counter() - start_t

        if result is None:
            entry = {
                "qid": qid, "query": query[:200], "gold_count": gold_count,
                "error": error_msg, "elapsed_seconds": round(elapsed, 1),
            }
            results.append(entry)
            save_incremental(results, output_json)
            continue

        # ---- Stage 1: 检索 ----
        candidate_pool = getattr(pipeline, "candidate_pool", [])
        pool_size = len(candidate_pool)
        pool_hits, _ = count_hits(candidate_pool, gold)
        stage1_prf = compute_prf(pool_hits, pool_size, gold_count)

        print(f"  Stage1 检索: pool={pool_size}, recall={stage1_prf['recall']:.4f}")

        # ---- Stage 2: 证据选择 ----
        selection_candidates = getattr(pipeline, "selection_candidates", [])
        selections = getattr(pipeline, "selections", [])
        ranked_papers = getattr(pipeline, "ranked_papers", [])
        # Stage 2 的 P/R: 在 ranked_papers 中命中了多少 gold
        ranked_hits, _ = count_hits([rp.paper for rp in ranked_papers], gold)
        stage2_prf = compute_prf(ranked_hits, len(ranked_papers), gold_count)

        print(f"  Stage2 证据选择: ranked={len(ranked_papers)}, recall={stage2_prf['recall']:.4f}")

        # ---- Stage 3: 合成输出 ----
        highly = result.highly_relevant_papers
        partial = result.partially_relevant_papers
        final_papers = [rp.paper for rp in highly] + [rp.paper for rp in partial]
        final_hits, _ = count_hits(final_papers, gold)
        stage3_prf = compute_prf(final_hits, len(final_papers), gold_count)

        print(f"  Stage3 合成: K={len(final_papers)}, P={stage3_prf['precision']:.4f}, "
              f"R={stage3_prf['recall']:.4f}, F1={stage3_prf['f1']:.4f}")

        # ---- Budget 聚合 ----
        run_metrics = result.run_metrics
        comp_metrics = run_metrics.component_metrics or []
        stage1_budget = aggregate_stage_budget(comp_metrics, "retrieval")
        stage2_budget = aggregate_stage_budget(comp_metrics, "evidence")
        stage3_budget = aggregate_stage_budget(comp_metrics, "synthesis")
        other_budget = aggregate_stage_budget(comp_metrics, "other")

        total_budget = {
            "elapsed_seconds": round(elapsed, 1),
            "llm_calls": run_metrics.llm_calls_used,
            "llm_elapsed": round(run_metrics.llm_elapsed_seconds, 1),
            "api_calls": run_metrics.api_calls_used,
            "tokens": run_metrics.token_estimate,
            "search_queries": run_metrics.search_queries_used,
        }

        print(f"  Budget: LLM={total_budget['llm_calls']}calls/{total_budget['llm_elapsed']}s, "
              f"API={total_budget['api_calls']}, tokens={total_budget['tokens']}, "
              f"time={total_budget['elapsed_seconds']}s")

        # ---- 输出论文列表 ----
        output_papers = []
        for i, rp in enumerate(highly, 1):
            p = rp.paper
            output_papers.append({
                "rank": i, "tier": "H", "score": round(rp.final_score, 4),
                "title": p.title[:100], "arxiv": p.arxiv_id,
            })
        for i, rp in enumerate(partial, 1):
            p = rp.paper
            output_papers.append({
                "rank": len(highly) + i, "tier": "P", "score": round(rp.final_score, 4),
                "title": p.title[:100], "arxiv": p.arxiv_id,
            })

        entry = {
            "qid": qid,
            "query": query[:200],
            "gold_count": gold_count,
            "gold_titles": [g.get("title", "")[:80] if isinstance(g, dict) else str(g)[:80]
                            for g in gold],
            # Stage 1
            "stage1": {
                "pool_size": pool_size,
                "pool_hits": pool_hits,
                "precision": stage1_prf["precision"],
                "recall": stage1_prf["recall"],
                "f1": stage1_prf["f1"],
                **stage1_budget,
            },
            # Stage 2
            "stage2": {
                "selection_candidates": len(selection_candidates),
                "selections": len(selections),
                "ranked_papers": len(ranked_papers),
                "ranked_hits": ranked_hits,
                "precision": stage2_prf["precision"],
                "recall": stage2_prf["recall"],
                "f1": stage2_prf["f1"],
                **stage2_budget,
            },
            # Stage 3
            "stage3": {
                "highly": len(highly),
                "partial": len(partial),
                "final_output": len(final_papers),
                "final_hits": final_hits,
                "precision": stage3_prf["precision"],
                "recall": stage3_prf["recall"],
                "f1": stage3_prf["f1"],
                "dynamic_k": result.dynamic_k_chosen,
                "tie_break": result.tie_break_reason,
                **stage3_budget,
            },
            "total_budget": total_budget,
            "output_papers": output_papers,
            "error": None,
        }

        results.append(entry)
        save_incremental(results, output_json)
        print(f"  → 已保存 (共 {len(results)} 条)")

    return results


def generate_report(results: list, output_md: str):
    """生成 Markdown 报告。"""
    valid = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    lines = []
    lines.append("# 20例测试集最终评估报告\n")
    lines.append(f"- 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"- 成功: {len(valid)} 例, 失败: {len(failed)} 例\n")

    if valid:
        # 微观平均
        s1_hits = sum(r["stage1"]["pool_hits"] for r in valid)
        s1_pool = sum(r["stage1"]["pool_size"] for r in valid)
        s1_gold = sum(r["gold_count"] for r in valid)

        s2_hits = sum(r["stage2"]["ranked_hits"] for r in valid)
        s2_ranked = sum(r["stage2"]["ranked_papers"] for r in valid)

        s3_hits = sum(r["stage3"]["final_hits"] for r in valid)
        s3_out = sum(r["stage3"]["final_output"] for r in valid)

        total_llm = sum(r["total_budget"]["llm_calls"] for r in valid)
        total_tokens = sum(r["total_budget"]["tokens"] for r in valid)
        total_time = sum(r["total_budget"]["elapsed_seconds"] for r in valid)
        total_api = sum(r["total_budget"]["api_calls"] for r in valid)

        lines.append("## 整体性能\n")
        lines.append("| 指标 | Stage 1 检索 | Stage 2 证据选择 | Stage 3 最终输出 |")
        lines.append("|------|-------------|-----------------|-----------------|")
        lines.append(f"| Precision | {s1_hits/s1_pool:.4f} | {s2_hits/s2_ranked:.4f} | {s3_hits/s3_out:.4f} |")
        lines.append(f"| Recall | {s1_hits/s1_gold:.4f} | {s2_hits/s1_gold:.4f} | {s3_hits/s1_gold:.4f} |")
        s1_f1 = 2*(s1_hits/s1_pool)*(s1_hits/s1_gold)/((s1_hits/s1_pool)+(s1_hits/s1_gold)) if (s1_hits/s1_pool)+(s1_hits/s1_gold) > 0 else 0
        s2_f1 = 2*(s2_hits/s2_ranked)*(s2_hits/s1_gold)/((s2_hits/s2_ranked)+(s2_hits/s1_gold)) if (s2_hits/s2_ranked)+(s2_hits/s1_gold) > 0 else 0
        s3_f1 = 2*(s3_hits/s3_out)*(s3_hits/s1_gold)/((s3_hits/s3_out)+(s3_hits/s1_gold)) if (s3_hits/s3_out)+(s3_hits/s1_gold) > 0 else 0
        lines.append(f"| F1 | {s1_f1:.4f} | {s2_f1:.4f} | {s3_f1:.4f} |")
        lines.append("")

        lines.append("## 整体Budget\n")
        lines.append(f"| 总时间(s) | 总LLM调用 | 总Tokens | 总API调用 | 平均时间(s) | 平均LLM | 平均Tokens |")
        lines.append(f"|-----------|----------|----------|----------|------------|---------|-----------|")
        lines.append(f"| {total_time:.1f} | {total_llm} | {total_tokens} | {total_api} | "
                      f"{total_time/len(valid):.1f} | {total_llm/len(valid):.1f} | {total_tokens/len(valid):.0f} |")
        lines.append("")

    # 逐条详情
    lines.append("## 逐条详情\n")
    lines.append("| QID | Gold | S1 Pool | S1 R | S2 Ranked | S2 R | S3 K | S3 Hits | S3 P | S3 R | S3 F1 | LLM | Tokens | Time(s) |")
    lines.append("|-----|------|---------|------|-----------|------|------|---------|------|------|-------|-----|--------|---------|")
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['qid']} | {r['gold_count']} | ERR | - | - | - | - | - | - | - | - | - | - | {r['elapsed_seconds']:.1f} |")
        else:
            s1 = r["stage1"]
            s2 = r["stage2"]
            s3 = r["stage3"]
            tb = r["total_budget"]
            lines.append(
                f"| {r['qid']} | {r['gold_count']} | {s1['pool_size']} | {s1['recall']:.2f} | "
                f"{s2['ranked_papers']} | {s2['recall']:.2f} | "
                f"{s3['final_output']} | {s3['final_hits']} | {s3['precision']:.2f} | {s3['recall']:.2f} | {s3['f1']:.2f} | "
                f"{tb['llm_calls']} | {tb['tokens']} | {tb['elapsed_seconds']:.1f} |"
            )
    lines.append("")

    # 逐条Budget分解
    lines.append("## 逐条Budget分解（按阶段）\n")
    lines.append("| QID | S1 Time | S1 LLM | S1 Tokens | S2 Time | S2 LLM | S2 Tokens | S3 Time | S3 LLM | S3 Tokens | Total Time | Total LLM | Total Tokens |")
    lines.append("|-----|---------|--------|-----------|---------|--------|-----------|---------|--------|-----------|-----------|-----------|-------------|")
    for r in valid:
        s1 = r["stage1"]
        s2 = r["stage2"]
        s3 = r["stage3"]
        tb = r["total_budget"]
        lines.append(
            f"| {r['qid']} | {s1['elapsed_seconds']:.1f} | {s1['llm_calls']} | {s1['tokens']} | "
            f"{s2['elapsed_seconds']:.1f} | {s2['llm_calls']} | {s2['tokens']} | "
            f"{s3['elapsed_seconds']:.1f} | {s3['llm_calls']} | {s3['tokens']} | "
            f"{tb['elapsed_seconds']:.1f} | {tb['llm_calls']} | {tb['tokens']} |"
        )
    lines.append("")

    # 失败案例
    if failed:
        lines.append("## 失败案例\n")
        for r in failed:
            lines.append(f"- QID={r['qid']}: {r.get('error', 'unknown')}")
        lines.append("")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已保存: {output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="20例最终评估")
    parser.add_argument("--qids", type=int, nargs="+", default=list(range(1, 21)),
                        help="QID列表（默认1-20）")
    parser.add_argument("--config", type=str, default="configs/effect_first.yaml")
    parser.add_argument("--dataset", type=str,
                        default="data/benchmarks/AutoScholarQuery_dev.jsonl")
    parser.add_argument("--output", type=str, default="outputs/eval_20cases.json")
    parser.add_argument("--report", type=str, default="outputs/eval_20cases_report.md")
    args = parser.parse_args()

    results = run_eval(
        qids=args.qids,
        config_path=args.config,
        dataset=args.dataset,
        output_json=args.output,
    )
    generate_report(results, args.report)
