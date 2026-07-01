#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三阶段逐步评测脚本  (Phase 1 -> Phase 2 -> Phase 3)

Phase 1: 检索 + 去重候选池召回率
Phase 2: Evidence Selection 送选候选的召回率 & 精确率
Phase 3: 精排 + F1-beta 截断后最终推荐的 F1 / 召回率 / 精确率 及推荐 K 值

运行: .venv\Scripts\python scripts\three_phase_eval.py
"""
from __future__ import annotations
import json, logging, os, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholar_agent.infra import OpenAICompatibleLLMClient, load_config
from scholar_agent.workflow.budget import BudgetManager
from scholar_agent.retrieval import build_providers
from scholar_agent.workflow.pipeline import PaperAgentPipeline

# =========================================================
# 匹配 / 评分函数 (来自 evaluate.py)
# =========================================================

def _norm_id(v):
    if not v: return None
    t = str(v).strip().lower()
    for pat, rep in [
        (r"^https?://(dx\.)?doi\.org/", ""),
        (r"^doi:", ""),
        (r"^https?://arxiv\.org/(abs|pdf)/", ""),
        (r"\.pdf$", ""),
        (r"v\d+$", ""),
    ]:
        t = re.sub(pat, rep, t)
    return t.strip() or None

def _norm_title(v):
    if not v: return None
    t = re.sub(r"[^a-z0-9]+", " ", str(v).lower()).strip()
    return re.sub(r"\s+", " ", t) or None

def paper_keys(p) -> set:
    meta = getattr(p, "metadata", {}) or {}
    keys = {
        _norm_id(getattr(p, "paper_id", None)),
        _norm_id(getattr(p, "doi", None)),
        _norm_id(getattr(p, "arxiv_id", None)),
        _norm_title(getattr(p, "title", None)),
    }
    for k in ("corpus_id", "corpusId"):
        v = meta.get(k)
        if v: keys.add(f"corpus:{str(v).strip().lower()}")
    return {k for k in keys if k}

def build_gold_sets(entry: dict) -> list[set]:
    """
    Build one key-set per gold paper.
    Merge title + its corresponding arxiv_id into the SAME set.
    """
    titles   = entry.get("answer", [])
    aids     = entry.get("answer_arxiv_id", [])
    gsets: list[set] = []
    for i, title in enumerate(titles):
        keys: set = set()
        n = _norm_title(title)
        if n: keys.add(n)
        if i < len(aids):
            n2 = _norm_id(aids[i])
            if n2: keys.add(n2)
        if keys:
            gsets.append(keys)
    # Extra arxiv_ids beyond the title list
    for aid in aids[len(titles):]:
        n = _norm_id(aid)
        if n: gsets.append({n})
    return gsets

def score_pool(papers, gold_sets) -> dict:
    matched: set[int] = set()
    for p in papers:
        keys = paper_keys(p)
        for i, g in enumerate(gold_sets):
            if i not in matched and keys & g:
                matched.add(i)
    tp   = len(matched)
    pred = len(papers)
    gold = len(gold_sets)
    prec = tp / pred if pred else 0.0
    rec  = tp / gold if gold else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return dict(tp=tp, pred=pred, gold=gold,
                precision=round(prec,4), recall=round(rec,4), f1=round(f1,4))

def from_ranked(ranked):
    return [getattr(rp, "paper", rp) for rp in ranked]

# =========================================================
# 主评测函数
# =========================================================

def evaluate_case(entry: dict, pipeline: PaperAgentPipeline, label: str) -> dict:
    query     = entry["question"]
    gold_sets = build_gold_sets(entry)
    gold_n    = len(entry.get("answer", []))

    print(f"\n{'='*68}")
    print(f"  {label}  |  QID: {entry.get('qid','?')}")
    print(f"  Gold数 = {gold_n}  |  Gold keys = {len(gold_sets)} 组")
    print(f"  Query  : {query[:100]}")
    print(f"{'='*68}")

    t0 = time.time()
    result = pipeline.run(query)
    elapsed = round(time.time() - t0, 1)

    # ---- Phase 1: 完整检索去重候选池 --------------------------------
    pool1 = getattr(pipeline, "candidate_pool", []) or []
    sc1   = score_pool(pool1, gold_sets)
    print(f"\n【Phase 1 — 检索召回】")
    print(f"  候选池 {len(pool1)} 篇  |  TP={sc1['tp']} / Gold={sc1['gold']}")
    print(f"  Recall={sc1['recall']:.4f}   Precision={sc1['precision']:.4f}   F1={sc1['f1']:.4f}")

    # ---- Phase 2: Evidence Selector 输入候选 ------------------------
    pool2 = getattr(pipeline, "selection_candidates", []) or []
    sc2   = score_pool(pool2, gold_sets)
    print(f"\n【Phase 2 — 筛选候选（Evidence Selector 输入）】")
    print(f"  筛选候选 {len(pool2)} 篇  |  TP={sc2['tp']} / Gold={sc2['gold']}")
    print(f"  Recall={sc2['recall']:.4f}   Precision={sc2['precision']:.4f}   F1={sc2['f1']:.4f}")

    # ---- Phase 3: 精排 + K 截断最终推荐 ----------------------------
    ranked  = getattr(pipeline, "ranked_papers", []) or []
    high    = result.highly_relevant_papers or []
    partial = result.partially_relevant_papers or []
    final   = high + partial
    k_high  = len(high)
    k_ext   = len(final)

    sc3_all   = score_pool(from_ranked(ranked), gold_sets)
    sc3_final = score_pool(from_ranked(final),  gold_sets)

    print(f"\n【Phase 3 — 精排 + K 截断最终推荐】")
    print(f"  精排全列表 ({len(ranked)} 篇): Recall={sc3_all['recall']:.4f}  Prec={sc3_all['precision']:.4f}  F1={sc3_all['f1']:.4f}")
    print(f"  最终推荐 K={k_ext}  (高相关K1={k_high} / 部分相关={len(partial)})")
    print(f"    TP={sc3_final['tp']}  Gold={sc3_final['gold']}")
    print(f"    F1={sc3_final['f1']:.4f}   Recall={sc3_final['recall']:.4f}   Precision={sc3_final['precision']:.4f}")

    if high:
        print(f"\n  ▶ 高度相关 ({k_high} 篇):")
        for rp in high[:6]:
            p = getattr(rp, "paper", rp)
            print(f"    [{getattr(rp,'final_score',0):.3f}] {getattr(p,'title','?')[:68]}")
    if partial:
        print(f"\n  ▶ 部分相关 ({len(partial)} 篇):")
        for rp in partial[:5]:
            p = getattr(rp, "paper", rp)
            print(f"    [{getattr(rp,'final_score',0):.3f}] {getattr(p,'title','?')[:68]}")

    print(f"\n  ⏱ 耗时: {elapsed}s")
    if result.run_metrics and result.run_metrics.errors:
        errs = result.run_metrics.errors
        print(f"  ⚠ {len(errs)} 个错误:")
        for e in errs[:5]:
            print(f"    - {e}")

    return {
        "label": label,
        "qid": entry.get("qid", ""),
        "gold_count": gold_n,
        "query": query[:80],
        "phase1": {"pool_size": len(pool1), **sc1},
        "phase2": {"pool_size": len(pool2), **sc2},
        "phase3": {
            "ranked_size": len(ranked),
            "k_high": k_high, "k_ext": k_ext,
            "full_ranked": sc3_all,
            "final_output": sc3_final,
        },
        "elapsed_s": elapsed,
    }

# =========================================================
# 入口
# =========================================================

def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    dev_path = "data/benchmarks/AutoScholarQuery_dev.jsonl"
    data = [json.loads(l) for l in open(dev_path, encoding="utf-8")]

    from collections import Counter
    dist = Counter(len(d.get("answer", [])) for d in data)
    print(f"[数据集] 共 {len(data)} 条，Gold 数分布: {sorted(dist.items())[:10]}")

    moderate = [d for d in data if 2 <= len(d.get("answer", [])) <= 5]
    print(f"[抽样] 从 {len(moderate)} 条 gold=2~5 中随机抽取 3 条")
    random.seed(42)
    cases = random.sample(moderate, min(3, len(moderate)))
    for i, c in enumerate(cases):
        print(f"  Case {i+1}: {c.get('qid','')} | gold={len(c.get('answer',[]))} | {c['question'][:70]}")

    # --- 初始化（关键：mode 必须设为 live）---
    config = load_config("configs/default.yaml")
    providers = build_providers(config, mode="live")   # <-- 强制 live 模式
    print(f"\n[初始化] 加载了 {len(providers)} 个检索提供商:")
    for p in providers:
        print(f"  - {type(p).__name__}")

    results = []
    for i, entry in enumerate(cases):
        budget = BudgetManager(config)
        llm    = OpenAICompatibleLLMClient(config.llm, budget)
        pipe   = PaperAgentPipeline(config, llm_client=llm, providers=providers)
        try:
            r = evaluate_case(entry, pipe, f"Case {i+1}/3")
            results.append(r)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"[ERROR] Case {i+1} failed: {exc}")

    # --- 汇总 ---
    print(f"\n\n{'='*72}")
    print("  三阶段评测汇总表")
    print(f"{'='*72}")
    print(f"{'Case':<11} {'Gold':>4} | {'P1 Rec':>7} | "
          f"{'P2 Rec':>7} {'P2 Prec':>8} | "
          f"{'P3 F1':>7} {'P3 Rec':>7} {'P3 Prec':>8} {'K':>5}")
    print("-"*78)
    for r in results:
        p1 = r["phase1"]; p2 = r["phase2"]; p3 = r["phase3"]["final_output"]
        print(f"{r['label']:<11} {r['gold_count']:>4} | "
              f"{p1['recall']:>7.4f} | "
              f"{p2['recall']:>7.4f} {p2['precision']:>8.4f} | "
              f"{p3['f1']:>7.4f} {p3['recall']:>7.4f} {p3['precision']:>8.4f} "
              f"{r['phase3']['k_ext']:>5}")
    if results:
        def _avg(key, sub): return sum(r[sub][key] for r in results) / len(results)
        def _avg3(key): return sum(r["phase3"]["final_output"][key] for r in results) / len(results)
        print("-"*78)
        print(f"{'平均':<11} {'':>4} | "
              f"{_avg('recall','phase1'):>7.4f} | "
              f"{_avg('recall','phase2'):>7.4f} {_avg('precision','phase2'):>8.4f} | "
              f"{_avg3('f1'):>7.4f} {_avg3('recall'):>7.4f} {_avg3('precision'):>8.4f}")

    Path("outputs").mkdir(exist_ok=True)
    out = "outputs/three_phase_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 详细报告已保存到 {out}")

if __name__ == "__main__":
    main()
