#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 dev_1000 基线结果：提取每个 case 的召回率，标注零召回案例。"""
import json, glob, os, sys, random

# 找最新的 pool_recall JSON
files = sorted(glob.glob("pool_recall_*_n30.json"), key=os.path.getmtime)
if not files:
    print("没有找到结果 JSON，任务可能还没完成")
    sys.exit(1)
latest = files[-1]
print(f"分析文件: {latest}\n")

with open(latest, encoding="utf-8") as f:
    data = json.load(f)

# 加载 dev_1000 数据集，建立 qid -> gold 映射
cases_ds = []
with open("data/开发+测试集/CNScholarQuery_ZH_v0.4_dev_1000.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            cases_ds.append(json.loads(line))

# eval 脚本用 1-indexed QID，对应 dataset 的 index+1
# seed=42 抽样 30 条
random.seed(42)
indices = sorted(random.sample(range(len(cases_ds)), 30))
qid_to_case = {}
for i, idx in enumerate(indices):
    qid = idx + 1  # 1-indexed
    qid_to_case[qid] = cases_ds[idx]

print("=" * 90)
print("整体结果")
print("=" * 90)
s = data["summary"]
print(f"Micro Pool Recall: {s['micro_pool_recall']}")
print(f"Avg Pool Recall:   {s['avg_pool_recall']}")
print(f"总命中: {s['total_hits']}/{s['total_gold']}  (达 0.5 需 {s['total_gold']//2+1})")
print(f"R@100: {s['avg_recall_at_100']}  R@300: {s['avg_recall_at_300']}  R@500: {s['avg_recall_at_500']}")
print(f"平均耗时: {s['avg_latency_s']}s")

print("\n" + "=" * 90)
print("每个 case 详情")
print("=" * 90)
print(f"{'QID':<6}{'gold':<6}{'hits':<6}{'recall':<9}{'pool':<7}{'状态':<10}{'query_en(前50字)'}")
print("-" * 90)
zero_cases = []
for c in data["cases"]:
    qid = c["qid"]
    ds = qid_to_case.get(qid, {})
    q_en = ds.get("question_en_original", "").replace("\n", " ")[:50]
    gold_titles = ds.get("answer", [])
    gold_arxiv = ds.get("answer_arxiv_id", [])
    status = "命中" if c["hits"] > 0 else "✗零召回"
    if c["hits"] == 0:
        zero_cases.append((qid, c, ds, gold_titles, gold_arxiv))
    print(f"{qid:<6}{c['gold_count']:<6}{c['hits']:<6}{c['pool_recall']:<9.4f}{c['pool_size']:<7}{status:<10}{q_en}")

print(f"\n零召回案例: {len(zero_cases)}/{len(data['cases'])}")
if zero_cases:
    print("\n" + "=" * 90)
    print("零召回案例的语义桥接分析")
    print("=" * 90)
    for qid, c, ds, gold_titles, gold_arxiv in zero_cases:
        q_en = ds.get("question_en_original", "").replace("\n", " ")[:80]
        print(f"\nQID{qid}: {q_en}")
        for i, (title, aid) in enumerate(zip(gold_titles, gold_arxiv)):
            title = title.replace("\n", " ")[:60]
            print(f"  gold[{i}]: {title} (arxiv={aid})")

# 对照预判
print("\n" + "=" * 90)
print("Single-gold case 命中统计")
print("=" * 90)
single_gold_hits = 0
single_gold_total = 0
for c in data["cases"]:
    if c["gold_count"] == 1:
        single_gold_total += 1
        if c["hits"] > 0:
            single_gold_hits += 1
print(f"Single-gold 命中: {single_gold_hits}/{single_gold_total} = {single_gold_hits/single_gold_total:.4f}" if single_gold_total else "无 single-gold case")

multi_gold = [c for c in data["cases"] if c["gold_count"] > 1]
if multi_gold:
    mg_hits = sum(c["hits"] for c in multi_gold)
    mg_total = sum(c["gold_count"] for c in multi_gold)
    print(f"Multi-gold 命中: {mg_hits}/{mg_total} = {mg_hits/mg_total:.4f}")
