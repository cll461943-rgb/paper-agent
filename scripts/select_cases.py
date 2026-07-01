#!/usr/bin/env python3
"""Select 3 cases with moderate gold count from dev set."""
import json, sys, random

data = [json.loads(l) for l in open('data/benchmarks/AutoScholarQuery_dev.jsonl', encoding='utf-8')]

# Show distribution first
from collections import Counter
dist = Counter()
for d in data:
    gold = d.get('gold_papers', d.get('relevant_papers', []))
    dist[len(gold)] += 1
print("Gold count distribution:", sorted(dist.items()))

moderate = []
for i, d in enumerate(data):
    gold = d.get('gold_papers', d.get('relevant_papers', []))
    # Use flexible range: 2-8 gold papers = moderate
    if 2 <= len(gold) <= 8:
        moderate.append((i, d))

print(f"Found {len(moderate)} cases with 2-8 gold papers")


random.seed(42)
selected = random.sample(moderate, 3)

for idx, (i, d) in enumerate(selected):
    gold = d.get('gold_papers', d.get('relevant_papers', []))
    qid = d.get('id', d.get('query_id', '?'))
    query = d.get('query', '?')[:100]
    print(f"Case {idx+1}: line_idx={i}, id={qid}, gold_count={len(gold)}")
    print(f"  Query: {query}")
    print()
