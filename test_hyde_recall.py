#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test whether longer, more specific search queries (HyDE-style) can find gold
papers on OpenAlex that short keyword queries miss.

For each of 5 zero-recall cases, we test four query variants:
  1. Original (short natural-language question)
  2. LLM-style term mapping (keyword bag)
  3. HyDE-style long hypothetical-abstract query
  4. Gold title itself (control)

Each variant is searched on OpenAlex with limit=50. We check whether any
returned paper matches the gold paper (by arXiv ID or normalized title).
"""

from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Make the project's src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.infra.config import load_config
from scholar_agent.retrieval import build_providers
from scholar_agent.models.schemas import SearchQuery


# ---------------------------------------------------------------------------
# Matching helpers (simplified port of evaluate.py logic)
# ---------------------------------------------------------------------------

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


def _normalize_arxiv(value: str | None) -> str | None:
    """Extract a clean arXiv id like '2012.07805' from various encodings."""
    if not value:
        return None
    text = value.strip().lower()
    # Strip common prefixes
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"^arxiv:", "", text)
    text = re.sub(r"^10\.48550/arxiv\.", "", text)
    text = re.sub(r"^10\.48550/arxiv_", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"v\d+$", "", text)
    text = text.strip()
    return text or None


def paper_matches_gold(paper, gold_arxiv: str, gold_title: str) -> bool:
    """Return True if the paper is the gold paper (arXiv id or title match)."""
    gold_arxiv_norm = _normalize_arxiv(gold_arxiv)
    gold_title_norm = _normalize_title(gold_title)

    # Build all string identifiers we can extract from the paper
    candidate_strings: list[str | None] = [
        paper.doi,
        paper.arxiv_id,
        paper.paper_id,
        paper.url,
    ]
    # Also look inside metadata for anything arxiv-flavoured
    md = paper.metadata or {}
    for v in md.values():
        if isinstance(v, str):
            candidate_strings.append(v)
        elif isinstance(v, list):
            candidate_strings.extend(s for s in v if isinstance(s, str))

    # Check arXiv-id match
    if gold_arxiv_norm:
        for raw in candidate_strings:
            norm = _normalize_arxiv(raw)
            if norm and (norm == gold_arxiv_norm or norm.endswith(gold_arxiv_norm) or gold_arxiv_norm.endswith(norm)):
                return True
            # Also substring check for "2012.07805" inside a DOI string
            if raw and gold_arxiv_norm in raw.lower():
                return True

    # Check normalized-title match
    paper_title_norm = _normalize_title(paper.title)
    if gold_title_norm and paper_title_norm:
        if paper_title_norm == gold_title_norm:
            return True
        # Partial containment either direction (handles subtitle differences)
        if gold_title_norm in paper_title_norm or paper_title_norm in gold_title_norm:
            return True

    return False


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "qid": 143,
        "original": "Which studies used perturbation techniques similar to ours for measuring contamination in the test questions?",
        "llm_mapped": "training data extraction perturbation contamination memorization",
        "hyde": (
            "This paper proposes a method for extracting training data from "
            "large language models, using perturbation techniques to measure "
            "data contamination and memorization in test questions"
        ),
        "gold_title": "Extracting Training Data from Large Language Models",
        "gold_arxiv": "2012.07805",
    },
    {
        "qid": 251,
        "original": "Which papers introduced semi-supervised methods that explore the entire eigenspectrum?",
        "llm_mapped": "semi-supervised eigenspectrum generalized pagerank graph neural network",
        "hyde": (
            "This paper introduces a semi-supervised method that explores the "
            "entire eigenspectrum for graph neural networks, proposing an "
            "adaptive universal generalized PageRank approach for node "
            "classification"
        ),
        "gold_title": "Adaptive Universal Generalized PageRank Graph Neural Network",
        "gold_arxiv": "2006.07988",
    },
    {
        "qid": 655,
        "original": "Which paper provides a discussion on the many-body representation hypothesis in context of voxel and point cloud representations?",
        "llm_mapped": "many-body representation hypothesis voxel point cloud 3D molecules",
        "hyde": (
            "This paper provides a discussion on the many-body representation "
            "hypothesis in the context of voxel and point cloud representations "
            "for tasks on molecules in three dimensions"
        ),
        "gold_title": "ATOM3D: Tasks On Molecules in Three Dimensions",
        "gold_arxiv": "2012.04035",
    },
    {
        "qid": 755,
        "original": "In which works generating sparse graphs resulted in more computationally tractable solutions?",
        "llm_mapped": "sparse graphs neural relational inference interacting systems tractable",
        "hyde": (
            "This paper presents a method for generating sparse graphs that "
            "results in more computationally tractable solutions for neural "
            "relational inference in interacting dynamical systems"
        ),
        "gold_title": "Neural Relational Inference for Interacting Systems",
        "gold_arxiv": "1802.04687",
    },
    {
        "qid": 760,
        "original": "Which works discuss the zero-violation approaches, where methods are initialized in the feasible region, and only updated in ways that are guaranteed not to leave the feasible region?",
        "llm_mapped": "zero constraint violation constrained MDP feasible region safe exploration",
        "hyde": (
            "This paper discusses zero-violation approaches for constrained "
            "Markov decision processes, where methods are initialized in the "
            "feasible region and only updated in ways that are guaranteed not "
            "to leave the feasible region"
        ),
        "gold_title": "Learning Policies with Zero or Bounded Constraint Violation for Constrained MDPs",
        "gold_arxiv": "2106.02684",
    },
]

VARIANT_NAMES = ["original", "llm_mapped", "hyde", "gold_title"]


def make_query(variant_value: str, route: str = "core_topic") -> SearchQuery:
    return SearchQuery(
        query=variant_value,
        route=route,
        intent="keyword",
        required_terms=[],
        optional_terms=[],
        filters={},
        priority=5,
    )


def run_variant(oa, variant_value: str, gold_arxiv: str, gold_title: str, limit: int = 50):
    """Search OpenAlex and return (found: bool, n_results: int, match_info: str)."""
    sq = make_query(variant_value)
    try:
        papers = oa.search(sq, limit=limit)
    except Exception as exc:
        return False, 0, f"ERROR: {exc}"

    for idx, paper in enumerate(papers):
        if paper_matches_gold(paper, gold_arxiv, gold_title):
            return True, len(papers), f"rank={idx + 1} title='{paper.title[:60]}'"
    return False, len(papers), ""


def truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def main() -> None:
    config = load_config("configs/effect_first.yaml")
    providers = build_providers(config, provider_names=["openalex"])
    if not providers:
        print("ERROR: OpenAlex provider not built. Check config / credentials.")
        sys.exit(1)
    oa = providers[0]
    print(f"Using provider: {oa.name}\n")

    # Results table: case -> variant -> (found, n_results, info)
    results: list[dict] = []

    for case in CASES:
        qid = case["qid"]
        print(f"{'=' * 80}")
        print(f"QID {qid}  gold='{truncate(case['gold_title'], 50)}'  arxiv={case['gold_arxiv']}")
        print(f"{'-' * 80}")

        case_row = {"qid": qid, "gold_title": case["gold_title"], "gold_arxiv": case["gold_arxiv"], "variants": {}}

        for vname in VARIANT_NAMES:
            vvalue = case[vname]
            # Use "hyde" route for hyde variant so _sanitize passes it through raw
            route = "hyde" if vname == "hyde" else "core_topic"
            sq = make_query(vvalue, route=route)
            # Temporarily override route on the query object
            # (make_query already sets route, but we want explicit control)
            found, n, info = run_variant(oa, vvalue, case["gold_arxiv"], case["gold_title"], limit=50)
            mark = "FOUND" if found else "miss  "
            label = {"original": "original (short)", "llm_mapped": "LLM term-mapped   ", "hyde": "HyDE-style (long) ", "gold_title": "gold title (ctrl)"}[vname]
            print(f"  [{mark}] {label}  n={n:<3} {info}")
            print(f"         query: {truncate(vvalue, 95)}")
            case_row["variants"][vname] = {"found": found, "n": n, "info": info}
            time.sleep(1.0)  # be polite to OpenAlex

        results.append(case_row)
        print()

    # ----- Summary table -----
    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = f"{'QID':<6}{'original':<12}{'LLM-mapped':<14}{'HyDE-style':<14}{'gold_title':<14}  Gold Paper"
    print(header)
    print("-" * 90)

    totals = {"original": 0, "llm_mapped": 0, "hyde": 0, "gold_title": 0}
    for row in results:
        cells = []
        for vname in VARIANT_NAMES:
            v = row["variants"][vname]
            cells.append("Y" if v["found"] else "-")
            if v["found"]:
                totals[vname] += 1
        line = f"{row['qid']:<6}{cells[0]:<12}{cells[1]:<14}{cells[2]:<14}{cells[3]:<14}  {truncate(row['gold_title'], 40)}"
        print(line)

    print("-" * 90)
    total_line = (
        f"{'TOTAL':<6}"
        f"{totals['original']}/{len(results):<9}"
        f"{totals['llm_mapped']}/{len(results):<11}"
        f"{totals['hyde']}/{len(results):<11}"
        f"{totals['gold_title']}/{len(results):<11}"
    )
    print(total_line)

    print()
    print("Variant hit rates:")
    for vname in VARIANT_NAMES:
        rate = totals[vname] / len(results) * 100
        print(f"  {vname:<14}: {totals[vname]}/{len(results)}  ({rate:.0f}%)")


if __name__ == "__main__":
    main()
