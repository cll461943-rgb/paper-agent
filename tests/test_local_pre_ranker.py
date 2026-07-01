"""Unit tests for local_pre_ranker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.models.schemas import Paper, QueryPlan
from scholar_agent.ranking.local_pre_ranker import (
    local_pre_rank,
    local_pre_rank_score,
    _title_similarity,
    _constraint_coverage,
    _citation_score,
    _recency_score,
    _route_prior,
    _source_agreement,
)


def make_paper(
    paper_id: str,
    title: str,
    abstract: str = "",
    year: int = 2024,
    citation_count: int = 10,
    retrieval_path: list[str] | None = None,
    bge_score: float | None = None,
    specter_score: float | None = None,
) -> Paper:
    metadata = {}
    if bge_score is not None:
        metadata["bge_score"] = bge_score
    if specter_score is not None:
        metadata["specter_score"] = specter_score
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year,
        citation_count=citation_count,
        retrieval_path=retrieval_path or [],
        metadata=metadata,
    )


def test_missing_bge_no_fake():
    """Missing BGE should NOT default to 0.5 — feature dropped, weights renormalized."""
    plan = QueryPlan(original_query="quantum monte carlo neural network")
    paper_no_bge = make_paper("p1", "Neural network quantum Monte Carlo for ground states")
    paper_with_bge = make_paper("p2", "Unrelated paper", bge_score=0.9)

    score_no_bge, subs_no_bge = local_pre_rank_score(paper_no_bge, plan, plan.original_query)
    score_with_bge, subs_with_bge = local_pre_rank_score(paper_with_bge, plan, plan.original_query)

    # BGE should be -1.0 (missing marker) for paper without BGE
    assert subs_no_bge["bge"] == -1.0, f"Expected bge=-1.0, got {subs_no_bge['bge']}"
    assert subs_with_bge["bge"] == 0.9, f"Expected bge=0.9, got {subs_with_bge['bge']}"

    # Score without BGE should still be valid (0-1 range)
    assert 0.0 <= score_no_bge <= 1.0, f"Score out of range: {score_no_bge}"
    print(f"  PASS: missing_bge_no_fake (no_bge={score_no_bge:.3f}, with_bge={score_with_bge:.3f})")


def test_title_exact_route_bonus():
    """Papers with title_exact in retrieval_path should get route_prior bonus."""
    plan = QueryPlan(original_query="attention is all you need")
    paper_exact = make_paper("p1", "Attention Is All You Need", retrieval_path=["provider:openalex", "title_exact"])
    paper_none = make_paper("p2", "Attention Is All You Need", retrieval_path=["provider:openalex"])

    score_exact, subs_exact = local_pre_rank_score(paper_exact, plan, plan.original_query)
    score_none, subs_none = local_pre_rank_score(paper_none, plan, plan.original_query)

    assert subs_exact["route_prior"] > subs_none["route_prior"], \
        f"title_exact should boost route_prior: {subs_exact['route_prior']} vs {subs_none['route_prior']}"
    assert score_exact > score_none, f"title_exact paper should score higher: {score_exact} vs {score_none}"
    print(f"  PASS: title_exact_route_bonus (exact={score_exact:.3f}, none={score_none:.3f})")


def test_topk_truncation():
    """local_pre_rank should return at most topk papers."""
    plan = QueryPlan(original_query="machine learning")
    papers = [make_paper(f"p{i}", f"Paper {i} about machine learning", citation_count=i * 10) for i in range(50)]

    result = local_pre_rank(papers, plan, plan.original_query, topk=10)
    assert len(result) == 10, f"Expected 10 papers, got {len(result)}"

    # Verify top papers are sorted descending by score
    scores = [p.metadata["local_pre_rank_score"] for p in result]
    assert scores == sorted(scores, reverse=True), "Results should be sorted descending"
    print(f"  PASS: topk_truncation (50 -> {len(result)}, scores descending)")


def test_constraint_coverage():
    """Constraint coverage should reflect how many plan constraints appear in paper text."""
    plan = QueryPlan(
        original_query="BERT on SQuAD dataset",
        datasets=["SQuAD"],
        methods=["BERT"],
    )
    paper_hit = make_paper("p1", "BERT fine-tuning on SQuAD for question answering", abstract="We use BERT on SQuAD dataset")
    paper_miss = make_paper("p2", "GPT-3 on WikiText", abstract="We use GPT-3 on WikiText")

    cov_hit = _constraint_coverage(paper_hit, plan)
    cov_miss = _constraint_coverage(paper_miss, plan)

    assert cov_hit > cov_miss, f"Hit paper should have higher coverage: {cov_hit} vs {cov_miss}"
    assert cov_hit >= 0.5, f"Hit paper coverage too low: {cov_hit}"
    print(f"  PASS: constraint_coverage (hit={cov_hit:.3f}, miss={cov_miss:.3f})")


def test_recency_score():
    """Recency score should decay with age."""
    paper_new = make_paper("p1", "New paper", year=2025)
    paper_old = make_paper("p2", "Old paper", year=2010)

    rec_new = _recency_score(paper_new)
    rec_old = _recency_score(paper_old)

    assert rec_new > rec_old, f"Newer paper should have higher recency: {rec_new} vs {rec_old}"
    print(f"  PASS: recency_score (2025={rec_new:.3f}, 2010={rec_old:.3f})")


if __name__ == "__main__":
    print("Running local_pre_ranker unit tests...")
    test_missing_bge_no_fake()
    test_title_exact_route_bonus()
    test_topk_truncation()
    test_constraint_coverage()
    test_recency_score()
    print("\nAll tests passed.")
