from scholar_agent.models.schemas import Paper, SelectionResult
from scholar_agent.ranking.final_reranker import compute_paper_score


def _paper(paper_id: str, local_score: float) -> Paper:
    return Paper(
        paper_id=paper_id,
        title="Cerebrospinal fluid amyloid beta and tau biomarkers",
        abstract="Alzheimer disease progression prediction using CSF biomarkers.",
        retrieval_path=["provider:openalex", "route:title_like", "route:broad_synonym"],
        metadata={"constraint_aware_local_score": local_score},
    )


def _selection(paper_id: str) -> SelectionResult:
    return SelectionResult(
        paper_id=paper_id,
        relevance_level="medium",
        matched_constraints=["cerebrospinal fluid biomarkers", "tau protein"],
        missing_constraints=[],
        evidence=[],
        reason="local fallback",
    )


def test_fallback_final_score_uses_constraint_aware_local_prerank():
    low_score, _ = compute_paper_score(_paper("low", 0.10), _selection("low"))
    high_score, subscores = compute_paper_score(_paper("high", 0.90), _selection("high"))

    assert subscores["Local_PreRank"] == 0.90
    assert high_score > low_score
