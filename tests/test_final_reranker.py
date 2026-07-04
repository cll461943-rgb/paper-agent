import pytest
from scholar_agent.models.schemas import Paper, SelectionResult, EvidenceItem, RankedPaper
from scholar_agent.ranking.final_reranker import compute_paper_score, rerank_papers


def test_compute_paper_score_defaults():
    # 测试默认兜底计算
    paper = Paper(
        paper_id="paper_1",
        title="Test Title",
        abstract="Test abstract",
        year=2025,
    )
    selection = SelectionResult(
        paper_id="paper_1",
        relevance_level="medium",
        matched_constraints=["constraint_1"],
        missing_constraints=[],
        evidence=[],
        reason="Good",
    )
    
    score, subscores = compute_paper_score(paper, selection)
    assert 0.0 <= score <= 1.0
    assert "LLM_Relevance" in subscores
    assert "Recency" in subscores
    # 2025 年新颖度应为 0.9
    assert abs(subscores["Recency"] - 0.9) < 1e-5


def test_rerank_papers_ordering():
    # 测试重排顺序
    p1 = Paper(
        paper_id="p1",
        title="High relevance recent paper",
        abstract="Very important",
        year=2026,
        citation_count=50,
        retrieval_path=["openalex", "semantic_scholar"],
    )
    s1 = SelectionResult(
        paper_id="p1",
        relevance_level="high",
        matched_constraints=["c1", "c2"],
        missing_constraints=[],
        evidence=[EvidenceItem(field="abstract", text="Very important")],
        reason="Excellent matches",
    )

    p2 = Paper(
        paper_id="p2",
        title="Low relevance old paper",
        abstract="Unimportant",
        year=2018,
        citation_count=2,
        retrieval_path=["openalex"],
    )
    s2 = SelectionResult(
        paper_id="p2",
        relevance_level="low",
        matched_constraints=[],
        missing_constraints=["c1", "c2"],
        evidence=[],
        reason="Poor matches",
    )

    papers = [p2, p1]
    selections = [s2, s1]

    ranked = rerank_papers(papers, selections)
    
    assert len(ranked) == 2
    # p1 应该排第一
    assert ranked[0].paper.paper_id == "p1"
    assert ranked[0].rank == 1
    assert ranked[1].paper.paper_id == "p2"
    assert ranked[1].rank == 2
    assert ranked[0].final_score > ranked[1].final_score


def test_biomedical_route_bonus_requires_specific_treatment_signal():
    base_kwargs = dict(
        abstract="Non-small cell lung cancer treatment study.",
        year=2025,
        citation_count=0,
        retrieval_path=["provider:pubmed", "route:biomedical"],
    )
    specific = Paper(
        paper_id="specific",
        title="RNA therapies for non-small cell lung cancer clinical trials",
        **base_kwargs,
    )
    generic = Paper(
        paper_id="generic",
        title="General review of non-small cell lung cancer treatment",
        **base_kwargs,
    )
    low_specific = Paper(
        paper_id="low-specific",
        title="RNA therapies for non-small cell lung cancer clinical trials",
        **base_kwargs,
    )
    high_selection = SelectionResult(
        paper_id="specific",
        relevance_level="high",
        matched_constraints=["lung cancer"],
        missing_constraints=[],
        evidence=[],
        reason="local fallback",
    )
    low_selection = SelectionResult(
        paper_id="low-specific",
        relevance_level="low",
        matched_constraints=[],
        missing_constraints=[],
        evidence=[],
        reason="not selected",
    )

    specific_score, specific_subscores = compute_paper_score(specific, high_selection)
    generic_score, generic_subscores = compute_paper_score(generic, high_selection)
    _, low_subscores = compute_paper_score(low_specific, low_selection)

    assert specific_subscores["Biomedical_Fallback_Bonus"] == pytest.approx(0.18)
    assert generic_subscores["Biomedical_Fallback_Bonus"] == 0.0
    assert low_subscores["Biomedical_Fallback_Bonus"] == 0.0
    assert specific_score > generic_score
