import pytest
from scholar_agent.models.schemas import Paper, QueryPlan, SelectionResult, EvidenceItem
from scholar_agent.selection.evidence_validator import validate_evidence, validate_selections


def test_validate_evidence_valid():
    # 正常符合情况
    paper = Paper(
        paper_id="paper_1",
        title="Deep learning for paper retrieval",
        abstract="We introduce a new deep learning method for scholarly paper retrieval.",
        year=2025,
    )
    plan = QueryPlan(
        original_query="scholarly paper retrieval",
        time_range={"start_year": 2024, "end_year": 2026},
        must_have_constraints=["scholarly"],
    )
    selection = SelectionResult(
        paper_id="paper_1",
        relevance_level="high",
        matched_constraints=["scholarly"],
        evidence=[EvidenceItem(field="abstract", text="deep learning method for scholarly paper retrieval")],
        reason="Matches scholarly query.",
        confidence=0.9,
    )

    validated = validate_evidence(paper, selection, plan)
    assert validated.is_validated is True
    assert validated.relevance_level == "high"


def test_validate_evidence_invalid_year():
    # 年份不符
    paper = Paper(
        paper_id="paper_2",
        title="Old paper retrieval",
        abstract="A methods study from long ago.",
        year=2020,  # 2020 不在 [2024, 2026] 内
    )
    plan = QueryPlan(
        original_query="retrieval",
        time_range={"start_year": 2024, "end_year": 2026},
    )
    selection = SelectionResult(
        paper_id="paper_2",
        relevance_level="high",
        evidence=[],
        reason="Matches query.",
    )

    validated = validate_evidence(paper, selection, plan)
    assert validated.is_validated is False
    assert validated.relevance_level == "low"  # 降级
    assert any("year" in note.lower() for note in validated.validation_notes)


def test_validate_evidence_hallucinated_evidence():
    # 虚假证据 (不在标题和摘要中)
    paper = Paper(
        paper_id="paper_3",
        title="Agent paper retrieval",
        abstract="We use agents.",
        year=2025,
    )
    plan = QueryPlan(
        original_query="agent",
        time_range={"start_year": 2024, "end_year": 2026},
    )
    selection = SelectionResult(
        paper_id="paper_3",
        relevance_level="high",
        evidence=[EvidenceItem(field="abstract", text="this sentence is not in abstract")],
        reason="Matches query.",
    )

    validated = validate_evidence(paper, selection, plan)
    assert validated.is_validated is False
    assert validated.relevance_level in ("medium", "low")  # 应该降级
    assert any("evidence" in note.lower() for note in validated.validation_notes)


def test_validate_evidence_missing_must_have():
    # 缺少 must_have 限制条件
    paper = Paper(
        paper_id="paper_4",
        title="Agent paper retrieval",
        abstract="We use agents.",
        year=2025,
    )
    plan = QueryPlan(
        original_query="agent",
        time_range={"start_year": 2024, "end_year": 2026},
        must_have_constraints=["Reranking"],  # 这个词没有在 paper 中出现
    )
    selection = SelectionResult(
        paper_id="paper_4",
        relevance_level="high",
        evidence=[],
        reason="Matches query.",
    )

    validated = validate_evidence(paper, selection, plan)
    assert validated.is_validated is False
    assert validated.relevance_level in ("medium", "low")  # 应该降级
    assert any("must-have" in note.lower() or "constraint" in note.lower() for note in validated.validation_notes)
