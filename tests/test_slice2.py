#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slice 2 单测：SelectionResult schema + batch_evidence_selector strict 分数"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.models.schemas import SelectionResult, EvidenceItem, Paper, QueryPlan
from scholar_agent.selection.evidence_selector import _safe_score, _fallback_numeric_scores, _create_fallback_selection


def test_safe_score():
    """_safe_score 正常值、None、非法值"""
    assert _safe_score(0.85, 0.5) == 0.85
    assert _safe_score(None, 0.5) == 0.5
    assert _safe_score("abc", 0.5) == 0.5
    assert _safe_score(1.5, 0.5) == 1.0  # clamped
    assert _safe_score(-0.3, 0.5) == 0.0  # clamped
    print("✅ test_safe_score passed")


def test_fallback_numeric_scores():
    """_fallback_numeric_scores 各 level 返回正确"""
    assert _fallback_numeric_scores("high") == (0.90, 0.75, 0.70)
    assert _fallback_numeric_scores("medium") == (0.55, 0.45, 0.45)
    assert _fallback_numeric_scores("low") == (0.20, 0.20, 0.15)
    assert _fallback_numeric_scores("irrelevant") == (0.0, 0.0, 0.0)
    print("✅ test_fallback_numeric_scores passed")


def test_selection_result_new_fields():
    """SelectionResult 新字段默认值和赋值"""
    # 默认值
    sr = SelectionResult(
        paper_id="test-1",
        relevance_level="high",
        reason="test",
    )
    assert sr.relevance_score is None
    assert sr.constraint_score is None
    assert sr.evidence_score is None
    assert sr.uncertainty == []

    # 赋值
    sr2 = SelectionResult(
        paper_id="test-2",
        relevance_level="medium",
        reason="test",
        relevance_score=0.65,
        constraint_score=0.40,
        evidence_score=0.50,
        uncertainty=["abstract too short"],
    )
    assert sr2.relevance_score == 0.65
    assert sr2.constraint_score == 0.40
    assert sr2.evidence_score == 0.50
    assert sr2.uncertainty == ["abstract too short"]
    print("✅ test_selection_result_new_fields passed")


def test_fallback_selection_has_numeric_scores():
    """_create_fallback_selection 返回的 SelectionResult 含数值分数"""
    paper = Paper(
        paper_id="p1",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        abstract="We introduce a new language representation model called BERT.",
        year=2019,
        citation_count=50000,
    )
    plan = QueryPlan(
        original_query="papers about BERT",
        query_type="specific_query",
        methods=["BERT"],
        datasets=[],
        entities=[],
    )
    result = _create_fallback_selection(paper, plan)
    assert result.relevance_score is not None
    assert result.constraint_score is not None
    assert result.evidence_score is not None
    assert result.uncertainty == []

    # BERT in title → keyword match → should be at least medium
    assert result.relevance_level in {"high", "medium"}
    assert 0.0 <= result.relevance_score <= 1.0
    print(f"✅ test_fallback_selection_has_numeric_scores passed (level={result.relevance_level}, rel_score={result.relevance_score})")


def test_batch_selector_parses_numeric_scores():
    """batch_select_and_extract_evidence 解析 LLM 返回的数值分数"""
    from scholar_agent.selection.batch_evidence_selector import batch_select_and_extract_evidence

    paper = Paper(
        paper_id="p1",
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex neural networks.",
        year=2017,
        citation_count=100000,
    )
    plan = QueryPlan(
        original_query="transformer attention mechanism",
        query_type="specific_query",
        methods=["transformer"],
        datasets=[],
        entities=[],
    )

    # Mock LLM client that returns numeric scores
    class MockLLMClient:
        def complete_json(self, system_prompt, user_prompt, model_type="flash"):
            return {
                "selections": [
                    {
                        "paper_id": "p1",
                        "relevance_level": "high",
                        "relevance_score": 0.92,
                        "constraint_score": 0.85,
                        "evidence_score": 0.80,
                        "matched_constraints": ["transformer"],
                        "missing_constraints": [],
                        "evidence": [{"field": "title", "text": "Attention Is All You Need"}],
                        "reason": "Directly about transformer attention",
                        "confidence": 0.95,
                        "uncertainty": ["abstract may not cover all constraints"],
                    }
                ]
            }

    results = batch_select_and_extract_evidence([paper], plan, MockLLMClient())
    assert len(results) == 1
    r = results[0]
    assert r.relevance_level == "high"
    assert r.relevance_score == 0.92
    assert r.constraint_score == 0.85
    assert r.evidence_score == 0.80
    assert r.uncertainty == ["abstract may not cover all constraints"]
    print(f"✅ test_batch_selector_parses_numeric_scores passed (rel={r.relevance_score}, const={r.constraint_score}, ev={r.evidence_score})")


def test_batch_selector_fallback_on_invalid_json():
    """LLM 返回无效 JSON 时 fallback 填入默认数值分数"""
    from scholar_agent.selection.batch_evidence_selector import batch_select_and_extract_evidence

    paper = Paper(
        paper_id="p1",
        title="Some Random Paper",
        abstract="A paper about something else entirely.",
        year=2020,
        citation_count=5,
    )
    plan = QueryPlan(
        original_query="transformer attention mechanism",
        query_type="specific_query",
        methods=["transformer"],
        datasets=[],
        entities=[],
    )

    # Mock LLM client that returns None (simulating failure)
    class MockLLMClient:
        def complete_json(self, system_prompt, user_prompt, model_type="flash"):
            return None

    results = batch_select_and_extract_evidence([paper], plan, MockLLMClient())
    assert len(results) == 1
    r = results[0]
    # Fallback should have numeric scores
    assert r.relevance_score is not None
    assert r.constraint_score is not None
    assert r.evidence_score is not None
    # "transformer" not in title/abstract → low
    assert r.relevance_level == "low"
    assert r.relevance_score == 0.20  # fallback for "low"
    print(f"✅ test_batch_selector_fallback_on_invalid_json passed (level={r.relevance_level}, rel_score={r.relevance_score})")


if __name__ == "__main__":
    test_safe_score()
    test_fallback_numeric_scores()
    test_selection_result_new_fields()
    test_fallback_selection_has_numeric_scores()
    test_batch_selector_parses_numeric_scores()
    test_batch_selector_fallback_on_invalid_json()
    print("\n🎉 All Slice 2 tests passed!")
