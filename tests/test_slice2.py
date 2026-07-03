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
        def complete_json(self, system_prompt, user_prompt, model_type="flash", **kwargs):
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
        def complete_json(self, system_prompt, user_prompt, model_type="flash", **kwargs):
            return None

    results = batch_select_and_extract_evidence([paper], plan, MockLLMClient())
    assert len(results) == 1
    r = results[0]
    # Fallback should have numeric scores
    assert r.relevance_score is not None
    assert r.constraint_score is not None
    assert r.evidence_score is not None
    # "transformer" not in title/abstract and only weak quality priors -> irrelevant
    assert r.relevance_level == "irrelevant"
    assert r.relevance_score is not None and 0.0 <= r.relevance_score < 0.08
    print(f"✅ test_batch_selector_fallback_on_invalid_json passed (level={r.relevance_level}, rel_score={r.relevance_score})")


def test_batch_selector_local_fallback_uses_entities_as_constraints():
    """LLM 不可用时，entities 也必须参与本地 matched/missing 与 final score。"""
    from scholar_agent.selection.batch_evidence_selector import batch_select_and_extract_evidence
    from scholar_agent.ranking.final_reranker import compute_paper_score

    plan = QueryPlan(
        original_query=(
            "How can deep neural networks enhance real-time facial recognition performance "
            "when a person is partially occluded, such as wearing a mask?"
        ),
        query_type="specific_query",
        methods=["deep neural network"],
        datasets=[],
        entities=["face recognition", "masked face recognition", "occluded face recognition"],
    )
    gold_like = Paper(
        paper_id="gold-like",
        title="Efficient Masked Face Recognition Method during the COVID-19 Pandemic",
        abstract="We propose an efficient masked face recognition method for occluded face recognition.",
        year=2022,
        citation_count=25,
        retrieval_path=["provider:openalex", "route:core_topic"],
    )
    broad_noise = Paper(
        paper_id="broad-noise",
        title="Deep learning for AI",
        abstract="A general overview of deep learning applications.",
        year=2022,
        citation_count=25,
        retrieval_path=["provider:openalex", "route:core_topic"],
    )

    results = batch_select_and_extract_evidence(
        [broad_noise, gold_like],
        plan,
        llm_client=None,
        original_query=plan.original_query,
    )
    by_id = {result.paper_id: result for result in results}

    assert "masked face recognition" in by_id["gold-like"].matched_constraints
    assert "face recognition" in by_id["gold-like"].matched_constraints
    assert len(by_id["gold-like"].matched_constraints) > len(by_id["broad-noise"].matched_constraints)

    gold_score, gold_subscores = compute_paper_score(gold_like, by_id["gold-like"])
    noise_score, noise_subscores = compute_paper_score(broad_noise, by_id["broad-noise"])
    assert gold_subscores["Constraint_Coverage"] > noise_subscores["Constraint_Coverage"]
    assert gold_score > noise_score
    print("✅ test_batch_selector_local_fallback_uses_entities_as_constraints passed")


if __name__ == "__main__":
    test_safe_score()
    test_fallback_numeric_scores()
    test_selection_result_new_fields()
    test_fallback_selection_has_numeric_scores()
    test_batch_selector_parses_numeric_scores()
    test_batch_selector_fallback_on_invalid_json()
    test_batch_selector_local_fallback_uses_entities_as_constraints()
    print("\n🎉 All Slice 2 tests passed!")
