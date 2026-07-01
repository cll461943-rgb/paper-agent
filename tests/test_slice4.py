#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slice 4 单测：final_reranker missing-aware 加权"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.models.schemas import Paper, SelectionResult, EvidenceItem
from scholar_agent.ranking.final_reranker import (
    compute_paper_score,
    _weighted_sum_missing_aware,
    _parse_validation_note_float,
    _unique_provider_agreement,
    _title_route_bonus,
    _calculate_title_similarity,
)


def test_weighted_sum_missing_aware_all_present():
    """所有特征都有值时，正常加权"""
    features = {"a": 0.8, "b": 0.6, "c": 1.0}
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    score, valid = _weighted_sum_missing_aware(features, weights)
    expected = 0.8 * 0.5 / 1.0 + 0.6 * 0.3 / 1.0 + 1.0 * 0.2 / 1.0
    assert abs(score - expected) < 1e-6, f"Expected {expected}, got {score}"
    assert len(valid) == 3
    print(f"✅ test_weighted_sum_missing_aware_all_present passed (score={score:.4f})")


def test_weighted_sum_missing_aware_bge_none():
    """BGE=None 时被丢弃，权重重分配到其他特征"""
    features = {"a": 0.8, "b": None, "c": 1.0}
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    score, valid = _weighted_sum_missing_aware(features, weights)
    # 只有 a 和 c 有效，总权重 = 0.5+0.2 = 0.7
    expected = (0.8 * 0.5 + 1.0 * 0.2) / 0.7
    assert abs(score - expected) < 1e-6, f"Expected {expected}, got {score}"
    assert len(valid) == 2  # b 被丢弃
    assert "b" not in valid
    print(f"✅ test_weighted_sum_missing_aware_bge_none passed (score={score:.4f})")


def test_compute_score_bge_none_no_fake():
    """BGE=None 时不假装，subscores 中 BGE_Reranker = -1.0"""
    paper = Paper(
        paper_id="p1",
        title="Test Paper",
        abstract="Test abstract.",
        year=2023,
        citation_count=100,
        metadata={},  # 无 bge_score
    )
    selection = SelectionResult(
        paper_id="p1",
        relevance_level="high",
        reason="test",
        relevance_score=0.90,
        constraint_score=0.80,
        evidence_score=0.70,
    )
    score, subscores = compute_paper_score(paper, selection)
    assert subscores["BGE_Reranker"] == -1.0, f"Expected -1.0, got {subscores['BGE_Reranker']}"
    assert subscores["LLM_Relevance"] == 0.90
    assert subscores["Constraint_Coverage"] == 0.80
    print(f"✅ test_compute_score_bge_none_no_fake passed (score={score:.4f}, BGE={subscores['BGE_Reranker']})")


def test_compute_score_bge_present():
    """BGE 有值时正常使用"""
    paper = Paper(
        paper_id="p1",
        title="Test Paper",
        abstract="Test abstract.",
        year=2023,
        citation_count=100,
        metadata={"bge_score": 0.85},
    )
    selection = SelectionResult(
        paper_id="p1",
        relevance_level="high",
        reason="test",
        relevance_score=0.90,
        constraint_score=0.80,
        evidence_score=0.70,
    )
    score, subscores = compute_paper_score(paper, selection)
    assert subscores["BGE_Reranker"] == 0.85
    print(f"✅ test_compute_score_bge_present passed (score={score:.4f}, BGE={subscores['BGE_Reranker']})")


def test_title_exact_bonus_high_similarity():
    """title_exact + title_similarity >= 0.92 → bonus=0.12"""
    paper = Paper(
        paper_id="p1",
        title="Attention Is All You Need",
        abstract="Transformer model.",
        year=2017,
        retrieval_path=["provider:arxiv", "title_exact:attention is all you need"],
    )
    selection = SelectionResult(
        paper_id="p1",
        relevance_level="high",
        reason="test",
        relevance_score=0.90,
    )
    # original_query 与 title 高度相似
    score, subscores = compute_paper_score(
        paper, selection, original_query="Attention Is All You Need"
    )
    assert subscores["Title_Bonus"] == 0.12, f"Expected 0.12, got {subscores['Title_Bonus']}"
    print(f"✅ test_title_exact_bonus_high_similarity passed (bonus={subscores['Title_Bonus']})")


def test_title_exact_bonus_low_similarity():
    """title_exact + title_similarity < 0.92 → bonus=0"""
    paper = Paper(
        paper_id="p1",
        title="Some Random Paper Title",
        abstract="About something else.",
        year=2020,
        retrieval_path=["provider:arxiv", "title_exact:some random paper title"],
    )
    selection = SelectionResult(
        paper_id="p1",
        relevance_level="medium",
        reason="test",
        relevance_score=0.50,
    )
    # original_query 与 title 不相似
    score, subscores = compute_paper_score(
        paper, selection, original_query="BERT pre-training for language understanding"
    )
    assert subscores["Title_Bonus"] == 0.0, f"Expected 0.0, got {subscores['Title_Bonus']}"
    print(f"✅ test_title_exact_bonus_low_similarity passed (bonus={subscores['Title_Bonus']})")


def test_parse_validation_note_float():
    """从 validation_notes 解析 key=value"""
    notes = ["constraint_coverage=0.750", "valid_evidence_ratio=1.000", "Some other note"]
    assert _parse_validation_note_float(notes, "constraint_coverage") == 0.75
    assert _parse_validation_note_float(notes, "valid_evidence_ratio") == 1.0
    assert _parse_validation_note_float(notes, "nonexistent", default=0.5) == 0.5
    assert _parse_validation_note_float(notes, "nonexistent") is None
    print("✅ test_parse_validation_note_float passed")


def test_score_falls_back_to_validation_notes():
    """selection 无 constraint_score/evidence_score 时从 validation_notes 解析"""
    paper = Paper(
        paper_id="p1",
        title="BERT on SQuAD",
        abstract="We test BERT on SQuAD.",
        year=2023,
        citation_count=50,
    )
    selection = SelectionResult(
        paper_id="p1",
        relevance_level="high",
        reason="test",
        relevance_score=0.90,
        # constraint_score 和 evidence_score 为 None
        validation_notes=["constraint_coverage=0.667", "valid_evidence_ratio=0.500"],
    )
    score, subscores = compute_paper_score(paper, selection)
    assert subscores["Constraint_Coverage"] == 0.667, f"Expected 0.667, got {subscores['Constraint_Coverage']}"
    assert subscores["Evidence_Completeness"] == 0.5, f"Expected 0.5, got {subscores['Evidence_Completeness']}"
    print(f"✅ test_score_falls_back_to_validation_notes passed (score={score:.4f})")


if __name__ == "__main__":
    test_weighted_sum_missing_aware_all_present()
    test_weighted_sum_missing_aware_bge_none()
    test_compute_score_bge_none_no_fake()
    test_compute_score_bge_present()
    test_title_exact_bonus_high_similarity()
    test_title_exact_bonus_low_similarity()
    test_parse_validation_note_float()
    test_score_falls_back_to_validation_notes()
    print("\n🎉 All Slice 4 tests passed!")
