#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slice 5 单测：dynamic_k 对齐 + synthesis 清理 + pipeline SelectionConfig"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scholar_agent.ranking.dynamic_k import decide_dynamic_k
from scholar_agent.infra.config import AppConfig, SelectionConfig


class MockRankedPaper:
    """简易 RankedPaper mock"""
    def __init__(self, paper_id: str, final_score: float):
        self.paper = type("P", (), {"paper_id": paper_id})()
        self.paper.paper_id = paper_id
        self.final_score = final_score
        self.subscores = {}
        self.selection = type("S", (), {
            "relevance_level": "high",
            "validation_notes": [],
            "reason": "test",
        })()


class MockQueryPlan:
    """简易 QueryPlan mock"""
    def __init__(self, query_type: str = "unknown", original_query: str = ""):
        self.query_type = query_type
        self.original_query = original_query


def test_survey_k_is_5():
    """survey 类型 → K=5（不是 20）"""
    config = AppConfig()
    plan = MockQueryPlan(query_type="survey", original_query="survey of NLP")
    papers = [MockRankedPaper(f"p{i}", 0.8 - i * 0.05) for i in range(10)]
    k = decide_dynamic_k(plan, papers, config)
    assert k == 5, f"Expected 5, got {k}"
    print(f"✅ test_survey_k_is_5 passed (k={k})")


def test_specific_paper_k_is_2():
    """specific_paper 类型 → K=2"""
    config = AppConfig()
    plan = MockQueryPlan(query_type="specific_paper", original_query="BERT paper")
    papers = [MockRankedPaper(f"p{i}", 0.8 - i * 0.05) for i in range(10)]
    k = decide_dynamic_k(plan, papers, config)
    assert k == 2, f"Expected 2, got {k}"
    print(f"✅ test_specific_paper_k_is_2 passed (k={k})")


def test_exact_title_k_is_1():
    """exact_title 类型 → K=1"""
    config = AppConfig()
    plan = MockQueryPlan(query_type="exact_title", original_query="exact title")
    papers = [MockRankedPaper(f"p{i}", 0.8 - i * 0.05) for i in range(10)]
    k = decide_dynamic_k(plan, papers, config)
    assert k == 1, f"Expected 1, got {k}"
    print(f"✅ test_exact_title_k_is_1 passed (k={k})")


def test_cliff_cutoff():
    """分数断崖 → K=1"""
    config = AppConfig()
    plan = MockQueryPlan(query_type="broad_topic", original_query="broad topic")
    # 第一名 0.95，第二名 0.70，差 0.25 >= 0.15
    papers = [MockRankedPaper("p0", 0.95), MockRankedPaper("p1", 0.70)]
    k = decide_dynamic_k(plan, papers, config)
    assert k == 1, f"Expected 1 (cliff cutoff), got {k}"
    print(f"✅ test_cliff_cutoff passed (k={k})")


def test_fallback_k_is_3():
    """unknown 类型 + 无分数门槛通过 → K=3（不是 5）"""
    config = AppConfig()
    plan = MockQueryPlan(query_type="unknown", original_query="something")
    # 所有分数 < 0.55 (min_final_score)，走 fallback
    papers = [MockRankedPaper(f"p{i}", 0.30 + i * 0.01) for i in range(10)]
    k = decide_dynamic_k(plan, papers, config)
    assert k == 3, f"Expected 3 (fallback), got {k}"
    print(f"✅ test_fallback_k_is_3 passed (k={k})")


def test_min_final_score_0_55():
    """min_final_score = 0.55：分数 0.50 的不通过门槛"""
    config = AppConfig()
    plan = MockQueryPlan(query_type="unknown", original_query="something")
    # 3 篇 >= 0.55, 2 篇 < 0.55
    papers = [
        MockRankedPaper("p0", 0.80),
        MockRankedPaper("p1", 0.70),
        MockRankedPaper("p2", 0.60),
        MockRankedPaper("p3", 0.50),  # < 0.55, 不通过
        MockRankedPaper("p4", 0.40),  # < 0.55, 不通过
    ]
    k = decide_dynamic_k(plan, papers, config)
    assert k == 3, f"Expected 3 (only 3 above 0.55), got {k}"
    print(f"✅ test_min_final_score_0_55 passed (k={k})")


def test_selection_config_exists():
    """SelectionConfig 在 AppConfig 中存在且有正确默认值"""
    config = AppConfig()
    assert hasattr(config, "selection")
    assert isinstance(config.selection, SelectionConfig)
    assert config.selection.exact_title_selection_topk == 8
    assert config.selection.constraint_selection_topk == 15
    assert config.selection.broad_selection_topk == 20
    assert config.selection.default_selection_topk == 12
    assert config.selection.batch_size == 5
    print("✅ test_selection_config_exists passed")


if __name__ == "__main__":
    test_survey_k_is_5()
    test_specific_paper_k_is_2()
    test_exact_title_k_is_1()
    test_cliff_cutoff()
    test_fallback_k_is_3()
    test_min_final_score_0_55()
    test_selection_config_exists()
    print("\n🎉 All Slice 5 tests passed!")
