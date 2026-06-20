import pytest
from types import SimpleNamespace
from scholar_agent.models.schemas import RankedPaper
from scholar_agent.ranking.dynamic_k import decide_dynamic_k

class DummyRankedPaper:
    def __init__(self, paper_id: str, final_score: float):
        self.paper = SimpleNamespace(paper_id=paper_id, title="", abstract="", year=2025, venue="")
        self.final_score = final_score
        # 兼容 RankedPaper 结构
        self.selection = SimpleNamespace(relevance_level="high", validation_notes=[])

def test_decide_dynamic_k_empty():
    assert decide_dynamic_k(None, [], None) == 0

def test_decide_dynamic_k_cliff():
    # 差值大于等于 0.15 (0.9 - 0.7 = 0.2 >= 0.15)
    papers = [
        DummyRankedPaper("p1", 0.9),
        DummyRankedPaper("p2", 0.7),
        DummyRankedPaper("p3", 0.65)
    ]
    query_contract = SimpleNamespace(query_type="survey")
    # 触发断崖切成 1
    assert decide_dynamic_k(query_contract, papers, None) == 1

def test_decide_dynamic_k_query_type_specific():
    # 差值不触发断崖
    papers = [
        DummyRankedPaper("p1", 0.8),
        DummyRankedPaper("p2", 0.75),
        DummyRankedPaper("p3", 0.7),
        DummyRankedPaper("p4", 0.6)
    ]
    
    # specific_paper -> 1
    q1 = SimpleNamespace(query_type="specific_paper", original_query="")
    assert decide_dynamic_k(q1, papers, None) == 1

    # dataset_constraint -> 3
    q2 = SimpleNamespace(query_type="dataset_constraint", original_query="")
    assert decide_dynamic_k(q2, papers, None) == 3

    # survey -> 5 (因为只有 4 篇，所以返回 4)
    q3 = SimpleNamespace(query_type="survey", original_query="")
    assert decide_dynamic_k(q3, papers, None) == 4

def test_decide_dynamic_k_heuristic():
    papers = [
        DummyRankedPaper("p1", 0.8),
        DummyRankedPaper("p2", 0.75),
        DummyRankedPaper("p3", 0.7)
    ]
    # query_type="unknown" 但 query 包含 "综述" -> survey -> 5 (截断 3)
    q = SimpleNamespace(query_type="unknown", original_query="关于LLM的最新综述")
    assert decide_dynamic_k(q, papers, None) == 3

def test_decide_dynamic_k_threshold():
    papers = [
        DummyRankedPaper("p1", 0.70),
        DummyRankedPaper("p2", 0.60),
        DummyRankedPaper("p3", 0.50), # 低于 0.55 阈值
        DummyRankedPaper("p4", 0.40)
    ]
    # query_type="unknown"
    q = SimpleNamespace(query_type="unknown", original_query="random")
    # 大于等于 0.55 的只有 p1, p2，返回 2
    assert decide_dynamic_k(q, papers, None) == 2

def test_decide_dynamic_k_with_config():
    config = SimpleNamespace(
        dynamic_k=SimpleNamespace(
            score_gap_top1=0.05, # 改为 0.05 触发断崖
            exact_title_k=1,
            method_comparison_k=3,
            survey_k=5,
            min_final_score=0.7,
            fallback_k=3
        ),
        ranking=SimpleNamespace(
            max_final_papers=10
        )
    )
    papers = [
        DummyRankedPaper("p1", 0.8),
        DummyRankedPaper("p2", 0.72)
    ]
    q = SimpleNamespace(query_type="survey", original_query="")
    # 差值 0.08 >= 0.05，触发 cliff cut-off -> 1
    assert decide_dynamic_k(q, papers, config) == 1
