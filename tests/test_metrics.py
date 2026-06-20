import pytest
from scholar_agent.models.schemas import Paper, SelectionResult
from scholar_agent.evaluation.metrics import (
    score_papers_against_gold,
    calculate_hit_at_k,
    compute_selector_stats,
    is_paper_match_gold,
)
from scholar_agent.evaluation.error_analysis import (
    compute_candidate_to_final_loss,
    classify_drop_reason,
)

class MockPaper:
    def __init__(self, paper_id, title, doi=None, arxiv_id=None, year=2025):
        self.paper_id = paper_id
        self.title = title
        self.doi = doi
        self.arxiv_id = arxiv_id
        self.year = year
        self.metadata = {}

class MockSelection:
    def __init__(self, paper_id, relevance_level, is_validated=True, validation_notes=None):
        self.paper_id = paper_id
        self.relevance_level = relevance_level
        self.is_validated = is_validated
        self.validation_notes = validation_notes or []


def test_is_paper_match_gold():
    p = MockPaper("p1", "Attention Is All You Need", doi="10.1111/attn", arxiv_id="1706.03762")
    
    # 1. 匹配 ID (relaxed)
    assert is_paper_match_gold(p, "1706.03762", strict=False)
    assert is_paper_match_gold(p, "doi:10.1111/attn", strict=False)
    
    # 2. 匹配标题 (relaxed)
    assert is_paper_match_gold(p, "attention is all you need", strict=False)
    assert is_paper_match_gold(p, "attention is all", strict=False) # 包含关系匹配
    
    # 3. 匹配 ID (strict)
    assert is_paper_match_gold(p, "1706.03762", strict=True)
    # 标题不一致时 strict 应该匹配失败
    assert not is_paper_match_gold(p, "Attention Is Not All You Need", strict=True)


def test_score_papers_against_gold():
    papers = [
        MockPaper("p1", "BERT: Pre-training of Deep Bidirectional Transformers"),
        MockPaper("p2", "Attention Is All You Need"),
    ]
    gold = [
        {"paper_id": "p1", "title": "BERT: Pre-training of Deep Bidirectional Transformers"},
        {"paper_id": "p3", "title": "GPT-3"}
    ]
    
    # 1. Relaxed score (p1 应该匹配上)
    relaxed_scores = score_papers_against_gold(papers, gold, strict=False)
    assert relaxed_scores["true_positive"] == 1
    assert relaxed_scores["predicted_count"] == 2
    assert relaxed_scores["gold_count"] == 2
    assert relaxed_scores["precision"] == 0.5
    assert relaxed_scores["recall"] == 0.5
    assert relaxed_scores["f1"] == 0.5
    
    # 2. Strict score
    strict_scores = score_papers_against_gold(papers, gold, strict=True)
    assert strict_scores["true_positive"] == 1


def test_calculate_hit_at_k():
    papers = [
        MockPaper("p1", "BERT"),
        MockPaper("p2", "Attention"),
    ]
    gold = ["Attention", "GPT-4"]
    
    assert calculate_hit_at_k(papers, gold, 1) == 0
    assert calculate_hit_at_k(papers, gold, 2) == 1


def test_compute_selector_stats():
    candidates = [
        MockPaper("p1", "BERT"),
        MockPaper("p2", "Attention"),
        MockPaper("p3", "ResNet"),
    ]
    selections = [
        MockSelection("p1", "high"),
        MockSelection("p2", "low"),
        MockSelection("p3", "high"),
    ]
    gold = ["BERT", "Attention"]
    
    stats = compute_selector_stats(candidates, selections, gold)
    # p3 判定为 high，但不在 gold 中，所以 false_positive = 1
    assert stats["selector_false_positive"] == 1
    # p2 属于 gold，但在 selections 中判定为 low，所以 false_negative = 1
    assert stats["selector_false_negative"] == 1


def test_error_analysis():
    candidates = [
        MockPaper("p1", "BERT"),
        MockPaper("p2", "Attention"),
    ]
    final = [
        MockPaper("p2", "Attention"),
    ]
    gold = ["BERT", "Attention", "GPT-3"]
    
    # 1. compute_candidate_to_final_loss
    loss = compute_candidate_to_final_loss(gold, candidates, final)
    assert loss["candidate_gold_count"] == 2 # BERT and Attention
    assert loss["final_gold_count"] == 1 # Attention
    assert loss["lost_after_candidate"] == 1 # BERT is lost
    assert len(loss["lost_papers"]) == 1
    
    # 2. classify_drop_reason
    selections = [
        MockSelection("p1", "low"), # BERT 被 selector 误判
        MockSelection("p2", "high"),
    ]
    # BERT 归因: llm_selector_misjudge
    reason_bert = classify_drop_reason("BERT", candidates, selections, final, 300, 3)
    assert reason_bert == "llm_selector_misjudge"
    
    # GPT-3 归因: source_missing (不在 candidates 中)
    reason_gpt = classify_drop_reason("GPT-3", candidates, selections, final, 300, 3)
    assert reason_gpt == "source_missing"
