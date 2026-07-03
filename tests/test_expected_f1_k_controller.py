from types import SimpleNamespace

from scholar_agent.models.schemas import Paper, RankedPaper, SelectionResult, QueryPlan
from scholar_agent.ranking.expected_f1_k_controller import decide_k_via_expected_f1


def _ranked_with_listwise_probs(probs: list[float]) -> tuple[list[RankedPaper], SimpleNamespace]:
    ranked: list[RankedPaper] = []
    relevance_probabilities: dict[str, float] = {}
    relevance_levels: dict[str, str] = {}
    relevance_confidences: dict[str, float] = {}
    for idx, prob in enumerate(probs, start=1):
        paper = Paper(
            paper_id=f"p{idx}",
            title=f"Paper {idx}",
            metadata={"local_pre_rank_score": 0.2},
        )
        level = "high" if idx == 1 else "medium"
        selection = SelectionResult(
            paper_id=paper.paper_id,
            relevance_level=level,
            reason="Listwise judged candidate.",
        )
        ranked.append(RankedPaper(paper=paper, selection=selection, final_score=0.9 - idx * 0.03, rank=idx))
        relevance_probabilities[paper.paper_id] = prob
        relevance_levels[paper.paper_id] = level
        relevance_confidences[paper.paper_id] = 0.9

    return ranked, SimpleNamespace(
        success=True,
        relevance_probabilities=relevance_probabilities,
        relevance_levels=relevance_levels,
        relevance_confidences=relevance_confidences,
    )


def test_expected_f1_controller_enforces_p_floor_for_focused_query():
    ranked, listwise = _ranked_with_listwise_probs([0.95, 0.70, 0.55, 0.41])
    result = decide_k_via_expected_f1(
        ranked,
        query_type="specific_paper",
        listwise_result=listwise,
        k_max=4,
        config=SimpleNamespace(dynamic_k=SimpleNamespace(hard_max_output=20)),
        all_candidates_count=100,
        query_plan=QueryPlan(
            original_query="Which papers extended IPS and SNIPS methods to implicit feedback data?",
            query_type="specific_paper",
            entities=["IPS", "SNIPS"],
        ),
    )

    assert result.chosen_k == 2
    assert "p_floor=0.45 capped candidate K at 2" in result.tie_break_reason
