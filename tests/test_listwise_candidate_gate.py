from scholar_agent.models.schemas import Paper, SelectionResult
from scholar_agent.workflow.pipeline import _select_listwise_input_candidates


def make_paper(
    paper_id: str,
    *,
    local_score: float,
    retrieval_path: list[str] | None = None,
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        retrieval_path=retrieval_path or [],
        metadata={"local_pre_rank_score": local_score},
    )


def make_selection(
    paper_id: str,
    level: str,
    *,
    relevance_score: float | None = None,
    matched_constraints: list[str] | None = None,
    is_validated: bool = True,
    validation_notes: list[str] | None = None,
) -> SelectionResult:
    return SelectionResult(
        paper_id=paper_id,
        relevance_level=level,
        matched_constraints=matched_constraints or [],
        reason="test selection",
        is_validated=is_validated,
        validation_notes=validation_notes or [],
        relevance_score=relevance_score,
    )


def test_listwise_input_keeps_high_and_medium_candidates():
    high = make_paper("high", local_score=0.40)
    medium = make_paper("medium", local_score=0.50)
    weak_medium = make_paper("weak_medium", local_score=0.60)

    selected, filtered_count, bridged_count = _select_listwise_input_candidates(
        [high, medium, weak_medium],
        [
            make_selection("high", "high", relevance_score=0.90),
            make_selection("medium", "medium", relevance_score=0.25),
            make_selection("weak_medium", "medium", relevance_score=0.20),
        ],
    )

    selected_ids = {paper.paper_id for paper in selected}
    assert selected_ids == {"high", "medium"}
    assert filtered_count == 1
    assert bridged_count == 0


def test_listwise_input_bridges_strong_low_title_exact_candidate():
    gold_like = make_paper(
        "gold_like",
        local_score=0.62,
        retrieval_path=["provider:openalex", "route:title_exact"],
    )
    weak_low = make_paper(
        "weak_low",
        local_score=0.62,
        retrieval_path=["provider:openalex"],
    )

    selected, filtered_count, bridged_count = _select_listwise_input_candidates(
        [weak_low, gold_like],
        [
            make_selection("weak_low", "low", relevance_score=0.20),
            make_selection("gold_like", "low", relevance_score=0.20),
        ],
    )

    selected_ids = [paper.paper_id for paper in selected]
    assert selected_ids == ["gold_like"]
    assert filtered_count == 0
    assert bridged_count == 1


def test_listwise_input_bridges_title_exact_with_missing_must_have_note():
    gold_like = make_paper(
        "gold_like",
        local_score=0.62,
        retrieval_path=["provider:openalex", "route:title_exact"],
    )

    selected, filtered_count, bridged_count = _select_listwise_input_candidates(
        [gold_like],
        [
            make_selection(
                "gold_like",
                "low",
                relevance_score=0.20,
                is_validated=False,
                validation_notes=["Missing must-have constraint: 'implicit feedback'"],
            ),
        ],
    )

    assert [paper.paper_id for paper in selected] == ["gold_like"]
    assert filtered_count == 0
    assert bridged_count == 1


def test_listwise_input_does_not_bridge_year_violations():
    stale = make_paper(
        "stale",
        local_score=0.80,
        retrieval_path=["provider:openalex", "route:title_exact"],
    )

    selected, filtered_count, bridged_count = _select_listwise_input_candidates(
        [stale],
        [
            make_selection(
                "stale",
                "low",
                relevance_score=0.20,
                is_validated=False,
                validation_notes=["Year constraint violated: Paper year 2010 is outside range."],
            ),
        ],
    )

    assert selected == []
    assert filtered_count == 0
    assert bridged_count == 0
