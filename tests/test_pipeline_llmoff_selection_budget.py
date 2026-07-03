from scholar_agent.workflow.pipeline import _selection_budget_for_query


def test_selection_budget_stays_small_when_llm_available():
    assert _selection_budget_for_query(
        "unknown",
        llm_off_fallback=False,
        pool_size=239,
    ) == 45
    assert _selection_budget_for_query(
        "latest_work",
        llm_off_fallback=False,
        pool_size=239,
    ) == 65


def test_selection_budget_expands_for_llm_off_local_fallback():
    assert _selection_budget_for_query(
        "unknown",
        llm_off_fallback=True,
        pool_size=239,
    ) == 200


def test_selection_budget_is_capped_by_pool_size():
    assert _selection_budget_for_query(
        "unknown",
        llm_off_fallback=True,
        pool_size=37,
    ) == 37
