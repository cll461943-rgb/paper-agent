from types import SimpleNamespace

from evaluate import apply_time_budget_to_config, select_case_indices


def test_limit_sampling_is_reproducible_for_same_seed():
    first = select_case_indices(1000, limit=5, random_seed=123)
    second = select_case_indices(1000, limit=5, random_seed=123)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_limit_sampling_changes_with_different_seed():
    first = select_case_indices(1000, limit=5, random_seed=123)
    second = select_case_indices(1000, limit=5, random_seed=456)

    assert first != second


def test_explicit_case_filter_takes_precedence_over_seed():
    assert select_case_indices(20, cases_filter="1,3-4", limit=5, random_seed=123) == [0, 2, 3]


def test_simple_takes_precedence_over_limit_sampling():
    assert select_case_indices(20, simple=True, limit=5, random_seed=123) == [0]


def test_explicit_time_budget_updates_pipeline_case_deadline():
    config = SimpleNamespace(budget=SimpleNamespace(case_deadline_seconds=180))

    effective_budget = apply_time_budget_to_config(config, 240)

    assert effective_budget == 240.0
    assert config.budget.case_deadline_seconds == 240.0


def test_missing_time_budget_uses_configured_case_deadline():
    config = SimpleNamespace(budget=SimpleNamespace(case_deadline_seconds=180))

    effective_budget = apply_time_budget_to_config(config, None)

    assert effective_budget == 180.0
    assert config.budget.case_deadline_seconds == 180
