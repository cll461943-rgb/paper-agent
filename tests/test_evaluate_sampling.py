from evaluate import select_case_indices


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
