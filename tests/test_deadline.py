from scholar_agent.workflow.deadline import Deadline


def test_child_with_reserve_caps_early_stage_to_preserve_parent_budget():
    case_deadline = Deadline(100, stage_name="case")

    child = case_deadline.child_with_reserve(
        50,
        reserve_seconds=80,
        stage_name="retrieval",
    )

    assert 0 < child.remaining() <= 20
