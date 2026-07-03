from types import SimpleNamespace

from scholar_agent.models.schemas import Paper, QueryPlan
from scholar_agent.selection.batch_evidence_selector import batch_select_and_extract_evidence


class ShortDeadline:
    def __init__(self, remaining: float) -> None:
        self._remaining = remaining

    def expired(self) -> bool:
        return self._remaining <= 0

    def remaining(self) -> float:
        return self._remaining


class CapturingLLMClient:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def complete_json(self, *args, timeout_seconds=None, **kwargs):
        self.timeouts.append(timeout_seconds)
        return {
            "selections": [
                {
                    "paper_id": "p1",
                    "relevance_level": "high",
                    "relevance_score": 0.90,
                    "constraint_score": 0.80,
                    "evidence_score": 0.70,
                    "matched_constraints": ["IPS"],
                    "missing_constraints": [],
                    "evidence": [{"field": "title", "text": "IPS implicit feedback"}],
                    "reason": "Matches the query.",
                    "confidence": 0.90,
                    "uncertainty": [],
                }
            ]
        }


def test_evidence_selector_timeout_respects_child_deadline():
    paper = Paper(paper_id="p1", title="IPS implicit feedback")
    plan = QueryPlan(original_query="IPS implicit feedback", methods=["IPS"])
    llm_client = CapturingLLMClient()
    config = SimpleNamespace(
        llm=SimpleNamespace(timeout_evidence_selection=35, max_tokens_evidence_selection=1024),
        selection=SimpleNamespace(batch_size=5),
        budget=SimpleNamespace(max_llm_selection_papers=5),
    )

    results = batch_select_and_extract_evidence(
        [paper],
        plan,
        llm_client,
        config=config,
        deadline=ShortDeadline(remaining=3.0),
    )

    assert results[0].relevance_level == "high"
    assert llm_client.timeouts == [3.0]
