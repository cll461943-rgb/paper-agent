from scholar_agent.infra.config import AppConfig
from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider
from scholar_agent.retrieval.multi_route import MultiRouteRetriever
from scholar_agent.workflow.budget import BudgetManager


class ReferenceProvider(PaperProvider):
    name = "semantic_scholar"

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        return []

    def get_references(self, paper: Paper, limit: int) -> list[Paper]:
        return [
            Paper(paper_id=f"ref-{index}", title=f"Reference {index}")
            for index in range(limit)
        ]


class CountingProvider(PaperProvider):
    name = "openalex"

    def __init__(self) -> None:
        self.search_calls = 0

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        self.search_calls += 1
        return [Paper(paper_id="p1", title="Paper 1")]


class ExpiredDeadline:
    def expired(self) -> bool:
        return True

    def remaining(self) -> float:
        return 0.0


def test_refchain_budget_exhaustion_skips_expansion_without_crashing():
    config = AppConfig()
    config.budget.max_api_calls = 0
    budget = BudgetManager(config)
    provider = ReferenceProvider()
    retriever = MultiRouteRetriever([provider], budget)
    seed = Paper(paper_id="seed", title="Seed Paper")

    expanded = retriever.expand_refchain([seed], [provider], limit_per_seed=2)

    assert expanded == []
    assert budget.api_calls_used == 0
    assert any("API call budget exceeded" in error for error in budget.errors)


def test_retrieval_deadline_skips_provider_search_when_expired():
    config = AppConfig()
    budget = BudgetManager(config)
    provider = CountingProvider()
    retriever = MultiRouteRetriever([provider], budget)
    query = SearchQuery(query="test", route="core_topic", intent="test")

    results, papers = retriever.retrieve([query], deadline=ExpiredDeadline())

    assert results == []
    assert papers == []
    assert provider.search_calls == 0
