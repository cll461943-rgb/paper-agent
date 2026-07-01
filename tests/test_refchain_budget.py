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
