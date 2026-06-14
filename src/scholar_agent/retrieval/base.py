from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from scholar_agent.models.schemas import Paper, SearchQuery


class PaperProvider(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        raise NotImplementedError

    def search_title_exact(self, title: str, limit: int = 10) -> list[Paper]:
        _ = (title, limit)
        return []

    def get_citations(self, paper: Paper, limit: int) -> list[Paper]:
        return []

    def get_references(self, paper: Paper, limit: int) -> list[Paper]:
        return []

    def get_related_works(self, paper: Paper, limit: int) -> list[Paper]:
        return []

    def is_available(self) -> bool:
        return True

    def query_cache_key(self, query: SearchQuery, limit: int) -> str:
        return f"{self.name}|{query.query}|{query.route}|{limit}|{sorted(query.filters.items())}"

    def enrich_paper(self, paper: Paper, query: SearchQuery) -> Paper:
        if f"provider:{self.name}" not in paper.retrieval_path:
            paper.retrieval_path.append(f"provider:{self.name}")
        if f"route:{query.route}" not in paper.retrieval_path:
            paper.retrieval_path.append(f"route:{query.route}")
        paper.source = self.name
        return paper

    def paper_cache_key(self, paper: Paper) -> str:
        return f"{self.name}|{paper.paper_id}"

    def get_paper(self, paper_id: str) -> Paper | None:
        return None
