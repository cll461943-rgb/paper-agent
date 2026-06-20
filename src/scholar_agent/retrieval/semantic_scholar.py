from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import quote

from scholar_agent.infra.cache import JsonFileCache
from scholar_agent.infra.config import ProviderConfig
from scholar_agent.infra.http_client import HttpClient
from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider

LOGGER = logging.getLogger(__name__)

SEARCH_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "title",
        "abstract",
        "year",
        "venue",
        "url",
        "citationCount",
        "authors",
        "externalIds",
        "referenceCount",
        "fieldsOfStudy",
    ]
)
GRAPH_FIELDS = SEARCH_FIELDS


def _clean_query_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:500] if normalized else "research paper"


class SemanticScholarProvider(PaperProvider):
    name = "semantic_scholar"

    def __init__(self, config: ProviderConfig, cache: JsonFileCache) -> None:
        self.config = config
        self.cache = cache
        self.http = HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_times=config.retry_times,
            min_interval_seconds=config.min_interval_seconds,
            backoff_base_seconds=config.backoff_base_seconds,
        )
        self.api_key = os.getenv(config.api_key_env) if config.api_key_env else None
        self.last_error: str | None = None

    def is_available(self) -> bool:
        if not self.config.base_url:
            self.last_error = "missing Semantic Scholar base_url"
            return False
        return True

    def _build_params(self, query: SearchQuery, limit: int) -> dict[str, Any]:
        raw_query = " ".join(
            part
            for part in [
                query.query,
                " ".join(query.required_terms),
                " ".join(query.optional_terms),
            ]
            if part
        )
        return {
            "query": _clean_query_text(raw_query),
            "limit": limit,
            "offset": 0,
            "fields": SEARCH_FIELDS,
        }

    def _headers(self) -> dict[str, str] | None:
        if not self.api_key:
            return None
        return {"x-api-key": self.api_key}

    @staticmethod
    def _semantic_scholar_lookup_id(paper: Paper) -> str | None:
        if paper.doi:
            return f"DOI:{paper.doi}"
        if paper.arxiv_id:
            return f"ARXIV:{paper.arxiv_id}"
        corpus_id = paper.metadata.get("corpus_id") or paper.metadata.get("corpusId")
        if corpus_id:
            return f"CorpusId:{corpus_id}"
        if paper.source == "semantic_scholar" and paper.paper_id:
            return paper.paper_id
        if paper.paper_id and not paper.paper_id.startswith(("http://", "https://")):
            return paper.paper_id
        return None

    def _paper_endpoint(self, paper_id: str, suffix: str) -> str:
        encoded_id = quote(paper_id, safe="")
        return self.config.base_url.replace("/paper/search", f"/paper/{encoded_id}/{suffix}")

    @staticmethod
    def _paper_from_semantic_scholar_item(item: dict[str, Any]) -> Paper:
        external_ids = item.get("externalIds") or {}
        paper_id = str(item.get("paperId") or item.get("url") or item.get("title") or "")
        authors = [
            author.get("name")
            for author in item.get("authors", [])
            if isinstance(author, dict) and author.get("name")
        ]
        return Paper(
            paper_id=paper_id,
            title=item.get("title") or "Untitled",
            abstract=item.get("abstract"),
            year=item.get("year"),
            venue=item.get("venue"),
            authors=authors,
            doi=external_ids.get("DOI"),
            arxiv_id=external_ids.get("ArXiv"),
            url=item.get("url"),
            citation_count=item.get("citationCount"),
            source="semantic_scholar",
            metadata={
                "raw_source": "semantic_scholar",
                "corpus_id": item.get("corpusId"),
                "external_ids": external_ids,
                "fields_of_study": item.get("fieldsOfStudy") or [],
                "reference_count": item.get("referenceCount"),
                "published_time": f"{item.get('year')}-12-31" if item.get("year") else None,
            },
        )

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        self.last_error = None
        if not self.is_available():
            LOGGER.info("Semantic Scholar is not available (missing API key or base_url), skipping.")
            return []
        cache_key = self.query_cache_key(query, limit)
        cached = self.cache.get_query(cache_key)
        if cached is not None:
            return [Paper.model_validate(item) for item in cached]

        try:
            payload = self.http.get_json(
                self.config.base_url,
                params=self._build_params(query, limit),
                headers=self._headers(),
            )
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("Semantic Scholar search failed for query=%s: %s", query.query, exc)
            return []

        papers: list[Paper] = []
        for item in payload.get("data", [])[:limit]:
            paper = self._paper_from_semantic_scholar_item(item)
            cached_paper_dict = self.cache.get_paper_dict(self.paper_cache_key(paper))
            if cached_paper_dict is not None:
                try:
                    paper = Paper.model_validate(cached_paper_dict)
                except Exception:
                    pass
            paper = self.enrich_paper(paper, query)
            self.cache.set_paper_dict(self.paper_cache_key(paper), paper.model_dump(mode="json"))
            papers.append(paper)

        self.cache.set_query(cache_key, [paper.model_dump(mode="json") for paper in papers])
        return papers

    def _expand_papers(self, paper: Paper, *, suffix: str, payload_key: str, route: str, limit: int) -> list[Paper]:
        self.last_error = None
        lookup_id = self._semantic_scholar_lookup_id(paper)
        if not lookup_id:
            return []
        query = SearchQuery(
            query=paper.title,
            route=route,
            intent=route,
            required_terms=[],
            optional_terms=[],
            filters={},
            priority=99,
        )
        cache_key = self.query_cache_key(
            query.model_copy(update={"query": f"{lookup_id}:{suffix}"}),
            limit,
        )
        cached = self.cache.get_query(cache_key)
        if cached is not None:
            return [Paper.model_validate(item) for item in cached]

        try:
            payload = self.http.get_json(
                self._paper_endpoint(lookup_id, suffix),
                params={"fields": GRAPH_FIELDS, "limit": limit, "offset": 0},
                headers=self._headers(),
            )
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("Semantic Scholar %s expansion failed for id=%s: %s", suffix, lookup_id, exc)
            return []

        papers: list[Paper] = []
        for item in payload.get("data", [])[:limit]:
            raw_paper = item.get(payload_key)
            if not isinstance(raw_paper, dict) or not raw_paper.get("paperId"):
                continue
            expanded = self._paper_from_semantic_scholar_item(raw_paper)
            expanded = self.enrich_paper(expanded, query)
            self.cache.set_paper_dict(self.paper_cache_key(expanded), expanded.model_dump(mode="json"))
            papers.append(expanded)

        self.cache.set_query(cache_key, [item.model_dump(mode="json") for item in papers])
        return papers

    def get_references(self, paper: Paper, limit: int) -> list[Paper]:
        return self._expand_papers(
            paper,
            suffix="references",
            payload_key="citedPaper",
            route="reference_expansion",
            limit=limit,
        )

    def get_citations(self, paper: Paper, limit: int) -> list[Paper]:
        return self._expand_papers(
            paper,
            suffix="citations",
            payload_key="citingPaper",
            route="citation_expansion",
            limit=limit,
        )
