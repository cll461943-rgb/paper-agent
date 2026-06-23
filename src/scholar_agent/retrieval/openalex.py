from __future__ import annotations

import logging
import re
from typing import Any

from scholar_agent.infra.cache import JsonFileCache
from scholar_agent.infra.config import ProviderConfig
from scholar_agent.infra.http_client import HttpClient
from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider

LOGGER = logging.getLogger(__name__)

OPENALEX_QUERY_STOPWORDS = {
    "a", "about", "all", "also", "an", "and", "any", "are",
    "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "instead", "is", "it", "me", "of", "on", "or",
    "paper", "papers", "research", "that", "the",
    "there", "to", "what", "which", "with",
}


def _decode_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    max_position = max((max(positions) for positions in index.values() if positions), default=-1)
    if max_position < 0:
        return None
    words = [""] * (max_position + 1)
    for word, positions in index.items():
        for position in positions:
            if 0 <= position <= max_position:
                words[position] = word
    return " ".join(token for token in words if token).strip() or None


def _sanitize_openalex_search(query: SearchQuery) -> str:
    # 如果是特定路由或者大模型生成的检索词本身就已经净化，避免去拼 required_terms / optional_terms
    if query.route in {"title_exact", "title_like", "query2doc", "hyde"} or len(query.query.split()) > 2:
        raw = query.query.strip()
    else:
        raw = " ".join(
            part
            for part in [
                query.query,
                " ".join(query.required_terms),
                " ".join(query.optional_terms),
            ]
            if part
        ).strip()
        
    if not raw:
        return "research paper"

    # 提取 token 并支持引号包裹的短语
    ascii_tokens = re.findall(r'"[^"]+"|[A-Za-z0-9][A-Za-z0-9\-\+\.]*', raw)
    has_non_ascii = any(ord(char) > 127 for char in raw)
    question_mark_ratio = raw.count("?") / max(len(raw), 1)
    looks_like_question = bool(re.match(r"(?i)^\s*(what|how|why|which|are|is|does|do|can)\b", raw)) or "?" in raw

    # 提高过滤与限制上限到 10 个词，防止过度截断导致零召回
    if has_non_ascii or question_mark_ratio > 0.15 or looks_like_question or len(ascii_tokens) > 5:
        if ascii_tokens:
            filtered = []
            for token in ascii_tokens:
                clean_tok = token.strip(' ."').lower()
                if clean_tok and clean_tok not in OPENALEX_QUERY_STOPWORDS:
                    filtered.append(token)
            result_tokens = dict.fromkeys(filtered or ascii_tokens)
            final_tokens = list(result_tokens.keys())[:10]
            return " ".join(final_tokens)[:180]
        return " ".join(token for token in ["research", "paper", query.route] if token)

    normalized = re.sub(r"\s+", " ", raw).strip()
    return normalized[:180] if normalized else "research paper"


def _normalize_openalex_work_url(work_id: str) -> str:
    work_id = (work_id or "").strip()
    if not work_id:
        return ""
    if work_id.startswith("https://api.openalex.org/works/"):
        return work_id
    if work_id.startswith("https://openalex.org/"):
        suffix = work_id.rsplit("/", 1)[-1]
        return f"https://api.openalex.org/works/{suffix}"
    if work_id.startswith("W") and work_id[1:].isdigit():
        return f"https://api.openalex.org/works/{work_id}"
    return work_id


def _looks_like_title(text: str) -> bool:
    words = text.split()
    bad_prefixes = [
        "give me",
        "find",
        "papers which",
        "papers that",
        "show that",
        "work on",
        "studies about",
    ]
    lowered = text.lower()
    return len(words) >= 5 and not any(prefix in lowered for prefix in bad_prefixes)


class OpenAlexProvider(PaperProvider):
    name = "openalex"

    def __init__(self, config: ProviderConfig, cache: JsonFileCache) -> None:
        self.config = config
        self.cache = cache
        self.http = HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_times=config.retry_times,
            min_interval_seconds=config.min_interval_seconds,
            backoff_base_seconds=config.backoff_base_seconds,
        )
        import os
        self.api_key = os.getenv(config.api_key_env) if config.api_key_env else None
        self.last_error: str | None = None

    def _build_params(self, query: SearchQuery, limit: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "search": _sanitize_openalex_search(query),
            "per-page": limit,
        }
        if self.config.mailto:
            params["mailto"] = self.config.mailto
        if self.api_key:
            params["api_key"] = self.api_key

        start_year = query.filters.get("start_year")
        end_year = query.filters.get("end_year")
        filters: list[str] = []
        if start_year:
            filters.append(f"from_publication_date:{start_year}-01-01")
        if end_year:
            filters.append(f"to_publication_date:{end_year}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        return params

    @staticmethod
    def _paper_from_openalex_item(item: dict[str, Any]) -> Paper:
        return Paper(
            paper_id=str(item.get("id", "")),
            title=item.get("title") or "Untitled",
            abstract=_decode_abstract(item.get("abstract_inverted_index")),
            year=item.get("publication_year"),
            venue=((item.get("primary_location") or {}).get("source") or {}).get("display_name"),
            authors=[
                (author.get("author") or {}).get("display_name")
                for author in item.get("authorships", [])
                if (author.get("author") or {}).get("display_name")
            ],
            doi=item.get("doi"),
            url=((item.get("primary_location") or {}).get("landing_page_url")) or item.get("id"),
            citation_count=item.get("cited_by_count"),
            source="openalex",
            references=[ref for ref in item.get("referenced_works", []) if ref],
            metadata={
                "raw_source": "openalex",
                "published_time": item.get("publication_date"),
                "concepts": [concept.get("display_name") for concept in item.get("concepts", []) if concept.get("display_name")],
                "cited_by_api_url": item.get("cited_by_api_url"),
                "related_works": [work for work in item.get("related_works", []) if work],
            },
        )

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        self.last_error = None
        cache_key = self.query_cache_key(query, limit)
        cached = self.cache.get_query(cache_key)
        if cached is not None:
            return [Paper.model_validate(item) for item in cached]

        try:
            payload = self.http.get_json(self.config.base_url, params=self._build_params(query, limit))
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("OpenAlex search failed for query=%s: %s", query.query, exc)
            return []

        papers: list[Paper] = []
        for item in payload.get("results", [])[:limit]:
            paper = self._paper_from_openalex_item(item)
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

    def search_title_exact(self, title: str, limit: int = 10) -> list[Paper]:
        self.last_error = None
        if len(title) > 180:
            title = title[:180]
        title = re.sub(r"[\r\n]+", " ", title).strip()
        if not title or not _looks_like_title(title):
            return []
        query = SearchQuery(
            query=title,
            route="title_exact",
            intent="title_exact_recall",
            required_terms=[],
            optional_terms=[],
            filters={},
            priority=0,
        )
        cache_key = self.query_cache_key(query, limit)
        cached = self.cache.get_query(cache_key)
        if cached is not None:
            return [Paper.model_validate(item) for item in cached]

        escaped_title = title.replace('"', '\\"')
        params = {"filter": f'title.search:"{escaped_title}"', "per-page": limit}
        if self.config.mailto:
            params["mailto"] = self.config.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            payload = self.http.get_json(self.config.base_url, params=params)
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("OpenAlex title search failed for title=%s: %s", title, exc)
            return []
        papers: list[Paper] = []
        for item in payload.get("results", [])[:limit]:
            paper = self._paper_from_openalex_item(item)
            paper = self.enrich_paper(paper, query)
            self.cache.set_paper_dict(self.paper_cache_key(paper), paper.model_dump(mode="json"))
            papers.append(paper)
        self.cache.set_query(cache_key, [paper.model_dump(mode="json") for paper in papers])
        return papers

    def _expand_works(self, work_ids: list[str], route: str, limit: int) -> list[Paper]:
        papers: list[Paper] = []
        for work_id in work_ids[:limit]:
            normalized_url = _normalize_openalex_work_url(work_id)
            if not normalized_url:
                continue
            params = {}
            if self.config.mailto:
                params["mailto"] = self.config.mailto
            if self.api_key:
                params["api_key"] = self.api_key
            try:
                item = self.http.get_json(normalized_url, params=params if params else None)
            except Exception as exc:
                self.last_error = str(exc)
                LOGGER.warning("OpenAlex work expansion failed for id=%s: %s", work_id, exc)
                continue
            paper = self._paper_from_openalex_item(item)
            paper = self.enrich_paper(
                paper,
                SearchQuery(
                    query=paper.title,
                    route=route,
                    intent=route,
                    required_terms=[],
                    optional_terms=[],
                    filters={},
                    priority=99,
                ),
            )
            papers.append(paper)
        return papers

    def get_references(self, paper: Paper, limit: int) -> list[Paper]:
        return self._expand_works(paper.references, "reference_expansion", limit)

    def get_citations(self, paper: Paper, limit: int) -> list[Paper]:
        cited_by_url = paper.metadata.get("cited_by_api_url")
        if not cited_by_url:
            return []

        citation_params = {
            "per-page": limit,
        }
        if self.config.mailto:
            citation_params["mailto"] = self.config.mailto
        if self.api_key:
            citation_params["api_key"] = self.api_key
        try:
            payload = self.http.get_json(
                cited_by_url,
                params=citation_params,
            )
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("OpenAlex citation expansion failed for id=%s: %s", paper.paper_id, exc)
            return []

        papers: list[Paper] = []
        for item in payload.get("results", [])[:limit]:
            expanded = self._paper_from_openalex_item(item)
            expanded = self.enrich_paper(
                expanded,
                SearchQuery(
                    query=paper.title,
                    route="citation_expansion",
                    intent="refchain_citation",
                    required_terms=[],
                    optional_terms=[],
                    filters={},
                    priority=99,
                ),
            )
            papers.append(expanded)

        return papers

    def get_related_works(self, paper: Paper, limit: int) -> list[Paper]:
        related_ids = paper.metadata.get("related_works", [])
        return self._expand_works(related_ids, "related_work_expansion", limit)
