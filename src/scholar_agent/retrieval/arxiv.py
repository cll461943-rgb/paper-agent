from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

from scholar_agent.infra.cache import JsonFileCache
from scholar_agent.infra.config import ProviderConfig
from scholar_agent.infra.http_client import HttpClient
from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider

LOGGER = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _sanitize_arxiv_search(query_text: str) -> str:
    raw = query_text.strip()
    if not raw:
        return ""
    
    # 提取 tokens，包括带双引号的短语、单字或带连字符的单词
    tokens = re.findall(r'"[^"]+"|[A-Za-z0-9][A-Za-z0-9\-\+\.]*', raw)
    if not tokens:
        tokens = raw.split()
    
    stop_words = {
        "a", "about", "all", "also", "an", "and", "any", "are", "as", "at", 
        "be", "by", "can", "do", "does", "for", "from", "how", "is", "it", 
        "of", "on", "or", "paper", "papers", "research", "that", "the", 
        "there", "to", "using", "what", "which", "with", "would", "first",
        "shows", "show", "proposes", "propose", "investigates", "investigate",
        "could", "recommend", "you", "me", "find", "list", "studies", "study",
        "demonstrate", "demonstrates", "explore", "explores", "introduced", "introduce"
    }
    
    filtered = []
    for token in tokens:
        clean_tok = token.strip("\"'").lower()
        if clean_tok and clean_tok not in stop_words:
            filtered.append(token)
            
    # 放宽限制到 10 个词以保护长关键词的完整性
    if len(filtered) > 10:
        filtered = filtered[:10]
    return " ".join(filtered)


class ArxivProvider(PaperProvider):
    name = "arxiv"

    def __init__(self, config: ProviderConfig, cache: JsonFileCache) -> None:
        self.config = config
        self.cache = cache
        self.http = HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_times=config.retry_times,
            min_interval_seconds=config.min_interval_seconds,
            backoff_base_seconds=config.backoff_base_seconds,
        )
        self.last_error: str | None = None

    def _build_params(self, query: SearchQuery, limit: int) -> dict[str, str | int]:
        sanitized = _sanitize_arxiv_search(query.query)
        # 兜底以防全部被过滤
        search_str = sanitized if sanitized else "machine learning"
        return {
            "search_query": f"all:{search_str}",
            "start": 0,
            "max_results": limit,
        }

    def _fetch(self, query: SearchQuery, params: dict[str, str | int], limit: int) -> list[Paper]:
        try:
            xml_text = self.http.get_text(
                self.config.base_url,
                params=params,
            )
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("arXiv search failed for query=%s: %s", query.query, exc)
            return []

        papers: list[Paper] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            self.last_error = str(exc)
            LOGGER.warning("arXiv response parse failed for query=%s: %s", query.query, exc)
            return []

        for entry in root.findall("atom:entry", ATOM_NS)[:limit]:
            entry_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").split())
            summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").split())
            published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
            authors = [
                name.text.strip()
                for name in entry.findall("atom:author/atom:name", ATOM_NS)
                if name.text
            ]
            paper = Paper(
                paper_id=entry_id or title,
                title=title or "Untitled",
                abstract=summary or None,
                year=int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None,
                authors=authors,
                arxiv_id=entry_id.rsplit("/", 1)[-1] if entry_id else None,
                url=entry_id or None,
                source=self.name,
                metadata={"raw_source": "arxiv"},
            )
            cached_paper_dict = self.cache.get_paper_dict(self.paper_cache_key(paper))
            if cached_paper_dict is not None:
                try:
                    paper = Paper.model_validate(cached_paper_dict)
                except Exception:
                    pass
            paper = self.enrich_paper(paper, query)
            self.cache.set_paper_dict(self.paper_cache_key(paper), paper.model_dump(mode="json"))
            papers.append(paper)

        return papers

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        self.last_error = None
        cache_key = self.query_cache_key(query, limit)
        cached = self.cache.get_query(cache_key)
        if cached is not None:
            return [Paper.model_validate(item) for item in cached]

        papers = self._fetch(query, self._build_params(query, limit), limit)
        self.cache.set_query(cache_key, [paper.model_dump(mode="json") for paper in papers])
        return papers

    def search_title_exact(self, title: str, limit: int = 10) -> list[Paper]:
        self.last_error = None
        title = re.sub(r"[\r\n]+", " ", title).strip()
        if not title:
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

        papers = self._fetch(
            query,
            {
                "search_query": f'ti:"{title}"',
                "start": 0,
                "max_results": limit,
            },
            limit,
        )
        self.cache.set_query(cache_key, [paper.model_dump(mode="json") for paper in papers])
        return papers
