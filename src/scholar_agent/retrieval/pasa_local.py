from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from threading import Lock

from scholar_agent.infra.config import ProviderConfig
from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider


def _keep_letters(text: str) -> str:
    return "".join(char for char in text if char.isalpha()).lower()


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(token) > 1]


class PasaLocalProvider(PaperProvider):
    name = "pasa_local"
    _CACHE_LOCK = Lock()
    _INDEX_CACHE: dict[str, dict] = {}

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.root = Path(config.base_url)
        self.id2paper_path = self.root / "id2paper.json"
        self.paper_zip_path = self.root / "cs_paper_2nd.zip"
        self._available = self.id2paper_path.exists() and self.paper_zip_path.exists()
        self._cache_key = str(self.root.resolve())
        self._index_ready = False
        self._paper_index: list[dict] = []
        self._title_to_record: dict[str, dict] = {}
        self._arxiv_to_record: dict[str, dict] = {}
        self._token_to_doc_ids: dict[str, list[int]] = {}

    def is_available(self) -> bool:
        return self._available

    def _ensure_index(self) -> None:
        if not self._available or self._index_ready:
            return
        with self._CACHE_LOCK:
            cached = self._INDEX_CACHE.get(self._cache_key)
            if cached is None:
                cached = self._build_index()
                self._INDEX_CACHE[self._cache_key] = cached
            self._paper_index = cached["paper_index"]
            self._title_to_record = cached["title_to_record"]
            self._arxiv_to_record = cached["arxiv_to_record"]
            self._token_to_doc_ids = cached["token_to_doc_ids"]
            self._index_ready = True

    def _build_index(self) -> dict[str, dict]:
        paper_index: list[dict] = []
        title_to_record: dict[str, dict] = {}
        arxiv_to_record: dict[str, dict] = {}
        token_to_doc_ids: dict[str, set[int]] = defaultdict(set)

        try:
            id2paper = json.loads(self.id2paper_path.read_text(encoding="utf-8"))
        except Exception:
            id2paper = {}

        if id2paper and self.paper_zip_path.exists():
            try:
                with zipfile.ZipFile(self.paper_zip_path, "r") as archive:
                    available_names = set(archive.namelist())
                    for arxiv_id, title in id2paper.items():
                        key = _keep_letters(title)
                        if key not in available_names:
                            continue
                        try:
                            paper_json = json.loads(archive.read(key).decode("utf-8"))
                        except Exception:
                            continue
                        text = f"{paper_json.get('title', '')} {paper_json.get('abstract', '')}"
                        tokens = Counter(_tokenize(text))
                        record = {
                            "arxiv_id": arxiv_id,
                            "title_key": key,
                            "title": paper_json.get("title", title),
                            "abstract": paper_json.get("abstract"),
                            "sections": paper_json.get("sections", []),
                            "tokens": tokens,
                        }
                        doc_id = len(paper_index)
                        paper_index.append(record)
                        title_to_record[key] = record
                        arxiv_to_record[arxiv_id] = record
                        for token in tokens.keys():
                            token_to_doc_ids[token].add(doc_id)
            except Exception:
                pass

        return {
            "paper_index": paper_index,
            "title_to_record": title_to_record,
            "arxiv_to_record": arxiv_to_record,
            "token_to_doc_ids": {token: sorted(doc_ids) for token, doc_ids in token_to_doc_ids.items()},
        }

    def _record_to_paper(self, record: dict, query: SearchQuery) -> Paper:
        paper = Paper(
            paper_id=f"ARXIV:{record['arxiv_id']}",
            title=record["title"],
            abstract=record.get("abstract"),
            arxiv_id=record["arxiv_id"],
            source=self.name,
            metadata={
                "raw_source": "pasa_local",
                "sections": record.get("sections", []),
            },
        )
        return self.enrich_paper(paper, query)

    def _score(self, record: dict, query_tokens: list[str]) -> float:
        tf = record["tokens"]
        if not query_tokens:
            return 0.0
        score = 0.0
        for token in query_tokens:
            score += 1 + math.log(tf[token]) if tf[token] > 0 else 0.0
        if _keep_letters(" ".join(query_tokens)) in _keep_letters(record["title"]):
            score += 5.0
        return score

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        self._ensure_index()
        if not self._available:
            return []
        query_tokens = _tokenize(query.query + " " + " ".join(query.required_terms + query.optional_terms))
        if not query_tokens:
            return []
        candidate_doc_ids = set()
        for token in query_tokens:
            candidate_doc_ids.update(self._token_to_doc_ids.get(token, []))
        scored = []
        for doc_id in candidate_doc_ids:
            record = self._paper_index[doc_id]
            score = self._score(record, query_tokens)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._record_to_paper(record, query) for _, record in scored[:limit]]

    def search_title_exact(self, title: str, limit: int = 10) -> list[Paper]:
        self._ensure_index()
        key = _keep_letters(title)
        record = self._title_to_record.get(key)
        if record is None:
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
        return [self._record_to_paper(record, query)][:limit]

    def get_references(self, paper: Paper, limit: int) -> list[Paper]:
        sections = paper.metadata.get("sections", [])
        if not sections:
            return []
        if isinstance(sections, dict):
            section_items = list(sections.values())
        elif isinstance(sections, list):
            section_items = sections
        else:
            return []
        query = SearchQuery(
            query=paper.title,
            route="reference_expansion",
            intent="reference_expansion",
            required_terms=[],
            optional_terms=[],
            filters={},
            priority=99,
        )
        references: list[Paper] = []
        for section in section_items[:limit]:
            title = section.get("title") if isinstance(section, dict) else None
            if not title:
                continue
            found = self.search_title_exact(title, limit=1)
            references.extend(found)
        return references[:limit]
