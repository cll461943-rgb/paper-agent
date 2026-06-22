from __future__ import annotations

import json
import math
import re
import sqlite3
import zipfile
from heapq import nlargest
from collections import Counter, defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

from scholar_agent.infra.config import ProviderConfig
from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider


def _keep_letters(text: str) -> str:
    return "".join(char for char in text if char.isalpha()).lower()


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(token) > 1]


QUERY_STOPWORDS = {
    "about", "aimed", "all", "also", "and", "any", "are", "can", "could",
    "describe", "discuss", "discussed", "do", "does", "effects", "for",
    "from", "give", "have", "how", "implemented", "in", "increasing",
    "is", "knowledge", "list", "me", "mention", "models", "of", "on",
    "papers", "provide", "related", "some", "studies", "study", "that",
    "the", "their", "to", "using", "what", "which", "with", "work",
    "works", "would", "you",
}


def _query_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in QUERY_STOPWORDS and len(token) > 2]


class PasaLocalProvider(PaperProvider):
    name = "pasa_local"
    max_token_postings = 80_000
    _CACHE_LOCK = Lock()
    _INDEX_CACHE: dict[str, dict[str, Any]] = {}

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.root = Path(config.base_url)
        self.id2paper_path = self.root / "id2paper.json"
        self.paper_zip_path = self.root / "cs_paper_2nd.zip"
        self.fts_path = self.root / "pasa_local_fts.sqlite"
        self.enable_fts = bool(getattr(config, "enable_fts", False))
        self._available = self.id2paper_path.exists() and self.paper_zip_path.exists()
        self._cache_key = str(self.root.resolve())
        self._index_ready = False
        self._paper_index: list[dict[str, Any]] = []
        self._title_to_record: dict[str, dict[str, Any]] = {}
        self._arxiv_to_record: dict[str, dict[str, Any]] = {}
        self._token_to_doc_ids: dict[str, list[int]] = {}
        self._id2paper: dict[str, str] = {}
        self._zip: zipfile.ZipFile | None = None
        self._fts_conn: sqlite3.Connection | None = None
        self._title_keys: set[str] = set()

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
            self._id2paper = cached["id2paper"]
            self._title_keys = cached["title_keys"]
            self._index_ready = True

    def _build_index(self) -> dict[str, Any]:
        paper_index: list[dict[str, Any]] = []
        title_to_record: dict[str, dict[str, Any]] = {}
        arxiv_to_record: dict[str, dict[str, Any]] = {}
        token_to_doc_ids: dict[str, set[int]] = defaultdict(set)
        id2paper: dict[str, str] = {}
        title_keys: set[str] = set()

        try:
            id2paper = json.loads(self.id2paper_path.read_text(encoding="utf-8"))
        except Exception:
            id2paper = {}

        for arxiv_id, title in id2paper.items():
            key = _keep_letters(title)
            if not key:
                continue
            record = {
                "arxiv_id": arxiv_id,
                "title_key": key,
                "title": title,
                "abstract": None,
                "sections": None,
                "tokens": Counter(_tokenize(title)),
            }
            doc_id = len(paper_index)
            paper_index.append(record)
            title_to_record[key] = record
            arxiv_to_record[arxiv_id] = record
            title_keys.add(key)
            for token in record["tokens"].keys():
                token_to_doc_ids[token].add(doc_id)

        return {
            "paper_index": paper_index,
            "title_to_record": title_to_record,
            "arxiv_to_record": arxiv_to_record,
            "token_to_doc_ids": {token: sorted(doc_ids) for token, doc_ids in token_to_doc_ids.items()},
            "id2paper": id2paper,
            "title_keys": title_keys,
        }

    def _open_zip(self) -> zipfile.ZipFile | None:
        if self._zip is None and self.paper_zip_path.exists():
            self._zip = zipfile.ZipFile(self.paper_zip_path, "r")
        return self._zip

    def _load_paper_json(self, title_key: str) -> dict[str, Any] | None:
        if title_key not in self._title_keys:
            return None
        archive = self._open_zip()
        if archive is None:
            return None
        try:
            return json.loads(archive.read(title_key).decode("utf-8"))
        except Exception:
            return None

    def _open_fts(self) -> sqlite3.Connection | None:
        if self._fts_conn is not None:
            return self._fts_conn
        if not self.enable_fts:
            return None
        if not self.fts_path.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{self.fts_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            self._fts_conn = conn
            return conn
        except sqlite3.Error:
            return None

    def _record_to_paper(self, record: dict[str, Any], query: SearchQuery) -> Paper:
        paper = Paper(
            paper_id=f"ARXIV:{record['arxiv_id']}",
            title=record["title"],
            abstract=None,
            arxiv_id=record["arxiv_id"],
            source=self.name,
            metadata={
                "raw_source": "pasa_local",
                "title_key": record["title_key"],
                "sections": None,
            },
        )
        return self.enrich_paper(paper, query)

    def _fts_row_to_paper(self, row: sqlite3.Row, query: SearchQuery) -> Paper:
        paper = Paper(
            paper_id=f"ARXIV:{row['arxiv_id']}",
            title=row["title"],
            abstract=row["abstract"],
            arxiv_id=row["arxiv_id"],
            source=self.name,
            metadata={
                "raw_source": "pasa_local_fts",
                "title_key": row["title_key"],
                "sections": None,
            },
        )
        return self.enrich_paper(paper, query)

    def _score(self, record: dict[str, Any], query_tokens: list[str]) -> float:
        tf = record["tokens"]
        if not query_tokens:
            return 0.0
        score = 0.0
        for token in query_tokens:
            score += 1 + math.log(tf[token]) if tf[token] > 0 else 0.0
        if _keep_letters(" ".join(query_tokens)) in record["title_key"]:
            score += 5.0
        return score

    @staticmethod
    def _fts_match_query(query_tokens: list[str]) -> str:
        terms = []
        seen = set()
        for token in query_tokens:
            if token in seen or not re.fullmatch(r"[A-Za-z0-9]+", token):
                continue
            seen.add(token)
            terms.append(f'"{token}"')
            if len(terms) >= 12:
                break
        return " OR ".join(terms)

    def _search_title_index(self, query: SearchQuery, query_tokens: list[str], limit: int) -> list[Paper]:
        candidate_doc_ids = set()
        for token in query_tokens:
            doc_ids = self._token_to_doc_ids.get(token, [])
            if len(doc_ids) > self.max_token_postings:
                continue
            candidate_doc_ids.update(doc_ids)
        if not candidate_doc_ids:
            for token in query_tokens:
                candidate_doc_ids.update(self._token_to_doc_ids.get(token, [])[: self.max_token_postings])
        scored = []
        for doc_id in candidate_doc_ids:
            record = self._paper_index[doc_id]
            score = self._score(record, query_tokens)
            if score > 0:
                scored.append((score, record))
        top_scored = nlargest(limit, scored, key=lambda item: item[0])
        return [self._record_to_paper(record, query) for _, record in top_scored]

    def _search_fts_index(self, query: SearchQuery, query_tokens: list[str], limit: int) -> list[Paper]:
        conn = self._open_fts()
        if conn is None:
            return []
        
        terms = []
        seen = set()
        for token in query_tokens:
            if token in seen or not re.fullmatch(r"[A-Za-z0-9]+", token):
                continue
            seen.add(token)
            terms.append(f'"{token}"')
            if len(terms) >= 12:
                break
        
        if not terms:
            return []
            
        # 1. 高精度：核心词 AND 模式
        and_query = " AND ".join(terms)
        rows = []
        try:
            rows = conn.execute(
                """
                SELECT arxiv_id, title, abstract, title_key
                FROM papers_fts
                WHERE papers_fts MATCH ?
                ORDER BY bm25(papers_fts, 3.0, 1.0, 0.35)
                LIMIT ?
                """,
                (and_query, limit),
            ).fetchall()
        except sqlite3.Error:
            pass
            
        # 2. 如果高精度返回的结果不足，退避到高召回：核心词 OR 模式
        if len(rows) < limit:
            or_query = " OR ".join(terms)
            try:
                or_rows = conn.execute(
                    """
                    SELECT arxiv_id, title, abstract, title_key
                    FROM papers_fts
                    WHERE papers_fts MATCH ?
                    ORDER BY bm25(papers_fts, 3.0, 1.0, 0.35)
                    LIMIT ?
                    """,
                    (or_query, limit),
                ).fetchall()
                existing_arxiv_ids = {row["arxiv_id"] for row in rows}
                for row in or_rows:
                    if row["arxiv_id"] not in existing_arxiv_ids:
                        rows.append(row)
                        if len(rows) >= limit:
                            break
            except sqlite3.Error:
                pass
                
        return [self._fts_row_to_paper(row, query) for row in rows[:limit]]

    @staticmethod
    def _dedupe_papers(papers: list[Paper], limit: int) -> list[Paper]:
        deduped: list[Paper] = []
        seen: set[str] = set()
        for paper in papers:
            key = (paper.arxiv_id or paper.paper_id or paper.title).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(paper)
            if len(deduped) >= limit:
                break
        return deduped

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        self._ensure_index()
        if not self._available:
            return []
        query_tokens = _query_tokens(query.query + " " + " ".join(query.required_terms + query.optional_terms))
        if not query_tokens:
            return []
        fts_results = self._search_fts_index(query, query_tokens, limit)
        title_results = self._search_title_index(query, query_tokens, limit)
        return self._dedupe_papers([*fts_results, *title_results], limit)

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
            title_key = paper.metadata.get("title_key") or _keep_letters(paper.title)
            paper_json = self._load_paper_json(title_key) or {}
            sections = paper_json.get("sections", [])
        if not sections:
            return []

        reference_titles: list[str] = []

        def add_title(value: Any) -> None:
            if isinstance(value, str):
                title = value.strip()
            elif isinstance(value, dict):
                raw_title = value.get("title")
                title = raw_title.strip() if isinstance(raw_title, str) else ""
            else:
                title = ""
            if title and title not in reference_titles:
                reference_titles.append(title)

        if isinstance(sections, dict):
            for section_refs in sections.values():
                if isinstance(section_refs, list):
                    for item in section_refs:
                        add_title(item)
                else:
                    add_title(section_refs)
        elif isinstance(sections, list):
            for item in sections:
                add_title(item)
        else:
            return []

        references: list[Paper] = []
        for title in reference_titles:
            found = self.search_title_exact(title, limit=1)
            references.extend(found)
            if len(references) >= limit:
                break
        return references[:limit]
