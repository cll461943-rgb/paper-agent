from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

from scholar_agent.infra.cache import JsonFileCache
from scholar_agent.infra.config import ProviderConfig
from scholar_agent.infra.http_client import HttpClient
from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider

LOGGER = logging.getLogger(__name__)


MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def _pubmed_publication_date(article: ET.Element) -> str | None:
    year = (
        article.findtext(".//Article/ArticleDate/Year")
        or article.findtext(".//JournalIssue/PubDate/Year")
    )
    if not year or not year.isdigit():
        return None
    month = (
        article.findtext(".//Article/ArticleDate/Month")
        or article.findtext(".//JournalIssue/PubDate/Month")
        or "12"
    )
    day = (
        article.findtext(".//Article/ArticleDate/Day")
        or article.findtext(".//JournalIssue/PubDate/Day")
        or "31"
    )
    month = MONTHS.get(month.strip().lower()[:3], month)
    try:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except ValueError:
        return f"{int(year):04d}-12-31"


class PubMedProvider(PaperProvider):
    name = "pubmed"

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

    def is_available(self) -> bool:
        return bool(self.config.base_url)

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        self.last_error = None
        cache_key = self.query_cache_key(query, limit)
        cached = self.cache.get_query(cache_key)
        if cached is not None:
            return [Paper.model_validate(item) for item in cached]

        # 1. 检索 PMID 列表 (esearch)
        esearch_url = self.config.base_url
        if esearch_url.endswith("efetch.fcgi"):
            esearch_url = esearch_url.replace("efetch.fcgi", "esearch.fcgi")
        elif not esearch_url.endswith("esearch.fcgi"):
            esearch_url = esearch_url.rstrip("/") + "/esearch.fcgi"

        esearch_params = {
            "db": "pubmed",
            "term": query.query,
            "retmode": "xml",
            "retmax": limit,
        }

        try:
            esearch_xml = self.http.get_text(esearch_url, params=esearch_params)
            esearch_root = ET.fromstring(esearch_xml)
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("PubMed esearch failed for query=%s: %s", query.query, exc)
            return []

        pmids = [id_elem.text for id_elem in esearch_root.findall(".//IdList/Id") if id_elem.text]
        if not pmids:
            self.cache.set_query(cache_key, [])
            return []

        # 2. 获取元数据 (efetch)
        efetch_url = self.config.base_url
        if efetch_url.endswith("esearch.fcgi"):
            efetch_url = efetch_url.replace("esearch.fcgi", "efetch.fcgi")
        elif not efetch_url.endswith("efetch.fcgi"):
            efetch_url = efetch_url.rstrip("/") + "/efetch.fcgi"

        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }

        try:
            efetch_xml = self.http.get_text(efetch_url, params=efetch_params)
            efetch_root = ET.fromstring(efetch_xml)
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("PubMed efetch failed for IDs=%s: %s", pmids, exc)
            return []

        papers: list[Paper] = []
        for article in efetch_root.findall(".//PubmedArticle")[:limit]:
            pmid = article.findtext(".//PMID")
            if not pmid:
                continue

            title = article.findtext(".//ArticleTitle") or "Untitled"
            
            abstract_texts = [text.text for text in article.findall(".//Abstract/AbstractText") if text.text]
            abstract = " ".join(abstract_texts) if abstract_texts else None

            # 提取发表年份
            year = None
            year_elem = article.find(".//JournalIssue/PubDate/Year")
            if year_elem is not None and year_elem.text and year_elem.text.isdigit():
                year = int(year_elem.text)
            else:
                medline_date = article.findtext(".//JournalIssue/PubDate/MedlineDate")
                if medline_date:
                    digit_match = re.search(r"\b(19|20)\d{2}\b", medline_date)
                    if digit_match:
                        year = int(digit_match.group(0))

            venue = article.findtext(".//Journal/Title") or article.findtext(".//Journal/ISOAbbreviation")

            # 提取作者
            authors = []
            for author in article.findall(".//AuthorList/Author"):
                lastname = author.findtext("LastName") or ""
                forename = author.findtext("ForeName") or ""
                name_parts = [forename, lastname]
                name = " ".join(part for part in name_parts if part).strip()
                if name:
                    authors.append(name)

            # 提取 DOI
            doi = None
            for article_id in article.findall(".//ArticleIdList/ArticleId"):
                if article_id.attrib.get("IdType") == "doi" and article_id.text:
                    doi = article_id.text.strip()
                    break

            paper = Paper(
                paper_id=f"pubmed:{pmid}",
                title=title,
                abstract=abstract,
                year=year,
                venue=venue,
                authors=authors,
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                source=self.name,
                metadata={
                    "raw_source": "pubmed",
                    "pmid": pmid,
                    "published_time": _pubmed_publication_date(article),
                },
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

        self.cache.set_query(cache_key, [paper.model_dump(mode="json") for paper in papers])
        return papers
