from __future__ import annotations

from scholar_agent.retrieval.base import PaperProvider
from scholar_agent.retrieval.arxiv import ArxivProvider
from scholar_agent.retrieval.factory import build_providers
from scholar_agent.retrieval.mock_provider import MockPaperProvider
from scholar_agent.retrieval.multi_route import MultiRouteRetriever
from scholar_agent.retrieval.openalex import OpenAlexProvider
from scholar_agent.retrieval.pasa_local import PasaLocalProvider
from scholar_agent.retrieval.pubmed import PubMedProvider
from scholar_agent.retrieval.refchain import expand_with_refchain, annotate_refchain_candidate
from scholar_agent.retrieval.semantic_scholar import SemanticScholarProvider

__all__ = [
    "PaperProvider",
    "ArxivProvider",
    "build_providers",
    "MockPaperProvider",
    "MultiRouteRetriever",
    "OpenAlexProvider",
    "PasaLocalProvider",
    "PubMedProvider",
    "expand_with_refchain",
    "annotate_refchain_candidate",
    "SemanticScholarProvider",
]
