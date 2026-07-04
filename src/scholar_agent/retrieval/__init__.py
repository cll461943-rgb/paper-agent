from __future__ import annotations

from scholar_agent.retrieval.base import PaperProvider
from scholar_agent.retrieval.arxiv import ArxivProvider
try:
    from scholar_agent.retrieval.faiss_vector import FaissVectorProvider
except ImportError:  # pragma: no cover - depends on optional faiss runtime.
    FaissVectorProvider = None
from scholar_agent.retrieval.factory import build_providers
from scholar_agent.retrieval.mock_provider import MockPaperProvider
from scholar_agent.retrieval.multi_route import MultiRouteRetriever
from scholar_agent.retrieval.openalex import OpenAlexProvider
from scholar_agent.retrieval.pasa_local import PasaLocalProvider
from scholar_agent.retrieval.pubmed import PubMedProvider
from scholar_agent.retrieval.refchain import expand_with_refchain, annotate_refchain_candidate
from scholar_agent.retrieval.semantic_scholar import SemanticScholarProvider
from scholar_agent.retrieval.embedding_service import EmbeddingService
from scholar_agent.retrieval.rrf_fusion import rrf_fuse, rrf_fuse_with_vector
try:
    from scholar_agent.retrieval.vector_rerank import VectorReranker
except ImportError:  # pragma: no cover - depends on optional faiss runtime.
    VectorReranker = None
from scholar_agent.retrieval.semantic_bridge import SemanticBridge

__all__ = [
    "PaperProvider",
    "ArxivProvider",
    "FaissVectorProvider",
    "build_providers",
    "MockPaperProvider",
    "MultiRouteRetriever",
    "OpenAlexProvider",
    "PasaLocalProvider",
    "PubMedProvider",
    "expand_with_refchain",
    "annotate_refchain_candidate",
    "SemanticScholarProvider",
    "EmbeddingService",
    "rrf_fuse",
    "rrf_fuse_with_vector",
    "VectorReranker",
    "SemanticBridge",
]
