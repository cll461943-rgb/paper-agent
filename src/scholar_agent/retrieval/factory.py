from __future__ import annotations

import os

from scholar_agent.infra.cache import JsonFileCache
from scholar_agent.infra.config import AppConfig
from scholar_agent.retrieval.arxiv import ArxivProvider
from scholar_agent.retrieval.base import PaperProvider
from scholar_agent.retrieval.mock_provider import MockPaperProvider
from scholar_agent.retrieval.openalex import OpenAlexProvider
from scholar_agent.retrieval.pasa_local import PasaLocalProvider
from scholar_agent.retrieval.pubmed import PubMedProvider
from scholar_agent.retrieval.semantic_scholar import SemanticScholarProvider


def build_providers(
    config: AppConfig,
    mode: str | None = None,
    provider_names: list[str] | None = None,
) -> list[PaperProvider]:
    selected_mode = mode or config.app.mode
    explicit_provider_names = provider_names is not None
    if explicit_provider_names:
        selected_names = provider_names
    elif selected_mode == "live":
        selected_names = ["pasa_local", "openalex", "semantic_scholar", "arxiv", "pubmed"]
    else:
        selected_names = list(config.app.providers)
    cache = JsonFileCache(config.app.cache_dir)

    if selected_mode == "mock" and not explicit_provider_names:
        return [MockPaperProvider()]

    providers: list[PaperProvider] = []
    provider_map = {
        "openalex": OpenAlexProvider,
        "arxiv": ArxivProvider,
        "pasa_local": PasaLocalProvider,
        "semantic_scholar": SemanticScholarProvider,
        "pubmed": PubMedProvider,
        "mock": MockPaperProvider,
    }

    # Respect OPENALEX_EMAIL when config file leaves mailto empty.
    if not config.providers.openalex.mailto:
        config.providers.openalex.mailto = os.getenv("OPENALEX_EMAIL", "")

    for name in selected_names:
        builder = provider_map.get(name)
        if builder is None:
            continue
        if name == "mock":
            providers.append(builder())
        elif name == "pasa_local" and (explicit_provider_names or config.providers.pasa_local.enabled):
            providers.append(builder(config.providers.pasa_local))
        elif name == "openalex" and (explicit_provider_names or config.providers.openalex.enabled):
            providers.append(builder(config.providers.openalex, cache))
        elif name == "arxiv" and (explicit_provider_names or config.providers.arxiv.enabled):
            providers.append(builder(config.providers.arxiv, cache))
        elif name == "semantic_scholar" and (explicit_provider_names or config.providers.semantic_scholar.enabled):
            providers.append(builder(config.providers.semantic_scholar, cache))
        elif name == "pubmed" and (explicit_provider_names or config.providers.pubmed.enabled):
            providers.append(builder(config.providers.pubmed, cache))

    return providers
