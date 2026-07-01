from __future__ import annotations

from collections.abc import Callable

from scholar_agent.models.schemas import Paper
from scholar_agent.retrieval.base import PaperProvider

RefchainOperation = tuple[str, str, Callable[[Paper, int], list[Paper]]]

REFCHAIN_ROUTE_SOURCES = {
    "reference_expansion": "references",
    "citation_expansion": "citations",
    "related_work_expansion": "related",
}

REFCHAIN_METHODS = [
    ("reference_expansion", "get_references"),
    ("citation_expansion", "get_citations"),
    ("related_work_expansion", "get_related_works"),
]


def _provider_overrides(method_name: str, provider: PaperProvider) -> bool:
    return getattr(type(provider), method_name) is not getattr(PaperProvider, method_name)


def refchain_operations(provider: PaperProvider) -> list[RefchainOperation]:
    operations: list[RefchainOperation] = []
    for route, method_name in REFCHAIN_METHODS:
        if _provider_overrides(method_name, provider):
            operations.append((route, REFCHAIN_ROUTE_SOURCES[route], getattr(provider, method_name)))
    return operations


def _append_unique(items: list[str], item: str) -> None:
    if item and item not in items:
        items.append(item)


def _append_unique_edge(edges: list[dict], edge: dict) -> None:
    if edge not in edges:
        edges.append(edge)


def annotate_refchain_candidate(
    paper: Paper,
    *,
    provider_name: str,
    route: str,
    source: str,
    seed_paper: Paper,
    seed_index: int,
    limit_per_seed: int,
) -> Paper:
    candidate = paper.model_copy(deep=True)
    candidate.source = provider_name
    _append_unique(candidate.retrieval_path, f"provider:{provider_name}")
    _append_unique(candidate.retrieval_path, f"route:{route}")
    _append_unique(candidate.retrieval_path, f"refchain_source:{source}")
    _append_unique(candidate.retrieval_path, f"refchain_seed:{seed_paper.paper_id}")

    edge = {
        "provider": provider_name,
        "route": route,
        "source": source,
        "seed_paper_id": seed_paper.paper_id,
        "seed_title": seed_paper.title,
        "seed_source": seed_paper.source,
        "seed_index": seed_index,
        "limit_per_seed": limit_per_seed,
    }
    candidate.metadata["refchain_provider"] = provider_name
    candidate.metadata["refchain_route"] = route
    candidate.metadata["refchain_source"] = source
    candidate.metadata["refchain_seed_paper_id"] = seed_paper.paper_id
    candidate.metadata["refchain_seed_title"] = seed_paper.title
    candidate.metadata["refchain_seed_source"] = seed_paper.source
    candidate.metadata["refchain_seed_index"] = seed_index
    candidate.metadata["refchain_limit_per_seed"] = limit_per_seed
    edges = candidate.metadata.get("refchain_edges", [])
    if not isinstance(edges, list):
        edges = []
    _append_unique_edge(edges, edge)
    candidate.metadata["refchain_edges"] = edges
    return candidate


def expand_with_refchain(seed_papers: list[Paper], providers: list[PaperProvider], limit_per_seed: int = 5) -> list[Paper]:
    expanded: list[Paper] = []
    if limit_per_seed <= 0:
        return expanded
    for provider in providers:
        operations = refchain_operations(provider)
        for seed_index, paper in enumerate(seed_papers, start=1):
            # Each route gets its own independent budget
            for route, source, operation in operations:
                papers = operation(paper, limit=limit_per_seed)[:limit_per_seed]
                expanded.extend(
                    annotate_refchain_candidate(
                        candidate,
                        provider_name=provider.name,
                        route=route,
                        source=source,
                        seed_paper=paper,
                        seed_index=seed_index,
                        limit_per_seed=limit_per_seed,
                    )
                    for candidate in papers
                )
    return expanded
