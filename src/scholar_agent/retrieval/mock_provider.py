from __future__ import annotations

from scholar_agent.models.schemas import Paper, SearchQuery
from scholar_agent.retrieval.base import PaperProvider


class MockPaperProvider(PaperProvider):
    name = "mock"

    def __init__(self) -> None:
        self._papers = [
            Paper(
                paper_id="paper-001",
                title="Multimodal Large Language Models for Radiology Report Generation on MIMIC-CXR",
                abstract=(
                    "We study multimodal large language models for radiology report generation "
                    "and validate on MIMIC-CXR with evidence-aware evaluation."
                ),
                year=2025,
                venue="MICCAI",
                authors=["A. Chen", "B. Li"],
                citation_count=28,
                url="https://example.org/paper-001",
                source="mock",
                retrieval_path=["mock_seed"],
                metadata={"datasets": ["MIMIC-CXR"], "methods": ["multimodal large language model"]},
            ),
            Paper(
                paper_id="paper-002",
                title="Medical Image Report Generation with Vision-Language Models on MIMIC-CXR",
                abstract=(
                    "A vision-language model is applied to chest X-ray report generation "
                    "with experiments on MIMIC-CXR."
                ),
                year=2024,
                venue="EMNLP Findings",
                authors=["C. Wang", "D. Xu"],
                citation_count=19,
                url="https://example.org/paper-002",
                source="mock",
                retrieval_path=["mock_seed"],
                metadata={"datasets": ["MIMIC-CXR"], "methods": ["vision-language model"]},
            ),
            Paper(
                paper_id="paper-003",
                title="Scaling Multimodal LLMs for Clinical Report Generation",
                abstract=(
                    "We scale multimodal LLMs for clinical report generation and include "
                    "a MIMIC-CXR evaluation subset."
                ),
                year=2026,
                venue="ACL",
                authors=["E. Zhao"],
                citation_count=7,
                url="https://example.org/paper-003",
                source="mock",
                retrieval_path=["mock_seed"],
                metadata={"datasets": ["MIMIC-CXR"], "methods": ["multimodal LLM"]},
            ),
            Paper(
                paper_id="paper-004",
                title="Retrieval-Augmented Chest X-Ray Report Generation",
                abstract=(
                    "A retrieval-augmented transformer improves chest X-ray report generation "
                    "on IU-Xray and partially on MIMIC-CXR."
                ),
                year=2023,
                venue="AAAI",
                authors=["F. Sun"],
                citation_count=33,
                url="https://example.org/paper-004",
                source="mock",
                retrieval_path=["mock_seed"],
                metadata={"datasets": ["IU-Xray", "MIMIC-CXR"], "methods": ["retrieval-augmented transformer"]},
            ),
            Paper(
                paper_id="paper-005",
                title="Large Language Models for Drug Discovery: A Survey",
                abstract="Survey of LLM applications in drug discovery without radiology datasets.",
                year=2024,
                venue="Nature Machine Intelligence",
                authors=["G. Luo"],
                citation_count=102,
                url="https://example.org/paper-005",
                source="mock",
                retrieval_path=["mock_seed"],
                metadata={"datasets": [], "methods": ["large language model"]},
            ),
        ]

    def search(self, query: SearchQuery, limit: int) -> list[Paper]:
        papers = [paper.model_copy(deep=True) for paper in self._papers[:limit]]
        return [self.enrich_paper(paper, query) for paper in papers]

    def search_title_exact(self, title: str, limit: int = 10) -> list[Paper]:
        matched = []
        for paper in self._papers:
            if paper.title.lower() == title.lower():
                matched.append(paper.model_copy(deep=True))
        if matched:
            query = SearchQuery(
                query=title,
                route="title_exact",
                intent="title_exact_recall",
                required_terms=[],
                optional_terms=[],
                filters={},
                priority=0,
            )
            return [self.enrich_paper(p, query) for p in matched[:limit]]
        return []

    def get_references(self, paper: Paper, limit: int) -> list[Paper]:
        # 返回 Mock 数据集的其他几篇作为 reference
        query = SearchQuery(
            query=paper.title,
            route="reference_expansion",
            intent="reference_expansion",
            required_terms=[],
            optional_terms=[],
            filters={},
            priority=99,
        )
        return [self.enrich_paper(p.model_copy(deep=True), query) for p in self._papers if p.paper_id != paper.paper_id][:limit]
