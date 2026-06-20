from __future__ import annotations

import json
import sqlite3
import zipfile

from scholar_agent.infra.config import ProviderConfig
from scholar_agent.models.schemas import SearchQuery
from scholar_agent.retrieval.pasa_local import PasaLocalProvider


def test_pasa_local_title_exact_search_does_not_open_zip_during_index_build(tmp_path, monkeypatch):
    root = tmp_path / "paper_database"
    root.mkdir()
    (root / "id2paper.json").write_text(
        json.dumps({"1706.03762": "Attention Is All You Need"}),
        encoding="utf-8",
    )
    (root / "cs_paper_2nd.zip").write_bytes(b"placeholder")

    provider = PasaLocalProvider(ProviderConfig(base_url=str(root)))

    class ForbiddenZip:
        def __init__(self, *args, **kwargs):
            raise AssertionError("zipfile should not be opened during title-only index build")

    monkeypatch.setattr(zipfile, "ZipFile", ForbiddenZip)

    papers = provider.search_title_exact("Attention Is All You Need", limit=1)

    assert len(papers) == 1
    assert papers[0].title == "Attention Is All You Need"
    assert papers[0].arxiv_id == "1706.03762"


def test_pasa_local_search_skips_overly_broad_title_tokens(tmp_path):
    root = tmp_path / "paper_database"
    root.mkdir()
    (root / "id2paper.json").write_text(
        json.dumps(
            {
                "1": "Common Token Noise",
                "2": "Common Token Signal",
                "3": "Common Token Extra",
                "4": "Rare Signal Paper",
            }
        ),
        encoding="utf-8",
    )
    (root / "cs_paper_2nd.zip").write_bytes(b"placeholder")

    provider = PasaLocalProvider(ProviderConfig(base_url=str(root)))
    provider.max_token_postings = 2

    papers = provider.search(
        SearchQuery(query="common rare", route="core_topic", intent="topic_recall"),
        limit=1,
    )

    assert papers[0].title == "Rare Signal Paper"


def test_pasa_local_get_references_reads_section_reference_titles(tmp_path):
    root = tmp_path / "paper_database"
    root.mkdir()
    (root / "id2paper.json").write_text(
        json.dumps(
            {
                "1": "Seed Paper",
                "2": "Reference Alpha",
                "3": "Reference Beta",
                "4": "Unrelated Paper",
            }
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(root / "cs_paper_2nd.zip", "w") as archive:
        archive.writestr(
            "seedpaper",
            json.dumps(
                {
                    "title": "Seed Paper",
                    "abstract": "A seed paper.",
                    "sections": {
                        "1 Introduction": [
                            "Reference Alpha",
                            {"title": "Reference Beta"},
                            "Missing Reference",
                        ],
                        "2 Related Work": ["Reference Alpha"],
                    },
                }
            ),
        )

    provider = PasaLocalProvider(ProviderConfig(base_url=str(root)))
    seed = provider.search_title_exact("Seed Paper", limit=1)[0]

    references = provider.get_references(seed, limit=2)

    assert [paper.title for paper in references] == ["Reference Alpha", "Reference Beta"]


def test_pasa_local_search_uses_optional_fts_sidecar_for_abstract_terms(tmp_path):
    root = tmp_path / "paper_database"
    root.mkdir()
    (root / "id2paper.json").write_text(
        json.dumps({"1": "Hidden Gold Paper"}),
        encoding="utf-8",
    )
    (root / "cs_paper_2nd.zip").write_bytes(b"placeholder")

    conn = sqlite3.connect(root / "pasa_local_fts.sqlite")
    conn.execute(
        """
        CREATE VIRTUAL TABLE papers_fts USING fts5(
            arxiv_id UNINDEXED,
            title,
            abstract,
            refs,
            title_key UNINDEXED,
            tokenize='unicode61'
        )
        """
    )
    conn.execute(
        "INSERT INTO papers_fts(arxiv_id, title, abstract, refs, title_key) VALUES (?, ?, ?, ?, ?)",
        (
            "1",
            "Hidden Gold Paper",
            "This work requires actual calls to a real API.",
            "",
            "hiddengoldpaper",
        ),
    )
    conn.commit()
    conn.close()

    provider = PasaLocalProvider(ProviderConfig(base_url=str(root), enable_fts=True))

    papers = provider.search(
        SearchQuery(query="actual real API calls", route="core_topic", intent="topic_recall"),
        limit=5,
    )

    assert [paper.title for paper in papers] == ["Hidden Gold Paper"]
    assert papers[0].abstract == "This work requires actual calls to a real API."
