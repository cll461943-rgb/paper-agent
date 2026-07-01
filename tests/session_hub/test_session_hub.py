"""Tests for the session_hub package.

Run from the repo root::

    .venv/Scripts/python -m pytest tests/session_hub -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from session_hub.connectors import (  # noqa: E402
    ClaudeCodeConnector,
    CodexConnector,
    AiderConnector,
    GeminiCliConnector,
)
from session_hub.index import SessionIndex  # noqa: E402
from session_hub.share import export_session, import_bundle  # noqa: E402
from session_hub.cli import main as cli_main  # noqa: E402
from session_hub.models import AgentKind, Conversation, Message, Snippet  # noqa: E402


# ---------------------------------------------------------------- fixtures
@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "index.sqlite"


# ---------------------------------------------------------------- connector
def test_claude_code_connector_parses_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "session-x.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": "fix the auth bug"},
                        "timestamp": "2026-06-30T10:00:00Z",
                        "cwd": "/tmp/proj",
                        "sessionId": "session-x",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Read",
                                    "input": {"path": "/tmp/proj/auth.py"},
                                }
                            ],
                        },
                        "timestamp": "2026-06-30T10:00:05Z",
                        "sessionId": "session-x",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    conn = ClaudeCodeConnector()
    convs = list(conn.iter_sessions(tmp_path))
    assert len(convs) == 1
    c = convs[0]
    assert c.agent == AgentKind.CLAUDE_CODE
    assert c.session_id == "session-x"
    assert c.workspace == "/tmp/proj"
    assert c.title.startswith("fix the auth bug")
    assert len(c.messages) == 2
    assert "[Tool: Read]" in c.messages[1].plain_text()


def test_codex_connector_parses_rollout(tmp_path: Path) -> None:
    p = tmp_path / "rollout-2026-04-12T20-42-04-abc123def456.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": "2026-04-12T20:42:04Z",
                        "payload": {
                            "role": "user",
                            "content": [{"text": "summarize the diff"}],
                            "cwd": "/tmp/proj",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": "2026-04-12T20:42:30Z",
                        "payload": {
                            "role": "assistant",
                            "content": [{"text": "The diff adds three new fields."}],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    conn = CodexConnector()
    convs = list(conn.iter_sessions(tmp_path))
    assert len(convs) == 1
    c = convs[0]
    assert c.agent == AgentKind.CODEX
    assert c.workspace == "/tmp/proj"
    assert "summarize the diff" in c.messages[0].plain_text()


def test_aider_connector_parses_markdown(tmp_path: Path) -> None:
    p = tmp_path / ".aider.chat.history.md"
    p.write_text(
        "#### <user>\nadd docstrings\n</user>\n#### <assistant>\nDone.\n</assistant>\n",
        encoding="utf-8",
    )
    # Aider connector looks under the parent of the file.
    conn = AiderConnector()
    convs = list(conn.iter_sessions(tmp_path))
    assert len(convs) == 1
    c = convs[0]
    assert c.agent == AgentKind.AIDER
    assert c.messages[0].role == "user"
    assert "add docstrings" in c.messages[0].plain_text()


def test_gemini_cli_connector_parses_json(tmp_path: Path) -> None:
    chat_dir = tmp_path / "tmp" / "proj" / "chats"
    chat_dir.mkdir(parents=True)
    p = chat_dir / "session-2026-05-01T10-00-00-abcd.json"
    p.write_text(
        json.dumps(
            {
                "sessionId": "abcd",
                "projectDir": "/tmp/proj",
                "messages": [
                    {"role": "user", "content": "explain transformers", "timestamp": "2026-05-01T10:00:00Z"},
                    {"role": "assistant", "content": "they use self-attention", "timestamp": "2026-05-01T10:00:30Z"},
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = GeminiCliConnector()
    convs = list(conn.iter_sessions(tmp_path))
    assert len(convs) == 1
    c = convs[0]
    assert c.agent == AgentKind.GEMINI_CLI
    assert c.workspace == "/tmp/proj"
    assert "transformers" in c.messages[0].plain_text()


# ------------------------------------------------------------------- indexer
def test_index_upsert_and_search(tmp_db: Path) -> None:
    idx = SessionIndex(tmp_db)
    conv = Conversation(
        agent=AgentKind.CLAUDE_CODE,
        session_id="s1",
        source_path="/tmp/s1.jsonl",
        workspace="/tmp/proj",
        title="auth bug",
        messages=[
            Message(role="user", ts="2026-06-30T10:00:00Z", snippets=[Snippet("text", "fix the auth bug")]),
            Message(role="assistant", ts="2026-06-30T10:00:05Z", snippets=[Snippet("text", "patching middleware now")]),
        ],
    )
    idx.upsert_conversation(conv)
    assert idx.get_session("claude_code", "s1")["message_count"] == 2
    hits = idx.search("auth")
    assert len(hits) >= 1
    assert "auth" in hits[0]["hit"].lower()
    # Re-upsert should replace, not duplicate.
    idx.upsert_conversation(conv)
    assert idx.list_sessions(limit=10).__len__() == 1


def test_index_search_filters_by_agent(tmp_db: Path) -> None:
    idx = SessionIndex(tmp_db)
    for agent, sid in [("claude_code", "s1"), ("codex", "s2")]:
        idx.upsert_conversation(
            Conversation(
                agent=AgentKind(agent),
                session_id=sid,
                source_path=f"/tmp/{sid}",
                messages=[Message(role="user", snippets=[Snippet("text", "fix the auth bug")])],
            )
        )
    hits = idx.search("auth", agent="codex")
    assert all(h["agent"] == "codex" for h in hits)


def test_index_stats(tmp_db: Path) -> None:
    idx = SessionIndex(tmp_db)
    idx.upsert_conversation(
        Conversation(
            agent=AgentKind.CLAUDE_CODE,
            session_id="a",
            source_path="/x",
            messages=[Message(role="user", snippets=[Snippet("text", "hi")])],
        )
    )
    s = idx.stats()
    assert s["messages_total"] == 1
    assert s["sessions_by_agent"]["claude_code"] == 1


# --------------------------------------------------------------------- share
def test_share_round_trip(tmp_path: Path) -> None:
    db_a = tmp_path / "a.sqlite"
    idx_a = SessionIndex(db_a)
    idx_a.upsert_conversation(
        Conversation(
            agent=AgentKind.CLAUDE_CODE,
            session_id="shared-1",
            source_path="/tmp/shared.jsonl",
            workspace="/tmp/proj",
            title="shared title",
            messages=[
                Message(role="user", ts="2026-06-30T10:00:00Z", snippets=[Snippet("text", "explain RAG")]),
                Message(role="assistant", ts="2026-06-30T10:00:05Z", snippets=[Snippet("text", "retrieval augmented generation")]),
            ],
        )
    )
    bundle = tmp_path / "shared.json"
    export_session(db_a, "claude_code", "shared-1", bundle)
    assert bundle.exists()
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    assert payload["bundle_version"] == 1
    assert payload["session"]["session_id"] == "shared-1"
    # Ingest into a second DB.
    db_b = tmp_path / "b.sqlite"
    idx_b = SessionIndex(db_b)
    res = import_bundle(bundle, idx_b)
    assert res["session_id"] == "shared-1"
    assert res["messages_indexed"] == 2
    assert idx_b.search("RAG")[0]["workspace"] == "/tmp/proj"


# ----------------------------------------------------------------------- CLI
def test_cli_search_json(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    idx = SessionIndex(db)
    idx.upsert_conversation(
        Conversation(
            agent=AgentKind.CLAUDE_CODE,
            session_id="cli-1",
            source_path="/x",
            messages=[Message(role="user", snippets=[Snippet("text", "rate limiting design")])],
        )
    )
    rc = cli_main(
        ["--db", str(db), "search", "rate", "--json"]
    )
    assert rc == 0


def test_cli_agents_lists_known(tmp_path: Path) -> None:
    rc = cli_main(["--db", str(tmp_path / "x.sqlite"), "agents", "--json"])
    assert rc == 0


def test_cli_share_then_ingest(tmp_path: Path) -> None:
    db_a = tmp_path / "a.sqlite"
    idx = SessionIndex(db_a)
    idx.upsert_conversation(
        Conversation(
            agent=AgentKind.CODEX,
            session_id="bundle-1",
            source_path="/x",
            messages=[Message(role="user", snippets=[Snippet("text", "shared session body")])],
        )
    )
    bundle = tmp_path / "b.json"
    cli_main(["--db", str(db_a), "share", "codex", "bundle-1", "-o", str(bundle)])
    assert bundle.exists()
    db_b = tmp_path / "b.sqlite"
    cli_main(["--db", str(db_b), "ingest", str(bundle)])
    assert SessionIndex(db_b).get_session("codex", "bundle-1") is not None