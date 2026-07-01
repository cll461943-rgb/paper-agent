"""SQLite-backed session index.

Schema
------
``sessions``
    One row per ``Conversation``: id, agent, workspace, timestamps,
    title, message count, source path.

``messages``
    One row per ``Message``: id, session_id, role, ts, ordinal,
    plain_text (denormalized for fast access).

``messages_fts``
    FTS5 virtual table mirroring ``messages.plain_text`` for BM25-style
    keyword search.

Why FTS5 and not full vectors
-----------------------------
Vector search is the right call at internet scale; for a single
user's local session history (typically <10k messages), FTS5 with the
default BM25 ranking is sub-millisecond and dependency-free. We keep
the door open for an optional embedding column later but ship the
simpler thing first — same philosophy as cass's "lexical is the
required fast path".
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

from .models import AgentKind, Conversation, Message, Snippet

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT NOT NULL,
    agent         TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    workspace     TEXT DEFAULT '',
    started_at    TEXT DEFAULT '',
    ended_at      TEXT DEFAULT '',
    title         TEXT DEFAULT '',
    message_count INTEGER DEFAULT 0,
    indexed_at    TEXT NOT NULL,
    PRIMARY KEY (agent, session_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent       TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    role        TEXT NOT NULL,
    ts          TEXT DEFAULT '',
    plain_text  TEXT NOT NULL,
    snippets    TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(agent, session_id, ordinal);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    plain_text,
    content='messages',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SessionIndex:
    """Owns the SQLite file. Thread-unsafe; one process per DB file."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ write
    def upsert_conversation(self, conv: Conversation) -> int:
        """Replace one conversation's rows. Returns message count."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM messages WHERE agent=? AND session_id=?",
                (conv.agent.value, conv.session_id),
            )
            conn.execute(
                "DELETE FROM sessions WHERE agent=? AND session_id=?",
                (conv.agent.value, conv.session_id),
            )
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, agent, source_path, workspace,
                    started_at, ended_at, title, message_count, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conv.session_id,
                    conv.agent.value,
                    conv.source_path,
                    conv.workspace,
                    conv.started_at,
                    conv.ended_at,
                    conv.title,
                    len(conv.messages),
                    conv.indexed_at,
                ),
            )
            for ordinal, msg in enumerate(conv.messages):
                conn.execute(
                    """
                    INSERT INTO messages (
                        agent, session_id, ordinal, role, ts, plain_text, snippets
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conv.agent.value,
                        conv.session_id,
                        ordinal,
                        msg.role,
                        msg.ts,
                        msg.plain_text(),
                        json.dumps(
                            [asdict(s) for s in msg.snippets], ensure_ascii=False
                        ),
                    ),
                )
                # External-content FTS5 tables need explicit sync after
                # INSERTs that bypass triggers; doing it per-row keeps the
                # index immediately queryable for the next command.
                conn.execute(
                    "INSERT INTO messages_fts(rowid, plain_text) VALUES (last_insert_rowid(), ?)",
                    (msg.plain_text(),),
                )
            return len(conv.messages)

    def delete_session(self, agent: AgentKind, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM messages WHERE agent=? AND session_id=?",
                (agent.value, session_id),
            )
            conn.execute(
                "DELETE FROM sessions WHERE agent=? AND session_id=?",
                (agent.value, session_id),
            )

    # ------------------------------------------------------------------- read
    def list_sessions(
        self,
        agent: Optional[str] = None,
        workspace: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        sql = "SELECT * FROM sessions WHERE 1=1"
        params: list = []
        if agent:
            sql += " AND agent=?"
            params.append(agent)
        if workspace:
            sql += " AND workspace LIKE ?"
            params.append(f"%{workspace}%")
        sql += " ORDER BY COALESCE(started_at, indexed_at) DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_session(self, agent: str, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE agent=? AND session_id=?",
                (agent, session_id),
            ).fetchone()
            return dict(row) if row else None

    def get_messages(self, agent: str, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT ordinal, role, ts, plain_text, snippets
                FROM messages
                WHERE agent=? AND session_id=?
                ORDER BY ordinal ASC
                """,
                (agent, session_id),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["snippets"] = json.loads(d["snippets"])
            except json.JSONDecodeError:
                d["snippets"] = []
            out.append(d)
        return out

    def search(
        self,
        query: str,
        agent: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """BM25 keyword search across ``messages_fts``.

        Returns rows with session context for re-grouping in the UI.
        """
        if not query.strip():
            return []
        # FTS5 prefix MATCH; quote user input to avoid syntax errors.
        fts_query = _to_fts_query(query)
        sql = """
        SELECT
            m.agent        AS agent,
            m.session_id   AS session_id,
            m.ordinal      AS ordinal,
            m.role         AS role,
            m.ts           AS ts,
            snippet(messages_fts, 0, '<<', '>>', '...', 12) AS hit,
            s.title        AS title,
            s.workspace    AS workspace,
            bm25(messages_fts) AS score
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        JOIN sessions s ON s.agent = m.agent AND s.session_id = m.session_id
        WHERE messages_fts MATCH ?
        """
        params: list = [fts_query]
        if agent:
            sql += " AND m.agent = ?"
            params.append(agent)
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        with self._conn() as conn:
            totals = {
                r["agent"]: r["n"]
                for r in conn.execute(
                    "SELECT agent, COUNT(*) AS n FROM sessions GROUP BY agent"
                ).fetchall()
            }
            msg_total = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
        return {"sessions_by_agent": totals, "messages_total": msg_total}


def _to_fts_query(q: str) -> str:
    """Turn ``"foo bar"`` into ``'"foo" OR "bar"'`` to be friendly to
    short user input. Long phrases stay quoted as a single unit."""
    tokens = [t for t in q.split() if t]
    if not tokens:
        return '""'
    if len(tokens) == 1:
        return f'"{tokens[0]}"'
    return " OR ".join(f'"{t}"' for t in tokens)