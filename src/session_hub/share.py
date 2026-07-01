"""Cross-machine session sharing via portable bundles.

``session-hub share <session-id>`` reads one session out of the local
SQLite index and emits a single JSON bundle that any other machine can
absorb with ``session-hub ingest <bundle.json>``. The bundle carries
enough provenance (agent, source path, workspace, timestamps) for the
receiver to attribute the session to the right connector on import.

This is the lightweight analog of cass's "remote sources" feature: no
server, no rsync config, no SSH keys — just a file you can email,
drop on a thumb drive, or attach to a bug report.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import AgentKind
from .index import SessionIndex

BUNDLE_VERSION = 1


def export_session(db_path: Path, agent: str, session_id: str, out_path: Path) -> Path:
    """Read one session from the DB and write a portable bundle."""
    idx = SessionIndex(db_path)
    session = idx.get_session(agent, session_id)
    if session is None:
        raise KeyError(f"session not found: {agent}/{session_id}")
    messages = idx.get_messages(agent, session_id)
    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": session,
        "messages": messages,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def import_bundle(bundle_path: Path, target_index: SessionIndex) -> dict:
    """Read a bundle file and ingest it into ``target_index``."""
    data = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    if data.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            f"unsupported bundle version: {data.get('bundle_version')}"
        )
    session = data["session"]
    agent_kind = AgentKind(session["agent"])
    from .models import Conversation, Message, Snippet  # local import: avoid cycle

    msgs = []
    for m in data["messages"]:
        snippets = [Snippet(**s) for s in m.get("snippets", []) if isinstance(s, dict)]
        msgs.append(
            Message(
                role=m["role"],
                ts=m.get("ts", ""),
                snippets=snippets,
                raw=m.get("plain_text", ""),
            )
        )
    conv = Conversation(
        agent=agent_kind,
        session_id=session["session_id"],
        source_path=session.get("source_path", ""),
        workspace=session.get("workspace", ""),
        started_at=session.get("started_at", ""),
        ended_at=session.get("ended_at", ""),
        title=session.get("title", ""),
        messages=msgs,
    )
    n = target_index.upsert_conversation(conv)
    return {
        "agent": agent_kind.value,
        "session_id": session["session_id"],
        "messages_indexed": n,
    }