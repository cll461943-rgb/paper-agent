"""session_hub: cross-agent session history manager.

A lightweight Python port of the core ideas behind
``coding-agent-search`` (cass) — normalize disparate agent session
formats into a single searchable SQLite index, then expose a tiny CLI
that other agents (or humans) can call to search, list, view and export
those sessions.

Design goals
------------
- **Source of truth = SQLite.** All derived views (search, list, share)
  read from the local DB; connectors only write.
- **Pluggable connectors.** Each agent (Claude Code, Codex CLI,
  Aider, Gemini CLI) is a thin subclass that yields normalized
  ``Conversation`` records. Adding a new agent = adding one file.
- **Sharing is a first-class verb.** ``session-hub share <session-id>``
  emits a self-contained bundle any other machine can ``index`` —
  no server required.
- **Robot-friendly.** Every command accepts ``--json`` for machine
  consumption, mirroring the cass ``--robot`` contract.
"""

from __future__ import annotations

from .models import Conversation, Message, Snippet, AgentKind
from .index import SessionIndex
from .connectors import discover_connectors, get_connector

__all__ = [
    "Conversation",
    "Message",
    "Snippet",
    "AgentKind",
    "SessionIndex",
    "discover_connectors",
    "get_connector",
]