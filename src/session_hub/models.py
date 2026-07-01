"""Normalized session schema.

The connectors convert each agent's native format into these three
record types. Keeping them small and explicit makes the indexer and
search code agent-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AgentKind(str, Enum):
    """Enumerates the agents we know how to read today."""

    CLAUDE_CODE = "claude_code"
    CODEX = "codex"
    AIDER = "aider"
    GEMINI_CLI = "gemini_cli"
    UNKNOWN = "unknown"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Snippet:
    """A piece of a message: text, tool invocation, or tool result.

    Storing snippets instead of opaque ``str`` lets the indexer tag
    ``[Tool: Read] path=/foo`` separately from user prose, and lets
    search results highlight the specific snippet that matched.
    """

    kind: str  # "text" | "tool_use" | "tool_result" | "thinking" | "system"
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A single turn in a conversation."""

    role: str  # "user" | "assistant" | "tool" | "system"
    snippets: list[Snippet] = field(default_factory=list)
    ts: str = ""  # ISO-8601 UTC; empty if unknown
    raw: str = ""  # best-effort flattened text for FTS indexing

    def plain_text(self) -> str:
        """Return a single text string for full-text indexing."""
        if self.raw:
            return self.raw
        return "\n".join(s.content for s in self.snippets if s.content)


@dataclass
class Conversation:
    """One full session produced by some agent."""

    agent: AgentKind
    session_id: str  # opaque id unique within the agent
    source_path: str  # file the conversation was read from
    workspace: str = ""  # best-effort working directory
    started_at: str = ""  # ISO-8601 UTC of first message
    ended_at: str = ""  # ISO-8601 UTC of last message
    title: str = ""  # first user prompt or auto-title
    messages: list[Message] = field(default_factory=list)
    indexed_at: str = field(default_factory=_utcnow_iso)

    def messages_plain(self) -> list[str]:
        return [m.plain_text() for m in self.messages]