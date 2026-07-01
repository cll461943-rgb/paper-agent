"""Agent connectors — one file per known agent.

A connector's only job is to walk one storage layout and yield
``Conversation`` records. The indexer knows nothing about agents.

Register every connector in ``ALL`` so ``discover_connectors()`` finds
them automatically.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..models import AgentKind, Conversation, Message, Snippet


class Connector(ABC):
    """Read one agent's session files into normalized ``Conversation``s."""

    agent: AgentKind

    @abstractmethod
    def default_roots(self) -> list[Path]:
        """Where to look for session files by default."""

    @abstractmethod
    def iter_sessions(self, root: Path) -> Iterator[Conversation]:
        """Yield each session found under ``root``."""


# --------------------------------------------------------------------- helpers
def _iso(ts: str | float | int | None) -> str:
    """Best-effort ISO-8601 UTC normalization."""
    if not ts:
        return ""
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(
                timespec="seconds"
            )
        s = str(ts).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc).isoformat(
            timespec="seconds"
        )
    except Exception:
        return str(ts)


def _flatten_content(content) -> str:
    """Claude Code stores content as either a string or a list of
    typed blocks. Convert either to a single text blob for indexing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("type")
                if t == "text":
                    out.append(blk.get("text", ""))
                elif t == "tool_use":
                    name = blk.get("name", "tool")
                    inp = blk.get("input", {})
                    summary = ", ".join(
                        f"{k}={str(v)[:80]}" for k, v in (inp or {}).items()
                    )
                    out.append(f"[Tool: {name}] {summary}")
                elif t == "tool_result":
                    inner = blk.get("content", "")
                    if isinstance(inner, list):
                        inner = _flatten_content(inner)
                    out.append(f"[ToolResult] {str(inner)[:400]}")
                elif t == "thinking":
                    out.append(f"[Thinking] {blk.get('thinking', '')[:400]}")
        return "\n".join(p for p in out if p)
    return str(content)


def _extract_text_snippets(content) -> list[Snippet]:
    if isinstance(content, str):
        return [Snippet(kind="text", content=content)] if content else []
    if isinstance(content, list):
        snippets = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            t = blk.get("type")
            if t == "text":
                snippets.append(Snippet(kind="text", content=blk.get("text", "")))
            elif t == "tool_use":
                snippets.append(
                    Snippet(
                        kind="tool_use",
                        content=f"[Tool: {blk.get('name', 'tool')}]",
                        meta={"input": blk.get("input", {})},
                    )
                )
            elif t == "tool_result":
                inner = blk.get("content", "")
                snippets.append(
                    Snippet(
                        kind="tool_result",
                        content=str(inner)[:2000],
                    )
                )
            elif t == "thinking":
                snippets.append(Snippet(kind="thinking", content=blk.get("thinking", "")))
        return snippets
    return [Snippet(kind="text", content=str(content))]


# ----------------------------------------------------------------- Claude Code
class ClaudeCodeConnector(Connector):
    """Claude Code stores sessions as JSONL under
    ``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl``."""

    agent = AgentKind.CLAUDE_CODE

    def default_roots(self) -> list[Path]:
        home = Path(os.path.expanduser("~"))
        return [home / ".claude" / "projects"]

    def iter_sessions(self, root: Path) -> Iterator[Conversation]:
        if not root.exists():
            return
        for path in sorted(root.rglob("*.jsonl")):
            conv = self._parse_file(path)
            if conv is not None and conv.messages:
                yield conv

    def _parse_file(self, path: Path) -> Optional[Conversation]:
        session_id = path.stem
        messages: list[Message] = []
        workspace = ""
        started = ""
        ended = ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type")
            if rtype in ("user", "assistant"):
                msg_obj = rec.get("message") or {}
                role = msg_obj.get("role", rtype)
                content = msg_obj.get("content", "")
                ts = _iso(rec.get("timestamp", ""))
                if not workspace:
                    workspace = rec.get("cwd", "")
                snippets = _extract_text_snippets(content)
                messages.append(
                    Message(
                        role=role,
                        ts=ts,
                        snippets=snippets,
                        raw=_flatten_content(content),
                    )
                )
                if ts:
                    if not started or ts < started:
                        started = ts
                    if not ended or ts > ended:
                        ended = ts
            elif rtype == "ai-title" and not messages:
                # First user prompt is more useful for the title.
                pass
        if not messages:
            return None
        title = ""
        for m in messages:
            if m.role == "user":
                first = m.plain_text().strip().splitlines()[0] if m.plain_text() else ""
                title = first[:120]
                break
        return Conversation(
            agent=self.agent,
            session_id=session_id,
            source_path=str(path),
            workspace=workspace,
            started_at=started,
            ended_at=ended,
            title=title,
            messages=messages,
        )


# -------------------------------------------------------------------- Codex CLI
class CodexConnector(Connector):
    """Codex stores sessions under
    ``~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl``."""

    agent = AgentKind.CODEX

    def default_roots(self) -> list[Path]:
        home = Path(os.path.expanduser("~"))
        return [home / ".codex" / "sessions", home / ".codex" / "archived_sessions"]

    def iter_sessions(self, root: Path) -> Iterator[Conversation]:
        if not root.exists():
            return
        for path in sorted(root.rglob("rollout-*.jsonl")):
            conv = self._parse_file(path)
            if conv is not None and conv.messages:
                yield conv

    def _parse_file(self, path: Path) -> Optional[Conversation]:
        # Filename: rollout-2026-04-12T20-42-04-<uuid>.jsonl
        m = re.search(r"rollout-([\dT-]+)-([0-9a-f-]+)\.jsonl$", path.name)
        session_id = m.group(2) if m else path.stem
        started_at = _iso(m.group(1)) if m else ""

        messages: list[Message] = []
        workspace = ""
        ended = ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type")
            ts = _iso(rec.get("timestamp", ""))
            payload = rec.get("payload") or {}
            cwd = payload.get("cwd") or rec.get("cwd")
            if cwd and not workspace:
                workspace = cwd
            if rtype == "user" or rtype == "user_message":
                content = payload.get("content") or payload.get("message") or ""
                if isinstance(content, str):
                    messages.append(
                        Message(role="user", ts=ts, snippets=[Snippet("text", content)])
                    )
            elif rtype in ("agent_message", "assistant"):
                content = payload.get("content") or payload.get("message") or ""
                if isinstance(content, str):
                    messages.append(
                        Message(
                            role="assistant",
                            ts=ts,
                            snippets=[Snippet("text", content)],
                        )
                    )
            elif rtype == "response_item":
                # Codex OpenAI-style message
                p = rec.get("payload", {})
                role = p.get("role")
                if role in ("user", "assistant"):
                    msg_content = p.get("content") or ""
                    if isinstance(msg_content, list):
                        msg_content = "\n".join(
                            b.get("text", "") for b in msg_content if isinstance(b, dict)
                        )
                    messages.append(
                        Message(
                            role=role,
                            ts=ts,
                            snippets=[Snippet("text", str(msg_content))],
                        )
                    )
            if ts:
                if ts < started_at or not started_at:
                    started_at = ts
                if ts > ended:
                    ended = ts
        if not messages:
            return None
        title = ""
        for m2 in messages:
            if m2.role == "user":
                first = m2.plain_text().strip().splitlines()[0] if m2.plain_text() else ""
                title = first[:120]
                break
        return Conversation(
            agent=self.agent,
            session_id=session_id,
            source_path=str(path),
            workspace=workspace,
            started_at=started_at,
            ended_at=ended,
            title=title,
            messages=messages,
        )


# -------------------------------------------------------------------------- Aider
class AiderConnector(Connector):
    """Aider keeps a markdown chat history at ``~/.aider.chat.history.md``."""

    agent = AgentKind.AIDER

    def default_roots(self) -> list[Path]:
        home = Path(os.path.expanduser("~"))
        return [home]

    def iter_sessions(self, root: Path) -> Iterator[Conversation]:
        path = root / ".aider.chat.history.md"
        if not path.exists():
            return
        conv = self._parse_file(path)
        if conv is not None and conv.messages:
            yield conv

    def _parse_file(self, path: Path) -> Optional[Conversation]:
        # Aider format:
        # #### <user> ... </user>
        # #### <assistant> ... </assistant>
        text = path.read_text(encoding="utf-8", errors="replace")
        pattern = re.compile(r"^####\s+<(\w+)>\s*$(.*?)^</\1>\s*$", re.M | re.S)
        messages: list[Message] = []
        for m in pattern.finditer(text):
            role = m.group(1)
            content = m.group(2).strip()
            messages.append(
                Message(role=role, ts="", snippets=[Snippet("text", content)])
            )
        if not messages:
            return None
        title = ""
        for msg in messages:
            if msg.role == "user":
                first = msg.plain_text().strip().splitlines()[0]
                title = first[:120]
                break
        # Session id = file mtime to detect "new file" boundaries on
        # re-index; collision-resistant enough for a single-user store.
        mtime = path.stat().st_mtime
        return Conversation(
            agent=self.agent,
            session_id=f"aider-{int(mtime)}",
            source_path=str(path),
            workspace="",
            started_at=_iso(mtime),
            ended_at=_iso(mtime),
            title=title,
            messages=messages,
        )


# -------------------------------------------------------------------- Gemini CLI
class GeminiCliConnector(Connector):
    """Gemini CLI keeps a JSON transcript under
    ``~/.gemini/tmp/<project>/chats/session-<ts>-<uuid>.json``."""

    agent = AgentKind.GEMINI_CLI

    def default_roots(self) -> list[Path]:
        home = Path(os.path.expanduser("~"))
        return [home / ".gemini"]

    def iter_sessions(self, root: Path) -> Iterator[Conversation]:
        chats = root / "tmp"
        if not chats.exists():
            return
        for path in sorted(chats.rglob("session-*.json")):
            conv = self._parse_file(path)
            if conv is not None and conv.messages:
                yield conv

    def _parse_file(self, path: Path) -> Optional[Conversation]:
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            return None
        session_id = data.get("sessionId") or data.get("id") or path.stem
        messages: list[Message] = []
        started = ""
        ended = ""
        workspace = data.get("projectDir") or data.get("cwd") or ""
        for m in data.get("messages", []):
            role = m.get("role") or m.get("type") or "user"
            content = m.get("content") or m.get("text") or ""
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            ts = _iso(m.get("timestamp") or m.get("ts") or "")
            messages.append(Message(role=role, ts=ts, snippets=[Snippet("text", content)]))
            if ts:
                if not started or ts < started:
                    started = ts
                if ts > ended:
                    ended = ts
        if not messages:
            return None
        title = ""
        for m2 in messages:
            if m2.role == "user":
                first = m2.plain_text().strip().splitlines()[0] if m2.plain_text() else ""
                title = first[:120]
                break
        return Conversation(
            agent=self.agent,
            session_id=session_id,
            source_path=str(path),
            workspace=workspace,
            started_at=started,
            ended_at=ended,
            title=title,
            messages=messages,
        )


# --------------------------------------------------------------------- registry
ALL: list[Connector] = [
    ClaudeCodeConnector(),
    CodexConnector(),
    AiderConnector(),
    GeminiCliConnector(),
]


def get_connector(agent: str) -> Optional[Connector]:
    for c in ALL:
        if c.agent.value == agent:
            return c
    return None


def discover_connectors() -> list[Connector]:
    return list(ALL)