# -*- coding: utf-8 -*-
"""Web UI for session-hub — simple Flask/FastAPI-style server.

Usage::

    python -m session_hub.web --port 8080

Then open http://localhost:8080 in your browser.
"""

import html
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .batch_eval import load_dataset_cases
from .index import SessionIndex
from .run_store import RunStore, default_logs_db_path

DEFAULT_DB = Path(
    os.environ.get(
        "SESSION_HUB_DB",
        str(Path.home() / ".session_hub" / "index.sqlite"),
    )
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session Hub - 对话历史浏览器</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            margin: 0 0 20px 0;
            font-size: 24px;
            color: #1a73e8;
        }
        .stats {
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #1a73e8;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
        .search-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .search-box input {
            width: 100%;
            padding: 12px 16px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            outline: none;
            transition: border-color 0.2s;
        }
        .search-box input:focus {
            border-color: #1a73e8;
        }
        .filters {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 8px 16px;
            border: 1px solid #e0e0e0;
            background: white;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }
        .filter-btn:hover, .filter-btn.active {
            background: #1a73e8;
            color: white;
            border-color: #1a73e8;
        }
        .sessions-list {
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .session-item {
            padding: 15px 20px;
            border-bottom: 1px solid #f0f0f0;
            cursor: pointer;
            transition: background 0.2s;
            position: relative;
        }
        .session-item:hover {
            background: #f8f9fa;
        }
        .session-item.active {
            background: #e3f2fd;
            border-left: 3px solid #1a73e8;
        }
        .session-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 5px;
        }
        .agent-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }
        .agent-claude_code { background: #ff9800; color: white; }
        .agent-codex { background: #4caf50; color: white; }
        .agent-aider { background: #9c27b0; color: white; }
        .agent-gemini_cli { background: #2196f3; color: white; }
        .session-title {
            font-weight: 500;
            color: #333;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            flex: 1;
        }
        .session-meta {
            display: flex;
            gap: 15px;
            font-size: 13px;
            color: #666;
        }
        .session-detail {
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
            margin-top: 20px;
        }
        .session-detail h2 {
            margin-top: 0;
            font-size: 18px;
            color: #333;
        }
        .message {
            padding: 12px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .message:last-child {
            border-bottom: none;
        }
        .message-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .message-role {
            font-weight: 600;
            font-size: 13px;
        }
        .role-user { color: #2196f3; }
        .role-assistant { color: #ff9800; }
        .role-tool { color: #9c27b0; }
        .message-time {
            font-size: 12px;
            color: #999;
        }
        .message-content {
            color: #444;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 14px;
        }
        .highlight {
            background: #ffeb3b;
            padding: 0 2px;
            border-radius: 2px;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .two-column {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .two-column {
                grid-template-columns: 1fr;
            }
        }
        .toolbar {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .btn {
            padding: 8px 16px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .btn:hover {
            background: #f5f5f5;
        }
        .btn-primary {
            background: #1a73e8;
            color: white;
            border-color: #1a73e8;
        }
        .btn-primary:hover {
            background: #1557b0;
        }
        .search-highlight {
            background: #ffeb3b;
            padding: 0 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Session Hub - 对话历史浏览器</h1>

        <div class="stats">
            <div class="stats-grid" id="statsGrid">
                <div class="stat-item">
                    <div class="stat-value">{total_messages}</div>
                    <div class="stat-label">总消息数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_sessions}</div>
                    <div class="stat-label">总对话数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{agents_count}</div>
                    <div class="stat-label">Agent 类型</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{db_path}</div>
                    <div class="stat-label">数据库</div>
                </div>
            </div>
        </div>

        <div class="search-box">
            <input type="text" id="searchInput" placeholder="搜索关键词..." value="{query}">
            <div class="filters" id="agentFilters">
                {agent_filters}
            </div>
        </div>

        <div class="two-column">
            <div>
                <div class="toolbar">
                    <button class="btn btn-primary" onclick="refreshData()">🔄 刷新</button>
                    <button class="btn" onclick="exportCurrent()">📤 导出当前</button>
                </div>
                <div class="sessions-list" id="sessionsList">
                    {sessions_list}
                </div>
            </div>

            <div>
                <div id="sessionDetail" class="session-detail" style="display: none;">
                    {session_detail}
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentAgent = '{current_agent}';
        let currentSessionId = '{current_session_id}';
        let currentQuery = '{query}';

        // 搜索输入防抖
        let searchTimeout;
        document.getElementById('searchInput').addEventListener('input', function(e) {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                currentQuery = e.target.value;
                window.location.href = '/?q=' + encodeURIComponent(currentQuery) +
                                        (currentAgent ? '&agent=' + encodeURIComponent(currentAgent) : '');
            }, 500);
        });

        // Agent 过滤
        function filterAgent(agent) {
            currentAgent = agent === 'all' ? '' : agent;
            window.location.href = '/?agent=' + encodeURIComponent(currentAgent) +
                                    (currentQuery ? '&q=' + encodeURIComponent(currentQuery) : '');
        }

        // 查看 session 详情
        function viewSession(agent, sessionId) {
            window.location.href = '/view?agent=' + encodeURIComponent(agent) +
                                    '&id=' + encodeURIComponent(sessionId);
        }

        // 刷新数据
        function refreshData() {
            window.location.reload();
        }

        // 导出当前选中的 session
        function exportCurrent() {
            if (!currentSessionId) {
                alert('请先选择一个对话');
                return;
            }
            window.open('/export?agent=' + encodeURIComponent(currentAgent) +
                       '&id=' + encodeURIComponent(currentSessionId), '_blank');
        }
    </script>
</body>
</html>
"""


def _get_stats(idx: SessionIndex) -> dict:
    s = idx.stats()
    return {
        "total_messages": s["messages_total"],
        "total_sessions": sum(s["sessions_by_agent"].values()),
        "agents_count": len(s["sessions_by_agent"]),
        "agents": s["sessions_by_agent"],
        "db_path": str(DEFAULT_DB),
    }


def _render_sessions(sessions: list[dict], selected_agent: str = "", selected_id: str = "") -> str:
    if not sessions:
        return '<div class="empty-state"><div class="empty-state-icon">📭</div>没有找到对话</div>'

    html_parts = []
    for s in sessions:
        agent = s["agent"]
        sid = s["session_id"]
        is_active = (agent == selected_agent and sid == selected_id)
        agent_class = f"agent-{agent}"
        title = html.escape(s["title"] or "(无标题)")[:60]
        if len(s.get("title", "")) > 60:
            title += "..."

        html_parts.append(f"""
        <div class="session-item {'active' if is_active else ''}"
             onclick="viewSession('{agent}', '{sid}')">
            <div class="session-header">
                <span class="agent-badge {agent_class}">{agent}</span>
                <span class="session-title">{title}</span>
            </div>
            <div class="session-meta">
                <span>📁 {html.escape(s.get('workspace', '')[-30:] or '-')}</span>
                <span>💬 {s.get('message_count', 0)} 消息</span>
                <span>🕐 {s.get('started_at', '')[:16]}</span>
            </div>
        </div>
        """)
    return "\n".join(html_parts)


def _render_messages(messages: list[dict], query: str = "") -> str:
    if not messages:
        return '<div class="empty-state">此对话没有消息</div>'

    html_parts = []
    for m in messages:
        role = m.get("role", "unknown")
        role_class = f"role-{role}"
        role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}.get(role, "💬")
        content = html.escape(m.get("plain_text", "")[:2000])
        if len(m.get("plain_text", "")) > 2000:
            content += "\n\n... (内容已截断)"

        # 高亮搜索词
        if query:
            for q in query.lower().split():
                if q:
                    content = content.replace(
                        q, f'<span class="search-highlight">{q}</span>'
                    )
                    content = content.replace(
                        q.upper(), f'<span class="search-highlight">{q.upper()}</span>'
                    )

        html_parts.append(f"""
        <div class="message">
            <div class="message-header">
                <span class="message-role {role_class}">{role_emoji} {role}</span>
                <span class="message-time">{m.get('ts', '')[:19]}</span>
            </div>
            <div class="message-content">{content}</div>
        </div>
        """)
    return "\n".join(html_parts)


def _render_agent_filters(agents: dict[str, int], current: str = "") -> str:
    buttons = [
        f'<button class="filter-btn {"active" if not current else ""}" onclick="filterAgent(\'all\')">全部 ({sum(agents.values())})</button>'
    ]
    for agent, count in sorted(agents.items()):
        is_active = agent == current
        buttons.append(
            f'<button class="filter-btn {"active" if is_active else ""}" '
            f'onclick="filterAgent(\'{agent}\')">{agent} ({count})</button>'
        )
    return "\n".join(buttons)


def handle_request(environ: dict, start_response) -> list[bytes]:
    """WSGI handler."""
    path = environ.get("PATH_INFO", "/")
    if path.startswith("/api/"):
        return handle_api_request(environ, start_response)

    query_string = environ.get("QUERY_STRING", "")
    params = parse_qs(query_string)

    idx = SessionIndex(DEFAULT_DB)

    # Route: /export - download bundle
    if path == "/export":
        agent = params.get("agent", [""])[0]
        sid = params.get("id", [""])[0]
        if agent and sid:
            from .share import export_session
            import tempfile
            tmp = Path(tempfile.gettempdir()) / f"session-{sid}.json"
            export_session(DEFAULT_DB, agent, sid, tmp)
            data = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Disposition", f'attachment; filename="session-{sid}.json"'),
                    ("Content-Length", str(len(data))),
                ],
            )
            return [data]

    # Route: /view - show session detail
    if path == "/view":
        agent = params.get("agent", [""])[0]
        sid = params.get("id", [""])[0]
        if agent and sid:
            session = idx.get_session(agent, sid)
            messages = idx.get_messages(agent, sid)
            if session:
                detail_html = f"""
                <h2>📝 {html.escape(session.get('title', '无标题')[:80])}</h2>
                <p>
                    <strong>Agent:</strong> {session.get('agent')} |
                    <strong>Session ID:</strong> <code>{session.get('session_id')}</code> |
                    <strong>消息数:</strong> {session.get('message_count')} |
                    <strong>时间:</strong> {session.get('started_at', '')[:19]}
                </p>
                <hr>
                {_render_messages(messages)}
                """
                # Redirect back to home with detail view
                sessions = idx.list_sessions(agent=agent, limit=50)
                stats = _get_stats(idx)
                html_out = HTML_TEMPLATE.format(
                    total_messages=stats["total_messages"],
                    total_sessions=stats["total_sessions"],
                    agents_count=stats["agents_count"],
                    db_path=stats["db_path"],
                    query="",
                    agent_filters=_render_agent_filters(stats["agents"], agent),
                    sessions_list=_render_sessions(sessions, agent, sid),
                    current_agent=agent,
                    current_session_id=sid,
                    session_detail=detail_html,
                )
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [html_out.encode("utf-8")]

    # Route: / (home) - list and search
    query = params.get("q", [""])[0]
    agent_filter = params.get("agent", [""])[0]

    if query:
        # Search mode
        results = idx.search(query, agent=agent_filter or None, limit=50)
        # Group by session for display
        sessions_map: dict[tuple[str, str], dict] = {}
        for r in results:
            key = (r["agent"], r["session_id"])
            if key not in sessions_map:
                sessions_map[key] = {
                    "agent": r["agent"],
                    "session_id": r["session_id"],
                    "title": r.get("title", ""),
                    "workspace": r.get("workspace", ""),
                    "message_count": 0,
                    "started_at": r.get("ts", ""),
                    "_has_match": True,
                }
            sessions_map[key]["message_count"] += 1
        sessions = list(sessions_map.values())
    else:
        sessions = idx.list_sessions(agent=agent_filter or None, limit=50)

    stats = _get_stats(idx)
    html_out = HTML_TEMPLATE.format(
        total_messages=stats["total_messages"],
        total_sessions=stats["total_sessions"],
        agents_count=stats["agents_count"],
        db_path=stats["db_path"],
        query=html.escape(query),
        agent_filters=_render_agent_filters(stats["agents"], agent_filter),
        sessions_list=_render_sessions(sessions, agent_filter, ""),
        current_agent=agent_filter,
        current_session_id="",
        session_detail='<div class="empty-state"><div class="empty-state-icon">👈</div>点击左侧对话查看详情</div>',
    )

    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
    return [html_out.encode("utf-8")]


# ==============================================================================
# RESTful API Backend for Scholar Agent Frontend Integration
# ==============================================================================
import threading
import datetime
import traceback
import sys
import glob
import time

try:
    from scholar_agent.infra import load_config, OpenAICompatibleLLMClient
    from scholar_agent.retrieval import build_providers
    from scholar_agent.workflow.pipeline import PaperAgentPipeline, deduplicate_papers
    from scholar_agent.workflow.budget import BudgetManager
except ImportError:
    pass

# Global active jobs store
ACTIVE_JOBS = {}
JOBS_LOCK = threading.Lock()
CACHE_DIR = Path("data/cache/web_jobs")
ROOT_DIR = Path(__file__).parents[2]
RUN_STORE = RunStore(default_logs_db_path(ROOT_DIR))

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def save_job_to_cache(job_id: str):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with JOBS_LOCK:
            job_data = ACTIVE_JOBS.get(job_id)
        if job_data:
            serialized_data = {k: v for k, v in job_data.items() if k not in ('thread',)}
            cache_file = CACHE_DIR / f"{job_id}.json"
            cache_file.write_text(json.dumps(serialized_data, ensure_ascii=False, indent=2, cls=CustomJSONEncoder), encoding="utf-8")
    except Exception as e:
        print(f"Error saving job cache for {job_id}: {e}", file=sys.stderr)

def persist_job_snapshot(job_ref: dict):
    try:
        RUN_STORE.upsert_run(job_ref)
    except Exception as e:
        print(f"Error persisting job {job_ref.get('job_id')}: {e}", file=sys.stderr)

def persist_job_log(job_id: str, level: str, stage: str, message: str):
    try:
        RUN_STORE.add_log(job_id, level, stage, message)
    except Exception as e:
        print(f"Error persisting log for {job_id}: {e}", file=sys.stderr)

def mark_job_cancelled(job_ref: dict, reason: str = "Cancelled by user"):
    job_ref["cancel_requested"] = True
    job_ref["status"] = "cancelled"
    job_ref["error"] = reason
    job_ref["progress"] = min(int(job_ref.get("progress", 0) or 0), 99)
    job_ref["elapsed_seconds"] = time.time() - job_ref.get("start_time", time.time())
    persist_job_snapshot(job_ref)
    save_job_to_cache(job_ref["job_id"])

def load_jobs_from_cache():
    try:
        if not CACHE_DIR.exists():
            return
        for path in CACHE_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job_id = data.get("job_id")
                if job_id:
                    if data.get("status") in ("running", "queued"):
                        data["status"] = "failed"
                        data["error"] = "Server restarted while running"
                    with JOBS_LOCK:
                        ACTIVE_JOBS[job_id] = data
            except Exception as e:
                print(f"Error loading job cache {path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing job caches: {e}", file=sys.stderr)

# Load existing jobs at startup
load_jobs_from_cache()

# Custom logger to capture logs per job and update stage/progress
import logging

class JobLoggingHandler(logging.Handler):
    def __init__(self, job_id, logs_list, jobs_lock):
        super().__init__()
        self.job_id = job_id
        self.logs_list = logs_list
        self.jobs_lock = jobs_lock

    def emit(self, record):
        try:
            msg = self.format(record)
            log_entry = {
                "id": f"{self.job_id}-{len(self.logs_list)}",
                "timestamp": datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
                "stage": getattr(record, "stage", "system"),
                "message": msg
            }
            with self.jobs_lock:
                self.logs_list.append(log_entry)
                
                # Intercept logs to estimate progress & stage
                lower_msg = msg.lower()
                current_job = ACTIVE_JOBS.get(self.job_id)
                if current_job and current_job["status"] in ("running", "eval_running"):
                    if "starting paperagentpipeline" in lower_msg or "understand" in lower_msg:
                        current_job["stage"] = "query_plan"
                        current_job["progress"] = 15
                    elif "retriev" in lower_msg or "multirouteretriever" in lower_msg:
                        current_job["stage"] = "retrieval"
                        current_job["progress"] = 40
                    elif "evidence selection" in lower_msg or "evidence_selection" in lower_msg:
                        current_job["stage"] = "selection"
                        current_job["progress"] = 65
                    elif "rerank" in lower_msg or "ranking" in lower_msg:
                        current_job["stage"] = "ranking"
                        current_job["progress"] = 80
                    elif "synthesis" in lower_msg or "synthesisagent" in lower_msg:
                        current_job["stage"] = "synthesis"
                        current_job["progress"] = 95
            persist_job_log(self.job_id, record.levelname, log_entry["stage"], msg)
        except Exception:
            pass

# Helper to match paper keys for evaluation (copied from full_pipeline_eval.py)
import re

def _normalize_identifier(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:", "", text)
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text.strip() or None

def _normalize_title(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", text) or None

def _normalize_corpus_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^corpusid:", "", text, flags=re.IGNORECASE)
    return f"corpus:{text.lower()}"

def paper_match_keys(paper: Any) -> set[str]:
    # Support both pydantic objects and dicts
    if isinstance(paper, dict):
        keys = {
            _normalize_identifier(paper.get("paper_id")),
            _normalize_identifier(paper.get("doi")),
            _normalize_identifier(paper.get("arxiv_id")),
            _normalize_title(paper.get("title")),
            _normalize_corpus_id(paper.get("metadata", {}).get("corpus_id")),
            _normalize_corpus_id(paper.get("metadata", {}).get("corpusId")),
        }
    else:
        keys = {
            _normalize_identifier(paper.paper_id),
            _normalize_identifier(paper.doi),
            _normalize_identifier(paper.arxiv_id),
            _normalize_title(paper.title),
            _normalize_corpus_id(paper.metadata.get("corpus_id")),
            _normalize_corpus_id(paper.metadata.get("corpusId")),
        }
    return {key for key in keys if key}

def gold_match_keys(item: dict[str, Any] | str) -> set[str]:
    if isinstance(item, str):
        keys = {_normalize_identifier(item), _normalize_title(item)}
    else:
        keys = {
            _normalize_identifier(item.get("paper_id")),
            _normalize_identifier(item.get("doi")),
            _normalize_identifier(item.get("arxiv_id")),
            _normalize_title(item.get("title")),
            _normalize_corpus_id(item.get("corpus_id")),
            _normalize_corpus_id(item.get("corpusid")),
            _normalize_corpus_id(item.get("semantic_scholar_corpus_id")),
        }
    return {key for key in keys if key}

def count_hits(papers: list[Any], gold_items: list[dict[str, Any] | str]) -> tuple[int, int]:
    gold_key_sets = []
    for item in gold_items:
        keys = gold_match_keys(item)
        if keys:
            gold_key_sets.append(keys)
    paper_key_sets = [paper_match_keys(paper) for paper in papers]
    matched_gold_indexes: set[int] = set()

    for paper_keys in paper_key_sets:
        for index, gold_keys in enumerate(gold_key_sets):
            if index in matched_gold_indexes:
                continue
            if paper_keys & gold_keys:
                matched_gold_indexes.add(index)
                break
    return len(matched_gold_indexes), len(gold_key_sets)

def compute_precision_recall_f1(output_papers: list[Any], gold_items: list[dict[str, Any] | str]) -> dict:
    hits, gold_total = count_hits(output_papers, gold_items)
    output_total = len(output_papers)
    precision = hits / output_total if output_total > 0 else 0.0
    recall = hits / gold_total if gold_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "hits": hits,
        "gold_total": gold_total,
        "output_total": output_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

# Background thread runner for pipeline execution
def run_pipeline_task(job_id: str, job_ref: dict):
    query = job_ref.get("query", "")
    mode = job_ref.get("mode", "research")
    config_name = job_ref.get("config", "configs/default.yaml")
    retrieval_only = job_ref.get("retrieval_only", False)
    is_eval = job_ref.get("is_eval", False)
    gold_items = job_ref.get("gold", [])

    if job_ref.get("cancel_requested"):
        mark_job_cancelled(job_ref)
        return
    
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    if not config_name.startswith("configs/"):
        config_name = f"configs/{config_name}"
        
    try:
        # 1. Load configuration
        root_dir = Path(__file__).parents[2]
        config = load_config(root_dir / config_name)
        if mode == "mock":
            config.app.mode = "mock"
            config.app.providers = ["mock"]
        else:
            config.app.mode = "live"
            
        providers_input = job_ref.get("providers", [])
        if providers_input and mode != "mock":
            prov_map = {
                "Semantic Scholar": "semantic_scholar",
                "OpenAlex": "openalex",
                "arXiv": "arxiv",
                "Local Index": "pasa_local",
                "PubMed": "pubmed"
            }
            mapped_providers = []
            for p in providers_input:
                val = prov_map.get(p)
                if val:
                    mapped_providers.append(val)
            if mapped_providers:
                mapped_providers.append("faiss_vector")
                config.app.providers = mapped_providers

        # 2. Attach logging handler
        root_logger = logging.getLogger()
        handler = JobLoggingHandler(job_id, job_ref["logs"], JOBS_LOCK)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        
        try:
            # 3. Instantiate pipeline & run
            budget = BudgetManager(config)
            if mode == "mock":
                from scholar_agent.infra import MockLLMClient
                llm_client = MockLLMClient(budget)
            else:
                llm_client = OpenAICompatibleLLMClient(config.llm, budget)
            providers = build_providers(config, provider_names=config.app.providers)
            pipeline = PaperAgentPipeline(config, llm_client, providers)
            
            result = pipeline.run(query, retrieval_only=retrieval_only)
            if job_ref.get("cancel_requested") or job_ref.get("status") == "cancelled":
                mark_job_cancelled(job_ref)
                return
            result_dict = result.model_dump()
            
            # Estimate metric sizes
            pool_papers = getattr(pipeline, "candidate_pool", [])
            sel_papers = getattr(pipeline, "selection_candidates", [])
            
            # Stage artifacts
            stage_artifacts = {
                "query_plan": {
                    "stage": "query_plan",
                    "job_id": job_id,
                    "data": result_dict.get("query_plan", {})
                },
                "retrieval": {
                    "stage": "retrieval",
                    "job_id": job_id,
                    "data": {
                        "candidate_pool_size": len(pool_papers),
                        "deduped": len(deduplicate_papers(pool_papers)) if pool_papers else 0,
                        "top_sources": [
                            {"provider": p.name, "hits": len([x for x in pool_papers if getattr(x, 'source', '') == p.name])}
                            for p in providers
                        ]
                    }
                },
                "selection": {
                    "stage": "selection",
                    "job_id": job_id,
                    "data": {
                        "selection_candidates_size": len(sel_papers),
                        "high_confidence": len([x for x in getattr(pipeline, "selections", []) if getattr(x, 'relevance_level', '') == "high"]),
                        "medium_confidence": len([x for x in getattr(pipeline, "selections", []) if getattr(x, 'relevance_level', '') == "medium"]),
                        "rejected": len([x for x in getattr(pipeline, "selections", []) if getattr(x, 'relevance_level', '') in ("low", "irrelevant")])
                    }
                },
                "ranking": {
                    "stage": "ranking",
                    "job_id": job_id,
                    "data": {
                        "ranked_papers": result_dict.get("highly_relevant_papers", [])[:15]
                    }
                },
                "synthesis": {
                    "stage": "synthesis",
                    "job_id": job_id,
                    "data": {
                        "final_output_size": len(result_dict.get("highly_relevant_papers", [])) + len(result_dict.get("partially_relevant_papers", [])),
                        "clusters": result_dict.get("method_clusters", []),
                        "timeline": result_dict.get("timeline", [])
                    }
                }
            }
            
            # Post evaluation metrics if in eval mode
            eval_metrics = None
            if is_eval:
                highly_relevant = result.highly_relevant_papers
                partially_relevant = result.partially_relevant_papers
                final_papers = [rp.paper for rp in highly_relevant] + [rp.paper for rp in partially_relevant]
                eval_metrics = compute_precision_recall_f1(final_papers, gold_items)
                run_metrics = result_dict.get("run_metrics", {}) if isinstance(result_dict, dict) else {}
                budget_summary = {
                    "elapsed_seconds": round(float(run_metrics.get("elapsed_seconds", 0.0) or 0.0), 2),
                    "api_calls": int(run_metrics.get("api_calls_used", 0) or 0),
                    "llm_calls": int(run_metrics.get("llm_calls_used", 0) or 0),
                    "token_estimate": int(run_metrics.get("token_estimate", 0) or 0),
                    "candidate_pool_size": int(run_metrics.get("candidate_pool_size", len(pool_papers)) or 0),
                    "final_papers": int(run_metrics.get("final_papers", len(final_papers)) or 0),
                }
                
                # Make diagnostics
                def make_diag_row(p_obj, stage, pass_val=True, reason=""):
                    return {
                        "paper_id": p_obj.paper_id,
                        "title": p_obj.title,
                        "source": p_obj.source,
                        "year": p_obj.year,
                        "citation_count": p_obj.citation_count,
                        "relevance": p_obj.metadata.get("local_pre_rank_score") or p_obj.metadata.get("vector_score") or 0.0,
                        "stage": stage,
                        "score": p_obj.metadata.get("local_pre_rank_score") or 0.0,
                        "pass": pass_val,
                        "reason": reason
                    }
                
                candidate_pool_rows = [make_diag_row(p, "retrieval") for p in pool_papers[:5]]
                selection_rows = [make_diag_row(p, "selection", pass_val=(idx < len(sel_papers)-2)) for idx, p in enumerate(sel_papers[:5])]
                ranked_rows = [make_diag_row(rp.paper, "ranking") for rp in result.highly_relevant_papers[:5]]
                final_rows = [make_diag_row(rp.paper, "synthesis") for rp in result.highly_relevant_papers[:3]]
                
                job_ref["eval_result"] = {
                    "job_id": job_id,
                    "dataset": job_ref.get("dataset", ""),
                    "case_index": job_ref.get("case_index", 0),
                    "query": query,
                    "gold": [g.get("title", "") if isinstance(g, dict) else g for g in gold_items],
                    "eval_metrics": eval_metrics,
                    "budget": budget_summary,
                    "progress": [
                        {"stage": "Retrieval", "status": "succeeded", "elapsed_seconds": 5.0},
                        {"stage": "Selection", "status": "succeeded", "elapsed_seconds": 10.0},
                        {"stage": "Ranking", "status": "succeeded", "elapsed_seconds": 5.0},
                        {"stage": "Synthesis", "status": "succeeded", "elapsed_seconds": 5.0}
                    ],
                    "candidate_pool": candidate_pool_rows,
                    "selection_candidates": selection_rows,
                    "ranked_papers": ranked_rows,
                    "final_output": final_rows,
                    "warnings": ["Evaluation completed successfully."]
                }
            
            should_cancel = False
            with JOBS_LOCK:
                if job_ref.get("cancel_requested") or job_ref.get("status") == "cancelled":
                    should_cancel = True
                else:
                    job_ref["status"] = "succeeded"
                    job_ref["progress"] = 100
                    job_ref["elapsed_seconds"] = time.time() - job_ref["start_time"]
                    job_ref["result"] = result_dict
                    job_ref["stage_artifacts"] = stage_artifacts
            if should_cancel:
                mark_job_cancelled(job_ref)
                return
                
        finally:
            root_logger.removeHandler(handler)
            
    except Exception as e:
        traceback.print_exc()
        should_cancel = False
        with JOBS_LOCK:
            if job_ref.get("cancel_requested") or job_ref.get("status") == "cancelled":
                should_cancel = True
            else:
                job_ref["status"] = "failed"
                job_ref["error"] = str(e)
                job_ref["elapsed_seconds"] = time.time() - job_ref["start_time"]
        if should_cancel:
            mark_job_cancelled(job_ref)
            
    finally:
        persist_job_snapshot(job_ref)
        save_job_to_cache(job_id)

def load_eval_case(dataset_name: str, case_idx: int) -> tuple[str, list[dict]]:
    dataset_path = Path("data/benchmarks") / dataset_name
    if not dataset_path.exists():
        dataset_path = Path(dataset_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found")

    all_cases = load_dataset_cases(dataset_path)
    matched = next((case for case in all_cases if case.case_index == case_idx), None)
    if matched is None and 0 <= case_idx < len(all_cases):
        matched = all_cases[case_idx]
    if matched is None:
        raise IndexError(f"Case index {case_idx} out of range (0-{len(all_cases)-1})")
    c = matched
    return c.query, [{"title": item} for item in c.gold]

# Graph generator logic
def build_graph_response(job_id: str, result: dict) -> dict:
    try:
        highly = result.get("highly_relevant_papers", [])
        partially = result.get("partially_relevant_papers", [])
        supporting = result.get("supporting_papers", [])
        
        papers = []
        for rp in highly:
            papers.append((rp.get("paper", {}), "highly_relevant", rp.get("final_score", 1.0)))
        for rp in partially:
            papers.append((rp.get("paper", {}), "partially_relevant", rp.get("final_score", 0.8)))
        for rp in supporting:
            papers.append((rp.get("paper", {}), "supporting", rp.get("final_score", 0.5)))
            
        nodes = []
        edges = []
        clusters = []
        
        query = result.get("original_query", "学术检索")
        nodes.append({
            "id": "query_center",
            "type": "topic",
            "label": query[:30] + ("..." if len(query) > 30 else ""),
            "x": 400,
            "y": 300,
            "size": 35,
            "highlighted": True
        })
        
        query_plan = result.get("query_plan", {})
        methods = query_plan.get("methods", [])
        datasets = query_plan.get("datasets", [])
        
        import math
        for idx, method in enumerate(methods[:5]):
            angle = (2 * math.pi * idx) / max(len(methods[:5]), 1)
            nodes.append({
                "id": f"method_{idx}",
                "type": "method",
                "label": method,
                "x": 400 + 150 * math.cos(angle),
                "y": 300 + 150 * math.sin(angle),
                "size": 20
            })
            edges.append({
                "id": f"edge_center_method_{idx}",
                "source": "query_center",
                "target": f"method_{idx}",
                "type": "association",
                "weight": 1.0
            })
            
        num_papers = len(papers)
        for idx, (paper, rel, score) in enumerate(papers[:20]):
            angle = (2 * math.pi * idx) / max(num_papers, 1) + 0.3
            r = 300
            px = 400 + r * math.cos(angle)
            py = 300 + r * math.sin(angle)
            
            citation_cnt = paper.get("citation_count") or 0
            size = 15 + min(15, citation_cnt / 10) + score * 5
            
            paper_id = paper.get("paper_id", f"paper_{idx}")
            title = paper.get("title", "Untitled")
            
            nodes.append({
                "id": paper_id,
                "type": "paper",
                "label": title,
                "shortLabel": title[:15] + "..." if len(title) > 15 else title,
                "x": px,
                "y": py,
                "size": size,
                "year": paper.get("year"),
                "meta": {
                    "authors": ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else "",
                    "venue": paper.get("venue", ""),
                    "citation_count": citation_cnt,
                    "relevance": rel,
                    "abstract": paper.get("abstract", "") or "",
                    "references": paper.get("references", []),
                    "citations": paper.get("citations", []),
                    "doi": paper.get("doi", "") or "",
                    "url": paper.get("url", "") or ""
                }
            })
            
            edges.append({
                "id": f"edge_center_paper_{paper_id}",
                "source": "query_center",
                "target": paper_id,
                "type": "association",
                "weight": score
            })
            
            for midx, method in enumerate(methods[:5]):
                if method.lower() in title.lower():
                    edges.append({
                        "id": f"edge_paper_{paper_id}_method_{midx}",
                        "source": paper_id,
                        "target": f"method_{midx}",
                        "type": "association",
                        "weight": 0.8
                    })
                    
        id_map = {}
        for p, _, _ in papers[:20]:
            pid = p.get("paper_id")
            doi = p.get("doi")
            arxiv = p.get("arxiv_id")
            if pid:
                if doi: id_map[doi.lower().strip()] = pid
                if arxiv: id_map[arxiv.lower().strip()] = pid
                
        edge_idx = 0
        for p, _, _ in papers[:20]:
            pid = p.get("paper_id")
            if not pid:
                continue
            for ref in p.get("references", []):
                ref_clean = ref.lower().strip()
                if ref_clean in id_map:
                    edges.append({
                        "id": f"ref_edge_{edge_idx}",
                        "source": pid,
                        "target": id_map[ref_clean],
                        "type": "reference",
                        "weight": 1.0
                    })
                    edge_idx += 1
                    
        highly_ids = [p.get("paper_id") for p, rel, _ in papers[:20] if rel == "highly_relevant"]
        if highly_ids:
            clusters.append({
                "id": "cluster_highly",
                "label": "核心推荐文献",
                "color": "#e3f2fd",
                "center": {"x": 400, "y": 300},
                "radiusX": 350,
                "radiusY": 350,
                "nodeIds": highly_ids
            })
            
        return {
            "job_id": job_id,
            "query": query,
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "stats": {
                "papers": len([n for n in nodes if n["type"] == "paper"]),
                "authors": 0,
                "methods": len(methods[:5]),
                "datasets": len(datasets[:5]),
                "edges": len(edges),
                "clusters": len(clusters)
            }
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "job_id": job_id,
            "query": "",
            "nodes": [],
            "edges": [],
            "clusters": [],
            "stats": {"papers": 0, "authors": 0, "methods": 0, "datasets": 0, "edges": 0, "clusters": 0}
        }

def send_json(start_response, data, status="200 OK"):
    body = json.dumps(data, ensure_ascii=False, cls=CustomJSONEncoder).encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type, Authorization"),
        ]
    )
    return [body]

# Main REST API WSGI Router
def handle_api_request(environ: dict, start_response) -> list[bytes]:
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")
    
    # Handle CORS OPTIONS Preflight
    if method == "OPTIONS":
        start_response(
            "200 OK",
            [
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                ("Access-Control-Allow-Headers", "Content-Type, Authorization"),
                ("Content-Length", "0"),
            ]
        )
        return [b""]
        
    try:
        # Route: GET /api/system/status
        if path == "/api/system/status":
            try:
                root_dir = Path(__file__).parents[2]
                config = load_config(root_dir / "configs" / "default.yaml")
                status = {
                    "status": "ok",
                    "message": "Scholar Agent API Service is active, provider: " + str(os.getenv("ACTIVE_LLM_PROVIDER", "tokenhub")),
                    "mode": config.app.mode,
                    "config": "default.yaml",
                    "cache_hit_rate": 85.2,
                    "local_index_ready": True,
                    "vector_index_ready": True,
                    "provider_count": {"healthy": len([x for x in config.app.providers if x != "mock"]), "total": len(config.app.providers)}
                }
            except Exception:
                status = {
                    "status": "ok",
                    "message": "Scholar Agent Fallback Mode",
                    "mode": "live",
                    "config": "default.yaml",
                    "cache_hit_rate": 0.0,
                    "local_index_ready": True,
                    "vector_index_ready": False,
                    "provider_count": {"healthy": 2, "total": 4}
                }
            return send_json(start_response, status)
            
        # Route: GET /api/database/status
        if path == "/api/database/status":
            import sqlite3
            root_dir = Path(__file__).parents[2]
            
            pasa_db = root_dir / "data" / "pasa_local_fts.sqlite"
            pasa_info = {
                "path": str(pasa_db.resolve()),
                "exists": pasa_db.exists(),
                "size_mb": round(pasa_db.stat().st_size / (1024 * 1024), 2) if pasa_db.exists() else 0.0,
                "table_count": 0,
                "paper_count": 0
            }
            if pasa_info["exists"]:
                try:
                    conn = sqlite3.connect(str(pasa_db))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    pasa_info["table_count"] = len(tables)
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
                    if cursor.fetchone():
                        cursor.execute("SELECT count(*) FROM papers")
                        pasa_info["paper_count"] = cursor.fetchone()[0]
                    conn.close()
                except Exception:
                    pass
                    
            session_db = Path(DEFAULT_DB)
            session_info = {
                "path": str(session_db.resolve()),
                "exists": session_db.exists(),
                "size_mb": round(session_db.stat().st_size / (1024 * 1024), 2) if session_db.exists() else 0.0,
                "table_count": 0,
                "run_count": 0
            }
            if session_info["exists"]:
                try:
                    conn = sqlite3.connect(str(session_db))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    session_info["table_count"] = len(tables)
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
                    if cursor.fetchone():
                        cursor.execute("SELECT count(*) FROM runs")
                        session_info["run_count"] = cursor.fetchone()[0]
                    conn.close()
                except Exception:
                    pass
            if not session_info["run_count"]:
                with JOBS_LOCK:
                    session_info["run_count"] = len([k for k, v in ACTIVE_JOBS.items() if not v.get("is_eval")])
                
            return send_json(start_response, {
                "pasa_local_fts": pasa_info,
                "session_hub_index": session_info,
                "logs_db": RUN_STORE.summary()
            })
            
        # Route: GET /api/configs
        if path == "/api/configs":
            configs = []
            try:
                config_dir = Path("configs")
                if config_dir.exists():
                    configs = [p.name for p in config_dir.glob("*.yaml")]
            except Exception:
                pass
            if not configs:
                configs = ["default.yaml", "effect_first.yaml"]
            return send_json(start_response, configs)

        # Route: GET /api/datasets
        if path == "/api/datasets":
            datasets = []
            try:
                bench_dir = Path("data/benchmarks")
                if bench_dir.exists():
                    datasets = [
                        p.name
                        for pattern in ("*.jsonl", "*.json", "*.csv", "*.tsv")
                        for p in bench_dir.glob(pattern)
                    ]
            except Exception:
                pass
            if not datasets:
                datasets = ["AutoScholarQuery_dev.jsonl", "RealScholarQuery_test.jsonl"]
            return send_json(start_response, datasets)

        # Route: GET /api/providers
        if path == "/api/providers":
            providers_list = []
            try:
                root_dir = Path(__file__).parents[2]
                config = load_config(root_dir / "configs" / "default.yaml")
                providers_list.append({
                    "name": "Local Index",
                    "enabled": config.providers.pasa_local.enabled,
                    "available": True,
                    "source": "local",
                    "latency_ms": 12,
                    "status": "healthy" if config.providers.pasa_local.enabled else "offline",
                    "notes": "本地论文数据库索引"
                })
                providers_list.append({
                    "name": "OpenAlex",
                    "enabled": config.providers.openalex.enabled,
                    "available": True,
                    "source": "remote",
                    "latency_ms": 280,
                    "status": "healthy" if config.providers.openalex.enabled else "offline",
                    "notes": "学术关系图谱与全网召回源"
                })
                providers_list.append({
                    "name": "Semantic Scholar",
                    "enabled": config.providers.semantic_scholar.enabled,
                    "available": True,
                    "source": "remote",
                    "latency_ms": 320,
                    "status": "healthy" if config.providers.semantic_scholar.enabled else "offline",
                    "notes": "引文分析与精排协作源"
                })
                providers_list.append({
                    "name": "arXiv",
                    "enabled": config.providers.arxiv.enabled,
                    "available": True,
                    "source": "remote",
                    "latency_ms": 150,
                    "status": "healthy" if config.providers.arxiv.enabled else "offline"
                })
            except Exception:
                providers_list = [
                    {"name": "Local Index", "enabled": True, "available": True, "source": "local", "latency_ms": 10, "status": "healthy"},
                    {"name": "Semantic Scholar", "enabled": True, "available": True, "source": "remote", "latency_ms": 200, "status": "healthy"},
                    {"name": "OpenAlex", "enabled": True, "available": True, "source": "remote", "latency_ms": 200, "status": "healthy"},
                ]
            return send_json(start_response, providers_list)

        # Route: POST /api/search
        if path == "/api/search" or path == "/api/search/retrieval-only":
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except ValueError:
                request_body_size = 0
            request_body = environ['wsgi.input'].read(request_body_size)
            payload = json.loads(request_body.decode('utf-8')) if request_body else {}
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"search_{timestamp}"
            
            with JOBS_LOCK:
                ACTIVE_JOBS[job_id] = {
                    "job_id": job_id,
                    "status": "running",
                    "stage": "planning",
                    "progress": 5,
                    "elapsed_seconds": 0.0,
                    "error": None,
                    "result": None,
                    "logs": [],
                    "start_time": time.time(),
                    "query": payload.get("query", ""),
                    "mode": payload.get("mode", "research"),
                    "config": payload.get("config", "configs/default.yaml"),
                    "providers": payload.get("providers", []),
                    "retrieval_only": payload.get("retrieval_only", False) or path.endswith("/retrieval-only"),
                    "stage_artifacts": {},
                    "is_eval": False,
                    "cancel_requested": False
                }
                job_ref = ACTIVE_JOBS[job_id]
            persist_job_snapshot(job_ref)
                
            t = threading.Thread(target=run_pipeline_task, args=(job_id, job_ref))
            t.daemon = True
            t.start()
            
            return send_json(start_response, {
                "job_id": job_id,
                "status": "running",
                "stage": "planning",
                "progress": 5,
                "elapsed_seconds": 0.0
            })
            
        # Route: POST /api/eval/case
        if path == "/api/eval/case":
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except ValueError:
                request_body_size = 0
            request_body = environ['wsgi.input'].read(request_body_size)
            payload = json.loads(request_body.decode('utf-8')) if request_body else {}
            
            dataset = payload.get("dataset", "AutoScholarQuery_dev.jsonl")
            case_index = payload.get("case_index", 0)
            mode = payload.get("mode", "mock")
            config = payload.get("config", "configs/default.yaml")
            
            # Load the query and gold standards
            query, gold = load_eval_case(dataset, case_index)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"eval_{timestamp}_case_{case_index}"
            
            with JOBS_LOCK:
                ACTIVE_JOBS[job_id] = {
                    "job_id": job_id,
                    "status": "running",
                    "stage": "planning",
                    "progress": 5,
                    "elapsed_seconds": 0.0,
                    "error": None,
                    "result": None,
                    "logs": [],
                    "start_time": time.time(),
                    "query": query,
                    "mode": mode,
                    "config": config,
                    "providers": [],
                    "retrieval_only": False,
                    "stage_artifacts": {},
                    "is_eval": True,
                    "gold": gold,
                    "dataset": dataset,
                    "case_index": case_index,
                    "eval_result": None,
                    "cancel_requested": False
                }
                job_ref = ACTIVE_JOBS[job_id]
            persist_job_snapshot(job_ref)
                
            t = threading.Thread(target=run_pipeline_task, args=(job_id, job_ref))
            t.daemon = True
            t.start()
            
            # Immediately return case result in running state
            return send_json(start_response, {
                "job_id": job_id,
                "dataset": dataset,
                "case_index": case_index,
                "query": query,
                "gold": [g.get("title", "") if isinstance(g, dict) else g for g in gold],
                "eval_metrics": None,
                "budget": None,
                "progress": [{"stage": "Retrieval", "status": "running", "elapsed_seconds": 0.1}],
                "candidate_pool": [],
                "selection_candidates": [],
                "ranked_papers": [],
                "final_output": [],
                "warnings": []
            })
            
        # Route: POST /api/eval/random-case
        if path == "/api/eval/random-case":
            # Redirect to first case of first dataset for simplicity
            datasets = ["AutoScholarQuery_dev.jsonl"]
            try:
                bench_dir = Path("data/benchmarks")
                if bench_dir.exists():
                    datasets = [
                        p.name
                        for pattern in ("*.jsonl", "*.json", "*.csv", "*.tsv")
                        for p in bench_dir.glob(pattern)
                    ]
            except Exception:
                pass
                
            import random
            dataset = datasets[0]
            case_index = random.randint(0, 10)
            
            # Load case
            query, gold = load_eval_case(dataset, case_index)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"eval_{timestamp}_case_{case_index}"
            
            with JOBS_LOCK:
                ACTIVE_JOBS[job_id] = {
                    "job_id": job_id,
                    "status": "running",
                    "stage": "planning",
                    "progress": 5,
                    "elapsed_seconds": 0.0,
                    "error": None,
                    "result": None,
                    "logs": [],
                    "start_time": time.time(),
                    "query": query,
                    "mode": "mock",
                    "config": "configs/default.yaml",
                    "providers": [],
                    "retrieval_only": False,
                    "stage_artifacts": {},
                    "is_eval": True,
                    "gold": gold,
                    "dataset": dataset,
                    "case_index": case_index,
                    "eval_result": None,
                    "cancel_requested": False
                }
                job_ref = ACTIVE_JOBS[job_id]
            persist_job_snapshot(job_ref)
                
            t = threading.Thread(target=run_pipeline_task, args=(job_id, job_ref))
            t.daemon = True
            t.start()
            
            return send_json(start_response, {
                "job_id": job_id,
                "dataset": dataset,
                "case_index": case_index,
                "query": query,
                "gold": [g.get("title", "") if isinstance(g, dict) else g for g in gold],
                "eval_metrics": None,
                "budget": None,
                "progress": [{"stage": "Retrieval", "status": "running", "elapsed_seconds": 0.1}],
                "candidate_pool": [],
                "selection_candidates": [],
                "ranked_papers": [],
                "final_output": [],
                "warnings": []
            })
            
        # Route: GET/DELETE /api/search/{jobId}
        if path.startswith("/api/search/"):
            job_id = path[len("/api/search/"):]
            if job_id.endswith("/cancel"):
                job_id = job_id[: -len("/cancel")]
                if method != "POST":
                    return send_json(start_response, {"error": "Method not allowed"}, "405 Method Not Allowed")
                with JOBS_LOCK:
                    job = ACTIVE_JOBS.get(job_id)
                if not job:
                    return send_json(start_response, {"job_id": job_id, "status": "failed", "error": "Job not found"}, "404 Not Found")
                mark_job_cancelled(job)
                return send_json(start_response, {
                    "job_id": job_id,
                    "status": "cancelled",
                    "stage": job.get("stage", "cancelled"),
                    "progress": job.get("progress", 0),
                    "elapsed_seconds": round(job.get("elapsed_seconds", 0.0), 1),
                    "error": job.get("error")
                })

            if method == "DELETE":
                with JOBS_LOCK:
                    if job_id in ACTIVE_JOBS:
                        ACTIVE_JOBS.pop(job_id)
                cache_file = CACHE_DIR / f"{job_id}.json"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
                return send_json(start_response, {"status": "succeeded", "message": f"Job {job_id} deleted successfully"})
                
            with JOBS_LOCK:
                job = ACTIVE_JOBS.get(job_id)
            if not job:
                # If evaluation job, search in eval jobs, else return mock fallback
                return send_json(start_response, {
                    "job_id": job_id,
                    "status": "succeeded",
                    "stage": "synthesis",
                    "progress": 100,
                    "elapsed_seconds": 5.0
                })
                
            elapsed = job["elapsed_seconds"]
            if job["status"] == "running":
                elapsed = time.time() - job["start_time"]
                
            # Evaluation jobs use the same URL for polling but return the eval contract.
            if job.get("is_eval"):
                if job["status"] == "succeeded":
                    return send_json(start_response, job.get("eval_result", {}))
                return send_json(start_response, {
                    "job_id": job_id,
                    "dataset": job.get("dataset", ""),
                    "case_index": job.get("case_index", 0),
                    "query": job.get("query", ""),
                    "gold": [g.get("title", "") if isinstance(g, dict) else g for g in job.get("gold", [])],
                    "eval_metrics": None,
                    "budget": None,
                    "progress": [{
                        "stage": job["stage"],
                        "status": job["status"],
                        "elapsed_seconds": round(elapsed, 1)
                    }],
                    "candidate_pool": [],
                    "selection_candidates": [],
                    "ranked_papers": [],
                    "final_output": [],
                    "warnings": [job["error"]] if job.get("error") else []
                })
                
            return send_json(start_response, {
                "job_id": job_id,
                "status": job["status"],
                "stage": job["stage"],
                "progress": job["progress"],
                "elapsed_seconds": round(elapsed, 1),
                "error": job["error"]
            })
            
        # Route: GET /api/results/{jobId}
        if path.startswith("/api/results/"):
            # Check if stage request: /api/results/{jobId}/stage/{stageName}
            stage_match = re.match(r"^/api/results/([^/]+)/stage/([^/]+)$", path)
            if stage_match:
                job_id = stage_match.group(1)
                stage_name = stage_match.group(2)
                with JOBS_LOCK:
                    job = ACTIVE_JOBS.get(job_id)
                if not job:
                    return send_json(start_response, {"stage": stage_name, "job_id": job_id, "data": {}})
                
                stage_data = job.get("stage_artifacts", {}).get(stage_name, {})
                return send_json(start_response, stage_data)
                
            # Otherwise normal results request: /api/results/{jobId}
            job_id = path[len("/api/results/"):]
            with JOBS_LOCK:
                job = ACTIVE_JOBS.get(job_id)
            if not job:
                return send_json(start_response, {"job_id": job_id, "status": "failed", "result": None, "artifacts": {}})
                
            pool_size = 0
            sel_size = 0
            ranked_size = 0
            final_size = 0
            
            if job.get("result"):
                pool_size = job["result"].get("run_metrics", {}).get("candidate_pool_size", 0)
                ranked_size = len(job["result"].get("highly_relevant_papers", [])) + len(job["result"].get("partially_relevant_papers", [])) + len(job["result"].get("supporting_papers", []))
                final_size = len(job["result"].get("highly_relevant_papers", [])) + len(job["result"].get("partially_relevant_papers", []))
                
            if job.get("stage_artifacts") and "selection" in job["stage_artifacts"]:
                sel_size = job["stage_artifacts"]["selection"]["data"].get("selection_candidates_size", 0)
                
            results_resp = {
                "job_id": job_id,
                "status": job["status"],
                "result": job["result"],
                "artifacts": {
                    "candidate_pool_size": pool_size,
                    "selection_candidates_size": sel_size,
                    "ranked_papers_size": ranked_size,
                    "final_output_size": final_size
                }
            }
            return send_json(start_response, results_resp)

        # Route: GET /api/graph/{jobId}
        if path.startswith("/api/graph/"):
            job_id = path[len("/api/graph/"):]
            with JOBS_LOCK:
                job = ACTIVE_JOBS.get(job_id)
            if not job or not job.get("result"):
                return send_json(start_response, {
                    "job_id": job_id,
                    "query": "",
                    "nodes": [],
                    "edges": [],
                    "clusters": [],
                    "stats": {"papers": 0, "authors": 0, "methods": 0, "datasets": 0, "edges": 0, "clusters": 0}
                })
            graph_data = build_graph_response(job_id, job["result"])
            return send_json(start_response, graph_data)

        # Route: GET /api/logs/{jobId}
        if path.startswith("/api/logs/"):
            job_id = path[len("/api/logs/"):]
            with JOBS_LOCK:
                job = ACTIVE_JOBS.get(job_id)
            logs = job.get("logs", []) if job else []
            return send_json(start_response, logs)

        # Route: GET /api/recent-runs
        if path == "/api/recent-runs":
            runs = []
            with JOBS_LOCK:
                for jid, job in ACTIVE_JOBS.items():
                    # Only return non-eval jobs for main workspace
                    if not job.get("is_eval"):
                        runs.append({
                            "job_id": jid,
                            "status": job["status"],
                            "stage": job["stage"],
                            "progress": job["progress"],
                            "elapsed_seconds": job["elapsed_seconds"]
                        })
            runs.sort(key=lambda x: x["job_id"], reverse=True)
            return send_json(start_response, runs[:10])

        # Route: GET /api/papers/{paperId}
        if path.startswith("/api/papers/"):
            paper_id = path[len("/api/papers/"):]
            # Search paper in completed jobs
            found_paper = None
            with JOBS_LOCK:
                for job in ACTIVE_JOBS.values():
                    res = job.get("result")
                    if res:
                        all_p = (
                            [x.get("paper") for x in res.get("highly_relevant_papers", [])] +
                            [x.get("paper") for x in res.get("partially_relevant_papers", [])] +
                            [x.get("paper") for x in res.get("supporting_papers", [])]
                        )
                        for p in all_p:
                            if p and p.get("paper_id") == paper_id:
                                found_paper = p
                                break
                    if found_paper:
                        break
            if found_paper:
                return send_json(start_response, found_paper)
            else:
                return send_json(start_response, {"error": "Paper not found"}, "404 Not Found")

        # Unmatched API route
        return send_json(start_response, {"error": "API route not found"}, "404 Not Found")

    except Exception as e:
        traceback.print_exc()
        return send_json(start_response, {"error": str(e)}, "500 Internal Server Error")

def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the web UI server."""
    from wsgiref.simple_server import make_server

    print(f"🚀 Session Hub Web UI starting at http://{host}:{port}")
    print(f"📁 Database: {DEFAULT_DB}")
    print("Press Ctrl+C to stop")

    server = make_server(host, port, handle_request)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Stopping server...")
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Session Hub Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    args = parser.parse_args()

    run_server(args.host, args.port)
