#!/usr/bin/env python3
"""
Claude Code 历史记录查看工具
读取 ~/.claude/history.jsonl 并展示历史会话
"""

import json
import sys
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path.home() / ".claude" / "history.jsonl"


def format_timestamp(ts_ms):
    """将毫秒时间戳转换为可读格式"""
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "Unknown"


def load_history():
    """加载历史记录"""
    if not HISTORY_FILE.exists():
        print(f"❌ 历史文件不存在: {HISTORY_FILE}")
        return []

    entries = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries


def group_by_session(entries):
    """按 sessionId 分组"""
    sessions = {}
    for entry in entries:
        sid = entry.get("sessionId", "unknown")
        if sid not in sessions:
            sessions[sid] = {
                "id": sid,
                "project": entry.get("project", "Unknown"),
                "messages": [],
                "first_time": entry.get("timestamp", 0),
                "last_time": entry.get("timestamp", 0),
            }
        sessions[sid]["messages"].append(entry)
        sessions[sid]["last_time"] = max(sessions[sid]["last_time"], entry.get("timestamp", 0))
        sessions[sid]["first_time"] = min(sessions[sid]["first_time"], entry.get("timestamp", 0))
    return sessions


def show_history(limit=20, project_filter=None):
    """展示历史记录"""
    entries = load_history()
    if not entries:
        print("没有历史记录")
        return

    sessions = group_by_session(entries)

    # 按最后时间排序
    sorted_sessions = sorted(
        sessions.values(),
        key=lambda x: x["last_time"],
        reverse=True
    )

    # 过滤项目
    if project_filter:
        sorted_sessions = [s for s in sorted_sessions if project_filter in s["project"]]

    print(f"\n📚 Claude Code 历史会话 (共 {len(sorted_sessions)} 个会话, {len(entries)} 条消息)\n")
    print(f"{'会话ID':<36} {'项目路径':<50} {'消息数':<8} {'最后活动时间'}")
    print("-" * 110)

    for session in sorted_sessions[:limit]:
        sid = session["id"][:36]
        project = session["project"]
        if len(project) > 47:
            project = project[:44] + "..."
        msg_count = len(session["messages"])
        last_time = format_timestamp(session["last_time"])
        print(f"{sid:<36} {project:<50} {msg_count:<8} {last_time}")

    # 显示当前项目
    current_project = None
    for entry in reversed(entries):
        if "project" in entry:
            current_project = entry["project"]
            break

    if current_project:
        print(f"\n▶ 当前项目: {current_project}")


def show_session_detail(session_id):
    """展示特定会话详情"""
    entries = load_history()
    sessions = group_by_session(entries)

    if session_id not in sessions:
        print(f"❌ 会话不存在: {session_id}")
        return

    session = sessions[session_id]
    print(f"\n📋 会话详情: {session_id}\n")
    print(f"项目: {session['project']}")
    print(f"消息数: {len(session['messages'])}")
    print(f"开始时间: {format_timestamp(session['first_time'])}")
    print(f"最后活跃: {format_timestamp(session['last_time'])}")
    print(f"\n消息列表:\n")

    for i, msg in enumerate(session["messages"][-20:], 1):  # 只显示最后20条
        display = msg.get("display", "")
        if display.startswith("/"):
            print(f"  {i}. [{format_timestamp(msg['timestamp'])}] 命令: {display}")
        else:
            preview = display[:60] + "..." if len(display) > 60 else display
            print(f"  {i}. [{format_timestamp(msg['timestamp'])}] {preview}")


def main():
    if len(sys.argv) < 2:
        show_history(limit=20)
        print("\n💡 提示:")
        print("  查看最近n个会话: python .claude/history_viewer.py list [n]")
        print("  查看会话详情: python .claude/history_viewer.py show <session_id>")
        print("  过滤项目: python .claude/history_viewer.py project <关键字>")
        return

    command = sys.argv[1].lower()

    if command == "list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        show_history(limit=limit)

    elif command == "show":
        if len(sys.argv) < 3:
            print("❌ 请提供会话ID")
            return
        show_session_detail(sys.argv[2])

    elif command == "project":
        if len(sys.argv) < 3:
            print("❌ 请提供项目关键字")
            return
        show_history(project_filter=sys.argv[2])

    else:
        print(f"❌ 未知命令: {command}")
        print("可用命令: list, show, project")


if __name__ == "__main__":
    main()
