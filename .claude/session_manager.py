#!/usr/bin/env python3
"""
Claude Code 会话历史管理工具
提供类似Codex的历史会话查看和恢复功能
"""

import json
import sys
from datetime import datetime
from pathlib import Path


HISTORY_DIR = Path(".claude/chat-history")
INDEX_FILE = HISTORY_DIR / "index.json"


def ensure_history_dir():
    """确保历史目录存在"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        save_index({"version": "1.0", "sessions": [], "current_session": None})


def load_index():
    """加载索引文件"""
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": "1.0", "sessions": [], "current_session": None}


def save_index(data):
    """保存索引文件"""
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_session(title: str, context: dict = None) -> str:
    """创建新会话"""
    ensure_history_dir()
    index = load_index()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = {
        "id": session_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "message_count": 0,
        "context": context or {}
    }

    index["sessions"].insert(0, session)
    index["current_session"] = session_id
    save_index(index)

    # 创建会话文件
    session_file = HISTORY_DIR / f"session_{session_id}.json"
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump({"messages": [], "context": context or {}}, f, indent=2, ensure_ascii=False)

    return session_id


def list_sessions(limit: int = 10):
    """列出最近会话"""
    ensure_history_dir()
    index = load_index()
    sessions = index.get("sessions", [])

    print(f"\n📚 最近会话 (共 {len(sessions)} 个):\n")
    print(f"{'ID':<20} {'标题':<40} {'消息数':<8} {'更新时间'}")
    print("-" * 90)

    for session in sessions[:limit]:
        sid = session["id"]
        title = session["title"][:37] + "..." if len(session["title"]) > 40 else session["title"]
        msg_count = session.get("message_count", 0)
        updated = session["updated_at"][:16] if "updated_at" in session else "N/A"
        print(f"{sid:<20} {title:<40} {msg_count:<8} {updated}")

    current = index.get("current_session")
    if current:
        print(f"\n▶ 当前会话: {current}")

    return sessions[:limit]


def switch_session(session_id: str):
    """切换到指定会话"""
    index = load_index()

    # 查找会话
    session = None
    for s in index.get("sessions", []):
        if s["id"] == session_id:
            session = s
            break

    if not session:
        print(f"❌ 会话不存在: {session_id}")
        return False

    index["current_session"] = session_id
    save_index(index)

    print(f"✅ 已切换到会话: {session_id}")
    print(f"📋 标题: {session.get('title', 'N/A')}")
    return True


def get_current_session():
    """获取当前会话信息"""
    index = load_index()
    current_id = index.get("current_session")
    if not current_id:
        return None

    for s in index.get("sessions", []):
        if s["id"] == current_id:
            return s
    return None


def show_help():
    """显示帮助信息"""
    print("""
📖 Claude Code 会话历史管理

用法:
  python .claude/session_manager.py <command> [args]

命令:
  list [n]              列出最近n个会话 (默认10个)
  create "标题"         创建新会话
  switch <session_id>   切换到指定会话
  current               显示当前会话
  help                  显示帮助

示例:
  python .claude/session_manager.py list
  python .claude/session_manager.py create "实现检索模块"
  python .claude/session_manager.py switch 20250701_143052
""")


def main():
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        list_sessions(limit)

    elif command == "create":
        if len(sys.argv) < 3:
            print("❌ 请提供会话标题")
            print("用法: python .claude/session_manager.py create \"标题\"")
            return
        title = sys.argv[2]
        session_id = create_session(title)
        print(f"✅ 已创建会话: {session_id}")

    elif command == "switch":
        if len(sys.argv) < 3:
            print("❌ 请提供会话ID")
            return
        switch_session(sys.argv[2])

    elif command == "current":
        session = get_current_session()
        if session:
            print(f"▶ 当前会话: {session['id']}")
            print(f"  标题: {session.get('title', 'N/A')}")
            print(f"  消息数: {session.get('message_count', 0)}")
            print(f"  更新时间: {session.get('updated_at', 'N/A')}")
        else:
            print("⚠ 没有活动的会话")

    elif command == "help":
        show_help()

    else:
        print(f"❌ 未知命令: {command}")
        show_help()


if __name__ == "__main__":
    main()
