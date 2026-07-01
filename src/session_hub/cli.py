"""CLI entry point — ``python -m session_hub ...``.

Subcommands mirror the cass contract where it makes sense, but stay
deliberately small: index / search / list / show / share / ingest /
stats. Every command supports ``--json`` for agent consumption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .connectors import ALL, get_connector
from .index import SessionIndex
from .models import AgentKind
from .share import export_session, import_bundle

DEFAULT_DB = Path(
    os.environ.get(
        "SESSION_HUB_DB",
        str(Path.home() / ".session_hub" / "index.sqlite"),
    )
)


def _print_json(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


# --------------------------------------------------------------- command impls
def cmd_index(args: argparse.Namespace) -> int:
    idx = SessionIndex(args.db)
    connectors = [get_connector(a) for a in (args.agent or [c.agent.value for c in ALL])]
    connectors = [c for c in connectors if c is not None]
    total_sessions = 0
    total_messages = 0
    per_agent: dict[str, int] = {}
    for conn in connectors:
        n_s = 0
        for root in conn.default_roots():
            for conv in conn.iter_sessions(root):
                n = idx.upsert_conversation(conv)
                n_s += 1
                total_messages += n
        per_agent[conn.agent.value] = n_s
        total_sessions += n_s
    if args.json:
        _print_json(
            {
                "ok": True,
                "db": str(args.db),
                "sessions_indexed": total_sessions,
                "messages_indexed": total_messages,
                "by_agent": per_agent,
            }
        )
    else:
        print(f"Indexed {total_sessions} sessions ({total_messages} messages) into {args.db}")
        for a, n in per_agent.items():
            print(f"  {a}: {n}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    idx = SessionIndex(args.db)
    rows = idx.search(args.query, agent=args.agent, limit=args.limit)
    if args.json:
        _print_json({"ok": True, "query": args.query, "hits": rows, "count": len(rows)})
        return 0
    if not rows:
        print("(no matches)")
        return 0
    for r in rows:
        print(f"[{r['agent']}] {r['title'] or r['session_id'][:8]}")
        print(f"  workspace: {r['workspace']}")
        print(f"  msg#{r['ordinal']} ({r['role']}, {r['ts']}): {r['hit']}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    idx = SessionIndex(args.db)
    rows = idx.list_sessions(agent=args.agent, workspace=args.workspace, limit=args.limit)
    if args.json:
        _print_json({"ok": True, "sessions": rows, "count": len(rows)})
        return 0
    print(f"{'agent':<14}{'msgs':>5}  {'started':<20}  title")
    for r in rows:
        print(
            f"{r['agent']:<14}{r['message_count']:>5}  "
            f"{(r['started_at'] or '')[:19]:<20}  {r['title']}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    idx = SessionIndex(args.db)
    session = idx.get_session(args.agent, args.session_id)
    if session is None:
        if args.json:
            _print_json({"ok": False, "error": "session_not_found"})
        else:
            print(f"session not found: {args.agent}/{args.session_id}", file=sys.stderr)
        return 1
    messages = idx.get_messages(args.agent, args.session_id)
    if args.json:
        _print_json({"ok": True, "session": session, "messages": messages})
        return 0
    print(f"# {session['title'] or '(untitled)'}")
    print(f"agent={session['agent']}  session={session['session_id']}")
    print(f"workspace={session['workspace']}")
    print(f"started={session['started_at']}  ended={session['ended_at']}")
    print("-" * 60)
    for m in messages:
        head = f"[{m['ordinal']:>3}] {m['role']:<9} {m['ts']}"
        print(head)
        body = m["plain_text"][: args.max_chars]
        for line in body.splitlines():
            print(f"    {line}")
        if len(m["plain_text"]) > args.max_chars:
            print(f"    ... ({len(m['plain_text']) - args.max_chars} more chars)")
    return 0


def cmd_share(args: argparse.Namespace) -> int:
    out = export_session(args.db, args.agent, args.session_id, args.output)
    if args.json:
        _print_json({"ok": True, "bundle_path": str(out)})
    else:
        print(f"wrote bundle: {out}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    idx = SessionIndex(args.db)
    results = []
    for bundle in args.bundle:
        results.append(import_bundle(Path(bundle), idx))
    if args.json:
        _print_json({"ok": True, "ingested": results, "count": len(results)})
    else:
        for r in results:
            print(f"ingested {r['agent']}/{r['session_id']} ({r['messages_indexed']} messages)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    idx = SessionIndex(args.db)
    s = idx.stats()
    s["db"] = str(args.db)
    s["ok"] = True
    if args.json:
        _print_json(s)
    else:
        print(f"db: {s['db']}")
        print(f"messages total: {s['messages_total']}")
        for a, n in s["sessions_by_agent"].items():
            print(f"  {a}: {n} sessions")
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    out = [
        {
            "agent": c.agent.value,
            "roots": [str(p) for p in c.default_roots()],
        }
        for c in ALL
    ]
    if args.json:
        _print_json({"ok": True, "agents": out})
    else:
        for a in out:
            print(f"{a['agent']}: {', '.join(a['roots'])}")
    return 0


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="session-hub",
        description="Cross-agent session history manager.",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite index path (default: {DEFAULT_DB})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index", help="scan agent roots and rebuild the index")
    s.add_argument("--agent", action="append", help="limit to one agent (repeatable)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="BM25 keyword search across all sessions")
    s.add_argument("query")
    s.add_argument("--agent", help="limit to one agent")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("list", help="list indexed sessions")
    s.add_argument("--agent", help="filter by agent")
    s.add_argument("--workspace", help="filter by workspace substring")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show full session content")
    s.add_argument("agent")
    s.add_argument("session_id")
    s.add_argument("--max-chars", type=int, default=400)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("share", help="export a session to a portable bundle")
    s.add_argument("agent")
    s.add_argument("session_id")
    s.add_argument("-o", "--output", type=Path, required=True)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_share)

    s = sub.add_parser("ingest", help="import a portable bundle into the index")
    s.add_argument("bundle", nargs="+", type=Path)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("stats", help="show index statistics")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("agents", help="list supported agents and their storage roots")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_agents)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())