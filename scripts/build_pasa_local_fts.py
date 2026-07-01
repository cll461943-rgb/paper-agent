from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any


def keep_letters(text: str | None) -> str:
    return "".join(char for char in (text or "") if char.isalpha()).lower()


def section_reference_text(sections: Any, *, max_chars: int = 12_000) -> str:
    parts: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            raw_title = value.get("title")
            text = raw_title if isinstance(raw_title, str) else ""
        else:
            text = ""
        if text:
            parts.append(text)

    if isinstance(sections, dict):
        for refs in sections.values():
            if isinstance(refs, list):
                for item in refs:
                    add(item)
            else:
                add(refs)
    elif isinstance(sections, list):
        for item in sections:
            add(item)

    text = " ".join(parts)
    return text[:max_chars]


def build_index(root: Path, output: Path) -> None:
    id2paper_path = root / "id2paper.json"
    paper_zip_path = root / "cs_paper_2nd.zip"
    if not id2paper_path.exists():
        raise FileNotFoundError(id2paper_path)
    if not paper_zip_path.exists():
        raise FileNotFoundError(paper_zip_path)

    id2paper = json.loads(id2paper_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix=output.name, suffix=".tmp", dir=output.parent, delete=False) as handle:
        temp_path = Path(handle.name)

    started = time.perf_counter()
    inserted = 0
    try:
        conn = sqlite3.connect(temp_path)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(
            """
            CREATE VIRTUAL TABLE papers_fts USING fts5(
                arxiv_id UNINDEXED,
                title,
                abstract,
                refs,
                title_key UNINDEXED,
                tokenize='unicode61'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("id2paper_mtime", str(id2paper_path.stat().st_mtime_ns)),
                ("zip_mtime", str(paper_zip_path.stat().st_mtime_ns)),
                ("id2paper_count", str(len(id2paper))),
            ],
        )

        with zipfile.ZipFile(paper_zip_path, "r") as archive:
            available = set(archive.namelist())
            batch: list[tuple[str, str, str, str, str]] = []
            for arxiv_id, fallback_title in id2paper.items():
                title_key = keep_letters(fallback_title)
                if title_key not in available:
                    continue
                try:
                    paper_json = json.loads(archive.read(title_key).decode("utf-8"))
                except Exception:
                    continue
                title = paper_json.get("title") or fallback_title
                abstract = paper_json.get("abstract") or ""
                refs = section_reference_text(paper_json.get("sections"))
                batch.append((arxiv_id, title, abstract, refs, title_key))
                if len(batch) >= 1000:
                    conn.executemany(
                        "INSERT INTO papers_fts(arxiv_id, title, abstract, refs, title_key) VALUES (?, ?, ?, ?, ?)",
                        batch,
                    )
                    inserted += len(batch)
                    batch.clear()
                    if inserted % 50_000 == 0:
                        elapsed = time.perf_counter() - started
                        print(f"indexed={inserted} elapsed={elapsed:.1f}s", flush=True)
            if batch:
                conn.executemany(
                    "INSERT INTO papers_fts(arxiv_id, title, abstract, refs, title_key) VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
                inserted += len(batch)

        conn.commit()
        conn.close()
        temp_path.replace(output)
        elapsed = time.perf_counter() - started
        print(f"done indexed={inserted} output={output} elapsed={elapsed:.1f}s", flush=True)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the optional PaSa local SQLite FTS sidecar index.")
    parser.add_argument("--root", default="data", help="Directory containing id2paper.json and cs_paper_2nd.zip.")
    parser.add_argument("--output", default=None, help="Output sqlite path. Defaults to <root>/pasa_local_fts.sqlite.")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output) if args.output else root / "pasa_local_fts.sqlite"
    build_index(root, output)


if __name__ == "__main__":
    main()
