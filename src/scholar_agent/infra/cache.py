from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonFileCache:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.query_dir = self.root_dir / "queries"
        self.paper_dir = self.root_dir / "papers"
        self.llm_dir = self.root_dir / "llm"
        self.query_dir.mkdir(parents=True, exist_ok=True)
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        self.llm_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _query_path(self, key: str) -> Path:
        return self.query_dir / f"{self._hash_key(key)}.json"

    def _paper_path(self, paper_key: str) -> Path:
        return self.paper_dir / f"{self._hash_key(paper_key)}.json"

    def _llm_path(self, key: str) -> Path:
        return self.llm_dir / f"{self._hash_key(key)}.json"

    def get_query(self, key: str) -> list[dict[str, Any]] | None:
        path = self._query_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set_query(self, key: str, value: list[dict[str, Any]]) -> None:
        path = self._query_path(key)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_paper_dict(self, paper_key: str) -> dict[str, Any] | None:
        path = self._paper_path(paper_key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set_paper_dict(self, paper_key: str, paper_dict: dict[str, Any]) -> None:
        path = self._paper_path(paper_key)
        path.write_text(json.dumps(paper_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_llm(self, system_prompt: str, user_prompt: str) -> Any | None:
        key = f"{system_prompt}\n{user_prompt}"
        path = self._llm_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set_llm(self, system_prompt: str, user_prompt: str, value: Any) -> None:
        key = f"{system_prompt}\n{user_prompt}"
        path = self._llm_path(key)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
