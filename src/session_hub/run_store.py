from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "timed_out"}


def default_logs_db_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[2]
    return base / "logs" / "logs.db"


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _timestamp_to_iso(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return _dt.datetime.fromtimestamp(value, _dt.timezone.utc).isoformat()


class RunStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_logs_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_runs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT,
                    progress REAL,
                    query TEXT,
                    dataset TEXT,
                    case_index INTEGER,
                    mode TEXT,
                    config TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    elapsed_seconds REAL,
                    run_metrics_json TEXT,
                    eval_metrics_json TEXT,
                    budget_json TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_run_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    stage TEXT,
                    message TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_runs_updated_at ON web_runs(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_run_logs_job_id ON web_run_logs(job_id)")

    def upsert_run(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("job_id") or "")
        if not job_id:
            return

        status = str(job.get("status") or "running")
        result = job.get("result") if isinstance(job.get("result"), dict) else None
        eval_result = job.get("eval_result") if isinstance(job.get("eval_result"), dict) else None
        run_metrics = result.get("run_metrics") if result else None
        eval_metrics = eval_result.get("eval_metrics") if eval_result else result.get("benchmark_metrics") if result else None
        budget = eval_result.get("budget") if eval_result else None
        now = _now_iso()
        started_at = _timestamp_to_iso(job.get("start_time"))
        finished_at = now if status in TERMINAL_STATUSES else None
        kind = "eval" if job.get("is_eval") or job_id.startswith("eval_") else "search"

        values = {
            "job_id": job_id,
            "kind": kind,
            "status": status,
            "stage": job.get("stage"),
            "progress": job.get("progress"),
            "query": job.get("query"),
            "dataset": job.get("dataset"),
            "case_index": job.get("case_index"),
            "mode": job.get("mode"),
            "config": job.get("config"),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": job.get("elapsed_seconds"),
            "run_metrics_json": _json_dumps(run_metrics),
            "eval_metrics_json": _json_dumps(eval_metrics),
            "budget_json": _json_dumps(budget),
            "result_json": _json_dumps(result),
            "error": job.get("error"),
            "created_at": now,
            "updated_at": now,
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_runs (
                    job_id, kind, status, stage, progress, query, dataset, case_index,
                    mode, config, started_at, finished_at, elapsed_seconds,
                    run_metrics_json, eval_metrics_json, budget_json, result_json,
                    error, created_at, updated_at
                ) VALUES (
                    :job_id, :kind, :status, :stage, :progress, :query, :dataset, :case_index,
                    :mode, :config, :started_at, :finished_at, :elapsed_seconds,
                    :run_metrics_json, :eval_metrics_json, :budget_json, :result_json,
                    :error, :created_at, :updated_at
                )
                ON CONFLICT(job_id) DO UPDATE SET
                    kind = excluded.kind,
                    status = excluded.status,
                    stage = excluded.stage,
                    progress = excluded.progress,
                    query = excluded.query,
                    dataset = excluded.dataset,
                    case_index = excluded.case_index,
                    mode = excluded.mode,
                    config = excluded.config,
                    started_at = COALESCE(web_runs.started_at, excluded.started_at),
                    finished_at = excluded.finished_at,
                    elapsed_seconds = excluded.elapsed_seconds,
                    run_metrics_json = excluded.run_metrics_json,
                    eval_metrics_json = excluded.eval_metrics_json,
                    budget_json = excluded.budget_json,
                    result_json = excluded.result_json,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                values,
            )

    def add_log(self, job_id: str, level: str, stage: str, message: str, timestamp: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_run_logs (job_id, timestamp, level, stage, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, timestamp or _now_iso(), level, stage, message),
            )

    def get_run(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM web_runs WHERE job_id = ?", (job_id,)).fetchone()
        return self._decode_run(row) if row else None

    def list_logs(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, level, stage, message FROM web_run_logs WHERE job_id = ? ORDER BY id ASC",
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        self.initialize()
        with self._connect() as conn:
            table_count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
            run_count = conn.execute("SELECT COUNT(*) FROM web_runs").fetchone()[0]
        size_mb = round(self.db_path.stat().st_size / (1024 * 1024), 2) if self.db_path.exists() else 0
        return {
            "path": str(self.db_path),
            "exists": self.db_path.exists(),
            "size_mb": size_mb,
            "run_count": run_count,
            "table_count": table_count,
        }

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["run_metrics"] = _json_loads(data.pop("run_metrics_json", None))
        data["eval_metrics"] = _json_loads(data.pop("eval_metrics_json", None))
        data["budget"] = _json_loads(data.pop("budget_json", None))
        data["result"] = _json_loads(data.pop("result_json", None))
        return data
