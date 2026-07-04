from __future__ import annotations

import sqlite3

from session_hub.run_store import RunStore


def test_run_store_initializes_schema_and_persists_result(tmp_path):
    db_path = tmp_path / "logs.db"
    store = RunStore(db_path)

    store.upsert_run(
        {
            "job_id": "search_test",
            "status": "succeeded",
            "stage": "synthesis",
            "progress": 100,
            "query": "graph neural networks for recommendation",
            "mode": "research",
            "config": "local_full_pipeline.yaml",
            "elapsed_seconds": 12.5,
            "result": {
                "run_metrics": {
                    "api_calls_used": 7,
                    "llm_calls_used": 2,
                    "token_estimate": 1800,
                    "candidate_pool_size": 42,
                    "final_papers": 5,
                    "elapsed_seconds": 12.5,
                }
            },
        }
    )

    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert {"web_runs", "web_run_logs"}.issubset(tables)

    row = store.get_run("search_test")
    assert row is not None
    assert row["status"] == "succeeded"
    assert row["query"] == "graph neural networks for recommendation"
    assert row["run_metrics"]["token_estimate"] == 1800


def test_run_store_records_eval_metrics_and_logs(tmp_path):
    store = RunStore(tmp_path / "logs.db")

    store.upsert_run(
        {
            "job_id": "eval_case_1",
            "status": "succeeded",
            "stage": "synthesis",
            "progress": 100,
            "query": "medical multimodal benchmark",
            "is_eval": True,
            "dataset": "AutoScholarQuery_dev.jsonl",
            "case_index": 3,
            "eval_result": {
                "eval_metrics": {"f1": 0.5, "precision": 0.3333, "recall": 1.0},
                "budget": {"token_estimate": 6820, "api_calls": 24, "llm_calls": 3, "elapsed_seconds": 18.7},
            },
        }
    )
    store.add_log("eval_case_1", "INFO", "retrieval", "retrieved candidate pool")

    row = store.get_run("eval_case_1")
    assert row is not None
    assert row["kind"] == "eval"
    assert row["dataset"] == "AutoScholarQuery_dev.jsonl"
    assert row["eval_metrics"]["f1"] == 0.5
    assert row["budget"]["token_estimate"] == 6820
    assert store.list_logs("eval_case_1")[0]["message"] == "retrieved candidate pool"
