from __future__ import annotations

import json
import time

from session_hub.batch_eval import BatchCase, load_dataset_cases, load_jsonl_cases, run_batch_cases


def test_load_jsonl_cases_uses_line_number_as_case_index(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"query": "first query", "gold": ["paper a"]}),
                json.dumps({"case_index": 12, "query": "second query", "gold": ["paper b"]}),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_jsonl_cases(dataset)

    assert [case.case_index for case in cases] == [0, 12]
    assert cases[0].dataset == str(dataset)
    assert cases[0].query == "first query"


def test_load_dataset_cases_supports_json_case_wrapper(tmp_path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {"name": "case-a", "query": "wrapped query", "gold": ["paper a"], "k": 20},
                    {"name": "case-b", "question": "second wrapped query", "references": [{"title": "paper b"}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_dataset_cases(dataset)

    assert [case.case_index for case in cases] == [0, 1]
    assert [case.query for case in cases] == ["wrapped query", "second wrapped query"]
    assert cases[1].gold == ("paper b",)


def test_load_dataset_cases_supports_csv_and_tsv(tmp_path):
    csv_dataset = tmp_path / "cases.csv"
    csv_dataset.write_text(
        "case_index,query,gold\n7,csv query,\"paper a; paper b\"\n",
        encoding="utf-8",
    )
    tsv_dataset = tmp_path / "cases.tsv"
    tsv_dataset.write_text(
        "question\tanswer\n"
        "tsv query\tpaper c|paper d\n",
        encoding="utf-8",
    )

    csv_cases = load_dataset_cases(csv_dataset)
    tsv_cases = load_dataset_cases(tsv_dataset)

    assert csv_cases[0].case_index == 7
    assert csv_cases[0].query == "csv query"
    assert csv_cases[0].gold == ("paper a", "paper b")
    assert tsv_cases[0].case_index == 0
    assert tsv_cases[0].query == "tsv query"
    assert tsv_cases[0].gold == ("paper c", "paper d")


def test_run_batch_cases_times_out_one_case_and_continues():
    cases = [
        BatchCase(dataset="demo.jsonl", case_index=0, query="slow"),
        BatchCase(dataset="demo.jsonl", case_index=1, query="fast"),
    ]

    def run_case(case: BatchCase):
        if case.case_index == 0:
            time.sleep(0.4)
        else:
            time.sleep(0.01)
        return {"case_index": case.case_index, "f1": 1.0}

    started = time.perf_counter()
    results = run_batch_cases(cases, run_case, per_case_timeout_seconds=0.05, max_workers=2)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.3
    assert [result.status for result in results] == ["timed_out", "succeeded"]
    assert results[0].error == "case exceeded 0.05s timeout"
    assert results[1].payload == {"case_index": 1, "f1": 1.0}
