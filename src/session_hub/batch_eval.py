from __future__ import annotations

import concurrent.futures
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class BatchCase:
    dataset: str
    case_index: int
    query: str = ""
    gold: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchCaseResult:
    case: BatchCase
    status: str
    elapsed_seconds: float
    payload: Any = None
    error: str | None = None


def load_jsonl_cases(path: str | Path, limit: int | None = None) -> list[BatchCase]:
    dataset_path = Path(path)
    cases: list[BatchCase] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if limit is not None and len(cases) >= limit:
                break
            clean_line = line.strip()
            if not clean_line:
                continue
            row = json.loads(clean_line)
            cases.append(_case_from_mapping(dataset_path, row, line_index))
    return cases


def load_dataset_cases(path: str | Path, limit: int | None = None) -> list[BatchCase]:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_cases(dataset_path, limit=limit)
    if suffix in {".csv", ".tsv"}:
        return _load_delimited_cases(dataset_path, delimiter="\t" if suffix == ".tsv" else ",", limit=limit)
    if suffix == ".json":
        return _load_json_cases(dataset_path, limit=limit)
    return _load_auto_detected_cases(dataset_path, limit=limit)


def _load_auto_detected_cases(dataset_path: Path, limit: int | None) -> list[BatchCase]:
    sample = dataset_path.read_text(encoding="utf-8-sig").lstrip()
    if not sample:
        return []
    if sample[0] in "[{":
        return _load_json_cases(dataset_path, limit=limit)
    if "\t" in sample.splitlines()[0]:
        return _load_delimited_cases(dataset_path, delimiter="\t", limit=limit)
    if "," in sample.splitlines()[0]:
        return _load_delimited_cases(dataset_path, delimiter=",", limit=limit)
    return load_jsonl_cases(dataset_path, limit=limit)


def _load_json_cases(dataset_path: Path, limit: int | None) -> list[BatchCase]:
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        try:
            return load_jsonl_cases(dataset_path, limit=limit)
        except json.JSONDecodeError:
            raise ValueError(f"{dataset_path} is neither valid JSON nor JSONL: {exc}") from exc

    rows = _extract_case_rows(payload)
    return [_case_from_mapping(dataset_path, row, index) for index, row in enumerate(rows[:limit])]


def _load_delimited_cases(dataset_path: Path, *, delimiter: str, limit: int | None) -> list[BatchCase]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    return [_case_from_mapping(dataset_path, row, index) for index, row in enumerate(rows[:limit])]


def _extract_case_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("cases", "data", "examples", "items", "queries"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [payload]
    raise ValueError("dataset JSON must be an object, an array, or an object containing a cases/data/examples/items/queries array")


def _case_from_mapping(dataset_path: Path, row: dict[str, Any], fallback_index: int) -> BatchCase:
    case_index = int(row.get("case_index", row.get("caseIndex", row.get("index", fallback_index))))
    return BatchCase(
        dataset=str(dataset_path),
        case_index=case_index,
        query=str(row.get("query") or row.get("question") or row.get("prompt") or ""),
        gold=tuple(_normalise_gold_items(row)),
    )


def _normalise_gold_items(row: dict[str, Any]) -> list[str]:
    raw_gold = (
        row.get("gold")
        or row.get("gold_titles")
        or row.get("references")
        or row.get("answer")
        or row.get("answers")
        or row.get("expected")
        or []
    )
    if isinstance(raw_gold, str):
        stripped = raw_gold.strip()
        if stripped.startswith("["):
            try:
                raw_gold = json.loads(stripped)
            except json.JSONDecodeError:
                raw_gold = stripped
        if isinstance(raw_gold, str):
            separators = [";", "|", "\n"]
            parts = [raw_gold]
            for separator in separators:
                if separator in raw_gold:
                    parts = raw_gold.split(separator)
                    break
            return [part.strip() for part in parts if part.strip()]
    if isinstance(raw_gold, dict):
        raw_gold = [raw_gold]
    if not isinstance(raw_gold, list):
        return [str(raw_gold)] if raw_gold else []
    result: list[str] = []
    for item in raw_gold:
        if isinstance(item, dict):
            value = item.get("title") or item.get("name") or item.get("paper_title") or item.get("id")
            if value:
                result.append(str(value))
        elif item:
            result.append(str(item))
    return result


def run_batch_cases(
    cases: Iterable[BatchCase],
    run_case: Callable[[BatchCase], Any],
    *,
    per_case_timeout_seconds: float,
    max_workers: int = 4,
    should_stop: Callable[[], bool] | None = None,
    on_result: Callable[[BatchCaseResult], None] | None = None,
) -> list[BatchCaseResult]:
    case_list = list(cases)
    if per_case_timeout_seconds <= 0:
        raise ValueError("per_case_timeout_seconds must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    results: list[BatchCaseResult | None] = [None] * len(case_list)
    running: dict[concurrent.futures.Future[Any], tuple[int, BatchCase, float]] = {}
    next_index = 0

    def submit_next() -> None:
        nonlocal next_index
        while next_index < len(case_list) and len(running) < max_workers and not (should_stop and should_stop()):
            case = case_list[next_index]
            future = executor.submit(run_case, case)
            running[future] = (next_index, case, time.perf_counter())
            next_index += 1

    def record_result(index: int, result: BatchCaseResult) -> None:
        results[index] = result
        if on_result:
            on_result(result)

    submit_next()
    try:
        while running:
            if should_stop and should_stop():
                for future, (index, case, started) in list(running.items()):
                    future.cancel()
                    record_result(
                        index,
                        BatchCaseResult(
                            case=case,
                            status="cancelled",
                            elapsed_seconds=time.perf_counter() - started,
                            error="batch stopped by user",
                        ),
                    )
                    running.pop(future, None)
                for index in range(next_index, len(case_list)):
                    record_result(
                        index,
                        BatchCaseResult(
                            case=case_list[index],
                            status="cancelled",
                            elapsed_seconds=0.0,
                            error="batch stopped by user",
                        ),
                    )
                break

            done, _ = concurrent.futures.wait(
                running.keys(),
                timeout=min(0.02, per_case_timeout_seconds),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                index, case, started = running.pop(future)
                elapsed = time.perf_counter() - started
                try:
                    payload = future.result()
                    record_result(index, BatchCaseResult(case=case, status="succeeded", elapsed_seconds=elapsed, payload=payload))
                except Exception as exc:  # noqa: BLE001 - public runner should isolate all case failures.
                    record_result(index, BatchCaseResult(case=case, status="failed", elapsed_seconds=elapsed, error=str(exc)))

            now = time.perf_counter()
            timed_out = [
                future
                for future, (_, _, started) in running.items()
                if now - started >= per_case_timeout_seconds
            ]
            for future in timed_out:
                index, case, started = running.pop(future)
                future.cancel()
                record_result(
                    index,
                    BatchCaseResult(
                        case=case,
                        status="timed_out",
                        elapsed_seconds=time.perf_counter() - started,
                        error=f"case exceeded {per_case_timeout_seconds:g}s timeout",
                    ),
                )

            submit_next()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return [result for result in results if result is not None]
