from __future__ import annotations

import concurrent.futures
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
            case_index = int(row.get("case_index", row.get("caseIndex", line_index)))
            gold = row.get("gold") or row.get("gold_titles") or row.get("references") or []
            cases.append(
                BatchCase(
                    dataset=str(dataset_path),
                    case_index=case_index,
                    query=str(row.get("query") or row.get("question") or ""),
                    gold=tuple(str(item.get("title", item)) if isinstance(item, dict) else str(item) for item in gold),
                )
            )
    return cases


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
