from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from threading import Lock
from typing import Any


class HttpClient:
    """Small stdlib HTTP client with timeout, retry, backoff, and request pacing."""

    def __init__(
        self,
        timeout_seconds: int = 20,
        retry_times: int = 2,
        min_interval_seconds: float = 0.0,
        backoff_base_seconds: float = 1.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.retry_times = retry_times
        self.min_interval_seconds = min_interval_seconds
        self.backoff_base_seconds = backoff_base_seconds
        self._lock = Lock()
        self._last_request_at = 0.0

    def _pace(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.perf_counter()
            delta = now - self._last_request_at
            if delta < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - delta)
            self._last_request_at = time.perf_counter()

    def _backoff_sleep(self, attempt: int, status_code: int | None = None) -> None:
        base = self.backoff_base_seconds
        if status_code == 429:
            base *= 2.0
        time.sleep(min(base * (2**attempt), 30.0))

    def _request(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        query_string = urllib.parse.urlencode(params or {}, doseq=True)
        final_url = f"{url}?{query_string}" if query_string else url
        request = urllib.request.Request(final_url, headers=headers or {})

        last_error: Exception | None = None
        attempts = self.retry_times + 1
        for attempt in range(attempts):
            try:
                self._pace()
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                last_error = exc
                if attempt >= self.retry_times:
                    break
                if exc.code in {429, 500, 502, 503, 504}:
                    self._backoff_sleep(attempt, status_code=exc.code)
                    continue
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                if attempt >= self.retry_times:
                    break
                self._backoff_sleep(attempt)

        if last_error is None:
            raise RuntimeError("http request failed without explicit error")
        raise last_error

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return json.loads(self._request(url, params=params, headers=headers))

    def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        return self._request(url, params=params, headers=headers)
