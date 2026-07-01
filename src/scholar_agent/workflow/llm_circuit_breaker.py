"""LLM Circuit Breaker — disables LLM calls after repeated timeouts.

Core principle: if the LLM keeps timing out, stop trying.
After MAX_TIMEOUTS LLM timeouts, the circuit trips and all subsequent
LLM calls in this case are skipped (heuristic fallback used instead).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


class LLMCircuitBreaker:
    """Tracks LLM call outcomes and trips after MAX_TIMEOUTS timeouts.

    Once tripped, can_call() returns False for the rest of the case.
    The pipeline should check can_call() before every LLM invocation
    and fall back to heuristic mode when the circuit is open.
    """

    def __init__(self, max_timeouts: int = 2, max_errors: int = 5) -> None:
        self._timeout_count = 0
        self._error_count = 0
        self._success_count = 0
        self._max_timeouts = max_timeouts
        self._max_errors = max_errors
        self._tripped = False
        self._trip_reason = ""
        self._last_call_time: float = 0.0

    def record_timeout(self) -> None:
        """Record an LLM timeout. Trips circuit after max_timeouts."""
        self._timeout_count += 1
        self._last_call_time = time.perf_counter()
        if self._timeout_count >= self._max_timeouts:
            self._tripped = True
            self._trip_reason = f"timeouts ({self._timeout_count})"
            LOGGER.warning(
                "LLM circuit breaker TRIPPED after %d timeouts. "
                "All subsequent LLM calls will use heuristic fallback.",
                self._timeout_count,
            )

    def record_error(self, error: str = "") -> None:
        """Record a non-timeout LLM error. Trips after max_errors."""
        self._error_count += 1
        self._last_call_time = time.perf_counter()
        lowered = (error or "").lower()
        is_timeout = any(k in lowered for k in ("timeout", "timed out", "read time"))
        if is_timeout:
            self.record_timeout()
            return
        if self._error_count >= self._max_errors:
            self._tripped = True
            self._trip_reason = f"errors ({self._error_count})"
            LOGGER.warning(
                "LLM circuit breaker TRIPPED after %d errors. "
                "All subsequent LLM calls will use heuristic fallback. Last: %s",
                self._error_count, error[:100],
            )

    def record_success(self) -> None:
        """Record a successful LLM call. Resets timeout/error counters."""
        self._success_count += 1
        self._last_call_time = time.perf_counter()
        # Partial recovery: reset timeout count on success
        if self._timeout_count > 0:
            self._timeout_count = max(0, self._timeout_count - 1)

    @property
    def is_tripped(self) -> bool:
        """True if the circuit breaker has tripped — LLM calls should be skipped."""
        return self._tripped

    def can_call(self) -> bool:
        """Check if LLM calls are allowed (circuit not tripped)."""
        return not self._tripped

    def get_stats(self) -> dict:
        """Return current circuit breaker state for logging."""
        return {
            "tripped": self._tripped,
            "reason": self._trip_reason,
            "timeouts": self._timeout_count,
            "errors": self._error_count,
            "successes": self._success_count,
        }

    def reset(self) -> None:
        """Reset the circuit breaker for a new case."""
        self._timeout_count = 0
        self._error_count = 0
        self._success_count = 0
        self._tripped = False
        self._trip_reason = ""
