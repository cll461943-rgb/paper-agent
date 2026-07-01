"""Source Health Manager — tracks provider health, applies cooldown on 429/timeout.

Core principle: external APIs can fail, but the Agent must not collapse.
This module provides proactive health gating so unhealthy providers are
skipped BEFORE making expensive HTTP calls, not just reactively after.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

LOGGER = logging.getLogger(__name__)


class ProviderState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COOLDOWN = "cooldown"
    BLOCKED = "blocked"


@dataclass
class HealthRecord:
    """Per-provider health record for the current case."""
    name: str
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    rate_limit_count: int = 0
    last_error: str = ""
    cooldown_until: float = 0.0  # epoch timestamp
    state: ProviderState = ProviderState.HEALTHY
    total_elapsed: float = 0.0
    total_items: int = 0
    # Configurable per-provider time budget (replaces hardcoded 60s)
    time_budget: float = 60.0


class SourceHealthManager:
    """Manages provider health across a single pipeline run (case).

    Strategy (per optimization doc):
    - HTTP 429 → immediate cooldown 120s, skip for rest of case
    - timeout → after 2 consecutive timeouts, skip for rest of case
    - 5xx → short cooldown 30s
    - items=0 + high elapsed → degrade priority (reduce time_budget)
    - success → decrease error count (recovery)
    """

    # Cooldown durations in seconds (reduced for time optimization)
    COOLDOWN_429 = 120.0  # long: provider blocked for rest of case, safety fallback bypasses
    COOLDOWN_5XX = 10.0   # was 30s — fail fast on server errors
    COOLDOWN_TIMEOUT = 10.0  # was 15s — shorter timeout cooldown
    MAX_TIMEOUTS_PER_CASE = 2
    MAX_RATE_LIMITS_PER_CASE = 2  # After 2x 429, block for the rest of the case

    # Per-provider 429 cooldown overrides (seconds)
    # S2 429s: 5s cooldown (was 15s) — short recovery, fail fast
    # OpenAlex 429s: persistent (per-IP rate limit), 120s = rest of case
    COOLDOWN_429_BY_PROVIDER = {
        "semantic_scholar": 5.0,   # was 15s — too long, wasted 75s+ on repeated 429s
        "openalex": 120.0,
    }

    # Per-provider max 429 count before blocking
    # S2: allow 3 retries (was 5) — block sooner to save time
    # OpenAlex: persistent, block after 2
    MAX_RATE_LIMITS_BY_PROVIDER = {
        "semantic_scholar": 3,  # was 5 — block sooner
        "openalex": 2,
    }

    # Per-provider time budget defaults (reduced for time optimization)
    DEFAULT_TIME_BUDGET = {
        "pasa_local": 10.0,     # Local, fast
        "openalex": 35.0,       # was 45s — primary remote source, but still time-limited
        "semantic_scholar": 20.0,  # was 30s — secondary, rate-limited, less time
        "arxiv": 15.0,          # was 20s
        "pubmed": 15.0,         # was 20s
        "faiss_vector": 30.0,   # Local vector search, model loading overhead
    }

    def __init__(self) -> None:
        self._records: dict[str, HealthRecord] = {}
        self._case_start: float = time.perf_counter()

    def reset(self) -> None:
        """Reset all health state for a new case."""
        self._records.clear()
        self._case_start = time.perf_counter()

    def _get_or_create(self, provider_name: str) -> HealthRecord:
        if provider_name not in self._records:
            budget = self.DEFAULT_TIME_BUDGET.get(provider_name, 45.0)
            self._records[provider_name] = HealthRecord(
                name=provider_name,
                time_budget=budget,
            )
        return self._records[provider_name]

    def before_call(self, provider_name: str) -> bool:
        """Check if a provider is healthy enough to call right now.

        Returns True if the call should proceed, False if the provider
        is in cooldown or blocked.
        """
        rec = self._get_or_create(provider_name)
        now = time.time()

        if rec.state == ProviderState.BLOCKED:
            return False

        if rec.state == ProviderState.COOLDOWN and now < rec.cooldown_until:
            return False

        # Cooldown expired → recover to degraded (not fully healthy)
        if rec.state == ProviderState.COOLDOWN and now >= rec.cooldown_until:
            rec.state = ProviderState.DEGRADED
            LOGGER.info(
                "Provider %s recovered from cooldown to degraded state",
                provider_name,
            )

        return True

    def record_success(
        self,
        provider_name: str,
        elapsed: float,
        items: int,
    ) -> None:
        """Record a successful provider call."""
        rec = self._get_or_create(provider_name)
        rec.success_count += 1
        rec.total_elapsed += elapsed
        rec.total_items += items

        # Recovery: decrease error count on success
        if rec.error_count > 0:
            rec.error_count = max(0, rec.error_count - 1)

        # If provider was degraded and now succeeding, restore to healthy
        if rec.state == ProviderState.DEGRADED and rec.success_count >= 2:
            rec.state = ProviderState.HEALTHY
            LOGGER.info("Provider %s restored to healthy state", provider_name)

        # Penalize items=0 with high elapsed (degrade priority)
        if items == 0 and elapsed > 10.0 and rec.success_count <= 1:
            # Reduce time budget for slow-but-empty provider
            rec.time_budget = max(15.0, rec.time_budget * 0.7)
            LOGGER.warning(
                "Provider %s returned 0 items in %.1fs, reducing time_budget to %.0fs",
                provider_name, elapsed, rec.time_budget,
            )

    def record_error(
        self,
        provider_name: str,
        error: str,
    ) -> None:
        """Record a provider error and apply cooldown strategy."""
        rec = self._get_or_create(provider_name)
        rec.error_count += 1
        rec.last_error = error

        lowered = error.lower()

        # HTTP 429 / rate limit
        if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
            rec.rate_limit_count += 1
            max_rl = self.MAX_RATE_LIMITS_BY_PROVIDER.get(provider_name, self.MAX_RATE_LIMITS_PER_CASE)
            if rec.rate_limit_count >= max_rl:
                rec.state = ProviderState.BLOCKED
                LOGGER.warning(
                    "Provider %s BLOCKED after %d rate-limit errors (429). "
                    "Skipping for rest of case.",
                    provider_name, rec.rate_limit_count,
                )
            else:
                cooldown = self.COOLDOWN_429_BY_PROVIDER.get(provider_name, self.COOLDOWN_429)
                rec.cooldown_until = time.time() + cooldown
                rec.state = ProviderState.COOLDOWN
                LOGGER.warning(
                    "Provider %s entering 429 cooldown (%.0fs). "
                    "Rate-limit count: %d/%d",
                    provider_name, cooldown,
                    rec.rate_limit_count, self.MAX_RATE_LIMITS_PER_CASE,
                )

        # Timeout
        elif "timeout" in lowered or "timed out" in lowered or "handshake" in lowered:
            rec.timeout_count += 1
            if rec.timeout_count >= self.MAX_TIMEOUTS_PER_CASE:
                rec.state = ProviderState.BLOCKED
                LOGGER.warning(
                    "Provider %s BLOCKED after %d consecutive timeouts. "
                    "Skipping for rest of case.",
                    provider_name, rec.timeout_count,
                )
            else:
                rec.cooldown_until = time.time() + self.COOLDOWN_TIMEOUT
                rec.state = ProviderState.COOLDOWN
                LOGGER.warning(
                    "Provider %s entering timeout cooldown (%.0fs). "
                    "Timeout count: %d/%d",
                    provider_name, self.COOLDOWN_TIMEOUT,
                    rec.timeout_count, self.MAX_TIMEOUTS_PER_CASE,
                )

        # 5xx server error
        elif any(code in lowered for code in ("500", "502", "503", "504")):
            rec.cooldown_until = time.time() + self.COOLDOWN_5XX
            rec.state = ProviderState.COOLDOWN
            LOGGER.warning(
                "Provider %s entering 5xx cooldown (%.0fs). Error: %s",
                provider_name, self.COOLDOWN_5XX, error[:100],
            )

        # Generic error
        else:
            if rec.error_count >= 5:
                rec.state = ProviderState.BLOCKED
                LOGGER.warning(
                    "Provider %s BLOCKED after %d generic errors. Last: %s",
                    provider_name, rec.error_count, error[:100],
                )

    def get_time_budget(self, provider_name: str) -> float:
        """Get the time budget for a provider (healthy providers get full budget)."""
        rec = self._get_or_create(provider_name)
        if rec.state == ProviderState.BLOCKED:
            return 0.0
        if rec.state == ProviderState.DEGRADED:
            return rec.time_budget * 0.5  # Degraded providers get half budget
        return rec.time_budget

    def get_state(self, provider_name: str) -> ProviderState:
        """Get the current state of a provider."""
        return self._get_or_create(provider_name).state

    def get_stats(self) -> dict[str, dict]:
        """Get health stats for all providers (for logging/reporting)."""
        stats = {}
        for name, rec in self._records.items():
            stats[name] = {
                "state": rec.state.value,
                "success": rec.success_count,
                "errors": rec.error_count,
                "timeouts": rec.timeout_count,
                "rate_limits": rec.rate_limit_count,
                "items": rec.total_items,
                "elapsed": round(rec.total_elapsed, 2),
                "time_budget": rec.time_budget,
            }
        return stats

    def is_blocked(self, provider_name: str) -> bool:
        """Check if a provider is permanently blocked for this case."""
        return self._get_or_create(provider_name).state == ProviderState.BLOCKED
