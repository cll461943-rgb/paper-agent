"""Global Deadline / Budget — prevents any single stage from monopolizing time.

Core principle: each stage gets remaining time budget, not a fixed timeout.
Once the case deadline expires, the pipeline enters fallback mode.
"""

from __future__ import annotations

import logging
import time

LOGGER = logging.getLogger(__name__)


class Deadline:
    """A hierarchical deadline that tracks remaining time.

    Usage:
        deadline = Deadline(total_seconds=90)
        query_plan = understand_query(query, deadline=deadline.child(5))
        candidates = retrieve(query_plan, deadline=deadline.child(45))
        selections = select_evidence(candidates, deadline=deadline.child(20))
        result = synthesize(selections, deadline=deadline.child(10))

    Each child deadline shares the parent's absolute end time but caps
    the maximum time the child can spend. This prevents early stages
    from consuming all the time that later stages need.
    """

    def __init__(
        self,
        total_seconds: float,
        parent: "Deadline | None" = None,
        stage_name: str = "",
    ) -> None:
        self._start = time.perf_counter()
        self._total = total_seconds
        self._parent = parent
        self._stage_name = stage_name or "root"

        # If we have a parent, our end time is min(our total, parent's remaining)
        if parent is not None:
            parent_remaining = parent.remaining()
            self._end = self._start + min(total_seconds, parent_remaining)
        else:
            self._end = self._start + total_seconds

    def remaining(self) -> float:
        """Return remaining seconds until deadline."""
        return max(0.0, self._end - time.perf_counter())

    def expired(self) -> bool:
        """Check if this deadline has expired."""
        return time.perf_counter() >= self._end

    def child(self, seconds: float, stage_name: str = "") -> "Deadline":
        """Create a child deadline with allocated seconds.

        The child's actual end time is min(seconds, parent.remaining()),
        so if the parent is almost out of time, the child gets less.
        """
        return Deadline(
            total_seconds=seconds,
            parent=self,
            stage_name=stage_name,
        )

    def timeout_for(self, config_timeout: float | None) -> float:
        """Return the effective timeout for an API/LLM call.

        timeout = min(config_timeout, remaining())
        If config_timeout is None, uses remaining() directly.
        Minimum return is 1.0s to avoid zero-timeout issues.
        """
        rem = self.remaining()
        if config_timeout is not None:
            return max(1.0, min(config_timeout, rem))
        return max(1.0, rem)

    def elapsed(self) -> float:
        """Return elapsed seconds since this deadline started."""
        return time.perf_counter() - self._start

    @property
    def stage_name(self) -> str:
        return self._stage_name

    def __repr__(self) -> str:
        return (
            f"Deadline(stage={self._stage_name}, "
            f"remaining={self.remaining():.1f}s, "
            f"elapsed={self.elapsed():.1f}s)"
        )


# Default deadlines per the optimization document
# Increased for DeepSeek v4-flash reasoning model (needs 15-20s per call:
# reasoning_content consumes time before content is produced)
DEFAULT_CASE_DEADLINE = 360.0         # Total time per case (was 180)
DEFAULT_QUERY_UNDERSTANDING = 25.0    # Query understanding phase (was 8)
DEFAULT_RETRIEVAL = 70.0              # All retrieval rounds (was 45)
DEFAULT_EVIDENCE_SELECTION = 100.0     # Evidence selection (was 45)
DEFAULT_LISTWISE_RERANK = 90.0        # Increased for v4-pro reasoning model
DEFAULT_SYNTHESIS = 25.0              # Final synthesis (was 12)
DEFAULT_RESULT_REVIEW = 15.0          # Per-round result review (was 8)
DEFAULT_STRATEGY_OPTIMIZATION = 15.0  # Per-round strategy optimization (was 8)
