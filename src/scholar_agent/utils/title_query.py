from __future__ import annotations

import re


_SEARCH_REQUEST_PREFIXES = (
    "are there",
    "can you",
    "do you",
    "find papers",
    "give me",
    "i am looking",
    "list all",
    "papers about",
    "papers on",
    "papers that",
    "provide",
    "research papers",
    "show me",
    "studies about",
    "work on",
)

_QUESTION_FRAGMENT_PREFIXES = (
    "how can",
    "how could",
    "how should",
    "how do",
    "how does",
    "how did",
    "how is",
    "how are",
    "what improvements",
    "what improvement",
    "what breakthrough",
    "what advancements",
    "what are",
    "what is",
    "what can",
    "what do",
    "what does",
    "which papers",
    "why do",
    "why does",
    "when do",
    "where can",
)

_GENERIC_SEARCH_FRAGMENTS = (
    "research papers from the past",
    "from the past five years",
    "past five years on",
    "latest papers",
    "recent papers",
)


def looks_like_paper_title(text: str) -> bool:
    """Return True only for strings plausible enough for title-like search."""
    raw = re.sub(r"\s+", " ", text or "").strip()
    if not raw:
        return False

    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", raw)
    if len(words) < 3 or len(words) > 24:
        return False

    lowered = " ".join(words).lower()
    if any(lowered.startswith(prefix) for prefix in _SEARCH_REQUEST_PREFIXES):
        return False
    if any(lowered.startswith(prefix) for prefix in _QUESTION_FRAGMENT_PREFIXES):
        return False
    if any(fragment in lowered for fragment in _GENERIC_SEARCH_FRAGMENTS):
        return False

    titleish_words = sum(1 for word in words if word[:1].isupper() or word.isupper())
    has_acronym = any(word.isupper() and len(word) >= 2 for word in words)
    has_title_marker = ":" in raw or titleish_words >= 3 or has_acronym
    if len(words) < 5 and not has_title_marker:
        return False

    return True
