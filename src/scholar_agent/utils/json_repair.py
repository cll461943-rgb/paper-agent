"""Local JSON repair — fixes common LLM JSON output issues without LLM retries.

Core principle: never use LLM to repair LLM output. Use deterministic
local repair instead. If repair fails, return None and use heuristic fallback.
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def extract_json_block(text: str) -> str | None:
    """Extract the first JSON object/array from text that may contain
    markdown code fences, prose, or other non-JSON content.
    """
    if not text:
        return None

    # Strip markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        text = fence_match.group(1)

    # Try to find the first { or [ and its matching close
    text = text.strip()
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        # Find the matching end by counting brackets
        depth = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
        # If we exhausted the string without matching, try the next pair
    return None


def repair_json(text: str) -> str | None:
    """Apply deterministic repairs to common JSON issues.

    Fixes:
    - Trailing commas before } or ]
    - Single quotes → double quotes
    - Unescaped newlines inside strings
    - Missing closing brackets
    - Python-style booleans (True/False/None → true/false/null)
    """
    if not text:
        return None

    repaired = text

    # Fix Python-style booleans and None
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)

    # Fix single-quoted strings → double-quoted
    # (only when they look like JSON values, not inside double-quoted strings)
    def _fix_single_quotes(m: re.Match) -> str:
        content = m.group(1)
        # Escape any double quotes inside the content
        content = content.replace('"', '\\"')
        return f'"{content}"'

    repaired = re.sub(r"'([^']*)'", _fix_single_quotes, repaired)

    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    # Fix unescaped newlines inside string values
    # (JSON strings cannot contain literal newlines)
    def _fix_string_newlines(m: re.Match) -> str:
        prefix = m.group(1)  # "key": "value start
        suffix = m.group(3)  # rest
        # Replace newlines with \\n
        middle = m.group(2).replace("\n", "\\n").replace("\r", "\\r")
        return f'{prefix}{middle}{suffix}'

    # This is tricky — only fix newlines inside quoted strings
    # Simple approach: replace all literal newlines inside quotes
    result_chars = []
    in_string = False
    escape = False
    for c in repaired:
        if escape:
            result_chars.append(c)
            escape = False
            continue
        if c == "\\":
            result_chars.append(c)
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            result_chars.append(c)
            continue
        if in_string and c == "\n":
            result_chars.append("\\n")
            continue
        if in_string and c == "\r":
            continue  # Skip carriage returns
        result_chars.append(c)
    repaired = "".join(result_chars)

    # Try to fix missing closing brackets
    opens_braces = repaired.count("{") - repaired.count("}")
    opens_brackets = repaired.count("[") - repaired.count("]")
    if opens_braces > 0:
        repaired += "}" * opens_braces
    if opens_brackets > 0:
        repaired += "]" * opens_brackets

    return repaired


def safe_json_loads(text: str | None) -> Any | None:
    """Try to parse JSON from text, with local repair fallback.

    Returns parsed object on success, None on failure.
    Never raises — always returns None or the parsed object.
    """
    if not text:
        return None

    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Extract JSON block from surrounding text
    extracted = extract_json_block(text)
    if extracted:
        try:
            return json.loads(extracted)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try repair
        repaired = repair_json(extracted)
        if repaired:
            try:
                return json.loads(repaired)
            except (json.JSONDecodeError, TypeError):
                LOGGER.debug("JSON repair failed for: %s", repaired[:200])

    # Last resort: try repairing the original text
    repaired = repair_json(text)
    if repaired:
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            pass

    LOGGER.warning("safe_json_loads: all parsing attempts failed for text: %s", (text or "")[:200])
    return None
