from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import json
import re
from pathlib import Path
from typing import Any, Iterable


def keep_letters(text: str | None) -> str:
    """PaSa title normalizer: lowercase letters only, no spaces or punctuation."""
    return "".join(char for char in (text or "") if char.isalpha()).lower()


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"^arxiv:", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"v\d+$", "", text)
    text = text.strip()
    return text or None


def _paper_arxiv_id(paper: Any) -> str | None:
    direct = normalize_arxiv_id(getattr(paper, "arxiv_id", None))
    if direct:
        return direct
    paper_id = normalize_arxiv_id(getattr(paper, "paper_id", None))
    if paper_id:
        return paper_id
    metadata = getattr(paper, "metadata", {}) or {}
    external_ids = metadata.get("external_ids") or metadata.get("externalIds") or {}
    if isinstance(external_ids, dict):
        return normalize_arxiv_id(external_ids.get("ArXiv") or external_ids.get("arxiv"))
    return None


def gold_match_keys(item: dict[str, Any] | str) -> set[str]:
    if isinstance(item, str):
        return {key for key in {keep_letters(item), normalize_arxiv_id(item)} if key}
    title = item.get("title") or item.get("paper_title")
    keys = {
        keep_letters(title),
        normalize_arxiv_id(item.get("arxiv_id") or item.get("paper_id")),
    }
    return {key for key in keys if key}


def gold_match_key(item: dict[str, Any] | str) -> str | None:
    if isinstance(item, str):
        return keep_letters(item) or normalize_arxiv_id(item)
    title_key = keep_letters(item.get("title") or item.get("paper_title"))
    return title_key or normalize_arxiv_id(item.get("arxiv_id") or item.get("paper_id"))


def paper_match_keys(paper: Any) -> set[str]:
    keys = {keep_letters(getattr(paper, "title", None)), _paper_arxiv_id(paper)}
    return {key for key in keys if key}


def _unique_gold_key_sets(gold_items: Iterable[dict[str, Any] | str]) -> list[set[str]]:
    seen_keys: set[str] = set()
    key_sets: list[set[str]] = []
    for item in gold_items:
        keys = gold_match_keys(item)
        if keys and keys.isdisjoint(seen_keys):
            seen_keys.update(keys)
            key_sets.append(keys)
    return key_sets


def _unique_prediction_key_sets(papers: Iterable[Any]) -> list[set[str]]:
    seen_keys: set[str] = set()
    key_sets: list[set[str]] = []
    for paper in papers:
        keys = paper_match_keys(paper)
        if keys and keys.isdisjoint(seen_keys):
            seen_keys.update(keys)
            key_sets.append(keys)
    return key_sets


def score_pasa(papers: list[Any], gold_items: list[dict[str, Any] | str]) -> dict[str, float | int]:
    gold_key_sets = _unique_gold_key_sets(gold_items)
    prediction_key_sets = _unique_prediction_key_sets(papers)
    if not gold_key_sets:
        return {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "predicted_count": len(prediction_key_sets),
            "gold_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    matched_gold_indexes: set[int] = set()
    false_positive = 0

    for prediction_keys in prediction_key_sets:
        matching_indexes = [
            index
            for index, gold_keys in enumerate(gold_key_sets)
            if not prediction_keys.isdisjoint(gold_keys)
        ]
        if not matching_indexes:
            false_positive += 1
        else:
            unmatched_index = next((index for index in matching_indexes if index not in matched_gold_indexes), None)
            if unmatched_index is not None:
                matched_gold_indexes.add(unmatched_index)

    tp = len(matched_gold_indexes)
    fp = false_positive
    fn = len(gold_key_sets) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_count": tp + fp,
        "gold_count": len(gold_key_sets),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


@dataclass
class PasaCase:
    question: str
    answer: list[dict[str, Any] | str]
    published_time: date | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    iso_text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_text).date()
    except ValueError:
        pass
    match = re.match(r"^(\d{4})(\d{2})(\d{2})$", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.match(r"^(\d{4})[-/](\d{1,2})(?:\D|$)", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), 28)
        except ValueError:
            return None
    match = re.match(r"^(\d{4})(?:\D|$)", text)
    if match:
        return date(int(match.group(1)), 12, 31)
    return None


def cutoff_from_source_meta(source_meta: dict[str, Any] | None, days: int = 7) -> date | None:
    if not source_meta:
        return None
    published = _parse_date(source_meta.get("published_time"))
    return published - timedelta(days=days) if published else None


def paper_published_date(paper: Any) -> date | None:
    metadata = getattr(paper, "metadata", {}) or {}
    for key in ("published_time", "publication_date", "published_date", "published", "updated"):
        parsed = _parse_date(metadata.get(key))
        if parsed:
            return parsed
    year = getattr(paper, "year", None) or metadata.get("year") or metadata.get("publication_year")
    if year:
        try:
            return date(int(year), 12, 31)
        except (TypeError, ValueError):
            return None
    return None


def filter_by_cutoff(papers: list[Any], cutoff: date | None) -> list[Any]:
    if cutoff is None:
        return papers
    filtered: list[Any] = []
    for paper in papers:
        published = paper_published_date(paper)
        if published is None or published <= cutoff:
            filtered.append(paper)
    return filtered


def gold_coverage(papers: list[Any], gold_items: list[dict[str, Any] | str]) -> list[dict[str, Any]]:
    prediction_key_sets = [paper_match_keys(paper) for paper in papers]
    report: list[dict[str, Any]] = []
    for item in gold_items:
        keys = gold_match_keys(item)
        matched_index = next(
            (
                index
                for index, prediction_keys in enumerate(prediction_key_sets)
                if keys and not keys.isdisjoint(prediction_keys)
            ),
            None,
        )
        if isinstance(item, str):
            label = item
        else:
            label = item.get("title") or item.get("paper_title") or item.get("arxiv_id") or item.get("paper_id") or ""
        report.append(
            {
                "gold": label,
                "match_key": gold_match_key(item),
                "matched": matched_index is not None,
                "candidate_rank": matched_index + 1 if matched_index is not None else None,
            }
        )
    return report


def load_pasa_jsonl(path: str | Path, limit: int | None = None) -> list[PasaCase]:
    cases: list[PasaCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            answer = raw.get("answer") or raw.get("answers") or []
            if isinstance(answer, str):
                answer = [answer]
            elif not isinstance(answer, list):
                answer = [answer]
            source_meta = raw.get("source_meta") or {}
            cases.append(
                PasaCase(
                    question=raw.get("question") or raw.get("query") or "",
                    answer=answer,
                    published_time=_parse_date(source_meta.get("published_time")),
                    raw=raw,
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def macro_average(scores: list[dict[str, float | int]]) -> dict[str, float]:
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": round(sum(float(item["precision"]) for item in scores) / len(scores), 4),
        "recall": round(sum(float(item["recall"]) for item in scores) / len(scores), 4),
        "f1": round(sum(float(item["f1"]) for item in scores) / len(scores), 4),
    }


def recall_at_k(papers: list[Any], gold_items: list[dict[str, Any] | str], k: int) -> float:
    return float(score_pasa(papers[:k], gold_items)["recall"])
