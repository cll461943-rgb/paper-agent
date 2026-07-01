from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from typing import Any

from scholar_agent.models.schemas import EvidenceItem, Paper, QueryPlan, SelectionResult

LOGGER = logging.getLogger(__name__)


def _safe_score(value: Any, default: float) -> float:
    """将 LLM 返回的分数值安全转为 0-1 float，失败时返回 default。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _fallback_numeric_scores(relevance: str) -> tuple[float, float, float]:
    """根据 relevance_level 返回默认的 (relevance_score, constraint_score, evidence_score)。"""
    if relevance == "high":
        return 0.90, 0.75, 0.70
    if relevance == "medium":
        return 0.55, 0.45, 0.45
    if relevance == "low":
        return 0.20, 0.20, 0.15
    return 0.0, 0.0, 0.0


def _create_fallback_selection(paper: Paper, plan: QueryPlan) -> SelectionResult:
    """LLM 失败时的启发式兜底，基于关键词+质量信号加权评分。"""
    abstract_text = (paper.abstract or "").lower()
    title_text = paper.title.lower()
    full_text = title_text + " " + abstract_text

    matched = []
    missing = []
    keyword_score = 0.0

    # 关键词匹配 (0-0.4)
    for method in plan.methods:
        if method.lower() in full_text:
            matched.append(method)
            keyword_score += 0.2
        else:
            missing.append(method)

    for dataset in plan.datasets:
        if dataset.lower() in full_text:
            matched.append(dataset)
            keyword_score += 0.2
        else:
            missing.append(dataset)

    keyword_score = min(keyword_score, 0.4)

    # 质量信号加权 (0-0.3)
    quality_score = 0.0

    # 引用数：高引用论文更可能有价值
    if paper.citation_count and paper.citation_count > 200:
        quality_score += 0.15
    elif paper.citation_count and paper.citation_count > 50:
        quality_score += 0.08
    elif paper.citation_count and paper.citation_count > 10:
        quality_score += 0.03

    # 发表时间：最近 3 年的论文得分更高
    if paper.year and paper.year >= 2022:
        quality_score += 0.15
    elif paper.year and paper.year >= 2020:
        quality_score += 0.08

    quality_score = min(quality_score, 0.3)

    # 总分 = 关键词 + 质量
    total_score = keyword_score + quality_score

    # 当前计划没有约束时，给中等分数
    if not plan.methods and not plan.datasets:
        total_score = 0.35

    # 转换为相关性等级
    if total_score >= 0.5:
        relevance_level = "high"
        confidence = min(0.8, total_score)
    elif total_score >= 0.3:
        relevance_level = "medium"
        confidence = min(0.7, total_score)
    else:
        relevance_level = "low"
        confidence = min(0.5, total_score)

    fb_rel, fb_constraint, fb_evidence = _fallback_numeric_scores(relevance_level)

    return SelectionResult(
        paper_id=paper.paper_id,
        relevance_level=relevance_level,
        matched_constraints=matched,
        missing_constraints=missing,
        evidence=[],
        reason=f"Heuristic fallback: keyword={keyword_score:.2f}, quality={quality_score:.2f}.",
        confidence=confidence,
        is_validated=True,
        relevance_score=fb_rel,
        constraint_score=fb_constraint,
        evidence_score=fb_evidence,
        uncertainty=[],
    )


def select_and_extract_evidence(
    papers: list[Paper],
    plan: QueryPlan,
    llm_client: object | None,
) -> list[SelectionResult]:
    """Evidence Selector Agent - 对候选论文进行相关性判断。

    策略：小批次评分 + 激进 fallback
    - 小批次 (3篇/批) 提高单个 LLM 调用的稳定性
    - LLM 失败时完全信任改进的启发式 fallback
    - 避免一个 LLM 失败影响全量论文的"全 OR 全无"局面
    """
    results: list[SelectionResult] = []
    if not papers:
        return results

    if llm_client is None:
        return [_create_fallback_selection(paper, plan) for paper in papers]

    system_prompt = (
        "You are a strict scholarly paper Evidence Selector Agent. "
        "Your task is to judge candidate papers for relevance against a scholarly QueryPlan contract. "
        "For each paper, classify it into one of: 'high' (fully relevant and satisfies constraints), "
        "'medium' (partially relevant), 'low' (hardly relevant), 'irrelevant' (completely unrelated). "
        "You must provide explicit text spans from the paper's title or abstract as evidence. "
        "\n"
        "STRICT RELEVANCE RULES:\n"
        "1. You can ONLY judge based on title, abstract, year, venue, and metadata provided. "
        "Do NOT speculate about the full text content.\n"
        "2. If the core constraints (methods, datasets, entities) have no explicit evidence in title/abstract, "
        "the paper CANNOT be classified as 'high'.\n"
        "3. If the paper is only topically related but the specific method/dataset/task constraints are missing, "
        "classify it as 'medium' at most.\n"
        "4. If the evidence list is empty, the paper CANNOT be 'high'.\n"
        "5. If the title/abstract is only vaguely in the same research domain as the query but lacks specific "
        "constraint matches, classify it as 'low'.\n"
        "6. Contrastive/Alternative innovations: If a paper addresses the exact core research problem but "
        "proposes an alternative method, it is still relevant ('high' or 'medium').\n"
        "7. Semantic Equivalents: Recognize semantic synonyms, not just word matching.\n"
        "8. Do NOT give 'high' just because the retrieval path contains 'title_exact' or 'title_like'. "
        "Judge by evidence and constraint coverage, not retrieval path.\n"
        "\n"
        "CRITICAL OUTPUT REQUIREMENTS:\n"
        "- Return VALID JSON ONLY. Never include markdown, code fences, or explanations.\n"
        "- Strictly follow this structure: {\"selections\": [{...}, ...]}\n"
        "- ALWAYS close all brackets and quotes properly. NO trailing commas.\n"
        "- Each object in selections must have ALL fields: paper_id, relevance_level, relevance_score, "
        "constraint_score, evidence_score, matched_constraints, missing_constraints, evidence, reason, "
        "confidence, uncertainty.\n"
        "- relevance_level must be exactly one of: \"high\", \"medium\", \"low\", \"irrelevant\".\n"
        "- relevance_score, constraint_score, evidence_score, confidence must be numbers between 0.0 and 1.0.\n"
        "- uncertainty is a list of strings describing what is uncertain about the classification.\n"
        "- Scoring guidelines: high→relevance_score>0.8, medium→0.4-0.8, low→0.1-0.4, irrelevant→<0.1."
    )

    # 改用小批次（2-3篇）评分，提高稳定性和错误隔离
    batch_size = 3
    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]

    def process_batch_simple(batch: list[Paper]) -> list[dict[str, Any]]:
        """简单批处理，无重试。失败则返回空列表，由 fallback 接管。"""
        paper_payload = [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
                "year": p.year,
                "venue": p.venue,
                "citation_count": p.citation_count,
                "retrieval_path": p.retrieval_path,
            }
            for p in batch
        ]

        user_prompt = (
            "Output format schema (JSON only, no markdown):\n"
            "{\n"
            "  \"selections\": [\n"
            "    {\"paper_id\": \"string\", \"relevance_level\": \"high\"|\"medium\"|\"low\"|\"irrelevant\", "
            "\"relevance_score\": 0.0-1.0, \"constraint_score\": 0.0-1.0, \"evidence_score\": 0.0-1.0, "
            "\"matched_constraints\": [...], \"missing_constraints\": [...], \"evidence\": [...], "
            "\"reason\": \"string\", \"confidence\": 0.0-1.0, \"uncertainty\": [\"...\"]}\n"
            "  ]\n"
            "}\n\n"
            f"QueryPlan: {plan.model_dump_json()}\n"
            f"Papers: {json.dumps(paper_payload, ensure_ascii=False)}"
        )

        try:
            response = getattr(llm_client, "complete_json", lambda *_: None)(
                system_prompt, user_prompt, model_type="pro"
            )
            if response is None:
                return []

            payload = response.get("selections") if isinstance(response, dict) else response
            if isinstance(payload, list):
                return payload
        except Exception as exc:
            LOGGER.debug(f"Batch LLM call failed: {type(exc).__name__}")

        return []

    all_payloads = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_batch_simple, b) for b in batches]
        for future in concurrent.futures.as_completed(futures):
            try:
                payload = future.result()
                all_payloads.extend(payload)
            except Exception as exc:
                LOGGER.debug(f"Thread error: {exc}")

    # 增量回退：为每篇论文单独处理，失败才用 fallback
    for paper in papers:
        fallback = _create_fallback_selection(paper, plan)
        selected = fallback

        for item in all_payloads:
            if isinstance(item, dict) and item.get("paper_id") == paper.paper_id:
                try:
                    evidence_items = []
                    for ev in item.get("evidence", []):
                        if isinstance(ev, dict) and "field" in ev and "text" in ev:
                            evidence_items.append(EvidenceItem(field=ev["field"], text=ev["text"]))

                    relevance = item.get("relevance_level")
                    if relevance not in {"high", "medium", "low", "irrelevant"}:
                        relevance = "low"

                    default_rel, default_constraint, default_evidence = _fallback_numeric_scores(relevance)

                    selected = SelectionResult(
                        paper_id=paper.paper_id,
                        relevance_level=relevance,
                        matched_constraints=item.get("matched_constraints", []),
                        missing_constraints=item.get("missing_constraints", []),
                        evidence=evidence_items,
                        reason=item.get("reason", ""),
                        confidence=_safe_score(item.get("confidence"), 0.6),
                        is_validated=True,
                        relevance_score=_safe_score(item.get("relevance_score"), default_rel),
                        constraint_score=_safe_score(item.get("constraint_score"), default_constraint),
                        evidence_score=_safe_score(item.get("evidence_score"), default_evidence),
                        uncertainty=item.get("uncertainty", []),
                    )
                except Exception as exc:
                    LOGGER.debug(f"Parse error for {paper.paper_id}, using fallback: {exc}")
                break

        results.append(selected)

    return results
