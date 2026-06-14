from __future__ import annotations

import logging
from typing import Any

from scholar_agent.models.schemas import EvidenceItem, Paper, QueryPlan, SelectionResult

LOGGER = logging.getLogger(__name__)


def _create_fallback_selection(paper: Paper, plan: QueryPlan) -> SelectionResult:
    """如果 LLM 挂了，给出一个无证据的中低相关性兜底。"""
    # 简单的字符串匹配判定
    abstract_text = (paper.abstract or "").lower()
    title_text = paper.title.lower()
    full_text = title_text + " " + abstract_text

    matched = []
    missing = []
    
    # 简单匹配方法
    for method in plan.methods:
        if method.lower() in full_text:
            matched.append(method)
        else:
            missing.append(method)

    # 简单匹配数据集
    for dataset in plan.datasets:
        if dataset.lower() in full_text:
            matched.append(dataset)
        else:
            missing.append(dataset)

    # 简单评定相关度
    relevance_level: Any = "low"
    if matched:
        relevance_level = "medium"
    if not plan.methods and not plan.datasets:
        relevance_level = "medium"

    return SelectionResult(
        paper_id=paper.paper_id,
        relevance_level=relevance_level,
        matched_constraints=matched,
        missing_constraints=missing,
        evidence=[],
        reason="Heuristic fallback selection.",
        confidence=0.5,
        is_validated=True
    )


def select_and_extract_evidence(
    papers: list[Paper],
    plan: QueryPlan,
    llm_client: object | None,
) -> list[SelectionResult]:
    """Evidence Selector Agent
    对候选论文进行细粒度相关性判断，区分 high, medium, low, irrelevant。
    并抽取出论文中匹配意图的具体文本证据片（Evidence Card）。
    """
    results: list[SelectionResult] = []
    if not papers:
        return results

    if llm_client is None:
        return [_create_fallback_selection(paper, plan) for paper in papers]

    system_prompt = (
        "You are a scholarly paper Evidence Selector Agent. "
        "Your task is to judge candidate papers for relevance against a scholarly QueryPlan contract. "
        "For each paper, classify it into one of: 'high' (fully relevant and satisfies constraints), "
        "'medium' (partially relevant), 'low' (hardly relevant), 'irrelevant' (completely unrelated). "
        "Provide explicit text spans from the paper's title or abstract as evidence. "
        "Return a JSON object with a single field 'selections' containing an array of SelectionResult objects. "
        "Return JSON only."
    )

    batch_size = 5  # 控制批处理大小以提升 DeepSeek-flash 细粒度判断精度

    for start in range(0, len(papers), batch_size):
        batch = papers[start : start + batch_size]
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
            "Output format schema (JSON):\n"
            "{\n"
            "  \"selections\": [\n"
            "    {\n"
            "      \"paper_id\": \"xxx\",\n"
            "      \"relevance_level\": \"high\"/\"medium\"/\"low\"/\"irrelevant\",\n"
            "      \"matched_constraints\": [\"constraint_name\", ...],\n"
            "      \"missing_constraints\": [\"constraint_name\", ...],\n"
            "      \"evidence\": [\n"
            "        {\"field\": \"title\"/\"abstract\", \"text\": \"exact sentence from text\"}\n"
            "      ],\n"
            "      \"reason\": \"explanation matching constraints\",\n"
            "      \"confidence\": 0.95\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"QueryPlan Contract: {plan.model_dump_json()}\n"
            f"Candidate Papers to judge: {paper_payload}"
        )

        # 细粒度相关性判断，使用 pro 模型保证效果
        response = getattr(llm_client, "complete_json", lambda *_: None)(system_prompt, user_prompt, model_type="pro")
        payload = response.get("selections") if isinstance(response, dict) else response

        for paper in batch:
            fallback = _create_fallback_selection(paper, plan)
            selected = fallback
            if isinstance(payload, list):
                for item in payload:
                    if item.get("paper_id") == paper.paper_id:
                        try:
                            # 细致解析字段并转换
                            evidence_items = []
                            for ev in item.get("evidence", []):
                                if isinstance(ev, dict) and "field" in ev and "text" in ev:
                                    evidence_items.append(EvidenceItem(field=ev["field"], text=ev["text"]))
                            
                            relevance = item.get("relevance_level")
                            if relevance not in {"high", "medium", "low", "irrelevant"}:
                                relevance = "low"

                            selected = SelectionResult(
                                paper_id=paper.paper_id,
                                relevance_level=relevance,
                                matched_constraints=item.get("matched_constraints", []),
                                missing_constraints=item.get("missing_constraints", []),
                                evidence=evidence_items,
                                reason=item.get("reason", "Parsed from LLM"),
                                confidence=float(item.get("confidence", 0.9)),
                                is_validated=True
                            )
                        except Exception as exc:
                            LOGGER.warning("Error parsing LLM selection result for paper %s: %s", paper.paper_id, exc)
                            selected = fallback
                        break
            results.append(selected)

    return results
