from __future__ import annotations

import concurrent.futures
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

    batch_size = 7  # 平衡：20 篇论文需 3 个并发（vs. 5 需 4 个、10 需 2 个），提升稳定性
    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]

    def process_batch(batch: list[Paper]) -> list[dict[str, Any]]:
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

        try:
            # 不使用局部超时，依赖整体 budget timeout 机制
            response = getattr(llm_client, "complete_json", lambda *_: None)(
                system_prompt, user_prompt, model_type="pro"
            )
            payload = response.get("selections") if isinstance(response, dict) else response
            if isinstance(payload, list):
                return payload
        except Exception as exc:
            LOGGER.warning("LLM batch evidence selection failed or timed out: %s", exc)
        return []

    all_payloads = []
    # 使用 ThreadPoolExecutor 并发发起 API 请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(batches))) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        for future in concurrent.futures.as_completed(futures):
            try:
                payload = future.result()
                all_payloads.extend(payload)
            except Exception as exc:
                LOGGER.warning("Thread execution error for batch selection: %s", exc)

    # 针对每篇 Paper 匹配并组合 SelectionResult
    for paper in papers:
        fallback = _create_fallback_selection(paper, plan)
        selected = fallback
        if all_payloads:
            for item in all_payloads:
                if isinstance(item, dict) and item.get("paper_id") == paper.paper_id:
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
                            reason=item.get("reason", item.get("explanation", "Parsed from LLM")),
                            confidence=float(item.get("confidence", 0.9)),
                            is_validated=True
                        )
                    except Exception as exc:
                        LOGGER.warning("Error parsing LLM selection result for paper %s: %s", paper.paper_id, exc)
                        selected = fallback
                    break
        results.append(selected)

    return results
