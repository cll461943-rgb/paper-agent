from __future__ import annotations

import json
import logging
from typing import Any

from scholar_agent.models.schemas import EvidenceItem, Paper, QueryPlan, SelectionResult
from scholar_agent.selection.evidence_selector import _create_fallback_selection

LOGGER = logging.getLogger(__name__)


def batch_select_and_extract_evidence(
    papers: list[Paper],
    plan: QueryPlan,
    llm_client: object | None,
) -> list[SelectionResult]:
    """批量 Evidence Selector Agent - 对候选论文进行批量相关性判断。

    策略：
    - 大批次 (5篇/批) 减少 LLM 请求次数，降低 Token 开销
    - 强大的容错和解析逻辑，出错时逐篇 Fallback
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
        "\n"
        "CRITICAL RELEVANCE RULES FOR SCHOLARLY PAPERS:\n"
        "1. Contrastive, Alternative & Opposing Innovations: If a paper addresses the exact core research problem "
        "but proposes a method to REPLACE, AVOID, CONTRAST, or CRITIQUE the traditional constraints/methods requested "
        "in the query (e.g. using bootstrapping/self-training to AVOID traditional word-removal data augmentation; "
        "or proving that incorrect/invalid reasoning steps in in-context examples can still maintain reasoning performance; "
        "or minimizing worst-case distribution risk to perform adversarial training WITHOUT constructing samples), "
        "this paper is HIGHLY RELEVANT and must be classified as 'high' or 'medium'. Do NOT mark it as 'low' or 'irrelevant'.\n"
        "2. Semantic Equivalents: Do not rely on naive word-matching. Academic papers use formal, abstract language. "
        "Recognize semantic synonyms (e.g. 'invalid/incorrect reasoning steps' or 'ground-truth not necessary' matches 'absurdly wrong examples'; "
        "'distribution shift risk minimization' or 'worst-case distribution optimization' matches 'DRO-like objective without constructing samples').\n"
        "3. Recall-Oriented Classification: When in doubt, prefer 'medium' over 'low' and 'low' over 'irrelevant'. "
        "Only classify a paper as 'irrelevant' if it has absolutely no relation to the research domain of the query.\n"
        "4. Retrieved Path Confidence Boost: Papers with retrieval paths containing 'title_exact', 'title_like', or 'translated' "
        "should be given higher relevance consideration as they matched specific query constraints.\n"
        "5. Top-Ranked Candidates: If a paper is in the top 10 candidates by rough ranking, it likely has strong semantic relevance "
        "and should not be classified as 'irrelevant' without strong evidence.\n"
        "\n"
        "CRITICAL OUTPUT REQUIREMENTS:\n"
        "- Return VALID JSON ONLY. Never include markdown, code fences, or explanations.\n"
        "- Strictly follow this structure: {\"selections\": [{...}, ...]}\n"
        "- ALWAYS close all brackets and quotes properly. NO trailing commas.\n"
        "- Each object in selections must have ALL fields: paper_id, relevance_level, matched_constraints, "
        "missing_constraints, evidence, reason, confidence.\n"
        "- relevance_level must be exactly one of: \"high\", \"medium\", \"low\", \"irrelevant\".\n"
        "- confidence must be a number between 0.0 and 1.0.\n"
        "- Use confidence threshold: >0.8 for 'high', >0.6 for 'medium', >0.4 for 'low', <=0.4 for 'irrelevant'."
    )

    # 使用更大批次 (8篇/批) 减少批次数量，用 flash 模型降低单次调用延迟
    # evidence_selector 负责粗粒度相关性判断，flash 精度已足够
    # pro 模型留给 synthesis 合成阶段（精度要求更高）
    batch_size = 8
    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]

    all_payloads: list[dict[str, Any]] = []

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
            "Output format schema (JSON only, no markdown):\n"
            "{\n"
            "  \"selections\": [\n"
            "    {\"paper_id\": \"string\", \"relevance_level\": \"high\"|\"medium\"|\"low\"|\"irrelevant\", "
            "\"matched_constraints\": [...], \"missing_constraints\": [...], \"evidence\": [{\"field\": \"title\"|\"abstract\", \"text\": \"...\"}], "
            "\"reason\": \"string\", \"confidence\": 0.95}\n"
            "  ]\n"
            "}\n\n"
            f"QueryPlan: {plan.model_dump_json()}\n"
            f"Papers: {json.dumps(paper_payload, ensure_ascii=False)}"
        )

        try:
            # 使用 flash 模型做粗粒度相关性判断（延迟低约 4s/次 vs pro 的 18s/次）
            # pro 模型保留给最终合成阶段（synthesis），以保证输出质量
            response = getattr(llm_client, "complete_json", lambda *_: None)(
                system_prompt, user_prompt, model_type="flash"
            )
            if response is not None:
                payload = response.get("selections") if isinstance(response, dict) else response
                if isinstance(payload, list):
                    return payload
        except Exception as exc:
            LOGGER.warning(f"Batch LLM Selection failed for batch of size {len(batch)}: {exc}")

        # Fallback
        LOGGER.info(f"LLM batch failed. Applying fallback selections for {len(batch)} papers.")
        fallback_results = []
        for paper in batch:
            fallback_item = _create_fallback_selection(paper, plan)
            fallback_results.append({
                "paper_id": paper.paper_id,
                "relevance_level": fallback_item.relevance_level,
                "matched_constraints": fallback_item.matched_constraints,
                "missing_constraints": fallback_item.missing_constraints,
                "evidence": [{"field": ev.field, "text": ev.text} for ev in fallback_item.evidence],
                "reason": fallback_item.reason,
                "confidence": fallback_item.confidence
            })
        return fallback_results

    import concurrent.futures
    if batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(batches))) as executor:
            futures = [executor.submit(process_batch, b) for b in batches]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    all_payloads.extend(res)

    # 统一转换并包装结果
    paper_map = {p.paper_id: p for p in papers}
    for paper in papers:
        selected = None
        # 寻找对应的 payload
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

                    selected = SelectionResult(
                        paper_id=paper.paper_id,
                        relevance_level=relevance,
                        matched_constraints=item.get("matched_constraints", []),
                        missing_constraints=item.get("missing_constraints", []),
                        evidence=evidence_items,
                        reason=item.get("reason", ""),
                        confidence=float(item.get("confidence", 0.8)),
                        is_validated=True
                    )
                except Exception as exc:
                    LOGGER.warning(f"Parse error for selection result of {paper.paper_id}: {exc}")
                break

        # 如果没有找到或者解析挂了，再次做一次安全保底 fallback
        if not selected:
            selected = _create_fallback_selection(paper, plan)

        results.append(selected)

    return results
