from __future__ import annotations

import logging
import re
from typing import Literal

from scholar_agent.models.schemas import EvidenceItem, Paper, QueryPlan, SelectionResult

LOGGER = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    """去标点符号和空格，转小写，增加匹配的鲁棒性。"""
    if not s:
        return ""
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", "", s).lower()


def validate_evidence(
    paper: Paper,
    selection: SelectionResult,
    plan: QueryPlan
) -> SelectionResult:
    """对单篇论文的 LLM 筛选结果进行硬性规则校验，并在不合规时进行降级。"""
    # 拷贝一份以避免直接修改原始对象
    validated_notes: list[str] = list(selection.validation_notes)
    is_validated = True
    relevance_level: Literal["high", "medium", "low", "irrelevant"] = selection.relevance_level

    # 1. 校验年份约束
    if plan.time_range:
        start_year = plan.time_range.get("start_year")
        end_year = plan.time_range.get("end_year")
        # 允许没有年份的论文（可能有少量解析缺失），但如果有年份则必须在范围内
        if paper.year is not None:
            if (start_year is not None and paper.year < start_year) or \
               (end_year is not None and paper.year > end_year):
                is_validated = False
                validated_notes.append(
                    f"Year constraint violated: Paper year {paper.year} is outside range [{start_year}, {end_year}]."
                )

    # 2. 校验证据片段真实性（必须存在于标题或摘要中）
    title_norm = _normalize(paper.title)
    abstract_norm = _normalize(paper.abstract or "")
    combined_norm = title_norm + abstract_norm

    for ev in selection.evidence:
        ev_norm = _normalize(ev.text)
        if not ev_norm:
            continue
        if ev_norm not in combined_norm:
            is_validated = False
            validated_notes.append(
                f"Hallucinated evidence detected: '{ev.text[:30]}...' could not be verified in title or abstract."
            )

    # 3. 校验 must-have 强约束词
    paper_content_norm = _normalize((paper.title or "") + " " + (paper.abstract or ""))
    for constraint in plan.must_have_constraints:
        constraint_norm = _normalize(constraint)
        if constraint_norm and constraint_norm not in paper_content_norm:
            is_validated = False
            validated_notes.append(
                f"Missing must-have constraint: '{constraint}' was not found in paper text."
            )

    # 4. 如果校验失败，执行降级逻辑
    if not is_validated:
        if relevance_level == "high":
            relevance_level = "medium"
        elif relevance_level == "medium":
            relevance_level = "low"
        elif relevance_level == "low":
            relevance_level = "irrelevant"
        
        # 为了进一步保护质量，如果有极其严重的虚假证据或年份不符，可直接降级至 low
        # 这里只降一级或者在有多个 notes 时降到 low
        if len(validated_notes) >= 2 or any("year" in note.lower() for note in validated_notes):
            relevance_level = "low"

    # 基于高置信度召回路径（精确/相似标题匹配）的硬性等级保底与校验旁路，确保极度相关的文献不被LLM主观偏差或字面约束字眼缺失所过滤
    paths = paper.retrieval_path or []
    is_exact_title = any("title_exact" in p for p in paths)
    is_like_title = any("title_like" in p for p in paths)

    if is_exact_title:
        relevance_level = "high"
        is_validated = True
        validated_notes.append("Path calibration: title_exact matched, force high relevance.")
    elif is_like_title:
        if relevance_level in ("low", "irrelevant"):
            relevance_level = "medium"
            validated_notes.append("Path calibration: title_like matched, set medium relevance floor.")

    return SelectionResult(
        paper_id=selection.paper_id,
        relevance_level=relevance_level,
        matched_constraints=selection.matched_constraints,
        missing_constraints=selection.missing_constraints,
        evidence=selection.evidence,
        reason=selection.reason,
        confidence=selection.confidence,
        is_validated=is_validated,
        validation_notes=validated_notes
    )


def validate_selections(
    papers: list[Paper],
    selections: list[SelectionResult],
    plan: QueryPlan
) -> list[SelectionResult]:
    """批量校验一组论文的选择结果。"""
    paper_map = {p.paper_id: p for p in papers}
    validated_selections: list[SelectionResult] = []

    for sel in selections:
        paper = paper_map.get(sel.paper_id)
        if not paper:
            # 找不到 paper 就不做修改
            validated_selections.append(sel)
        else:
            validated_selections.append(validate_evidence(paper, sel, plan))

    return validated_selections
