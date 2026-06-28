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


def _check_constraint_match(constraint: str, paper_content_norm: str, paper_title: str, paper_abstract: str) -> bool:
    constraint_norm = _normalize(constraint)
    if not constraint_norm:
        return True
    
    # 1. 尝试直接子串匹配
    if constraint_norm in paper_content_norm:
        return True
        
    # 2. 如果是多词组合，去除停用词后核心词必须全部包含（支持顺序和定语变化）
    import re
    words = re.findall(r"\b\w{3,}\b", constraint.lower())
    stop_words = {"the", "a", "an", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from", "that", "this", "these", "those", "language", "model", "models"}
    core_words = [w for w in words if w not in stop_words]
    
    if core_words:
        raw_paper_text = (paper_title or "") + " " + (paper_abstract or "")
        paper_words = set(re.findall(r"\b\w+\b", raw_paper_text.lower()))
        if all(cw in paper_words for cw in core_words):
            return True
            
    # 3. 常见缩写或同义词映射
    synonyms = {
        "llm": {"llm", "llms", "large language model", "large language models", "foundation model", "foundation models"},
        "nlp": {"nlp", "natural language processing"},
        "multilingual": {"multilingual", "multi-lingual", "cross-lingual", "crosslingual"},
        "cross-lingual": {"multilingual", "multi-lingual", "cross-lingual", "crosslingual"},
        "contrastive learning": {"contrastive learning", "contrastive loss", "contrastive pretraining", "contrastive pre-training"},
    }
    
    constraint_lower = constraint.lower()
    for key, syns in synonyms.items():
        if key in constraint_lower:
            for syn in syns:
                syn_norm = _normalize(syn)
                if syn_norm and syn_norm in paper_content_norm:
                    return True
                    
    return False


def _selection_constraint_coverage(selection: SelectionResult, plan: QueryPlan) -> float:
    """计算 selection 对 plan 约束的覆盖比例。"""
    constraints = []
    constraints.extend(plan.datasets or [])
    constraints.extend(plan.methods or [])
    constraints.extend(plan.must_have_constraints or [])

    constraints = list(dict.fromkeys([c for c in constraints if c]))
    if not constraints:
        return 1.0 if selection.relevance_level in {"high", "medium"} else 0.3

    matched = set(x.lower() for x in selection.matched_constraints)
    hit = 0
    for c in constraints:
        c_low = c.lower()
        if c_low in matched:
            hit += 1
        elif any(c_low in m or m in c_low for m in matched):
            hit += 1

    return hit / max(len(constraints), 1)


def _valid_evidence_ratio(selection: SelectionResult, paper: Paper) -> float:
    """计算 selection 中证据片段真实存在于 paper 文本的比例。"""
    if not selection.evidence:
        return 0.0

    text = f"{paper.title} {paper.abstract or ''}".lower()
    valid = 0

    for ev in selection.evidence:
        ev_text = getattr(ev, "text", "") or ""
        ev_text_low = ev_text.lower().strip()
        if ev_text_low and ev_text_low in text:
            valid += 1

    return valid / max(len(selection.evidence), 1)


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
        if not _check_constraint_match(constraint, paper_content_norm, paper.title, paper.abstract):
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

    # 5. 计算 constraint coverage 和 valid evidence ratio，写入 validation_notes
    coverage = _selection_constraint_coverage(selection, plan)
    evidence_ratio = _valid_evidence_ratio(selection, paper)
    validated_notes.append(f"constraint_coverage={coverage:.3f}")
    validated_notes.append(f"valid_evidence_ratio={evidence_ratio:.3f}")

    # 6. 基于证据和约束覆盖的降级规则
    is_local_fallback = not selection.evidence and "local" in (selection.reason or "").lower()

    if relevance_level == "high":
        if not is_local_fallback and evidence_ratio < 0.5:
            relevance_level = "medium"
            validated_notes.append("Downgraded: high relevance without enough verifiable evidence.")
        if plan.must_have_constraints and coverage < 0.5:
            relevance_level = "medium"
            validated_notes.append("Downgraded: high relevance with low must-have coverage.")

    if relevance_level == "medium":
        if not is_local_fallback and evidence_ratio < 0.2:
            relevance_level = "low"
            validated_notes.append("Downgraded: medium relevance with no substantive verifiable evidence.")
        elif not is_local_fallback and (coverage < 0.25 and evidence_ratio < 0.5):
            relevance_level = "low"
            validated_notes.append("Downgraded: medium relevance with weak constraint/evidence support.")

    # 7. 检索路径信息保留（语义中性，不影响 level）
    paths = paper.retrieval_path or []
    if any("title_exact" in p for p in paths):
        validated_notes.append("Path: title_exact matched.")
    elif any("title_like" in p for p in paths):
        validated_notes.append("Path: title_like matched.")

    return SelectionResult(
        paper_id=selection.paper_id,
        relevance_level=relevance_level,
        matched_constraints=selection.matched_constraints,
        missing_constraints=selection.missing_constraints,
        evidence=selection.evidence,
        reason=selection.reason,
        confidence=selection.confidence,
        is_validated=is_validated,
        validation_notes=validated_notes,
        relevance_score=selection.relevance_score,
        constraint_score=selection.constraint_score,
        evidence_score=selection.evidence_score,
        uncertainty=selection.uncertainty,
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
