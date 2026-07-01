"""LLM Pool Reviewer: LLM 审阅候选池，替代纯启发式 local_pre_rank。

核心设计：
- 启发式只做"明显无关剔除"（去噪），不做精排，保留尽可能多的候选
- LLM 分批审阅候选池，做池级判断：覆盖了哪些方面、缺哪些方面、哪些明显无关要剔除
- LLM 输出：保留集 + 缺失方面 + query 理解修正建议（供闭环迭代用）
- RECALL-FIRST：宁可保留 borderline 也不误杀 gold

替代 local_pre_ranker.py 的纯启发式 8 维加权打分。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from scholar_agent.models.schemas import Paper, QueryPlan

LOGGER = logging.getLogger(__name__)


def _lightweight_dedup_sort(papers: list[Paper], plan: QueryPlan, original_query: str) -> list[Paper]:
    """轻量去噪：只剔除明显无关（title 零词重叠且无 route_bonus），不做精排。

    保留尽可能多的候选给 LLM 审阅。最多剔除池子的 20%，避免过度过滤。
    """
    import re

    stop_words = {"the", "a", "an", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from", "that", "this"}
    q_words = set(re.findall(r"\b\w{3,}\b", original_query.lower())) - stop_words
    if plan.research_topic:
        q_words |= set(re.findall(r"\b\w{3,}\b", plan.research_topic.lower())) - stop_words

    # 扩展：query_plan 的 methods/entities 也作为 query 词
    for field in ("methods", "entities", "datasets"):
        for term in getattr(plan, field, []) or []:
            q_words |= set(re.findall(r"\b\w{3,}\b", term.lower())) - stop_words

    kept: list[Paper] = []
    dropped: list[Paper] = []
    for p in papers:
        title_words = set(re.findall(r"\b\w{3,}\b", (p.title or "").lower())) - stop_words
        has_title_overlap = bool(q_words & title_words)
        has_route_bonus = any(
            "title_exact" in path or "title_like" in path or "core_topic" in path
            for path in (p.retrieval_path or [])
        )
        # 明显无关：title 零词重叠 + 无 route_bonus + 有 abstract 但 abstract 也零词重叠
        if not has_title_overlap and not has_route_bonus:
            abstract_words = set(re.findall(r"\b\w{3,}\b", (p.abstract or "").lower())) - stop_words
            if not (q_words & abstract_words):
                dropped.append(p)
                continue
        kept.append(p)

    # 限制：最多剔除 20%，避免过度过滤
    max_drop = max(1, len(papers) // 5)
    if len(dropped) > max_drop:
        # 把多出来的 dropped 放回 kept（按 citation_count 降序优先放回高引的）
        dropped.sort(key=lambda p: getattr(p, "citation_count", 0) or 0, reverse=True)
        kept.extend(dropped[:len(dropped) - max_drop])
        dropped = dropped[len(dropped) - max_drop:]

    LOGGER.info(
        "Lightweight dedup: %d -> %d (dropped %d obvious noise)",
        len(papers), len(kept), len(dropped),
    )
    return kept


def _build_pool_review_prompt(
    plan: QueryPlan,
    papers: list[Paper],
    original_query: str,
    abstract_limit: int = 400,
) -> tuple[str, str]:
    """构建 LLM 池审阅 prompt。

    与 batch_evidence_selector 不同：不是 per-paper 打分，而是池级判断——
    覆盖了哪些方面、缺哪些方面、哪些明显无关要剔除、query 理解修正建议。
    """
    system_prompt = (
        "You are an expert scholarly search Pool Reviewer. Your job is to review a candidate paper pool "
        "and make POOL-LEVEL decisions with RECALL-FIRST strategy.\n"
        "Your PRIMARY goal is to MAXIMIZE RECALL — do NOT discard any paper that could be relevant.\n"
        "False negatives (discarding gold papers) are MUCH worse than false positives (keeping borderline papers).\n\n"
        "For each paper, output a verdict:\n"
        "- 'keep': Paper is relevant or potentially relevant to the query.\n"
        "- 'soft_drop': Paper seems less relevant, but might still be useful — it will be deprioritized, NOT deleted.\n"
        "- 'noise': Paper is likely off-topic, but keep it in archive just in case.\n"
        "- 'discard': ONLY for papers that violate hard constraints (wrong year, wrong venue, metadata error). Almost never use this.\n"
        "When uncertain between keep and soft_drop, use 'keep'.\n"
        "When uncertain between soft_drop and noise, use 'soft_drop'.\n"
        "Reserve 'discard' for clear metadata violations only — NEVER use it for topic mismatch alone.\n\n"
        "Then analyze the POOL as a whole:\n"
        "- covered_aspects: which methods/datasets/entities/constraints from the QueryPlan are covered by the pool.\n"
        "- missing_aspects: which are NOT covered — these will trigger another retrieval round.\n"
        "- noise_patterns: if any, patterns of irrelevant papers (e.g. 'too many medical papers for a CS query').\n"
        "- re_understanding_feedback: if missing_aspects non-empty, suggest how to refine the query understanding.\n"
        "Return VALID JSON ONLY."
    )

    paper_payload = []
    for p in papers:
        paper_payload.append({
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": (p.abstract or "")[:abstract_limit],
            "year": p.year,
            "citation_count": p.citation_count,
            "retrieval_path": p.retrieval_path,
        })

    user_prompt = (
        'Output format (JSON only):\n'
        '{\n'
        '  "per_paper": [{"paper_id": "string", "verdict": "keep"|"soft_drop"|"noise"|"discard", "reason": "string"}],\n'
        '  "covered_aspects": ["string"],\n'
        '  "missing_aspects": ["string"],\n'
        '  "noise_patterns": ["string"],\n'
        '  "re_understanding_feedback": "string",\n'
        '  "coverage_score": 0.0-1.0,\n'
        '  "converged": true|false\n'
        '}\n\n'
        f"Original query: {original_query!r}\n"
        f"QueryPlan: {plan.model_dump_json()}\n"
        f"Papers ({len(papers)} total): {json.dumps(paper_payload, ensure_ascii=False)}"
    )

    return system_prompt, user_prompt


def _parse_pool_review_response(
    response: Any,
    paper_ids: set[str],
) -> dict[str, Any]:
    """解析 LLM 池审阅响应。"""
    if response is None or not isinstance(response, dict):
        return {}
    return response


def llm_review_pool(
    papers: list[Paper],
    plan: QueryPlan,
    llm_client: object | None,
    original_query: str,
    config: Any | None = None,
    deadline: Any | None = None,
    batch_size: int = 10,
    max_review_papers: int = 200,
) -> dict[str, Any]:
    """LLM 审阅候选池，返回保留集 + 缺失方面 + 修正建议。

    Returns:
        dict with keys:
            - kept_papers: list[Paper] (keep + borderline + LLM 未覆盖的)
            - dropped_papers: list[Paper]
            - covered_aspects: list[str]
            - missing_aspects: list[str]
            - noise_patterns: list[str]
            - re_understanding_feedback: str
            - coverage_score: float
            - converged: bool
    """
    if not papers:
        return {
            "kept_papers": [],
            "dropped_papers": [],
            "covered_aspects": [],
            "missing_aspects": [],
            "noise_patterns": [],
            "re_understanding_feedback": "",
            "coverage_score": 0.0,
            "converged": True,
        }

    # Step 1: 轻量去噪
    pre_filtered = _lightweight_dedup_sort(papers, plan, original_query)

    # Step 2: 取 top-N 送 LLM（按已有 rough_score 排序，用 local_pre_rank_score 如果有）
    review_candidates = pre_filtered[:max_review_papers]

    if not llm_client:
        # LLM 不可用：fallback 到 local_pre_rank 截断
        LOGGER.info("LLM pool review: LLM unavailable, using pre_filtered pool as-is (%d papers)", len(pre_filtered))
        return {
            "kept_papers": pre_filtered,
            "dropped_papers": [],
            "covered_aspects": [],
            "missing_aspects": [],
            "noise_patterns": [],
            "re_understanding_feedback": "",
            "coverage_score": 0.5,
            "converged": True,
        }

    # Step 3: 分批送 LLM
    # 收集所有批次的 per_paper verdicts
    all_verdicts: dict[str, str] = {}  # paper_id -> verdict
    covered_aspects: list[str] = []
    missing_aspects: list[str] = []
    noise_patterns: list[str] = []
    re_understanding_feedback = ""
    coverage_score = 0.5
    converged = True

    # 每批处理
    num_batches = (len(review_candidates) + batch_size - 1) // batch_size
    LOGGER.info(
        "LLM pool review: %d papers in %d batches (batch_size=%d)",
        len(review_candidates), num_batches, batch_size,
    )

    _consecutive_none = 0  # Fast-fail: skip remaining batches if LLM consistently returns None
    for i in range(0, len(review_candidates), batch_size):
        if deadline is not None and deadline.expired():
            LOGGER.info(
                "LLM pool review: deadline expired at batch %d/%d, keeping remaining papers",
                i // batch_size, num_batches,
            )
            break

        # Fast-fail: if 2 consecutive batches returned None, skip remaining
        if _consecutive_none >= 2:
            LOGGER.info(
                "LLM pool review: fast-fail after %d consecutive None batches, keeping remaining papers",
                _consecutive_none,
            )
            break

        batch = review_candidates[i:i + batch_size]
        system_prompt, user_prompt = _build_pool_review_prompt(plan, batch, original_query)

        try:
            response = getattr(llm_client, "complete_json", lambda *_: None)(
                system_prompt, user_prompt, model_type="flash"
            )
            if response and isinstance(response, dict):
                # 收集 per_paper verdicts
                for item in response.get("per_paper", []):
                    if isinstance(item, dict) and item.get("paper_id"):
                        pid = item["paper_id"]
                        verdict = item.get("verdict", "keep")
                        # Map legacy verdicts to non-destructive types
                        if verdict == "borderline":
                            verdict = "soft_drop"
                        elif verdict == "drop":
                            verdict = "noise"  # Legacy "drop" → "noise" (non-destructive)
                        if verdict not in ("keep", "soft_drop", "noise", "discard"):
                            verdict = "keep"
                        all_verdicts[pid] = verdict

                # 只在最后一批收集池级分析（避免重复覆盖）
                if i + batch_size >= len(review_candidates):
                    covered_aspects = response.get("covered_aspects", []) or []
                    missing_aspects = response.get("missing_aspects", []) or []
                    noise_patterns = response.get("noise_patterns", []) or []
                    re_understanding_feedback = response.get("re_understanding_feedback", "") or ""
                    coverage_score = response.get("coverage_score", 0.5) or 0.5
                    converged = bool(response.get("converged", True))
                _consecutive_none = 0  # Reset on success
            else:
                _consecutive_none += 1
                LOGGER.warning("LLM pool review: batch %d returned None, keeping all papers in batch",
                               i // batch_size)
        except Exception as exc:
            _consecutive_none += 1
            LOGGER.warning("LLM pool review: batch %d failed: %s, keeping all papers in batch",
                           i // batch_size, exc)

        # 批间停顿避免限流
        if i + batch_size < len(review_candidates):
            time.sleep(0.3)

    # Step 4: 非破坏式标记 — 给每篇论文打 pool_review_action 标签
    # 不再删除论文，而是用 metadata 标记 keep/soft_drop/noise/discard
    # 只有 discard（硬约束违反）才真正剔除，其余全部保留
    tagged_papers: list[Paper] = []
    discarded: list[Paper] = []
    action_counts = {"keep": 0, "soft_drop": 0, "noise": 0, "discard": 0}

    for p in pre_filtered:
        verdict = all_verdicts.get(p.paper_id, "keep")  # LLM 未覆盖的默认 keep
        # 写入 metadata
        if p.metadata is None:
            p.metadata = {}
        p.metadata["pool_review_action"] = verdict
        p.metadata["pool_review_reason"] = ""  # reason 可从 LLM 响应补充

        if verdict == "discard":
            discarded.append(p)
        else:
            tagged_papers.append(p)
        action_counts[verdict] = action_counts.get(verdict, 0) + 1

    # 补充 reason
    for item in response.get("per_paper", []) if isinstance(response, dict) else []:
        if isinstance(item, dict) and item.get("paper_id"):
            pid = item["paper_id"]
            reason = item.get("reason", "")
            for p in tagged_papers:
                if p.paper_id == pid and p.metadata:
                    p.metadata["pool_review_reason"] = reason
                    break

    # RECALL-FIRST: 保留所有 non-discard 论文（包括 noise），只剔除真正的 discard
    # 但如果 discard 太多（超过 10%），把高引的放回（防止误杀）
    max_discard_count = max(1, len(pre_filtered) // 10)
    if len(discarded) > max_discard_count:
        discarded.sort(key=lambda p: getattr(p, "citation_count", 0) or 0, reverse=True)
        recover = discarded[:len(discarded) - max_discard_count]
        for p in recover:
            if p.metadata is None:
                p.metadata = {}
            p.metadata["pool_review_action"] = "noise"  # 降级为 noise 而非 discard
            tagged_papers.append(p)
        discarded = discarded[len(discarded) - max_discard_count:]
        LOGGER.info("RECALL guard: recovered %d high-cite papers from discard list", len(recover))

    LOGGER.info(
        "LLM pool review done (non-destructive): keep=%d, soft_drop=%d, noise=%d, discard=%d | "
        "missing_aspects=%d, coverage=%.2f, converged=%s",
        action_counts["keep"], action_counts["soft_drop"],
        action_counts["noise"], action_counts["discard"],
        len(missing_aspects), coverage_score, converged,
    )

    return {
        "kept_papers": tagged_papers,  # 兼容：所有 non-discard 论文（含 soft_drop/noise 标签）
        "dropped_papers": discarded,   # 兼容：仅真正 discard 的论文
        "all_papers": tagged_papers,   # 新字段：非破坏式全量论文（带标签）
        "covered_aspects": covered_aspects,
        "missing_aspects": missing_aspects,
        "noise_patterns": noise_patterns,
        "re_understanding_feedback": re_understanding_feedback,
        "coverage_score": coverage_score,
        "converged": converged,
    }
