from __future__ import annotations

import logging
from typing import Any

from scholar_agent.models.schemas import Paper, QueryPlan, SearchQuery

LOGGER = logging.getLogger(__name__)


def optimize_search_strategy(
    plan: QueryPlan,
    review_result: dict[str, Any],
    candidates: list[Paper],
    round_index: int,
    llm_client: object | None = None,
) -> dict[str, Any]:
    """Search Strategy Optimization Agent
    根据 Review 结论，调整并生成下一轮检索的查询策略。
    """
    # 启发式兜底数据，以防 LLM 挂了或在 Mock 下运行
    fallback_seeds = [p.paper_id for p in candidates[:3]] if candidates else []
    fallback = {
        "round": round_index + 1,
        "search_goal": "补充缺失主题与硬约束项",
        "new_subqueries": [
            {
                "query": " ".join(review_result.get("suggested_new_keywords", []) or [plan.research_topic or ""]),
                "reason": "基于审阅建议补充缺失的关键词"
            }
        ] if review_result.get("suggested_new_keywords") else [],
        "citation_expansion_seeds": fallback_seeds,
        "negative_filters": review_result.get("suggested_excluded_terms", []),
        "stop_after_this_round": False
    }

    if llm_client is None:
        return fallback

    system_prompt = (
        "You are the Search Strategy Optimization Agent for a scholarly search workflow. "
        "Based on the Result Review Agent's feedback on covered/missing aspects, "
        "you formulate the next search plan (Round 2 or 3). "
        "You can propose new search queries, select candidate papers from the pool as seeds for citation expansion, "
        "and define negative keyword filters to prune search noise. "
        "Return a JSON object only. JSON ONLY."
    )

    user_prompt = (
        "Output JSON with these fields:\n"
        "{\n"
        "  \"round\": int,\n"
        "  \"search_goal\": \"description\",\n"
        "  \"new_subqueries\": [\n"
        "    {\"query\": \"expanded keywords query\", \"reason\": \"why we search this\"}\n"
        "  ],\n"
        "  \"citation_expansion_seeds\": [\"paper_id_1\", ...],\n"
        "  \"negative_filters\": [\"noise_word_1\", ...],\n"
        "  \"stop_after_this_round\": true/false\n"
        "}\n\n"
        f"QueryPlan Contract: {plan.model_dump_json()}\n"
        f"Review Agent Findings: {review_result}\n"
        f"Current Candidates Metadata: {[{'paper_id': p.paper_id, 'title': p.title} for p in candidates[:15]]}\n"
        f"Next Round Index: {round_index + 1}"
    )

    response = getattr(llm_client, "complete_json", lambda *_: None)(system_prompt, user_prompt, model_type="flash")
    if response is None:
        return fallback
    try:
        for key in ["round", "search_goal", "new_subqueries", "citation_expansion_seeds", "stop_after_this_round"]:
            if key not in response:
                return fallback
        return dict(response)
    except Exception as exc:
        LOGGER.warning("Strategy Optimization Agent parse response failed: %s", exc)
        return fallback
