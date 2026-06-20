from __future__ import annotations

import logging
from typing import Any

from scholar_agent.models.schemas import Paper, QueryPlan

LOGGER = logging.getLogger(__name__)


def review_retrieval_results(
    plan: QueryPlan,
    papers: list[Paper],
    round_index: int,
    llm_client: object | None = None,
) -> dict[str, Any]:
    """Result Review Agent
    每轮检索后，让大语言模型审阅候选论文（摘要和标题等），判断是否满足查询约束，
    并产生覆盖率分析、发现缺口（漏掉的方法/数据集等）、识别噪声，并决定是否停止。
    """
    # 启发式兜底数据，以防 LLM 挂了或在 Mock 下运行
    fallback = {
        "coverage_analysis": {
            "covered_aspects": [plan.research_topic or "core topic"] if plan.research_topic else [],
            "missing_aspects": [],
            "noise_patterns": []
        },
        "candidate_quality": {
            "enough_candidates": len(papers) >= 20,
            "hard_constraint_coverage": "high" if len(papers) >= 15 else "medium",
            "precision_risk": "low",
            "recall_risk": "low" if len(papers) >= 20 else "medium"
        },
        "next_action": "stop_search" if len(papers) >= 30 or round_index >= 3 else "continue_search",
        "suggested_new_keywords": [],
        "suggested_excluded_terms": [],
        "need_citation_expansion": len(papers) > 0 and len(papers) < 15,
        "reason": "已找到足够候选论文，或已达最大迭代轮数。" if len(papers) >= 20 else "候选论文较少，建议继续检索。"
    }

    if llm_client is None or not papers or len(papers) >= 30 or round_index >= 3:
        return fallback

    # 仅向大模型呈现 Top 20 篇候选论文进行审阅，以降低 Token 消耗并控制预算
    sample_papers = papers[:20]
    papers_payload = [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": (p.abstract[:300] + "...") if p.abstract else None,
            "year": p.year,
            "venue": p.venue,
            "source": p.source,
        }
        for p in sample_papers
    ]

    system_prompt = (
        "You are the Result Review Agent for a scholarly search workflow. "
        "Your job is to review the current retrieved candidate papers and analyze "
        "whether they satisfy the query contract (QueryPlan). "
        "Explain what aspects are covered, what is missing (gaps), and what noise exists. "
        "Then decide if we need another round of search or if we can stop. "
        "Return a JSON object matching the requested schema exactly. JSON ONLY."
    )

    user_prompt = (
        "Output JSON with these fields:\n"
        "{\n"
        "  \"coverage_analysis\": {\n"
        "    \"covered_aspects\": [\"aspect1\", ...],\n"
        "    \"missing_aspects\": [\"aspect2\", ...],\n"
        "    \"noise_patterns\": [\"noise_type\", ...]\n"
        "  },\n"
        "  \"candidate_quality\": {\n"
        "    \"enough_candidates\": true/false,\n"
        "    \"hard_constraint_coverage\": \"high\"/\"medium\"/\"low\",\n"
        "    \"precision_risk\": \"high\"/\"medium\"/\"low\",\n"
        "    \"recall_risk\": \"high\"/\"medium\"/\"low\"\n"
        "  },\n"
        "  \"next_action\": \"continue_search\" / \"stop_search\",\n"
        "  \"suggested_new_keywords\": [\"keyword1\", ...],\n"
        "  \"suggested_excluded_terms\": [\"term1\", ...],\n"
        "  \"need_citation_expansion\": true/false,\n"
        "  \"reason\": \"reasons for stopping or continuing\"\n"
        "}\n\n"
        f"QueryPlan Contract: {plan.model_dump_json()}\n"
        f"Current Candidate Papers (Top {len(papers_payload)}): {papers_payload}\n"
        f"Current Retrieval Round: {round_index}\n"
        "CRITICAL: When analyzing missing_aspects, check if the candidate papers cover ALL methods, datasets, and entities specified in the QueryPlan. "
        "If any method/dataset/entity is NOT mentioned in the titles or abstracts of the candidate papers, add it to missing_aspects. "
        "Be thorough and prefer recall over precision - if in doubt, add it to missing_aspects."
    )

    # 审阅工作需要相对深度的理解，使用 pro 级模型
    response = getattr(llm_client, "complete_json", lambda *_: None)(system_prompt, user_prompt, model_type="pro")
    if response is None:
        return fallback
    try:
        # 对输出进行基本结构校验
        for key in ["coverage_analysis", "candidate_quality", "next_action", "reason"]:
            if key not in response:
                return fallback

        # 增强 recall_risk 判断逻辑
        coverage = response.get("coverage_analysis", {})
        missing_aspects = coverage.get("missing_aspects", [])
        candidate_quality = response.get("candidate_quality", {})

        # 如果有明显 missing_aspects，强制 recall_risk 为 high
        if missing_aspects and len(missing_aspects) > 0:
            candidate_quality["recall_risk"] = "high"
            # 且如果还没达到硬上限，强制 next_action 为 continue_search
            if len(papers) < 30 and round_index < 3:
                response["next_action"] = "continue_search"
                response["reason"] = f"Missing aspects detected: {missing_aspects}. Continue search to improve recall."

        return dict(response)
    except Exception as exc:
        LOGGER.warning("Result Review Agent parse response failed: %s", exc)
        return fallback
