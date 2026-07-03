"""Dynamic Router: based on query_type, dynamically allocate provider组合、查询路由、配额.

替代 multi_route.py 中硬编码的 route_priority 和 max_by_provider 静态表。
支持闭环迭代：根据 LLM pool review 的 feedback 调整路由策略。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from scholar_agent.models.schemas import QueryPlan

LOGGER = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """动态路由配置：决定每 provider 接收哪些路由的查询、配额上限。"""

    # 每 provider 允许的路由集合（None=全路由）
    routes_per_provider: dict[str, set[str] | None]
    # 每 provider 查询配额上限
    caps_per_provider: dict[str, int]
    # 是否触发安全兜底搜索（池子不足时）
    need_broad_safety: bool = True
    # 安全兜底的池子阈值
    broad_safety_threshold: int = 120
    # 路由调整原因（日志用）
    reason: str = ""


# ── 路由策略表 ────────────────────────────────────────────────────────────

_BASE_PROVIDERS = {"pasa_local", "openalex", "semantic_scholar"}
_BASE_PLUS_ARXIV = {"pasa_local", "openalex", "semantic_scholar", "arxiv"}
_ALL_PROVIDERS = {"pasa_local", "openalex", "semantic_scholar", "arxiv", "pubmed"}

# query_type → 路由配置
_ROUTING_TABLE: dict[str, dict[str, Any]] = {
    "exact_title": {
        "providers": _BASE_PLUS_ARXIV,
        "routes": {
            "openalex": {"title_like", "title_exact", "core_topic"},
            "semantic_scholar": {"title_like", "title_exact", "core_topic"},
            "arxiv": {"title_like", "title_exact", "core_topic"},
            "pasa_local": None,  # 全路由
            "pubmed": set(),
        },
        "caps": {"openalex": 5, "semantic_scholar": 5, "arxiv": 5, "pubmed": 0},
        "safety_threshold": 60,
        "reason": "exact_title: 集中 title_exact 通道，最小化 broad 路由噪声",
    },
    "single_gold": {
        "providers": _BASE_PLUS_ARXIV,
        "routes": {
            "openalex": {"title_like", "title_exact", "core_topic"},
            "semantic_scholar": {"title_like", "title_exact", "core_topic"},
            "arxiv": {"title_like", "title_exact", "core_topic"},
            "pasa_local": None,
            "pubmed": set(),
        },
        "caps": {"openalex": 5, "semantic_scholar": 5, "arxiv": 5, "pubmed": 0},
        "safety_threshold": 60,
        "reason": "single_gold: 集中 title_exact 通道",
    },
    "specific_paper": {
        "providers": _BASE_PLUS_ARXIV,
        "routes": {
            "openalex": {"title_like", "title_exact", "core_topic", "method_task"},
            "semantic_scholar": {"title_like", "title_exact", "core_topic"},
            "arxiv": {"title_like", "title_exact", "core_topic"},
            "pasa_local": None,
            "pubmed": set(),
        },
        "caps": {"openalex": 6, "semantic_scholar": 5, "arxiv": 3, "pubmed": 0},
        "safety_threshold": 80,
        "reason": "specific_paper: title 优先 + core_topic 补充",
    },
    "dataset_constraint": {
        "providers": _ALL_PROVIDERS,
        "routes": {
            "openalex": {"entity_dataset", "dataset", "core_topic", "title_like"},
            "semantic_scholar": {"entity_dataset", "dataset", "core_topic"},
            "arxiv": {"entity_dataset", "dataset", "core_topic"},
            "pasa_local": None,
            "pubmed": {"dataset", "method_task"},
        },
        "caps": {"openalex": 8, "semantic_scholar": 5, "arxiv": 3, "pubmed": 2},
        "safety_threshold": 120,
        "reason": "dataset_constraint: 聚焦 dataset/entity 路由",
    },
    "method_comparison": {
        "providers": _BASE_PLUS_ARXIV,
        "routes": {
            "openalex": {"method_task", "core_topic", "broad_synonym", "title_like"},
            "semantic_scholar": {"method_task", "core_topic", "broad_synonym"},
            "arxiv": {"method_task", "core_topic", "broad_synonym"},
            "pasa_local": None,
            "pubmed": set(),
        },
        "caps": {"openalex": 8, "semantic_scholar": 6, "arxiv": 3, "pubmed": 0},
        "safety_threshold": 150,
        "reason": "method_comparison: method_task 优先 + broad_synonym 扩覆盖",
    },
    "survey": {
        "providers": _ALL_PROVIDERS,
        "routes": {
            "openalex": {"broad_synonym", "core_topic", "method_task", "entity_dataset", "title_like", "survey"},
            "semantic_scholar": {"broad_synonym", "core_topic", "method_task", "title_like"},
            "arxiv": {"broad_synonym", "core_topic", "latest", "title_like"},
            "pasa_local": None,
            "pubmed": {"method_task", "dataset"},
        },
        "caps": {"openalex": 10, "semantic_scholar": 8, "arxiv": 5, "pubmed": 3},
        "safety_threshold": 250,
        "reason": "survey: 全路由广覆盖，高配额",
    },
    "broad_topic": {
        "providers": _ALL_PROVIDERS,
        "routes": {
            "openalex": {"broad_synonym", "core_topic", "method_task", "entity_dataset", "title_like"},
            "semantic_scholar": {"broad_synonym", "core_topic", "method_task", "title_like"},
            "arxiv": {"broad_synonym", "core_topic", "latest", "title_like"},
            "pasa_local": None,
            "pubmed": {"method_task", "dataset"},
        },
        "caps": {"openalex": 10, "semantic_scholar": 8, "arxiv": 5, "pubmed": 3},
        "safety_threshold": 250,
        "reason": "broad_topic: 全路由广覆盖，高配额",
    },
    "latest_work": {
        "providers": _BASE_PLUS_ARXIV,
        "routes": {
            "openalex": {"latest", "core_topic", "title_like"},
            "semantic_scholar": {"latest", "core_topic", "title_like"},
            "arxiv": {"latest", "core_topic", "title_like"},
            "pasa_local": None,
            "pubmed": set(),
        },
        "caps": {"openalex": 6, "semantic_scholar": 5, "arxiv": 5, "pubmed": 0},
        "safety_threshold": 100,
        "reason": "latest_work: latest 路由优先，arxiv 高配额",
    },
    "unknown": {
        "providers": _BASE_PLUS_ARXIV,
        "routes": {
            "openalex": {"core_topic", "broad_synonym", "title_like"},
            "semantic_scholar": {"core_topic", "broad_synonym", "title_like"},
            "arxiv": {"core_topic", "broad_synonym", "title_like"},
            "pasa_local": None,
            "pubmed": set(),
        },
        "caps": {"openalex": 8, "semantic_scholar": 5, "arxiv": 3, "pubmed": 0},
        "safety_threshold": 120,
        "reason": "unknown: 默认 core_topic + broad_synonym 均衡覆盖",
    },
}

# 默认 fallback 配置（当 query_type 不在表中时）
_DEFAULT_CONFIG = _ROUTING_TABLE["unknown"]


def _is_biomedical_topic(query_plan: QueryPlan) -> bool:
    text = " ".join(
        [
            query_plan.original_query or "",
            query_plan.research_topic or "",
            *query_plan.methods,
            *query_plan.datasets,
            *query_plan.entities,
        ]
    ).lower()
    biomedical_terms = {
        "antibody",
        "biomedical",
        "cancer",
        "clinical",
        "diagnosis",
        "disease",
        "drug",
        "gene",
        "lung cancer",
        "medical",
        "protein",
        "therapy",
        "treatment",
    }
    return any(term in text for term in biomedical_terms)


def get_routing_config(
    query_plan: QueryPlan,
    review_feedback: dict[str, Any] | None = None,
) -> RoutingConfig:
    """根据 query_type 返回动态路由配置。

    Args:
        query_plan: 当前查询计划
        review_feedback: 上一轮 LLM pool review 的反馈（可选），包含：
            - missing_aspects: list[str] 缺失的方面
            - noise_patterns: list[str] 噪声模式
            - kept_count: int 保留的论文数
            - target_size: int 目标池规模
    """
    qt = getattr(query_plan, "query_type", "unknown")
    base = _ROUTING_TABLE.get(qt, _DEFAULT_CONFIG)

    routes = {k: (set(v) if v is not None else None) for k, v in base["routes"].items()}
    caps = dict(base["caps"])
    safety_threshold = base["safety_threshold"]
    reasons = [base["reason"]]

    if qt == "latest_work" and _is_biomedical_topic(query_plan):
        routes["pubmed"] = {"core_topic", "method_task", "broad_synonym", "entity_dataset", "title_like"}
        caps["pubmed"] = max(caps.get("pubmed", 0), 3)
        reasons.append("biomedical latest_work: enable PubMed core routes")

    # 闭环调整：根据 review_feedback 动态调整
    if review_feedback:
        missing_aspects = review_feedback.get("missing_aspects", []) or []
        noise_patterns = review_feedback.get("noise_patterns", []) or []
        kept_count = review_feedback.get("kept_count", 0) or 0
        target_size = review_feedback.get("target_size", 100) or 100

        # 缺失方面 → 放宽配额，确保 evolved 路由能被接收
        if missing_aspects:
            for prov in caps:
                caps[prov] = min(caps[prov] + 2, 12)
            # 确保 evolved/feedback_expansion 路由被所有 provider 接收
            for prov in routes:
                if routes[prov] is not None:
                    routes[prov] = routes[prov] | {"evolved", "feedback_expansion", "translated", "query2doc", "hyde"}
            reasons.append(f"missing_aspects={len(missing_aspects)}: 放宽配额+追加 evolved 路由")

        # 噪声过多 → 收紧路由，移除 broad_synonym
        if noise_patterns and len(noise_patterns) >= 2:
            for prov in routes:
                if routes[prov] is not None and "broad_synonym" in routes[prov]:
                    routes[prov].discard("broad_synonym")
            reasons.append(f"noise_patterns={len(noise_patterns)}: 移除 broad_synonym 降噪")

        # 池子不足 → 降安全兜底阈值，触发更多兜底搜索
        if kept_count < target_size * 0.5:
            safety_threshold = max(60, safety_threshold - 40)
            for prov in caps:
                caps[prov] = min(caps[prov] + 1, 12)
            reasons.append(f"kept={kept_count}<{target_size*0.5:.0f}: 降阈值+加配额")

    return RoutingConfig(
        routes_per_provider=routes,
        caps_per_provider=caps,
        need_broad_safety=True,
        broad_safety_threshold=safety_threshold,
        reason=" | ".join(reasons),
    )
