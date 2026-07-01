"""PromptRegistry — central registry for all PromptContracts.

Every LLM stage registers its PromptContract here. The registry provides:
- get(name): look up a contract by name
- list_contracts(): list all registered contracts (for auditing)
- render(name, **kwargs): get (system, user) prompts for a stage

This replaces the old approach where prompts were inline strings scattered
across 7+ Python files with no central management.
"""
from __future__ import annotations

import logging
from typing import Any

from scholar_agent.prompts.schemas import PromptContract, PromptVersion

LOGGER = logging.getLogger(__name__)


class PromptRegistry:
    """Central registry for all LLM prompt contracts in the pipeline."""

    _contracts: dict[str, PromptContract] = {}

    @classmethod
    def register(cls, contract: PromptContract) -> None:
        if contract.name in cls._contracts:
            LOGGER.debug("PromptRegistry: overwriting '%s'", contract.name)
        cls._contracts[contract.name] = contract

    @classmethod
    def get(cls, name: str) -> PromptContract | None:
        return cls._contracts.get(name)

    @classmethod
    def all_contracts(cls) -> dict[str, PromptContract]:
        return dict(cls._contracts)

    @classmethod
    def list_names(cls) -> list[str]:
        return list(cls._contracts.keys())

    @classmethod
    def summaries(cls) -> list[str]:
        return [c.summary() for c in cls._contracts.values()]


# ---------------------------------------------------------------------------
# Contract definitions — one per LLM stage
# ---------------------------------------------------------------------------

QUERY_UNDERSTANDING = PromptContract(
    name="query_understanding",
    stage="Stage 0: Query Understanding",
    objective="Parse natural language query into structured QueryPlan (hard/soft/negative/scope).",
    input_schema={"original_query": "str", "language": "str", "feedback": "optional dict for closed-loop"},
    output_schema={"query_type": "str", "research_topic": "str", "must_have_constraints": "list[str]",
                   "nice_to_have_constraints": "list[str]", "exclude_terms": "list[str]"},
    forbidden_actions=["Do NOT recommend specific papers", "Do NOT generate search queries"],
    failure_policy="Fall back to heuristic_understand_query (rule-based parsing).",
    prompt_version=PromptVersion(1, 0, 0),
    system_prompt_template=(
        "You are an expert academic search Query Understanding Agent. "
        "Parse a natural language research query into a structured QueryPlan. "
        "Extract: research topic, task, methods, datasets, entities, "
        "must-have constraints, nice-to-have constraints, exclude terms, time range. "
        "If Chinese, extract English equivalent terms. Return JSON only."
    ),
)

QUERY_GENERATION = PromptContract(
    name="query_generation",
    stage="Stage 2: Query Generation",
    objective="Generate structured search queries from QueryPlan.",
    input_schema={"query_plan": "QueryPlan", "original_query": "str", "round_idx": "int"},
    output_schema={"queries": "list of {query, route, intent, sources}"},
    forbidden_actions=["Do NOT make API queries too long (3-8 words)", "Do NOT ignore hard constraints"],
    failure_policy="Fall back to heuristic_generate_search_queries.",
    prompt_version=PromptVersion(1, 0, 0),
    system_prompt_template=(
        "You are a Search Query Generation Agent. Generate structured search queries for academic API "
        "and local vector search from a QueryPlan. Keep API queries 3-8 words. Return JSON only."
    ),
)

POOL_REVIEW = PromptContract(
    name="pool_review",
    stage="Stage 4: Non-destructive Pool Review",
    objective="Review candidate pool for coverage gaps. Tag papers keep/soft_drop/noise/discard. NEVER delete.",
    input_schema={"query_plan": "QueryPlan", "papers": "list[Paper] (batch)", "original_query": "str"},
    output_schema={"per_paper": "list of {paper_id, verdict, reason}",
                   "covered_aspects": "list[str]", "missing_aspects": "list[str]",
                   "coverage_score": "float", "converged": "bool"},
    forbidden_actions=[
        "Do NOT delete papers (non-destructive: only tag)",
        "Do NOT use 'discard' for topic mismatch (reserve for metadata violations)",
        "When uncertain, use 'keep' (RECALL-FIRST)",
    ],
    failure_policy="Keep all papers as 'keep' (non-destructive default).",
    prompt_version=PromptVersion(1, 0, 0),
    system_prompt_template=(
        "You are a Pool Reviewer. Review candidate papers with RECALL-FIRST strategy. "
        "Tag each: keep/soft_drop/noise/discard. Reserve 'discard' for metadata violations ONLY. "
        "NEVER delete papers — only tag. Return JSON only."
    ),
)

EVIDENCE_SELECTION = PromptContract(
    name="evidence_selection",
    stage="Stage 7: Evidence Selection",
    objective="Judge paper relevance with evidence grounding. Output relevance_level, probability, evidence.",
    input_schema={"query_plan": "QueryPlan", "papers": "list[Paper] (batch)", "original_query": "str"},
    output_schema={"results": "list of {paper_id, relevance_level, relevance_probability, evidence, reason}"},
    forbidden_actions=[
        "Do NOT use listwise_score from a later stage (no future-feature dependency)",
        "Do NOT fabricate evidence (must come from title/abstract/metadata)",
        "Do NOT rank papers (that's Listwise Reranker's job)",
    ],
    failure_policy="Fall back to local_score_paper (heuristic scoring).",
    prompt_version=PromptVersion(1, 0, 0),
    model_type="flash",
    system_prompt_template=(
        "You are an Evidence Selection Agent. Judge each paper's relevance with evidence from "
        "title/abstract/metadata. Output relevance_level, probability, evidence, reason. "
        "RECALL-FIRST: prefer 'medium' over 'low'. Return JSON only."
    ),
)

LISTWISE_RERANK = PromptContract(
    name="listwise_rerank",
    stage="Stage 9: Listwise Reranking",
    objective="Rank papers by 'probability of being gold-standard relevant' — NOT by popularity.",
    input_schema={"query_plan": "QueryPlan", "papers": "list[Paper] (top candidates)", "original_query": "str"},
    output_schema={"ranked_papers": "list of {paper_id, rank, listwise_score, relevance_probability}",
                   "recommended_k_min": "int", "recommended_k_max": "int"},
    forbidden_actions=[
        "Do NOT rank by citation count, fame, or reputation",
        "Do NOT output more papers than provided",
    ],
    failure_policy="Fall back to local_pre_rank_score ordering.",
    prompt_version=PromptVersion(1, 0, 0),
    system_prompt_template=(
        "You are a Listwise Reranker. Rank papers by probability of being gold-standard relevant. "
        "NOT by popularity or citations. Output listwise_score, relevance_probability, relevance_level. "
        "Return JSON only."
    ),
)

SYNTHESIS = PromptContract(
    name="synthesis",
    stage="Stage 11: Structured Synthesis",
    objective="Organize papers into method_clusters, timeline, self-report. NO paper changes.",
    input_schema={"recommended_papers": "list[Paper]", "query_plan": "QueryPlan", "search_rounds": "list"},
    output_schema={"method_clusters": "list", "timeline": "list", "agent_self_report": "dict"},
    forbidden_actions=[
        "Do NOT add new papers",
        "Do NOT delete or remove papers",
        "Do NOT rerank or change paper order",
        "Do NOT reference paper_ids not in the provided list",
    ],
    failure_policy="Fall back to _create_fallback_synthesis (heuristic).",
    prompt_version=PromptVersion(1, 0, 0),
    model_type="pro",
    system_prompt_template=(
        "You are a Synthesis Agent. Organize papers into method_clusters and timeline. "
        "CRITICAL: You must NOT add, delete, or rerank any papers. The paper set is FROZEN. "
        "Return JSON only."
    ),
)

HYDE = PromptContract(
    name="hyde",
    stage="Stage 5.5: HyDE Generation (disabled)",
    objective="Generate hypothetical paper abstract for semantic bridge.",
    input_schema={"original_query": "str", "query_plan": "QueryPlan"},
    output_schema={"hypothetical_abstract": "str"},
    forbidden_actions=["Do NOT cite real papers", "Do NOT use vague language"],
    failure_policy="Fall back to original query (HyDE disabled in current config).",
    prompt_version=PromptVersion(1, 0, 0),
    system_prompt_template=(
        "Generate a hypothetical paper abstract (3-4 sentences) that answers the research question. "
        "Use precise academic terminology. Return JSON: {\"hypothetical_abstract\": \"...\"}"
    ),
)

K_CONTROLLER = PromptContract(
    name="k_controller",
    stage="Stage 12: Expected-Fβ Output Controller",
    objective="Estimate relevant set size (g_hat) and recommend K range. Does NOT directly decide K.",
    input_schema={"query_type": "str", "candidate_probabilities": "list[float]", "pool_size": "int"},
    output_schema={"g_hat_estimate": "int", "k_min": "int", "k_max": "int", "confidence": "float"},
    forbidden_actions=["Do NOT directly set final K", "Do NOT use gold answer information"],
    failure_policy="Fall back to rule-based g_hat from query_type profile.",
    prompt_version=PromptVersion(1, 0, 0),
    system_prompt_template=(
        "You are a K Controller. Estimate the number of relevant papers (g_hat) and recommend K range. "
        "Return JSON: {\"g_hat_estimate\": int, \"k_min\": int, \"k_max\": int, \"confidence\": float}."
    ),
)

# Register all contracts
for _c in [QUERY_UNDERSTANDING, QUERY_GENERATION, POOL_REVIEW, EVIDENCE_SELECTION,
           LISTWISE_RERANK, SYNTHESIS, HYDE, K_CONTROLLER]:
    PromptRegistry.register(_c)


def get_prompt(name: str, **kwargs: Any) -> tuple[str, str] | None:
    return PromptRegistry.get(name).render(**kwargs) if PromptRegistry.get(name) else None


def list_contracts() -> list[str]:
    return PromptRegistry.summaries()
