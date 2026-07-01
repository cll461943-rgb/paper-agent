"""Prompt Control Layer — centralized prompt management with PromptContract.

This module provides the registry and contract infrastructure for all LLM
prompts in the Scholar Agent pipeline. Each LLM stage has a PromptContract
that defines its objective, input/output schema, forbidden actions, and
failure policy.

Usage:
    from scholar_agent.prompts import PromptRegistry, PromptContract

    # Look up a contract
    contract = PromptRegistry.get("pool_review")
    print(contract.summary())
    print(f"Forbidden: {contract.forbidden_actions}")

    # List all contracts
    for name, contract in PromptRegistry.all_contracts().items():
        print(contract.summary())

Contracts defined:
    query_understanding — Stage 0
    query_generation    — Stage 2
    pool_review         — Stage 4 (non-destructive)
    evidence_selection  — Stage 7
    listwise_rerank     — Stage 9
    synthesis           — Stage 11 (no add/delete/rerank)
    hyde                — Stage 5.5 (disabled)
    k_controller        — Stage 12 (Expected-Fβ)
"""
from scholar_agent.prompts.schemas import PromptContract, PromptVersion
from scholar_agent.prompts.registry import (
    PromptRegistry,
    QUERY_UNDERSTANDING,
    QUERY_GENERATION,
    POOL_REVIEW,
    EVIDENCE_SELECTION,
    LISTWISE_RERANK,
    SYNTHESIS,
    HYDE,
    K_CONTROLLER,
)

__all__ = [
    "PromptContract",
    "PromptVersion",
    "PromptRegistry",
    "QUERY_UNDERSTANDING",
    "QUERY_GENERATION",
    "POOL_REVIEW",
    "EVIDENCE_SELECTION",
    "LISTWISE_RERANK",
    "SYNTHESIS",
    "HYDE",
    "K_CONTROLLER",
]
