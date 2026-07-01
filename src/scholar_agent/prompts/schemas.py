"""PromptContract — structured contract for every LLM prompt in the pipeline.

Each LLM stage (query understanding, pool review, evidence selection,
listwise rerank, synthesis) has a PromptContract that defines:

1. Objective: what the LLM should accomplish
2. Input schema: what fields the prompt expects
3. Output schema: what JSON structure the LLM must return
4. Forbidden actions: what the LLM must NOT do
5. Failure policy: what the pipeline does when the LLM fails

This replaces the old approach where prompts were scattered as inline
string literals across 7+ Python files with no version tracking or
constraint enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PromptVersion:
    """Semantic version for prompt contracts."""
    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class PromptContract:
    """Contract for a single LLM prompt stage."""

    name: str                           # unique identifier (e.g. "pool_review")
    stage: str                          # pipeline stage (e.g. "Stage 4")
    objective: str                      # what the LLM should do
    input_schema: dict[str, str]        # expected input fields: {field_name: description}
    output_schema: dict[str, str]       # expected output fields: {field_name: type_description}
    forbidden_actions: list[str]        # what the LLM must NOT do
    failure_policy: str                 # what to do when LLM fails (e.g. "keep all papers")
    prompt_version: PromptVersion = field(default_factory=PromptVersion)  # version tracking
    model_type: str = "flash"           # recommended model type: "flash" or "pro"
    system_prompt_template: str = ""    # system prompt with {placeholders}
    user_prompt_template: str = ""      # user prompt template with {placeholders}

    # Rendering function: builds (system_prompt, user_prompt) from kwargs
    # Each stage module sets this to its _build_*_prompt function
    _render_fn: Callable[..., tuple[str, str]] | None = field(default=None, repr=False)

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Build the actual (system_prompt, user_prompt) for this contract.

        If a render function is registered, use it. Otherwise, fall back to
        simple {placeholder} substitution in the templates.
        """
        if self._render_fn is not None:
            return self._render_fn(**kwargs)
        # Fallback: simple string substitution
        system = self.system_prompt_template
        user = self.user_prompt_template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in system:
                system = system.replace(placeholder, str(value))
            if placeholder in user:
                user = user.replace(placeholder, str(value))
        return system, user

    def set_render(self, fn: Callable[..., tuple[str, str]]) -> None:
        """Register the rendering function for this contract."""
        self._render_fn = fn

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"[{self.name}] stage={self.stage} v{self.prompt_version} "
            f"model={self.model_type} | objective: {self.objective[:80]}"
        )
