from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

from scholar_agent.infra.config import LLMConfig
from scholar_agent.utils.json_repair import safe_json_loads
from scholar_agent.workflow.budget import BudgetExceededError, BudgetManager
from scholar_agent.workflow.llm_circuit_breaker import LLMCircuitBreaker


@dataclass
class LLMResponse:
    content: str
    elapsed_seconds: float
    token_estimate: int


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMConfig, budget: BudgetManager) -> None:
        self.config = config
        self.budget = budget
        self.session = requests.Session()
        self.session.trust_env = getattr(config, "trust_env", True)
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "scholar-agent/2.0.0",
                "Connection": "close",  # 显式关闭连接复用，防止 Keep-Alive 长链接被网关意外掐断
            }
        )
        # P2: Dynamic timeout override (set by pipeline Deadline) and circuit breaker
        self._timeout_override: float | None = None
        self._circuit_breaker: LLMCircuitBreaker | None = None

    @property
    def timeout(self) -> float | None:
        """Dynamic timeout override — set by pipeline to enforce Deadline."""
        return self._timeout_override

    @timeout.setter
    def timeout(self, value: float | None) -> None:
        self._timeout_override = value

    def set_circuit_breaker(self, breaker: LLMCircuitBreaker) -> None:
        """Attach a circuit breaker to track LLM call outcomes."""
        self._circuit_breaker = breaker

    @property
    def api_key(self) -> str:
        import os
        provider = os.getenv("ACTIVE_LLM_PROVIDER", "deepseek").strip().lower()
        if provider == "gpt55":
            return os.getenv("GPT55_API_KEY", "")
        if provider == "tokenhub":
            return os.getenv("TOKENHUB_API_KEY", "")
        return os.getenv(self.config.api_key_env, os.getenv("DEEPSEEK_API_KEY", ""))

    @property
    def base_url(self) -> str:
        import os
        provider = os.getenv("ACTIVE_LLM_PROVIDER", "deepseek").strip().lower()
        if provider == "gpt55":
            url = os.getenv("GPT55_BASE_URL", "https://cdn.coderelay.cn/v1")
            return url.rstrip("/")
        if provider == "tokenhub":
            url = os.getenv("TOKENHUB_BASE_URL", "https://api.tokenhub.market/v1")
            return url.rstrip("/")
        url = os.getenv(self.config.base_url_env, os.getenv("DEEPSEEK_BASE_URL", self.config.base_url))
        return url.rstrip("/")

    @property
    def model(self) -> str:
        import os
        provider = os.getenv("ACTIVE_LLM_PROVIDER", "deepseek").strip().lower()
        if provider == "gpt55":
            return os.getenv("GPT55_MODEL", "gpt-5.5")
        if provider == "tokenhub":
            return os.getenv("TOKENHUB_MODEL", "kimi-k2.7-code")
        return os.getenv("DEEPSEEK_MODEL", self.config.model)

    def is_available(self) -> bool:
        return self.config.enabled and self.config.mode != "off" and bool(self.api_key)

    def _endpoint(self) -> str:
        return self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"

    def _post_json(self, payload: dict[str, Any], timeout_seconds: float | None = None) -> dict[str, Any]:
        read_timeout = timeout_seconds if timeout_seconds is not None else self.config.timeout_seconds
        max_retries = getattr(self.config, "max_retries", 3)
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self._endpoint(),
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=(10, read_timeout),
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_exc = exc
                should_retry = False
                if isinstance(exc, requests.exceptions.HTTPError):
                    status_code = exc.response.status_code
                    if status_code == 429 or 500 <= status_code < 600:
                        should_retry = True
                elif isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                    should_retry = True
                
                if should_retry and attempt < max_retries - 1:
                    LOGGER.warning(f"LLM request transient error: {exc}. Retrying in 1s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(1)
                else:
                    raise exc
        if last_exc:
            raise last_exc

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def extract_json_payload(text: str) -> Any:
        candidate = text.strip()
        if not candidate:
            raise ValueError("empty llm response")

        fenced = candidate
        if "```" in candidate:
            parts = candidate.split("```")
            fenced_candidates = [part for part in parts if part.strip() and not part.strip().startswith("json")]
            if fenced_candidates:
                fenced = fenced_candidates[0].replace("json", "", 1).strip()
        for raw in (candidate, fenced):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

        starts = [index for index in (candidate.find("{"), candidate.find("[")) if index >= 0]
        if not starts:
            raise ValueError("no json block found")
        start = min(starts)
        end_obj = candidate.rfind("}")
        end_arr = candidate.rfind("]")
        end = max(end_obj, end_arr)
        if end <= start:
            raise ValueError("invalid json boundaries")
        return json.loads(candidate[start : end + 1])

    def complete_json(self, system_prompt: str, user_prompt: str, model_type: str = "flash", timeout_seconds: float | None = None, max_tokens: int | None = None) -> Any | None:
        # P2: Check circuit breaker first — if tripped, skip all LLM calls
        if self._circuit_breaker is not None and not self._circuit_breaker.can_call():
            LOGGER.debug("LLM circuit breaker tripped, skipping call")
            return None
        if not self.is_available():
            return None
        # P2: Use timeout_override from pipeline Deadline when explicit timeout_seconds not provided
        effective_timeout = timeout_seconds if timeout_seconds is not None else self._timeout_override
        try:
            self.budget.reserve_llm_call()
        except BudgetExceededError as exc:
            self.budget.record_error(str(exc))
            return None

        # 根据 model_type 选择 model，并根据 ACTIVE_LLM_PROVIDER 决定默认配置或覆盖
        import os
        provider = os.getenv("ACTIVE_LLM_PROVIDER", "deepseek").strip().lower()
        if provider == "gpt55":
            model_name = (
                os.getenv("GPT55_MODEL_PRO", "gpt-5.5")
                if model_type == "pro"
                else os.getenv("GPT55_MODEL_FLASH", "gpt-5.5")
            )
        elif provider == "tokenhub":
            model_name = (
                os.getenv("TOKENHUB_MODEL_PRO", "kimi-k2.7-code")
                if model_type == "pro"
                else os.getenv("TOKENHUB_MODEL_FLASH", "kimi-k2.7-code")
            )
        else:
            model_name = (
                os.getenv("DEEPSEEK_MODEL_PRO", self.config.model_pro)
                if model_type == "pro"
                else os.getenv("DEEPSEEK_MODEL_FLASH", self.config.model_flash)
            )

        # Effect-first: per-task max_tokens override
        effective_max_tokens = max_tokens if (max_tokens is not None and max_tokens > 0) else self.config.max_tokens

        payload = {
            "model": model_name,
            "temperature": self.config.temperature,
            "max_tokens": effective_max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        # tokenhub (kimi-k2.7-code) is a reasoning model — response_format: json_object
        # conflicts with its internal reasoning mode and causes silent timeouts.
        # Other providers (deepseek, gpt55) work fine with json_object.
        import os as _os
        _provider = _os.getenv("ACTIVE_LLM_PROVIDER", "deepseek").strip().lower()
        if _provider != "tokenhub":
            payload["response_format"] = {"type": "json_object"}
        started_at = time.perf_counter()
        try:
            raw = self._post_json(payload, timeout_seconds=effective_timeout)
        except Exception as exc:
            # P2: Record to circuit breaker
            if self._circuit_breaker is not None:
                if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                    self._circuit_breaker.record_timeout()
                else:
                    self._circuit_breaker.record_error(str(exc))

            # 仅在怀疑是格式不支持时才去掉 response_format 重试，其他网络错误直接认输，避免 2x3=6 次重试卡死
            is_format_error = False
            if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                if exc.response.status_code == 400:
                    is_format_error = True

            exc_str = str(exc).lower()
            if "response_format" in exc_str or "json_object" in exc_str or "json" in exc_str:
                is_format_error = True

            if is_format_error:
                try:
                    if "response_format" in payload:
                        del payload["response_format"]
                    raw = self._post_json(payload, timeout_seconds=effective_timeout)
                except Exception as retry_exc:
                    self.budget.record_error(f"llm request failed: {retry_exc} (orig: {exc})")
                    self.budget.record_llm_elapsed(time.perf_counter() - started_at)
                    return None
            else:
                self.budget.record_error(f"llm request failed: {exc}")
                self.budget.record_llm_elapsed(time.perf_counter() - started_at)
                return None

        elapsed = time.perf_counter() - started_at
        self.budget.record_llm_elapsed(elapsed)
        # P2: Record success to circuit breaker
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_success()

        # Extract content — reasoning models (kimi-k2.7-code) may put output in
        # reasoning_content or embed <思维> tags in content
        _msg = (raw.get("choices") or [{}])[0].get("message") or {}
        content = _msg.get("content") or _msg.get("text") or ""
        # Strip <思维>...</思维> tags that kimi-k2.7-code embeds in content
        if "<思维>" in content or "<思维 " in content:
            import re as _re
            content = _re.sub(r"<思维[^>]*>.*?</思维>", "", content, flags=_re.DOTALL).strip()
        # If content is empty after stripping, try reasoning_content (some providers
        # put the actual answer there for reasoning models)
        if not content:
            _rc = _msg.get("reasoning_content") or ""
            if _rc:
                # reasoning_content may itself contain <思维> tags — strip them
                if "<思维>" in _rc or "<思维 " in _rc:
                    import re as _re2
                    _rc = _re2.sub(r"<思维[^>]*>.*?</思维>", "", _rc, flags=_re2.DOTALL).strip()
                content = _rc

        usage = raw.get("usage") or {}
        token_estimate = int(
            usage.get("total_tokens")
            or self._estimate_tokens(system_prompt + user_prompt + content)
        )
        self.budget.record_token_estimate(token_estimate)
        # P2: Local JSON repair — no LLM repair calls (per optimization doc)
        result = safe_json_loads(content)
        if result is not None:
            return result

        # Fallback 1: Extract JSON block from mixed content (reasoning text + JSON)
        # kimi-k2.7-code returns reasoning text followed by JSON, e.g.:
        # "Let me analyze... The answer is:\n{"key": "value"}"
        # safe_json_loads fails on this because it starts with non-JSON text.
        # Fix: find the first '{' and last '}' and extract the JSON between them.
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            json_substr = content[first_brace:last_brace + 1]
            result = safe_json_loads(json_substr)
            if result is not None:
                return result

        # Fallback 2: try legacy extract_json_payload
        try:
            return self.extract_json_payload(content)
        except Exception as exc:
            self.budget.record_error(f"llm json parse failed: {exc}")
            return None

    def _repair_json_content(self, broken_content: str, model_name: str) -> Any | None:
        if not broken_content.strip():
            return None
        try:
            self.budget.reserve_llm_call()
        except BudgetExceededError as exc:
            self.budget.record_error(str(exc))
            return None

        payload = {
            "model": model_name,
            "temperature": 0.0,
            "max_tokens": min(2048, self.config.max_tokens),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Repair the user's broken JSON into valid JSON. "
                        "Return JSON only. Preserve the original structure and values as much as possible."
                    ),
                },
                {"role": "user", "content": broken_content},
            ],
        }
        # tokenhub reasoning model conflicts with response_format
        import os as _os2
        _provider2 = _os2.getenv("ACTIVE_LLM_PROVIDER", "deepseek").strip().lower()
        if _provider2 != "tokenhub":
            payload["response_format"] = {"type": "json_object"}
        started_at = time.perf_counter()
        try:
            raw = self._post_json(payload, timeout_seconds=10)  # P0-6: explicit 10s timeout for repair
        except Exception as exc:
            self.budget.record_error(f"llm json repair failed: {exc}")
            self.budget.record_llm_elapsed(time.perf_counter() - started_at)
            return None

        elapsed = time.perf_counter() - started_at
        self.budget.record_llm_elapsed(elapsed)
        _msg = (raw.get("choices") or [{}])[0].get("message") or {}
        content = _msg.get("content") or _msg.get("text") or ""
        # Strip <思维> tags for reasoning models
        if "<思维>" in content or "<思维 " in content:
            import re as _re3
            content = _re3.sub(r"<思维[^>]*>.*?</思维>", "", content, flags=_re3.DOTALL).strip()
        if not content:
            _rc = _msg.get("reasoning_content") or ""
            if _rc:
                if "<思维>" in _rc or "<思维 " in _rc:
                    import re as _re4
                    _rc = _re4.sub(r"<思维[^>]*>.*?</思维>", "", _rc, flags=_re4.DOTALL).strip()
                content = _rc
        usage = raw.get("usage") or {}
        token_estimate = int(
            usage.get("total_tokens")
            or self._estimate_tokens(content)
        )
        self.budget.record_token_estimate(token_estimate)
        try:
            return self.extract_json_payload(content)
        except Exception:
            return None


class MockLLMClient(OpenAICompatibleLLMClient):
    def __init__(self, budget: BudgetManager, responses: dict[str, Any] | None = None) -> None:
        super().__init__(LLMConfig(enabled=False, mode="off"), budget)
        self.responses = responses or {}

    def is_available(self) -> bool:
        return True

    def complete_json(self, system_prompt: str, user_prompt: str, model_type: str = "flash", timeout_seconds: float | None = None, max_tokens: int | None = None) -> Any | None:
        try:
            self.budget.reserve_llm_call()
        except BudgetExceededError as exc:
            self.budget.record_error(str(exc))
            return None
        started_at = time.perf_counter()
        key = "default"
        lowered = f"{system_prompt}\n{user_prompt}".lower()
        
        if "queryplan" in lowered:
            key = "query_understanding"
        elif "search_goal" in lowered or "round" in lowered:
            key = "search_planning"
        elif "coverage_analysis" in lowered or "resultreview" in lowered or "审阅" in user_prompt:
            key = "result_review"
        elif "relevance_level" in lowered or "evidence" in lowered:
            key = "evidence_selector"
        elif "highly_relevant_papers" in lowered or "structuredsynthesis" in lowered:
            key = "synthesis"

        # 如果预置了响应就直接返回，否则返回生成默认响应
        response = self.responses.get(key, self._get_default_mock_response(key, user_prompt))
        self.budget.record_llm_elapsed(time.perf_counter() - started_at)
        self.budget.record_token_estimate(self._estimate_tokens(system_prompt + user_prompt))
        return response

    def _get_default_mock_response(self, key: str, user_prompt: str) -> Any:
        if key == "query_understanding":
            return {
                "original_query": "mock query",
                "language": "zh",
                "research_topic": "Radiology Report Generation",
                "task": "find_relevant_papers",
                "methods": ["multimodal large language model", "vision-language model"],
                "datasets": ["MIMIC-CXR"],
                "entities": ["report generation"],
                "time_range": {"start_year": 2024, "end_year": 2026},
                "venues": ["MICCAI", "EMNLP"],
                "must_have_constraints": ["MIMIC-CXR"],
                "nice_to_have_constraints": ["multimodal large language model"],
                "exclude_terms": ["non-scholarly"],
                "expected_output": "top_papers",
                "uncertainty": []
            }
        elif key == "search_planning":
            return {
                "round": 1,
                "search_goal": "构建临床报告生成的候选集",
                "subqueries": [
                    {
                        "query": "Radiology Report Generation MIMIC-CXR",
                        "purpose": "检索核心数据集文献",
                        "sources": ["mock"],
                        "retrievers": ["BM25"]
                    }
                ],
                "expected_evidence": ["title contains report", "abstract mentions MIMIC-CXR"]
            }
        elif key == "result_review":
            return {
                "coverage_analysis": {
                    "covered_aspects": ["Radiology Report Generation"],
                    "missing_aspects": ["multimodal LLMs"],
                    "noise_patterns": []
                },
                "candidate_quality": {
                    "enough_candidates": True,
                    "hard_constraint_coverage": "high",
                    "precision_risk": "low",
                    "recall_risk": "low"
                },
                "next_action": "stop_search",
                "suggested_new_keywords": [],
                "suggested_excluded_terms": [],
                "need_citation_expansion": False,
                "reason": "已找到 MIMIC-CXR 临床报告生成的代表性论文。"
            }
        elif key == "evidence_selector":
            return {
                "selections": [
                    {
                        "paper_id": "paper-001",
                        "relevance_level": "high",
                        "matched_constraints": ["MIMIC-CXR"],
                        "missing_constraints": [],
                        "evidence": [
                            {"field": "abstract", "text": "multimodal large language models for radiology report generation"}
                        ],
                        "reason": "直接在 MIMIC-CXR 评估了多模态大模型，满足硬约束条件。",
                        "confidence": 0.98
                    },
                    {
                        "paper_id": "paper-002",
                        "relevance_level": "medium",
                        "matched_constraints": ["MIMIC-CXR"],
                        "missing_constraints": [],
                        "evidence": [
                            {"field": "title", "text": "Medical Image Report Generation"}
                        ],
                        "reason": "匹配医学图像报告生成与 MIMIC-CXR 数据集。",
                        "confidence": 0.85
                    }
                ]
            }
        elif key == "synthesis":
            return {
                "highly_relevant_papers": ["paper-001"],
                "partially_relevant_papers": ["paper-002"],
                "method_clusters": [
                    {"cluster_name": "Multimodal Radiology Report Generation", "paper_ids": ["paper-001", "paper-002"], "summary": "MIMIC-CXR models"}
                ],
                "timeline": [
                    {"year": 2025, "event": "Multimodal models peak"}
                ],
                "agent_self_report": {
                    "strategy_summary": "Retrieved MIMIC-CXR papers and filtered by constraints.",
                    "gaps_identified": [],
                    "refinement_history": ["Round 1: Initial retrieval"]
                }
            }
        return {"default": "mock"}
