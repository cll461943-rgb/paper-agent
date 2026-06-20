from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests
from urllib3.util import connection

_original_create_connection = connection.create_connection

def patched_create_connection(address, *args, **kwargs):
    host, port = address
    if host == "api.deepseek.com":
        return _original_create_connection(("3.173.21.63", port), *args, **kwargs)
    return _original_create_connection(address, *args, **kwargs)

connection.create_connection = patched_create_connection

from scholar_agent.infra.config import LLMConfig
from scholar_agent.workflow.budget import BudgetExceededError, BudgetManager


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
        self.session.trust_env = False  # 直连国内 DeepSeek 接口，不走代理以规避 SSL 握手冲突
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "scholar-agent/2.0.0",
                "Connection": "close",  # 显式关闭连接复用，防止 Keep-Alive 长链接被网关意外掐断
            }
        )

    @property
    def api_key(self) -> str:
        import os
        return os.getenv(self.config.api_key_env, os.getenv("DEEPSEEK_API_KEY", ""))

    @property
    def base_url(self) -> str:
        import os
        url = os.getenv(self.config.base_url_env, os.getenv("DEEPSEEK_BASE_URL", self.config.base_url))
        return url.rstrip("/")

    @property
    def model(self) -> str:
        import os
        # 允许直接读取 model
        return os.getenv("DEEPSEEK_MODEL", self.config.model)

    def is_available(self) -> bool:
        return self.config.enabled and self.config.mode != "off" and bool(self.api_key)

    def _endpoint(self) -> str:
        return self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"

    def _post_json(self, payload: dict[str, Any], timeout_seconds: float | None = None) -> dict[str, Any]:
        read_timeout = timeout_seconds if timeout_seconds is not None else self.config.timeout_seconds
        max_retries = 3
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

    def complete_json(self, system_prompt: str, user_prompt: str, model_type: str = "flash", timeout_seconds: float | None = None) -> Any | None:
        if not self.is_available():
            return None
        try:
            self.budget.reserve_llm_call()
        except BudgetExceededError as exc:
            self.budget.record_error(str(exc))
            return None

        # 根据 model_type 选择 deepseek-v4-flash 还是 pro，并允许环境变量覆盖
        import os
        model_name = (
            os.getenv("DEEPSEEK_MODEL_PRO", self.config.model_pro)
            if model_type == "pro"
            else os.getenv("DEEPSEEK_MODEL_FLASH", self.config.model_flash)
        )

        payload = {
            "model": model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        started_at = time.perf_counter()
        try:
            raw = self._post_json(payload, timeout_seconds=timeout_seconds)
        except Exception as exc:
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
                    raw = self._post_json(payload, timeout_seconds=timeout_seconds)
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
        content = (
            ((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ((raw.get("choices") or [{}])[0].get("text"))
            or ""
        )
        usage = raw.get("usage") or {}
        token_estimate = int(
            usage.get("total_tokens")
            or self._estimate_tokens(system_prompt + user_prompt + content)
        )
        self.budget.record_token_estimate(token_estimate)
        try:
            return self.extract_json_payload(content)
        except Exception as exc:
            repaired = self._repair_json_content(content, model_name)
            if repaired is not None:
                return repaired
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
            "response_format": {"type": "json_object"},
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
        started_at = time.perf_counter()
        try:
            raw = self._post_json(payload)
        except Exception as exc:
            self.budget.record_error(f"llm json repair failed: {exc}")
            self.budget.record_llm_elapsed(time.perf_counter() - started_at)
            return None

        elapsed = time.perf_counter() - started_at
        self.budget.record_llm_elapsed(elapsed)
        content = (
            ((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ((raw.get("choices") or [{}])[0].get("text"))
            or ""
        )
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

    def complete_json(self, system_prompt: str, user_prompt: str, model_type: str = "flash", timeout_seconds: float | None = None) -> Any | None:
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
