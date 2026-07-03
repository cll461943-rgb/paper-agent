from types import SimpleNamespace
import os
import time

import requests

from scholar_agent.infra.llm_client import OpenAICompatibleLLMClient
from scholar_agent.workflow.llm_circuit_breaker import LLMCircuitBreaker


class SlowTimeoutSession:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def post(self, *args, timeout=None, **kwargs):
        self.calls.append(timeout)
        time.sleep(0.03)
        raise requests.exceptions.Timeout("simulated read timeout")


class ForbiddenResponse:
    status_code = 403

    def raise_for_status(self):
        raise requests.exceptions.HTTPError("403 Client Error: Forbidden", response=self)


class ForbiddenSession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return ForbiddenResponse()


class RecordingBudget:
    def __init__(self) -> None:
        self.llm_calls = 0
        self.errors: list[str] = []
        self.elapsed: list[float] = []

    def reserve_llm_call(self):
        self.llm_calls += 1

    def record_error(self, message: str):
        self.errors.append(message)

    def record_llm_elapsed(self, elapsed: float):
        self.elapsed.append(elapsed)


def test_post_json_timeout_seconds_is_total_retry_budget():
    config = SimpleNamespace(
        trust_env=True,
        timeout_seconds=30,
        max_retries=3,
        enabled=True,
        mode="live",
        api_key_env="NO_API_KEY",
        base_url_env="NO_BASE_URL",
        base_url="https://example.invalid/v1",
        model="mock",
        model_flash="mock-flash",
        model_pro="mock-pro",
        temperature=0.0,
        max_tokens=1024,
    )
    client = OpenAICompatibleLLMClient(config, SimpleNamespace())
    fake_session = SlowTimeoutSession()
    client.session = fake_session

    try:
        client._post_json({"messages": []}, timeout_seconds=0.01)
    except requests.exceptions.Timeout:
        pass
    else:
        raise AssertionError("expected timeout")

    assert len(fake_session.calls) == 1
    connect_timeout, read_timeout = fake_session.calls[0]
    assert 0 < connect_timeout <= 0.02
    assert 0 < read_timeout <= 0.02


def test_circuit_breaker_permanent_error_survives_phase_reset():
    breaker = LLMCircuitBreaker(max_errors=5)

    breaker.record_permanent_error("403 Forbidden")

    assert not breaker.can_call()
    assert breaker.get_stats()["tripped"] is True
    assert breaker.get_stats()["permanent"] is True

    breaker.reset_transient()

    assert not breaker.can_call()
    assert breaker.get_stats()["reason"] == "permanent_error"


def test_llm_client_403_trips_circuit_breaker_permanently():
    config = SimpleNamespace(
        trust_env=True,
        timeout_seconds=30,
        max_retries=1,
        enabled=True,
        mode="live",
        api_key_env="NO_API_KEY",
        base_url_env="NO_BASE_URL",
        base_url="https://example.invalid/v1",
        model="mock",
        model_flash="mock-flash",
        model_pro="mock-pro",
        temperature=0.0,
        max_tokens=1024,
    )
    budget = RecordingBudget()
    client = OpenAICompatibleLLMClient(config, budget)
    client.session = ForbiddenSession()
    breaker = LLMCircuitBreaker(max_errors=5)
    client.set_circuit_breaker(breaker)

    previous_provider = os.environ.get("ACTIVE_LLM_PROVIDER")
    previous_key = os.environ.get("NO_API_KEY")
    previous_base_url = os.environ.get("NO_BASE_URL")
    os.environ["ACTIVE_LLM_PROVIDER"] = "deepseek"
    os.environ["NO_API_KEY"] = "test-key"
    os.environ["NO_BASE_URL"] = "https://example.invalid/v1"
    try:
        result = client.complete_json("system", "user")

        assert result is None
        assert client.session.calls == 1
        assert breaker.get_stats()["tripped"] is True
        assert breaker.get_stats()["permanent"] is True
        assert breaker.get_stats()["errors"] == 1

        result = client.complete_json("system", "user")

        assert result is None
        assert client.session.calls == 1
    finally:
        if previous_provider is None:
            os.environ.pop("ACTIVE_LLM_PROVIDER", None)
        else:
            os.environ["ACTIVE_LLM_PROVIDER"] = previous_provider
        if previous_key is None:
            os.environ.pop("NO_API_KEY", None)
        else:
            os.environ["NO_API_KEY"] = previous_key
        if previous_base_url is None:
            os.environ.pop("NO_BASE_URL", None)
        else:
            os.environ["NO_BASE_URL"] = previous_base_url
