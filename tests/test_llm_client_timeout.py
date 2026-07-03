from types import SimpleNamespace
import time

import requests

from scholar_agent.infra.llm_client import OpenAICompatibleLLMClient


class SlowTimeoutSession:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def post(self, *args, timeout=None, **kwargs):
        self.calls.append(timeout)
        time.sleep(0.03)
        raise requests.exceptions.Timeout("simulated read timeout")


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
