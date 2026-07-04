from types import SimpleNamespace

from scholar_agent.infra.llm_client import OpenAICompatibleLLMClient


class RecordingResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "{\"ok\": true}"}}],
            "usage": {"total_tokens": 7},
        }


class RecordingSession:
    def __init__(self) -> None:
        self.headers = {}
        self.trust_env = True
        self.calls = []

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "headers": headers or {},
                "json": json,
                "timeout": timeout,
            }
        )
        return RecordingResponse()


class RecordingBudget:
    def __init__(self) -> None:
        self.llm_calls = 0
        self.errors = []
        self.elapsed = []
        self.tokens = []

    def reserve_llm_call(self):
        self.llm_calls += 1

    def record_error(self, message: str):
        self.errors.append(message)

    def record_llm_elapsed(self, elapsed: float):
        self.elapsed.append(elapsed)

    def record_token_estimate(self, tokens: int):
        self.tokens.append(tokens)


def make_config():
    return SimpleNamespace(
        trust_env=True,
        timeout_seconds=30,
        max_retries=1,
        enabled=True,
        mode="live",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        model_flash="deepseek-v4-flash",
        model_pro="deepseek-v4-pro",
        temperature=0.0,
        max_tokens=1024,
    )


def test_openai_provider_uses_official_key_pool_and_model_env(monkeypatch):
    monkeypatch.setenv("ACTIVE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEYS", "key-a, key-b")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.example/v1/")
    monkeypatch.setenv("OPENAI_MODEL_FLASH", "gpt-test-mini")
    monkeypatch.setenv("OPENAI_MODEL_PRO", "gpt-test-pro")

    budget = RecordingBudget()
    client = OpenAICompatibleLLMClient(make_config(), budget)
    session = RecordingSession()
    client.session = session

    result = client.complete_json("system", "user", model_type="flash")
    result_pro = client.complete_json("system", "user", model_type="pro")

    assert result == {"ok": True}
    assert result_pro == {"ok": True}
    assert [call["headers"]["Authorization"] for call in session.calls] == [
        "Bearer key-a",
        "Bearer key-b",
    ]
    assert [call["json"]["model"] for call in session.calls] == [
        "gpt-test-mini",
        "gpt-test-pro",
    ]
    assert {call["url"] for call in session.calls} == {
        "https://api.openai.example/v1/chat/completions"
    }


def test_openai_provider_requires_key(monkeypatch):
    monkeypatch.setenv("ACTIVE_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)

    client = OpenAICompatibleLLMClient(make_config(), RecordingBudget())

    assert client.is_available() is False


def test_deepseek_provider_preserves_config_base_url_env(monkeypatch):
    config = make_config()
    config.base_url_env = "CUSTOM_DEEPSEEK_BASE_URL"
    monkeypatch.setenv("ACTIVE_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("CUSTOM_DEEPSEEK_BASE_URL", "https://deepseek.example/v1")

    client = OpenAICompatibleLLMClient(config, RecordingBudget())

    assert client.base_url == "https://deepseek.example/v1"
