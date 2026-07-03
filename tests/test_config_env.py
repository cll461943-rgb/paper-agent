import os

from scholar_agent.infra.config import _load_dotenv_if_present


def test_dotenv_does_not_override_existing_environment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ACTIVE_LLM_PROVIDER=tokenhub\n"
        "DEEPSEEK_BASE_URL=https://from-dotenv.example\n",
        encoding="utf-8",
    )

    previous_provider = os.environ.get("ACTIVE_LLM_PROVIDER")
    previous_base_url = os.environ.get("DEEPSEEK_BASE_URL")
    try:
        os.environ["ACTIVE_LLM_PROVIDER"] = "deepseek"
        os.environ.pop("DEEPSEEK_BASE_URL", None)

        _load_dotenv_if_present(env_file)

        assert os.environ["ACTIVE_LLM_PROVIDER"] == "deepseek"
        assert os.environ["DEEPSEEK_BASE_URL"] == "https://from-dotenv.example"
    finally:
        if previous_provider is None:
            os.environ.pop("ACTIVE_LLM_PROVIDER", None)
        else:
            os.environ["ACTIVE_LLM_PROVIDER"] = previous_provider
        if previous_base_url is None:
            os.environ.pop("DEEPSEEK_BASE_URL", None)
        else:
            os.environ["DEEPSEEK_BASE_URL"] = previous_base_url
