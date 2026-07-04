from __future__ import annotations

from scholar_agent.infra.config import load_config
from scholar_agent.retrieval import build_providers


def test_missing_config_falls_back_to_project_default():
    config = load_config("missing.yaml")

    assert config.app.mode


def test_mock_provider_builds_without_optional_faiss():
    config = load_config("configs/default.yaml")
    config.app.mode = "mock"
    config.app.providers = ["mock"]

    providers = build_providers(config)

    assert [provider.name for provider in providers] == ["mock"]
