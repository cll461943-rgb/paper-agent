import pytest
from scholar_agent.infra.config import AppConfig
from scholar_agent.workflow.budget import BudgetManager
from scholar_agent.infra.llm_client import MockLLMClient
from scholar_agent.retrieval.factory import build_providers
from scholar_agent.workflow.pipeline import PaperAgentPipeline


def test_pipeline_workflow_mock():
    # 测试完整的 pipeline 工作流 (在 mock 模式下)
    # 构造默认配置
    config_dict = {
        "app": {
            "name": "scholar-agent-test",
            "version": "2.0.0",
            "mode": "mock",
            "cache_dir": ".cache_test",
            "providers": ["mock"],
            "max_retrieval_rounds": 2,
            "max_results_per_query": 5,
            "max_api_calls": 20,
            "max_llm_calls": 20,
        },
        "llm": {
            "enabled": True,
            "mode": "mock",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com",
            "model_flash": "deepseek-v4-flash",
            "model_pro": "deepseek-v4-pro",
            "temperature": 0.0,
            "max_tokens": 1024,
            "timeout_seconds": 30,
        },
        "providers": {
            "openalex": {"enabled": False, "email": "test@test.com"},
            "semantic_scholar": {"enabled": False},
            "arxiv": {"enabled": False},
            "pasa_local": {"enabled": False},
            "pubmed": {"enabled": False},
        }
    }
    
    # 我们构造一个简单的 AppConfig Mock 对象或者真实的 AppConfig
    # 由于 AppConfig 读 yaml 配置，我们可以直接实例化
    config = AppConfig()
    config.app.mode = "mock"
    config.app.providers = ["mock"]
    config.budget.max_retrieval_rounds = 2
    config.budget.max_results_per_query = 5
    config.budget.max_llm_calls = 20
    
    # 实例化 budget 和 mock LLM client
    budget = BudgetManager(config)
    llm_client = MockLLMClient(budget)
    
    # 实例化 providers
    providers = build_providers(config)
    
    # 运行 Pipeline
    pipeline = PaperAgentPipeline(config, llm_client, providers)
    result = pipeline.run("帮我检索2024-2026年期间大语言模型检索 Agent 相关的论文")
    
    # 验证最终结果
    assert result.original_query == "帮我检索2024-2026年期间大语言模型检索 Agent 相关的论文"
    assert result.query_plan is not None
    assert len(result.highly_relevant_papers) > 0 or len(result.partially_relevant_papers) > 0
    assert result.run_metrics.llm_calls_used > 0
    assert len(result.search_process) > 0
    
    # 验证引文图
    assert "nodes" in result.citation_graph
