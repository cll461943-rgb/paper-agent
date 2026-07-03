from scholar_agent.infra.config import AppConfig
from scholar_agent.models.schemas import QueryPlan
from scholar_agent.planning.query_generation import generate_search_queries
from scholar_agent.retrieval.multi_route import MultiRouteRetriever
from scholar_agent.utils.title_query import looks_like_paper_title
from scholar_agent.workflow.budget import BudgetManager


class TitleFilteringLLM:
    def complete_json(self, system_prompt, user_prompt, model_type="flash", timeout_seconds=None, max_tokens=None):
        if "EXACT paper titles" in system_prompt:
            return {
                "title_queries": [
                    "How can large-scale language models improve automated legal",
                    "What improvements are needed in vaccine development efficiency",
                    "research papers from the past five years on",
                    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                    "Demonstrating quantum error correction that scales",
                ]
            }
        if "ALTERNATIVE" in system_prompt:
            return {"term_queries": []}
        return {"search_queries": []}


def test_title_candidate_filter_rejects_spar_question_fragments():
    assert not looks_like_paper_title("How can large-scale language models improve automated legal")
    assert not looks_like_paper_title("What improvements are needed in vaccine development efficiency")
    assert not looks_like_paper_title("research papers from the past five years on")
    assert looks_like_paper_title("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding")
    assert looks_like_paper_title("Demonstrating quantum error correction that scales")


def test_llm_title_queries_keep_real_titles_and_drop_question_fragments():
    config = AppConfig()
    plan = QueryPlan(
        original_query="How can large-scale language models improve automated legal reasoning?",
        language="en",
        research_topic="automated legal reasoning",
        methods=["large-scale language models"],
        entities=["legal reasoning"],
    )

    queries = generate_search_queries(plan, BudgetManager(config), llm_client=TitleFilteringLLM())
    title_queries = [query.query for query in queries if query.route == "title_like"]

    assert "How can large-scale language models improve automated legal" not in title_queries
    assert "What improvements are needed in vaccine development efficiency" not in title_queries
    assert "research papers from the past five years on" not in title_queries
    assert "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" in title_queries
    assert "Demonstrating quantum error correction that scales" in title_queries


def test_title_exact_gate_reuses_question_fragment_filter():
    assert not MultiRouteRetriever._looks_like_title_query("How can large-scale language models improve automated legal")
    assert not MultiRouteRetriever._looks_like_title_query("research papers from the past five years on")
    assert MultiRouteRetriever._looks_like_title_query("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding")
