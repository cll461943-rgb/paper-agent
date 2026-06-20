from datetime import date
from scholar_agent.evaluation.pasa import (
    cutoff_from_source_meta,
    gold_coverage,
    filter_by_cutoff,
    keep_letters,
    load_pasa_jsonl,
    normalize_arxiv_id,
    recall_at_k,
    score_pasa,
)
from scholar_agent.infra.config import AppConfig
from scholar_agent.models.schemas import Paper
from scholar_agent.models.schemas import QueryPlan
from scholar_agent.models.schemas import RunMetrics
from scholar_agent.models.schemas import SearchProcessRound
from scholar_agent.models.schemas import WorkflowResult
from scholar_agent.planning.query_generation import generate_search_queries, heuristic_generate_search_queries
from scholar_agent.workflow.budget import BudgetManager
from evaluate_pasa import _apply_budget_overrides, _build_optional_llm_client, _result_search_queries, _select_case_groups
from evaluate_pasa import _prepare_config


def test_keep_letters_matches_pasa_title_normalization():
    assert keep_letters("BART: Denoising Sequence-to-Sequence Pre-training!") == "bartdenoisingsequencetosequencepretraining"


def test_score_pasa_uses_keep_letters_and_arxiv_id():
    papers = [
        Paper(paper_id="x", title="Attention Is All You Need"),
        Paper(paper_id="ARXIV:1706.03762v2", title="Different Parsed Title"),
        Paper(paper_id="noise", title="Unrelated Paper"),
    ]
    gold = [
        "attention-is all you need",
        {"arxiv_id": "1706.03762", "title": ""},
    ]
    scores = score_pasa(papers, gold)
    assert scores["true_positive"] == 2
    assert scores["false_positive"] == 1
    assert scores["recall"] == 1.0


def test_score_pasa_dedupes_duplicate_predictions():
    papers = [
        Paper(paper_id="p1", title="Attention Is All You Need"),
        Paper(paper_id="p2", title="Attention is all you need"),
    ]
    scores = score_pasa(papers, ["Attention Is All You Need"])
    assert scores["true_positive"] == 1
    assert scores["false_positive"] == 0
    assert scores["predicted_count"] == 1


def test_recall_at_k_is_macro_ready_per_case_value():
    papers = [
        Paper(paper_id="noise", title="Unrelated"),
        Paper(paper_id="p1", title="Gold Title"),
    ]
    assert recall_at_k(papers, ["Gold Title"], 1) == 0.0
    assert recall_at_k(papers, ["Gold Title"], 2) == 1.0


def test_filter_by_cutoff_removes_future_papers_but_keeps_unknown_dates():
    papers = [
        Paper(paper_id="old", title="Old", year=2022),
        Paper(paper_id="future", title="Future", year=2025),
        Paper(paper_id="unknown", title="Unknown"),
    ]
    filtered = filter_by_cutoff(papers, date(2024, 1, 1))
    assert [paper.paper_id for paper in filtered] == ["old", "unknown"]


def test_filter_by_cutoff_uses_exact_published_time_metadata():
    papers = [
        Paper(paper_id="old", title="Old", metadata={"published_time": "2024-02-20"}),
        Paper(paper_id="future", title="Future", metadata={"published_time": "2024-02-21T10:00:00Z"}),
    ]
    filtered = filter_by_cutoff(papers, date(2024, 2, 20))
    assert [paper.paper_id for paper in filtered] == ["old"]


def test_cutoff_from_source_meta_subtracts_default_pasa_window():
    cutoff = cutoff_from_source_meta({"published_time": "20240301"})
    assert cutoff == date(2024, 2, 23)


def test_load_pasa_jsonl_reads_question_answers_and_source_meta(tmp_path):
    dataset = tmp_path / "pasa.jsonl"
    dataset.write_text(
        '{"question":"q","answer":["Gold"],"source_meta":{"published_time":"2024-03"}}\n',
        encoding="utf-8",
    )
    cases = load_pasa_jsonl(dataset)
    assert cases[0].question == "q"
    assert cases[0].answer == ["Gold"]
    assert cases[0].published_time == date(2024, 3, 28)


def test_score_pasa_matches_official_empty_gold_behavior():
    scores = score_pasa([Paper(paper_id="p1", title="Any Paper")], [])
    assert scores["true_positive"] == 0
    assert scores["false_positive"] == 0
    assert scores["false_negative"] == 0
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0


def test_score_pasa_dedupes_same_gold_matched_by_title_and_arxiv_id():
    papers = [
        Paper(paper_id="p1", title="Attention Is All You Need"),
        Paper(paper_id="ARXIV:1706.03762v2", title="Different Parsed Title"),
    ]
    gold = [{"title": "Attention Is All You Need", "arxiv_id": "1706.03762"}]
    scores = score_pasa(papers, gold)
    assert scores["true_positive"] == 1
    assert scores["false_positive"] == 0


def test_paper_match_prefers_metadata_arxiv_id_when_title_differs():
    scores = score_pasa(
        [Paper(paper_id="paper-x", title="Different", arxiv_id="1706.03762v2")],
        [{"title": "Attention Is All You Need", "arxiv_id": "1706.03762"}],
    )
    assert scores["true_positive"] == 1


def test_gold_coverage_reports_candidate_rank():
    report = gold_coverage(
        [
            Paper(paper_id="p1", title="Noise"),
            Paper(paper_id="p2", title="Attention Is All You Need"),
        ],
        [{"title": "Attention Is All You Need"}],
    )
    assert report[0]["matched"] is True
    assert report[0]["candidate_rank"] == 2


def test_normalize_arxiv_id_strips_versions_and_urls():
    assert normalize_arxiv_id("https://arxiv.org/pdf/1706.03762v5.pdf") == "1706.03762"


def test_known_title_clues_disabled_for_heuristic_and_budget_paths():
    plan = QueryPlan(
        original_query="I am looking to understand more about denoising sequence-to-sequence pre-training.",
        language="en",
        methods=["denoising", "sequence-to-sequence", "pre-training"],
    )
    canonical_title = "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"

    heuristic_queries = heuristic_generate_search_queries(plan, enabled=False)
    assert canonical_title not in [item.query for item in heuristic_queries]

    config = AppConfig()
    config.known_title_clues.enabled = False
    budget_queries = generate_search_queries(plan, BudgetManager(config), llm_client=None)
    assert canonical_title not in [item.query for item in budget_queries]


def test_evaluate_pasa_can_disable_llm_runtime():
    config = AppConfig()
    assert _build_optional_llm_client("live", config, disable_llm=True) is None
    assert _build_optional_llm_client("live", config, disable_llm=False) is not None


def test_evaluate_pasa_selects_reproducible_budgeted_sample_groups():
    dataset = list(range(20))
    first = _select_case_groups(dataset, sample_size=5, sample_runs=4, random_seed=123)
    second = _select_case_groups(dataset, sample_size=5, sample_runs=4, random_seed=123)

    assert first == second
    assert len(first) == 4
    assert all(len(group) == 5 for group in first)
    assert all(len(set(group)) == 5 for group in first)


def test_evaluate_pasa_applies_budget_overrides():
    config = AppConfig()
    _apply_budget_overrides(
        config,
        max_llm_calls=2,
        max_api_calls=8,
        max_search_queries=6,
        max_retrieval_rounds=1,
        max_results_per_query=12,
        max_seed_papers=4,
    )

    assert config.budget.max_llm_calls == 2
    assert config.budget.max_api_calls == 8
    assert config.budget.max_search_queries == 6
    assert config.budget.max_retrieval_rounds == 1
    assert config.budget.max_results_per_query == 12
    assert config.budget.max_seed_papers == 4


def test_evaluate_pasa_records_search_queries_from_result_rounds():
    result = WorkflowResult(
        original_query="q",
        query_plan=QueryPlan(original_query="q"),
        search_process=[
            SearchProcessRound(round_index=1, search_goal="g", queries=["alpha", "beta"]),
            SearchProcessRound(round_index=2, search_goal="g", queries=["gamma"]),
        ],
        run_metrics=RunMetrics(),
    )

    assert _result_search_queries(result) == ["alpha", "beta", "gamma"]


def test_evaluate_pasa_can_enable_pasa_local_fts():
    class Args:
        config = None
        mode = "live"
        parallel_retrieval = False
        enable_known_title_clues = False
        enable_pasa_local_fts = True
        providers = "pasa_local"
        max_llm_calls = None
        max_api_calls = None
        max_search_queries = None
        max_retrieval_rounds = None
        max_results_per_query = None
        max_seed_papers = None

    config, _ = _prepare_config(Args())

    assert config.providers.pasa_local.enable_fts is True
