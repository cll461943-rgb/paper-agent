import pytest
from scholar_agent.models.schemas import Paper, SelectionResult, EvidenceItem, RankedPaper, QueryPlan, SearchProcessRound, RunMetrics
from scholar_agent.synthesis.synthesis_agent import SynthesisAgent


class MockLLMClient:
    def complete_json(self, system_prompt, user_prompt, model_type="pro"):
        return {
            "method_clusters": [
                {"cluster_name": "Agent Retrieval", "paper_ids": ["p1"]}
            ],
            "timeline": [
                {"year": 2026, "event": "Agent Search peak"}
            ],
            "agent_self_report": {
                "strategy_summary": "Successful multi-round expansion.",
                "gaps_identified": ["None"]
            }
        }


def test_synthesis_agent_grouping_and_graph():
    # 测试分组和引文图构建
    p1 = Paper(
        paper_id="p1",
        title="Paper A",
        year=2026,
        references=["p2"],
    )
    s1 = SelectionResult(
        paper_id="p1",
        relevance_level="high",
        reason="Perfect match",
    )
    rp1 = RankedPaper(paper=p1, selection=s1, final_score=0.9, rank=1)

    p2 = Paper(
        paper_id="p2",
        title="Paper B",
        year=2025,
    )
    s2 = SelectionResult(
        paper_id="p2",
        relevance_level="medium",
        reason="Partial match",
    )
    rp2 = RankedPaper(paper=p2, selection=s2, final_score=0.8, rank=2)

    p3 = Paper(
        paper_id="p3",
        title="Paper C",
        year=2024,
    )
    s3 = SelectionResult(
        paper_id="p3",
        relevance_level="low",
        reason="Poor match",
    )
    rp3 = RankedPaper(paper=p3, selection=s3, final_score=0.3, rank=3)

    plan = QueryPlan(original_query="scholarly retrieval")
    rounds = [SearchProcessRound(round_index=1, search_goal="Initial search")]
    metrics = RunMetrics()

    agent = SynthesisAgent(llm_client=MockLLMClient())
    result = agent.synthesize(
        original_query="scholarly retrieval",
        query_plan=plan,
        search_rounds=rounds,
        ranked_papers=[rp1, rp2, rp3],
        metrics=metrics
    )

    # 验证分类
    assert len(result.highly_relevant_papers) == 1
    assert result.highly_relevant_papers[0].paper.paper_id == "p1"
    assert len(result.partially_relevant_papers) == 1
    assert result.partially_relevant_papers[0].paper.paper_id == "p2"

    # 验证 citation_graph 生成 (p1 -> p2 有引用关系)
    graph = result.citation_graph
    assert "nodes" in graph
    assert "links" in graph
    # 包含了 p1 和 p2，p3 虽是 low 但也在重排列表中（不过由于合成被过滤，nodes 里可能只包含被推荐的 highly/partially 的 nodes，或者全候选集的 nodes。我们将在代码中使其只包含推荐文章）
    node_ids = {n["id"] for n in graph["nodes"]}
    assert "p1" in node_ids
    assert "p2" in node_ids
    assert len(graph["links"]) == 1
    assert graph["links"][0]["source"] == "p1"
    assert graph["links"][0]["target"] == "p2"
