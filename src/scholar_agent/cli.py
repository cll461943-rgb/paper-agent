from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from scholar_agent.infra import (
    MockLLMClient,
    OpenAICompatibleLLMClient,
    load_config,
    setup_logging,
)
from scholar_agent.retrieval import build_providers
from scholar_agent.workflow.budget import BudgetManager
from scholar_agent.workflow.pipeline import PaperAgentPipeline

LOGGER = logging.getLogger("scholar_agent.cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scholar Agent V2.0 - 智能学术论文搜索与推荐命令行工具"
    )
    parser.add_argument(
        "query",
        type=str,
        help="学术检索与推荐的复杂用户查询"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["live", "mock"],
        help="运行模式：live (联网大模型与数据源) 还是 mock (脱网模拟环境)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径，默认为 configs/default.yaml"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="结果保存的本地 JSON 文件路径"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="是否打印调试日志"
    )

    args = parser.parse_args()

    # 1. 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # 2. 加载配置
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Error loading configuration file {args.config}: {exc}", file=sys.stderr)
        sys.exit(1)

    # 3. 命令行参数覆盖配置
    if args.mode:
        config.app.mode = args.mode

    print(f"🚀 Initializing Scholar Agent in '{config.app.mode}' mode...")
    
    # 4. 初始化 Budget 和 LLM 客户端
    budget = BudgetManager(config)
    if config.app.mode == "mock":
        llm_client = MockLLMClient(budget)
    else:
        llm_client = OpenAICompatibleLLMClient(config.llm, budget)

    # 5. 构建检索器提供商
    try:
        providers = build_providers(config)
    except Exception as exc:
        print(f"Error building search providers: {exc}", file=sys.stderr)
        sys.exit(1)

    # 6. 执行 Pipeline
    print(f"🔍 Searching and synthesizing papers for: '{args.query}'")
    pipeline = PaperAgentPipeline(config, llm_client, providers)
    
    try:
        result = pipeline.run(args.query)
    except Exception as exc:
        LOGGER.exception("Pipeline run failed")
        print(f"❌ Error during pipeline execution: {exc}", file=sys.stderr)
        sys.exit(1)

    # 7. 打印主要推荐结果到终端
    print("\n" + "=" * 50)
    print("🎓 SCHOLAR AGENT RECOMMENDATION REPORT")
    print("=" * 50)
    print(f"Original Query: {result.original_query}")
    print(f"Research Topic: {result.query_plan.research_topic or 'N/A'}")
    print(f"Time Range constraint: {result.query_plan.time_range or 'None'}")
    print("-" * 50)
    
    print(f"\n🔥 Highly Relevant Papers ({len(result.highly_relevant_papers)}):")
    for rp in result.highly_relevant_papers:
        print(f"  [{rp.rank}] Score: {rp.final_score:.3f} | {rp.paper.title} ({rp.paper.year})")
        print(f"      Source: {rp.paper.source} | Citations: {rp.paper.citation_count or 0}")
        print(f"      Reason: {rp.selection.reason}")
        if rp.selection.validation_notes:
            print(f"      ⚠️ Notes: {'; '.join(rp.selection.validation_notes)}")
            
    print(f"\n⭐ Partially Relevant Papers ({len(result.partially_relevant_papers)}):")
    for rp in result.partially_relevant_papers:
        print(f"  [{rp.rank}] Score: {rp.final_score:.3f} | {rp.paper.title} ({rp.paper.year})")
        print(f"      Source: {rp.paper.source} | Citations: {rp.paper.citation_count or 0}")
        print(f"      Reason: {rp.selection.reason}")

    print("\n" + "-" * 50)
    print("📈 AGENT PROCESS SELF-REFLECTION SUMMARY:")
    print(f"  Summary: {result.agent_self_report.get('strategy_summary', 'N/A')}")
    if result.agent_self_report.get("gaps_identified"):
        print("  Identified Gaps:")
        for gap in result.agent_self_report["gaps_identified"]:
            print(f"    - {gap}")

    print("\n📊 RUN METRICS:")
    m = result.run_metrics
    print(f"  Total Time: {m.elapsed_seconds:.2f}s | LLM Time: {m.llm_elapsed_seconds:.2f}s")
    print(f"  LLM Calls: {m.llm_calls_used} | Token Estimate: {m.token_estimate}")
    print(f"  Search Queries Used: {m.search_queries_used} | Retrieval Rounds: {m.retrieval_rounds_used}")
    print(f"  Cache Hits: {m.cache_hits} | Candidate Pool Size: {m.candidate_pool_size}")
    if m.errors:
        print(f"  ⚠️ Logged Errors ({len(m.errors)}):")
        for err in m.errors[:5]:
            print(f"    - {err}")
    print("=" * 50)

    # 8. 保存输出文件
    if args.output:
        out_path = Path(args.output)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"\n💾 Full structured report saved to: {out_path.resolve()}")
        except Exception as exc:
            print(f"Error saving output file {args.output}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
