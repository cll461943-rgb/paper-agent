# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Global Skill Priority (强制应用)

本项目的所有工作必须遵循以下skill优先级，这些来自Codex AGENTS.md的核心原则：

### 工作流优先级顺序

对于每个实质性任务，按以下顺序应用：

1. **prompt-adapter** - 在执⾏前将⽤户请求适配为结构化的执⾏提示
2. **skill-router** - 根据任务类型选择最⼩相关的skill集合
3. **using-agent-skills** (Addy Osmani工程生命周期) - 代码项目的主要工程原则
4. **matt-pocock-engineering** - 编程任务的首要gate
5. **karpathy-guidelines** - 编码行为准则

### Prompt Adapter (强制)

在执⾏任何实质性⽤户请求前，内部重写为适配版执⾏提示：

```
⽬标：
上⽂：
约束：
需要使⽤的技能/⼯具：
执⾏步骤：
验收标准：
需要向⽤户展⽰的结果：
```

**规则**：
- 保留⽤户最新请求作为真相来源
- 不要扩⼤范围、发明缺失的可交付成果
- 不要静默覆盖⽤户明确约束
- ⾮平凡⼯作前显⽰⼀⾏：`适配版任务：<简洁描述>`

### Skill Router (强制)

任务分类后立即路由：

| 任务类型 | 路由路径 |
|---------|---------|
| Bug/失败测试/性能退化 | diagnose |
| 功能实现或bug修复 | tdd |
| 架构/重构/可测试性改进 | improve-codebase-architecture |
| 不熟悉的代码区域 | zoom-out |
| PRD⼯作 | to-prd |
| 计划分解 | to-issues |
| Issue审查 | triage |

### Using Agent Skills - 工程生命周期 (强制)

**默认项目节奏**（代码项目必须遵循）：

```
/spec → /plan → /build → /test → /review → /code-simplify → /ship
```

1. `/spec` - 定义要构建什么；spec before code
2. `/plan` - 将工作分解为小的、原子的任务
3. `/build` - 一次实现一个薄切片
4. `/test` - 用测试或等效检查证明行为
5. `/review` - 在合并或交接前改善代码健康
6. `/code-simplify` - 在行为受保护后简化
7. `/ship` - 准备最小的安全发布路径

**核心操作行为**（始终适用）：

1. **Surface Assumptions** - 实现前显式陈述假设
2. **Manage Confusion Actively** - 遇到不一致时STOP，命名困惑，提问
3. **Push Back When Warranted** - 不是yes-machine，指出问题，解释 downside
4. **Enforce Simplicity** - 主动抵制过度复杂化
5. **Maintain Scope Discipline** - 只触及被要求触及的内容
6. **Verify, Don't Assume** - 直到验证通过才算完成

### Matt Pocock Engineering (强制)

编程任务的priority-1 gate：

- **构建快速、确定的反馈循环** before trusting hypotheses
- **垂直切片和tracer bullets**，而不是水平批次
- **通过公共接口验证可观察行为** 的测试
- **仅在系统边界mock**；默认不mock内部协作者
- **深度模块**：小接口、深实现、高杠杆、强局部性
- **编辑要外科手术式**，删除临时调试instrumentation

### Karpathy Guidelines (强制)

编写、编辑、审查、重构代码时：

1. **Think Before Coding** - 明确陈述假设，不确定时提问，存在多种解释时呈现它们
2. **Simplicity First** - 只编写解决问题的最少代码，单次使用的代码不创建抽象
3. **Surgical Changes** - 只触及任务需要的文件和行，匹配现有风格，清理自己的孤儿代码
4. **Goal-Driven Execution** - 为非平凡代码变更定义可验证的成功标准

---

## Project Overview

This is **Scholar Agent V2.0** - an AI-powered academic paper retrieval and recommendation system. It takes natural language research queries and returns relevant academic papers with evidence-based ranking.

The system implements a multi-stage pipeline: query understanding → multi-source retrieval → evidence-based selection → ranking → synthesis.

## Architecture

### Core Pipeline (src/scholar_agent/workflow/pipeline.py)

The `PaperAgentPipeline` orchestrates the workflow:

1. **Planning** (`planning/`): Query understanding and search strategy generation
2. **Retrieval** (`retrieval/`): Multi-source paper fetching from academic APIs
3. **Selection** (`selection/`): Evidence-based paper relevance validation
4. **Ranking** (`ranking/`): Multi-factor scoring and final reranking
5. **Synthesis** (`synthesis/`): Result compilation and evidence summarization

### Key Modules

- **infra/**: LLM client, HTTP client with retry/backoff, caching, config loading
- **models/**: Pydantic schemas (Paper, QueryPlan, SelectionResult, RankedPaper, etc.)
- **retrieval/**: Provider implementations (OpenAlex, Semantic Scholar, PubMed, arXiv, PASA local)
- **ranking/**: Local pre-ranker, listwise LLM reranker, dynamic K controller, expected F1 K controller
- **selection/**: Evidence extraction, batch selection, evidence validation
- **prompts/**: LLM prompt registry and schemas

### Data Flow

```
User Query → Query Understanding → Search Queries → Multi-Route Retrieval → 
Candidate Pool → Evidence Selection → Ranking → Dynamic K Selection → Final Output
```

## Development Commands

### Environment Setup

```bash
# Using uv (recommended)
uv sync

# Or using pip with venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy and edit environment variables
cp .env.example .env
# Edit .env with your API keys (DeepSeek, OpenAlex, etc.)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest --cov=src/scholar_agent --cov-report=html
```

### Evaluation

```bash
# Full evaluation on dev set (mock mode - no API calls)
python evaluate.py -f data/benchmarks/AutoScholarQuery_dev.jsonl -l 5 -m mock

# Live evaluation with real APIs (requires API keys)
python evaluate.py -f data/benchmarks/AutoScholarQuery_dev.jsonl -l 5 -m live

# Ablation studies
python run_ablation.py --config configs/ablation_full.yaml --limit 10

# Full pipeline evaluation for specific cases
python full_pipeline_eval.py --qid AutoScholarQuery_dev_0 --config configs/default.yaml
```

### Using the CLI

```bash
# Run a single query through the pipeline
python -m scholar_agent.cli "Your research question here" --config configs/default.yaml --output result.json
```

## Project Structure

```
.
├── src/scholar_agent/     # Main source code
├── configs/               # YAML configuration files
│   ├── default.yaml       # Main configuration
│   ├── effect_first.yaml   # Effect-first routing config
│   └── ablation_*.yaml     # Ablation study configs
├── data/
│   ├── benchmarks/        # Evaluation datasets
│   └── cache/             # Runtime cache (gitignored)
├── tests/                 # Test files
├── scripts/               # Utility scripts
└── evaluate.py            # Main evaluation script
```

## Key Configuration Files

- **configs/default.yaml**: Main configuration controlling providers, LLM settings, budget limits, ranking weights
- **.env**: API keys and credentials (not committed)
- **pyproject.toml**: Package dependencies and pytest settings

## Testing Notes

- Tests use pytest and are located in `tests/`
- `conftest.py` adds `src/` to Python path automatically
- Mock LLM client available for testing without API calls
- Test data fixtures in `tests/` include sample papers and queries

## Common Tasks

### Adding a New Provider

1. Create provider class in `src/scholar_agent/retrieval/` inheriting from `PaperProvider`
2. Implement `search()` and `fetch_paper()` methods
3. Register in `src/scholar_agent/retrieval/factory.py`
4. Add configuration in `configs/default.yaml` under `providers:` section

### Modifying Ranking Weights

Edit `configs/default.yaml` under `ranking.weights:` - weights are normalized automatically.

### Debugging Pipeline Issues

```python
# Enable debug logging in config
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## Issue Tracking (Beads)

This project uses **bd (beads)** for issue tracking. Run `bd prime` for full workflow context.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

---

## Session Completion (强制)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

---

## Chat History Management

本系统自动保存会话快照到 `.claude/chat-history/`。

### 触发历史记录查看

输入以下任一触发词查看历史：
- `/history`
- `历史记录`
- `查看历史`

### 自动保存时机

- 完成重大操作后
- 会话自然结束前  
- 用户明确要求时

### 手动管理历史

```bash
# 查看历史索引
ls -la .claude/chat-history/

# 读取特定会话
cat .claude/chat-history/session_YYYYMMDD_HHMMSS.json
```

---

## Performance-First Constraint (强制)

当活跃工作涉及 `scholar-agent` 基准测试、F1/Recall、检索模型性能、BGE-M3、reranking、引用扩展或查询重写时，必须将其视为性能优先的基准任务：

**硬性优先级顺序**：

1. 在声称进展之前建立或刷新当前基准基线
2. 一次执行一个完整的中期报告步骤，然后停止并报告
3. 在第一个性能阶段，优先处理上层模型/检索变更：BGE-M3密集检索优先，然后是BGE-reranker-v2重排
4. 对于每个步骤，运行配对的前后基准或当完整基准不可能时运行明确命名的静态验证
5. 仅报告测量的F1、Recall、Precision、成本、延迟、候选池或top-k变更。没有证据不要声称步骤完成

**在此阶段禁止**（除非用户明确重定向）：
- 独立的UI优化、广泛的架构清理、广泛的文档、新的agent编排或无关的工作流/工具变更
- 重复Query2doc、RefChain、引用扩展或BM25/本地索引工作，除非它是命名的当前步骤并有直接基准假设
- 并行启动多个改进轨道或从几个步骤中做部分片段

---

## Non-Interactive Shell Commands (强制)

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest

# Other commands
scp -o BatchMode=yes        # For non-interactive scp
ssh -o BatchMode=yes        # To fail instead of prompting
apt-get -y                  # Auto-confirm apt
HOMEBREW_NO_AUTO_UPDATE=1   # For brew
```
