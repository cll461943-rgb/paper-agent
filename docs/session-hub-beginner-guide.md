# Session Hub 新手指南

> 跨 agent Session 共享与检索工具（受 `coding-agent-search` 启发）

---

## 快速开始（5 分钟）

### 1. 安装

```bash
# 进入项目目录
cd /c/Users/33316/Desktop/claude-code-src-main/paper-agent

# 确保虚拟环境已激活
. .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 安装 session_hub 包
pip install -e .

# 测试
session-hub --help
```

### 2. 首次索引

扫描本地所有支持的 agent（Claude Code、Codex CLI 等）并建立 SQLite 索引：

```bash
session-hub index
# 输出示例：Indexed 99 sessions (8348 messages) into C:\Users\...\.session_hub\index.sqlite
```

默认索引位置：`~/.session_hub/index.sqlite`
可通过环境变量或 `--db` 参数指定其他路径：

```bash
export SESSION_HUB_DB=/path/to/my_index.sqlite
session-hub index
```

### 3. 跨 Agent 检索

搜索关键词命中所有 agent 的 session：

```bash
session-hub search "rate limiting" --limit 5
# 输出示例：
# [codex] rate limiting implementation
#   workspace: D:\project
#   msg#15 (assistant, 2026-04-12...): ...rate limiting middleware...
```

---

## 核心功能

### 列出 Session

```bash
session-hub list --limit 10
session-hub list --agent claude_code      # 只看 Claude Code
session-hub list --workspace "paper-agent" # 按工作目录过滤
```

### 查看单个 Session

```bash
session-hub show claude_code 193c1d77-58b9-4d0c-a377-d3dc8a0c0993
```

### 统计信息

```bash
session-hub stats
# 输出：
# db: .../index.sqlite
# messages total: 8348
#   claude_code: 6 sessions
#   codex: 93 sessions
```

### 支持的 Agent

```bash
session-hub agents
# 输出：
# claude_code: C:\Users\...\.claude\projects
# codex: C:\Users\...\.codex\sessions, C:\Users\...\.codex\archived_sessions
# aider: C:\Users\...
# gemini_cli: C:\Users\...\.gemini
```

---

## 跨 Agent 共享 Session（你最想要的功能）

这是 session-hub 的核心特性：将任意 agent 的 session 导出为可移植的 bundle，在其他机器上导入，实现**跨 agent、跨机器**的 session 共享。

### 场景 1：本地机器上不同 Agent 之间共享

假设你在 Claude Code 中解决了一个棘手的问题，想把完整的 session 交给 Codex CLI 继续：

```bash
# 步骤 1：找到 session ID
session-hub list --agent claude_code --limit 5

# 步骤 2：导出为 bundle
session-hub share claude_code <session-id> -o /tmp/auth-bug-fix.json

# 步骤 3：在另一台机器上导入（或同机器的另一个索引）
# 机器 B 或新索引：
session-hub --db ~/.session_hub/secondary.sqlite ingest /tmp/auth-bug-fix.json

# 现在可以在机器 B 上搜索到之前 Claude Code 的 session
session-hub --db ~/.session_hub/secondary.sqlite search "auth bug"
```

### 场景 2：团队/协作场景

```bash
# A 同学解决问题后导出
session-hub share claude_code <session-id> -o ~/Desktop/fix-auth-bug.json
# 发送给 B 同学（邮件、Slack、网盘等）

# B 同学导入到自己的索引
session-hub ingest ~/Downloads/fix-auth-bug.json
# 现在 B 同学可以检索、查看完整的 session 上下文
session-hub show claude_code <session-id>
```

### 场景 3：Agent 作为调用者（机器人模式）

```bash
# 所有命令支持 --json 输出，供其他程序/Agent 消费
session-hub list --json | jq '.sessions[] | select(.agent=="claude_code")'

# 搜索并获取机器可读结果
session-hub search "refactor" --json --limit 10 > results.json

# 导出并验证
session-hub share claude_code <id> -o /tmp/x.json --json

# 导入并确认
session-hub ingest /tmp/x.json --json
```

---

## 高级用法

### 1. 增量索引

每次运行 `session-hub index` 会自动检测新 session，已有 session 会覆盖更新（去重）：

```bash
# 第一次建立完整索引
session-hub index

# 每天增量更新
session-hub index  # 只会索引新增的 sessions
```

### 2. 仅索引特定 Agent

```bash
session-hub index --agent claude_code
session-hub index --agent codex
```

### 3. 使用环境变量

```bash
# 自定义数据目录
export SESSION_HUB_DB=/data/shared_sessions/index.sqlite

# 然后所有命令自动使用该路径
session-hub index
session-hub search "bug"
```

### 4. 多索引管理

可以维护多个索引文件，用于不同目的：

```bash
# 个人工作索引
session-hub --db ~/.session_hub/work.sqlite index

# 团队共享索引
session-hub --db ~/team/shared.sqlite ingest <bundle>

# 临时分析索引
session-hub --db /tmp/analysis.sqlite index --agent claude_code
```

---

## 技术架构（可选阅读）

### 存储设计

```
~/.session_hub/
  └── index.sqlite          # SQLite 主索引
      ├── sessions 表        # 每个 session 的元数据
      ├── messages 表        # 所有消息
      ├── messages_fts 表    # FTS5 全文索引（BM25 搜索）
      └── meta 表            # 配置/元数据
```

### Session Bundle 格式

共享的 bundle 是自包含的 JSON 文件：

```json
{
  "bundle_version": 1,
  "exported_at": "2026-07-01T16:28:00+00:00",
  "session": {
    "session_id": "xxx",
    "agent": "claude_code",
    "title": "...",
    "workspace": "/path/to/project",
    ...
  },
  "messages": [
    {"role": "user", "ts": "...", "plain_text": "...", "snippets": [...]},
    ...
  ]
}
```

这种设计确保 bundle 不依赖任何外部文件，可以跨平台传输。

### Connector 设计

每个 agent 有独立的 connector 模块，负责：

1. 发现 agent 的数据目录
2. 解析原生格式（JSONL、Markdown、SQLite 等）
3. 归一化为统一的 `Conversation/Message/Snippet` schema

新增 agent 只需添加一个 connector 类，无需改动索引逻辑。

---

## FAQ

**Q: 为什么不用原生的 agent 历史功能？**  
A: Claude Code / Codex 等都有自己的历史，但格式各异且无法互通。Session Hub 提供跨 agent 的统一搜索和共享能力。

**Q: 数据会泄露吗？**  
A: 所有数据存储在本地 SQLite。分享的 bundle 由你主动生成，可以安全地传输给协作者。

**Q: 支持哪些 agent？**  
A: 目前支持 Claude Code、Codex CLI、Aider、Gemini CLI。未来可轻松扩展。

**Q: 支持语义搜索吗？**  
A: 当前使用 FTS5 BM25 全文搜索，速度极快（亚毫秒级）。语义搜索（向量匹配）可后续添加为可选增强。

**Q: 如何处理大文件？**  
A: SQLite 索引结构能轻松处理数万条消息。bundle 是按需导出的单个 session，不会有大小问题。

---

## 命令速查表

| 命令 | 说明 |
|------|------|
| `session-hub index` | 建立/更新索引 |
| `session-hub search "query"` | 关键词搜索 |
| `session-hub list` | 列出 sessions |
| `session-hub show <agent> <id>` | 查看完整内容 |
| `session-hub share <agent> <id> -o <file>` | 导出为 bundle |
| `session-hub ingest <file>` | 导入 bundle |
| `session-hub stats` | 显示统计信息 |
| `session-hub agents` | 列出支持的 agent |
| `session-hub <cmd> --json` | 机器可读输出 |

---

## 下一步

- [ ] 为团队建立共享索引
- [ ] 配置定时任务自动索引
- [ ] 探索 `--json` 输出与其他工具的集成
- [ ] 贡献新的 agent connector

---

> 本指南由 Claude Code 生成，基于 `coding-agent-search` 的核心设计理念。