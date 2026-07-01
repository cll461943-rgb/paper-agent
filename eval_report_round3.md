# CNScholarQuery_ZH_dev_1000 召回优化实验报告

## 实验配置
- 数据集: CNScholarQuery_ZH_v0.4_dev_1000.jsonl (1000 cases, avg_gold=2.54)
- 抽样: n=30, seed=42 (固定 QID 集合, 可复现)
- 配置: effect_first.yaml, retrieval_only=True (仅测 Phase 1 检索召回)
- 模型: DeepSeek v4-pro, temperature=0.0
- 金标匹配: arxiv_id 强匹配 (消除标题归一化假零召回)

## 四轮实验对比

| Run | Micro Recall | Macro Avg | R@100 | R@300 | Avg Latency | Zero-Recall |
|-----|-------------|-----------|-------|-------|-------------|-------------|
| Baseline | 0.3784 (28/74) | 0.4075 | 0.2285 | 0.3798 | 96.9s | 13/30 |
| Round 1 | 0.3784 (28/74) | 0.4021 | 0.2672 | 0.3731 | 97.0s | 14/30 |
| Round 2 | ~0.29 (aborted) | ~0.33 | ~0.22 | ~0.30 | ~85s | N/A |
| Round 3 | 0.3378 (25/74) | 0.3651 | 0.2185 | 0.3651 | 66.4s | 16/30 |

## 各轮改动详情

### Round 1: LLM timeout + arxiv + S2 路由扩展
- LLM timeout: 15s → 45s (timeout 率 47% → 5.6%)
- arxiv provider: 启用 (每案贡献 120-253 篇)
- S2 路由: 新增 core_topic/broad_synonym/method_task, cap 3 → 5
- case_deadline: 240s → 360s
- 结果: Micro 不变 (0.3784), R@100 +17% (0.2285→0.2672)
- 改进 4 case (+4 hits): QID 105(6→7), 143(0→1), 617(1→2), 759(2→3), 760(0→1)
- 回退 4 case (-4 hits): QID 28(1→0), 282(3→1), 693(1→0), 734(1→0)

### Round 2: refchain 无条件启用 (已回退)
- retrieval_only 模式下无条件启用 refchain (pool < 600)
- limit_per_seed: 5 → 10
- 结果: API 429 级联 (S2 429 从 38→155, 4x), Pool=0 灾难性失败
- 根因: refchain 每案最多 100 次 API 调用 (5 seeds × 10 × 2 ops), 触发全局 rate limit
- 决策: 立即回退

### Round 3: LLM title candidates 5 → 10
- _llm_title_queries 候选数从 5 增至 10
- 验证: LLM title queries 正常工作 (每案 7-10 个标题候选)
- 结果: Micro 下降至 0.3378 (低于基线)
- 根因: LLM 非确定性 (temp=0 不收敛) 产生不同 query understanding, 导致不同搜索结果

## 核心瓶颈分析

### 语义桥接失败 (Semantic Bridging Failure)
- 13-16/30 cases 零召回 (43-53%)
- 零召回 case 的 pool 通常 200-550 篇 (足够大), 但 gold 论文不在其中
- 根因: 搜索查询术语与 gold 论文术语不匹配
- 示例: QID 31 查询 "VAEs + 3D CNN + voxelized molecules"
  - LLM 回忆: PointNet, VoxNet, 3D ShapeNets (通用 3D 论文)
  - Gold 论文: 特定 VAE+voxel 分子生成论文 (LLM 无法回忆)

### LLM 非确定性
- DeepSeek API temp=0 不收敛 (服务端批处理随机性)
- 同配置同 QID 多次运行, 同一 case 在 0.0 和 1.0 间跳动
- 每轮 ±5% Micro Recall 方差
- 改进与回退相互抵消, 净变化为零

## 结论

当前架构 Micro Recall 天花板 ≈ 0.38-0.41 (CNScholarQuery_ZH_dev_1000, n=30)。

Round 1 改动值得保留:
- LLM timeout 45s (减少 timeout, 改善 R@100)
- arxiv 启用 (增加检索源)
- S2 路由扩展 (增加覆盖率)

## 达 0.5+ 的架构级方案

1. **多轮 ensemble**: 每 case 跑 3 次, 取 pool 并集 (平均 LLM 非确定性, 预期 +5-10% recall)
2. **Embedding 语义检索**: 用 BGE-M3 dense retrieval 替代/补充关键词匹配 (直接解决语义桥接)
3. **更强 LLM 模型**: 用 GPT-4/Claude 做 title recall (更好的学术知识覆盖)
