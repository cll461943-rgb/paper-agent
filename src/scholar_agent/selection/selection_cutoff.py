from __future__ import annotations

"""F1 / F_beta 最优截断（probabilistic plug-in cutoff）。

把"覆盖率与精确率平衡"这一多目标问题落到工程里：reranker 只负责把候选
按相关性降序排好（决定 P-R 曲线），本模块负责在这条曲线上选一个原则性的
操作点 k*，并据此把结果切成"高度相关 / 部分相关 / 丢弃"三段。

数学依据（见推导）：
    设第 i 名论文相关的（校准后）概率为 p_i（已降序），
        r_hat(k) = Σ_{i<=k} p_i          期望命中数
        R_hat    = Σ_i p_i               期望相关总数
        f_beta(k) = (1+β²)·r_hat(k) / (k + β²·R_hat)
    取 k* = argmax_k f_beta(k)。可证其等价于停机规则
        纳入第 k 项 ⇔ p_k > f_beta(k-1)/(1+β²)，
    即"边际精确率 > 当前 F_beta 的 1/(1+β²)"。β=1 即 marginal precision = F1/2。

双截断分层：
    核心集（高度相关） = β=1     的最优截断 top-k1（召回/精确平衡）
    扩展集（部分相关） = β>1     的最优截断 top-k2 中超出 k1 的部分（偏召回）
    其余                = 丢弃
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

LOGGER = logging.getLogger(__name__)

Tier = Literal["high", "partial", "excluded"]
_TIER_CODE: dict[Tier, float] = {"high": 2.0, "partial": 1.0, "excluded": 0.0}


# ---------------------------------------------------------------------------
# 数学核心：与 schema 解耦，仅作用于概率列表，便于单测
# ---------------------------------------------------------------------------
@dataclass
class CutoffResult:
    """单个 F_beta 截断点的求解结果。"""

    beta: float
    k_star: int           # F_beta 最优截断 k*
    threshold: float      # 纳入阈值 τ = f_beta(k*) / (1 + β²)
    f_at_k: float         # 截断点处的期望 F_beta
    r_hat: float          # 期望相关总数 R_hat = Σ p_i
    precision_at_k: float = 0.0   # 期望精确率 r_hat(k*)/k*
    recall_at_k: float = 0.0      # 期望召回率 r_hat(k*)/R_hat
    probs: list[float] = field(default_factory=list)     # 校准后的相关概率（降序）
    f_curve: list[float] = field(default_factory=list)   # f_beta_hat(k), k=1..N


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def f_beta_optimal_cutoff(probs: Sequence[float], beta: float = 1.0) -> CutoffResult:
    """对一组降序排列的相关概率，求使期望 F_beta 最大的截断 k*。

    复杂度 O(N)。probs 必须已按相关性降序（reranker 输出即满足）。
    """
    probs = [clamp01(float(p)) for p in probs]
    n = len(probs)
    if n == 0:
        return CutoffResult(beta=beta, k_star=0, threshold=0.0, f_at_k=0.0, r_hat=0.0)

    r_hat_total = math.fsum(probs)
    b2 = beta * beta

    if r_hat_total <= 0.0:  # 全 0 概率，F 无定义，返回空集
        return CutoffResult(
            beta=beta, k_star=0, threshold=0.0, f_at_k=0.0, r_hat=0.0,
            probs=probs, f_curve=[0.0] * n,
        )

    f_curve: list[float] = []
    cum = 0.0
    best_k, best_f = 0, -1.0
    for k in range(1, n + 1):
        cum += probs[k - 1]
        denom = k + b2 * r_hat_total
        f_k = (1.0 + b2) * cum / denom if denom > 0 else 0.0
        f_curve.append(f_k)
        if f_k > best_f + 1e-12:
            best_f, best_k = f_k, k

    cum_best = math.fsum(probs[:best_k])
    return CutoffResult(
        beta=beta,
        k_star=best_k,
        threshold=best_f / (1.0 + b2),
        f_at_k=best_f,
        r_hat=r_hat_total,
        precision_at_k=(cum_best / best_k) if best_k else 0.0,
        recall_at_k=(cum_best / r_hat_total) if r_hat_total else 0.0,
        probs=probs,
        f_curve=f_curve,
    )


# ---------------------------------------------------------------------------
# 校准：把 reranker 的 final_score 映射成伪概率（单调，不改变排序）
# ---------------------------------------------------------------------------
def identity_calibration(scores: Sequence[float]) -> list[float]:
    """默认校准：直接把 [0,1] 的 final_score 当作相关概率，与排名严格一致。"""
    return [clamp01(float(s)) for s in scores]


def logistic_calibration(
    midpoint: float = 0.5, temperature: float = 0.12
) -> Callable[[Sequence[float]], list[float]]:
    """可选 logistic 校准：拉开高低分对比，得到更"硬"的概率分隔。

    temperature 越小越接近硬阈值；midpoint 是 p=0.5 对应的分数位置。
    单调变换不改变排序，故与 reranker 排名一致。无标注数据时 identity 即可。
    """

    def _cal(scores: Sequence[float]) -> list[float]:
        return [1.0 / (1.0 + math.exp(-(float(s) - midpoint) / max(1e-6, temperature))) for s in scores]

    return _cal


# ---------------------------------------------------------------------------
# Score-gap 检测：利用得分断崖定位自然分界点
# ---------------------------------------------------------------------------
def find_score_gap_k(
    scores: Sequence[float],
    *,
    gap_threshold: float = 0.05,
    relative_threshold: float = 0.08,
    max_k: int = 20,
    min_k: int = 1,
) -> int | None:
    """在 top-N 得分序列中寻找最大的断崖位置作为推荐 K。

    断崖条件（双重门控）：
      1. 绝对落差 gap = scores[i] - scores[i+1] > gap_threshold
      2. 相对落差 gap / scores[i] > relative_threshold

    返回断崖后的位置 (i+1)，即"断崖之前有多少篇论文"。
    如果找不到满足条件的断崖，返回 None（调用方应回退到 F_beta 截断）。

    Parameters
    ----------
    scores : 降序排列的 final_score 序列。
    gap_threshold : 最小绝对落差（默认 0.05）。
    relative_threshold : 最小相对落差（默认 0.08，即 8%）。
    max_k : 搜索范围上限（默认 20）。
    min_k : 返回值的下限（默认 1）。
    """
    if len(scores) < 2:
        return min(min_k, len(scores)) if scores else None

    best_k: int | None = None
    best_gap = 0.0
    search_limit = min(max_k + 5, len(scores) - 1)

    for i in range(search_limit):
        s_curr = float(scores[i])
        s_next = float(scores[i + 1])
        gap = s_curr - s_next
        if gap <= 0:
            continue
        relative = gap / max(s_curr, 1e-6)
        if gap > gap_threshold and relative > relative_threshold and gap > best_gap:
            best_gap = gap
            best_k = i + 1

    if best_k is not None:
        best_k = max(best_k, min_k)
        best_k = min(best_k, max_k)

    return best_k


def find_percentile_drop_k(
    scores: Sequence[float],
    *,
    drop_ratio: float = 0.15,
    max_k: int = 20,
    min_k: int = 1,
) -> int:
    """平滑分布下的自适应 K 选择（score-gap 失败时的后备策略）。

    两级策略：
      1. 在 top-max_k 范围内寻找最大的相邻得分落差（自适应落差检测）。
         如果最大落差与中位数落差的比值 > 1.5，说明存在"相对断崖"，
         以该位置作为 K。
      2. 如果没有显著的相对断崖（分布极为均匀），回退到固定衰减截断：
         保留得分 >= top_score × (1 - drop_ratio) 的论文。

    Parameters
    ----------
    scores : 降序排列的 final_score 序列。
    drop_ratio : 固定衰减截断的衰减比例（默认 0.10 = 10%）。
    max_k : 搜索范围上限。
    min_k : 输出下限。
    """
    if not scores:
        return min_k

    search_limit = min(max_k, len(scores))

    # ── 第 1 级：自适应最大落差检测 ──
    gaps: list[tuple[int, float]] = []  # (position_after_gap, gap_size)
    for i in range(search_limit - 1):
        gap = float(scores[i]) - float(scores[i + 1])
        if gap > 0:
            gaps.append((i + 1, gap))

    if gaps:
        gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)
        largest_gap = gaps_sorted[0]
        gap_values = [g[1] for g in gaps_sorted]
        median_gap = sorted(gap_values)[len(gap_values) // 2] if gap_values else 0.0

        # 最大落差显著大于中位数落差 → 存在"相对断崖"
        # Ratio threshold 2.5 (raised from 1.5): only truly exceptional gaps
        # qualify as a cliff. Normal score variation (ratio 1.2-2.0) should NOT
        # trigger early cutoff — it was causing K=1 on cases with 5 gold papers.
        if median_gap > 0 and largest_gap[1] / median_gap > 2.5:
            # Minimum K floor: at least 25% of search range, capped by max_k.
            # search_limit = min(max_k, len(scores)), so floor ≤ max_k//4.
            k_floor = max(min_k, search_limit // 4)
            result = max(k_floor, min(largest_gap[0], max_k))
            LOGGER.info(
                "Adaptive-gap K: largest_gap=%.3f at pos=%d, median_gap=%.3f, "
                "ratio=%.1f → K=%d (floor=%d)",
                largest_gap[1], largest_gap[0], median_gap,
                largest_gap[1] / median_gap, result, k_floor,
            )
            return result

    # ── 第 2 级：固定衰减截断 ──
    top_score = float(scores[0])
    if top_score <= 0:
        return max(min_k, search_limit // 4) if scores else min_k

    threshold = top_score * (1.0 - drop_ratio)
    k = min_k
    for i in range(search_limit):
        if float(scores[i]) >= threshold:
            k = i + 1
        else:
            break

    # Minimum K floor: 25% of search range, capped by max_k
    k_floor = max(min_k, search_limit // 4)
    result = max(k_floor, min(k, max_k))
    LOGGER.info(
        "Percentile-drop K: top=%.3f, threshold=%.3f (drop=%.0f%%) → K=%d (floor=%d)",
        top_score, threshold, drop_ratio * 100, result, k_floor,
    )
    return result


# ---------------------------------------------------------------------------
# 工程入口：作用于 list[RankedPaper]
# ---------------------------------------------------------------------------
@dataclass
class FrontierPartition:
    """按 F1 / F_beta 双截断切出的三段结果。"""

    high: list           # RankedPaper：高度相关核心集（top-k1）
    partial: list        # RankedPaper：部分相关扩展集（k1+1 .. k2）
    excluded: list       # RankedPaper：丢弃
    core: CutoffResult       # β=1 的核心截断
    extended: CutoffResult   # β>1 的扩展截断

    @property
    def output(self) -> list:
        """送评/返回的最终集合 = 核心集 + 扩展集（top-k2）。"""
        return self.high + self.partial


def _annotate(rp, tier: Tier, core: CutoffResult, k_high: int, k_ext: int, *, set_should_output: bool) -> None:
    """把分层信息写回 subscores（dict[str,float]），不改动 schema 字段定义。"""
    try:
        sub = getattr(rp, "subscores", None)
        if isinstance(sub, dict):
            sub["F1_Tier"] = _TIER_CODE[tier]
            sub["F1_Core_K"] = float(k_high)
            sub["F1_Extended_K"] = float(k_ext)
            sub["F1_Include_Threshold"] = float(core.threshold)
    except Exception:  # pragma: no cover - 注释失败不应影响主流程
        pass
    if set_should_output:
        try:
            sel = getattr(rp, "selection", None)
            if sel is not None and hasattr(sel, "should_output"):
                setattr(sel, "should_output", tier in {"high", "partial"})
        except Exception:
            pass


def partition_ranked_papers(
    ranked_list: Sequence,
    *,
    recall_beta: float = 1.5,
    score_getter: Callable[[object], float] | None = None,
    calibration: Callable[[Sequence[float]], list[float]] | None = None,
    annotate: bool = True,
    set_should_output: bool = False,
    min_high: int = 0,
    max_output: int | None = None,
) -> FrontierPartition:
    """对已按 final_score 降序的 RankedPaper 列表施加 F1 / F_beta 双截断。

    参数
    ----
    recall_beta:        扩展集（部分相关）使用的 β，>1 偏召回，默认 1.5。
    score_getter:       从 RankedPaper 取相关概率的函数，默认取 final_score。
    calibration:        分数 -> 概率的单调校准，默认 identity。
    annotate:           是否把分层信息写回 subscores。
    set_should_output:  是否据分层同步 selection.should_output（默认不改）。
    min_high:           核心集的最小规模下限（保证至少返回若干结果）。
    max_output:         输出（核心+部分）的硬上限，控制成本/噪声。
    """
    ranked = list(ranked_list)
    if not ranked:
        empty = CutoffResult(beta=1.0, k_star=0, threshold=0.0, f_at_k=0.0, r_hat=0.0)
        return FrontierPartition([], [], [], empty, empty)

    getter = score_getter or (lambda rp: float(getattr(rp, "final_score", 0.0)))
    cal = calibration or identity_calibration
    probs = cal([getter(rp) for rp in ranked])

    core = f_beta_optimal_cutoff(probs, beta=1.0)
    extended = f_beta_optimal_cutoff(probs, beta=max(1.0, recall_beta))

    k_high = core.k_star
    if min_high > 0:
        k_high = max(k_high, min(min_high, len(ranked)))
    k_ext = max(extended.k_star, k_high)          # 扩展集不应小于核心集
    if max_output is not None and max_output >= 0:
        k_ext = min(k_ext, max_output)
        k_high = min(k_high, k_ext)

    high = ranked[:k_high]
    partial = ranked[k_high:k_ext]
    excluded = ranked[k_ext:]

    if annotate:
        for rp in high:
            _annotate(rp, "high", core, k_high, k_ext, set_should_output=set_should_output)
        for rp in partial:
            _annotate(rp, "partial", core, k_high, k_ext, set_should_output=set_should_output)
        for rp in excluded:
            _annotate(rp, "excluded", core, k_high, k_ext, set_should_output=set_should_output)

    # Write calibrated probabilities to paper metadata for evaluate.py metrics
    for idx, rp in enumerate(ranked):
        p_val = probs[idx]
        paper = getattr(rp, "paper", None)
        if paper is not None and hasattr(paper, "metadata"):
            if paper.metadata is None:
                paper.metadata = {}
            paper.metadata["calibrated_probability"] = p_val

    LOGGER.info(
        "F1 cutoff: N=%d R_hat=%.2f | core k1=%d (F1=%.3f P=%.3f R=%.3f tau=%.3f) "
        "| extended k2=%d (F%.1f=%.3f)",
        len(ranked), core.r_hat, k_high, core.f_at_k, core.precision_at_k,
        core.recall_at_k, core.threshold, k_ext, extended.beta, extended.f_at_k,
    )
    return FrontierPartition(high=high, partial=partial, excluded=excluded, core=core, extended=extended)


def summarize_partition(partition: FrontierPartition) -> dict:
    """生成结构化展示用的 dict（满足赛题"结果结构化"要求，可直接转 JSON）。"""

    def _row(rp, tier: Tier) -> dict:
        paper = getattr(rp, "paper", None)
        return {
            "rank": getattr(rp, "rank", None),
            "paper_id": getattr(paper, "paper_id", None),
            "title": getattr(paper, "title", None),
            "year": getattr(paper, "year", None),
            "venue": getattr(paper, "venue", None),
            "score": round(float(getattr(rp, "final_score", 0.0)), 4),
            "tier": tier,
            "label": getattr(getattr(rp, "selection", None), "label", None),
        }

    return {
        "cutoff": {
            "core_k": partition.core.k_star,
            "extended_k": partition.extended.k_star,
            "include_threshold": round(partition.core.threshold, 4),
            "expected_F1": round(partition.core.f_at_k, 4),
            "expected_precision": round(partition.core.precision_at_k, 4),
            "expected_recall": round(partition.core.recall_at_k, 4),
            "R_hat": round(partition.core.r_hat, 3),
        },
        "high_relevance": [_row(rp, "high") for rp in partition.high],
        "partial_relevance": [_row(rp, "partial") for rp in partition.partial],
    }
