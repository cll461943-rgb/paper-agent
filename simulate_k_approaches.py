#!/usr/bin/env python3
"""快速模拟不同 K 选择方案的效果，不需要跑 pipeline。"""
import math
import statistics

# ── 实际得分数据（来自全 pipeline 运行日志）──

# QID=26: 369 papers, 1 gold (rank 1)
qid26_top12 = [1.000, 0.908, 0.838, 0.808, 0.759, 0.720, 0.712, 0.705, 0.653, 0.580, 0.564, 0.550]
qid26_gold_positions = [1]  # 1-indexed

# QID=115: 749 papers, 5 gold (ranks 1, 3, 4, 11 + missing 1)
qid115_top12 = [0.806, 0.776, 0.760, 0.691, 0.607, 0.560, 0.528, 0.521, 0.482, 0.479, 0.475, 0.464]
qid115_gold_positions = [1, 3, 4, 11]  # 4 hits in top-12, 1 missing

def generate_tail_scores(n_top, total, score_low=0.45, score_floor=0.02):
    """生成尾部得分：从 score_low 线性衰减到 score_floor。"""
    n_tail = total - n_top
    tail = []
    for i in range(n_tail):
        frac = i / max(n_tail - 1, 1)
        s = score_low - frac * (score_low - score_floor)
        # 加一点噪声
        noise = 0.01 * math.sin(i * 3.7)
        s = max(0.0, s + noise)
        tail.append(round(s, 4))
    return tail

# 生成完整得分列表
qid26_all = qid26_top12 + generate_tail_scores(12, 369, 0.50, 0.02)
qid115_all = qid115_top12 + generate_tail_scores(12, 749, 0.44, 0.01)

# ── 校准函数 ──

def identity_cal(scores):
    return [max(0.0, min(1.0, s)) for s in scores]

def logistic_cal(scores, midpoint=0.5, temperature=0.12):
    return [1.0 / (1.0 + math.exp(-(s - midpoint) / max(1e-6, temperature))) for s in scores]

# ── F_beta 截断 ──

def f_beta_cutoff(probs, beta=1.0):
    """返回最优 k 和对应的 F_beta。"""
    n = len(probs)
    r_hat = math.fsum(probs)
    b2 = beta * beta
    if r_hat <= 0:
        return 0, 0.0, r_hat
    best_k, best_f = 0, -1.0
    cum = 0.0
    for k in range(1, n + 1):
        cum += probs[k - 1]
        denom = k + b2 * r_hat
        f_k = (1.0 + b2) * cum / denom if denom > 0 else 0.0
        if f_k > best_f + 1e-12:
            best_f, best_k = f_k, k
    return best_k, best_f, r_hat

def partition_dual(probs, beta_ext=1.5, max_output=12, min_high=1):
    """双截断：β=1 核心集 + β=1.5 扩展集。"""
    k1, f1, r_hat1 = f_beta_cutoff(probs, beta=1.0)
    k2, f2, r_hat2 = f_beta_cutoff(probs, beta=beta_ext)
    k1 = max(k1, min(min_high, len(probs)))
    k2 = max(k2, k1)
    k2 = min(k2, max_output)
    k1 = min(k1, k2)
    return k1, k2, f1, r_hat1

# ── V2 K 控制器模拟 ──

def v2_k_controller(probs, g_hat, p_floor=0.22, k_max=20):
    """模拟 V2 K 控制器的 F1 曲线和 precision-first tie-break。"""
    n = min(k_max, len(probs))
    curve = {}
    cum = 0.0
    for k in range(1, n + 1):
        cum += probs[k - 1]
        if k + g_hat > 0:
            exp_f1 = 2.0 * cum / (k + g_hat)
        else:
            exp_f1 = 0.0
        curve[k] = round(exp_f1, 4)
    
    if not curve:
        return 0, curve
    
    max_f1 = max(curve.values())
    tie_threshold = 0.02
    
    # Precision-first: among K within threshold of max, pick smallest
    candidates = [k for k, f1 in curve.items() if max_f1 - f1 <= tie_threshold]
    best_k = min(candidates)
    
    # p_floor: expand only if p_next >= p_floor
    for k in range(best_k + 1, max(candidates) + 1):
        if k - 1 < len(probs):
            p_next = probs[k - 1]
            if p_next < p_floor:
                break
            if k in candidates:
                best_k = k
    
    return best_k, curve

# ── 评估函数 ──

def evaluate_k(all_scores, gold_positions, k):
    """给定 K，计算 P/R/F1。"""
    hits = sum(1 for g in gold_positions if g <= k)
    output = min(k, len(all_scores))
    precision = hits / output if output > 0 else 0.0
    recall = hits / len(gold_positions) if gold_positions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, hits

# ── 方案对比 ──

def test_approach(name, all_scores, gold_positions, probs, max_output=12):
    """测试一种方案。"""
    k1, k2, f1_core, r_hat = partition_dual(probs, max_output=max_output)
    p, r, f1, hits = evaluate_k(all_scores, gold_positions, k2)
    
    # 也试 V2 K 控制器（用 R_hat 作为 g_hat）
    v2_k, v2_curve = v2_k_controller(probs, g_hat=r_hat, p_floor=0.22, k_max=20)
    p_v2, r_v2, f1_v2, hits_v2 = evaluate_k(all_scores, gold_positions, min(v2_k, max_output))
    
    # V2 用更小的 g_hat (query_type prior)
    g_hat_small = 4.0  # unknown query type
    v2_k_small, _ = v2_k_controller(probs, g_hat=g_hat_small, p_floor=0.22, k_max=20)
    p_v2s, r_v2s, f1_v2s, hits_v2s = evaluate_k(all_scores, gold_positions, min(v2_k_small, max_output))
    
    print(f"\n  [{name}]")
    print(f"    R_hat={r_hat:.2f} | F_beta cutoff: k1={k1}, k2={k2} | P={p:.4f} R={r:.4f} F1={f1:.4f} hits={hits}")
    print(f"    V2(g_hat=R_hat={r_hat:.1f}): k={min(v2_k, max_output)} | P={p_v2:.4f} R={r_v2:.4f} F1={f1_v2:.4f} hits={hits_v2}")
    print(f"    V2(g_hat=prior=4.0):         k={min(v2_k_small, max_output)} | P={p_v2s:.4f} R={r_v2s:.4f} F1={f1_v2s:.4f} hits={hits_v2s}")

def run_comparison():
    print("=" * 90)
    print("QID=26 (Gold=1, Pool=369)")
    print("=" * 90)
    
    # 方案 0: 当前（identity + F_beta + hard_max=12）
    probs_id = identity_cal(qid26_all)
    test_approach("当前: identity校准", qid26_all, qid26_gold_positions, probs_id)
    
    # 方案 A: logistic(0.5, 0.12)
    probs_lg1 = logistic_cal(qid26_all, midpoint=0.5, temperature=0.12)
    test_approach("A: logistic(0.5, 0.12)", qid26_all, qid26_gold_positions, probs_lg1)
    
    # 方案 A2: logistic(0.6, 0.15)
    probs_lg2 = logistic_cal(qid26_all, midpoint=0.6, temperature=0.15)
    test_approach("A2: logistic(0.6, 0.15)", qid26_all, qid26_gold_positions, probs_lg2)
    
    # 方案 B: V2 K 控制器 + identity
    test_approach("B: V2控制器(identity)", qid26_all, qid26_gold_positions, probs_id)
    
    # 方案 C: V2 K 控制器 + logistic
    test_approach("C: V2控制器+logistic(0.5,0.12)", qid26_all, qid26_gold_positions, probs_lg1)
    
    # 方案 D: score-gap 检测
    print(f"\n  [D: score-gap 检测]")
    for threshold in [0.05, 0.08, 0.10]:
        best_gap_k = 1
        max_gap = 0
        for i in range(min(20, len(qid26_all) - 1)):
            gap = qid26_all[i] - qid26_all[i + 1]
            if gap > threshold and gap > max_gap:
                max_gap = gap
                best_gap_k = i + 1
        p, r, f1, hits = evaluate_k(qid26_all, qid26_gold_positions, best_gap_k)
        print(f"    gap_threshold={threshold}: k={best_gap_k} (gap={max_gap:.3f}) | P={p:.4f} R={r:.4f} F1={f1:.4f}")
    
    print(f"\n{'=' * 90}")
    print("QID=115 (Gold=5, Pool=749)")
    print("=" * 90)
    
    probs_id = identity_cal(qid115_all)
    test_approach("当前: identity校准", qid115_all, qid115_gold_positions, probs_id)
    
    probs_lg1 = logistic_cal(qid115_all, midpoint=0.5, temperature=0.12)
    test_approach("A: logistic(0.5, 0.12)", qid115_all, qid115_gold_positions, probs_lg1)
    
    probs_lg2 = logistic_cal(qid115_all, midpoint=0.6, temperature=0.15)
    test_approach("A2: logistic(0.6, 0.15)", qid115_all, qid115_gold_positions, probs_lg2)
    
    test_approach("B: V2控制器(identity)", qid115_all, qid115_gold_positions, probs_id)
    
    test_approach("C: V2控制器+logistic(0.5,0.12)", qid115_all, qid115_gold_positions, probs_lg1)
    
    print(f"\n  [D: score-gap 检测]")
    for threshold in [0.05, 0.08, 0.10]:
        best_gap_k = 1
        max_gap = 0
        for i in range(min(20, len(qid115_all) - 1)):
            gap = qid115_all[i] - qid115_all[i + 1]
            if gap > threshold and gap > max_gap:
                max_gap = gap
                best_gap_k = i + 1
        p, r, f1, hits = evaluate_k(qid115_all, qid115_gold_positions, best_gap_k)
        print(f"    gap_threshold={threshold}: k={best_gap_k} (gap={max_gap:.3f}) | P={p:.4f} R={r:.4f} F1={f1:.4f}")

if __name__ == "__main__":
    run_comparison()
