from __future__ import annotations

from typing import Any
from scholar_agent.evaluation.metrics import is_paper_match_gold

def compute_candidate_to_final_loss(
    gold_items: list[dict[str, Any] | str],
    candidate_pool: list[Any],
    final_papers: list[Any]
) -> dict[str, Any]:
    """
    计算从候选池到最终结果的损失。
    - candidate_gold_count: 候选池中包含的 gold 数
    - final_gold_count: 最终推荐集中包含的 gold 数
    - lost_after_candidate: 被排序/筛选组件淘汰的 gold 数
    - lost_papers: 被淘汰的 gold 论文的 ID 或 Title
    """
    in_candidate = []
    in_final = []
    
    # 查找哪些金标在候选池里
    for gold in gold_items:
        found_in_cand = False
        for p in candidate_pool:
            if is_paper_match_gold(p, gold, strict=False):
                found_in_cand = True
                in_candidate.append(gold)
                break
                
        # 查找哪些金标在最终结果里
        for p in final_papers:
            if is_paper_match_gold(p, gold, strict=False):
                in_final.append(gold)
                break
                
    # 去重
    in_candidate_ids = set()
    for item in in_candidate:
        if isinstance(item, str):
            in_candidate_ids.add(item)
        else:
            in_candidate_ids.add(item.get("paper_id") or item.get("title"))
            
    in_final_ids = set()
    for item in in_final:
        if isinstance(item, str):
            in_final_ids.add(item)
        else:
            in_final_ids.add(item.get("paper_id") or item.get("title"))
            
    lost_papers_ids = in_candidate_ids - in_final_ids
    
    # 根据 id 还原 lost_papers 详细结构
    lost_papers = []
    for item in in_candidate:
        key = item if isinstance(item, str) else (item.get("paper_id") or item.get("title"))
        if key in lost_papers_ids:
            lost_papers.append(item)

    return {
        "candidate_gold_count": len(in_candidate_ids),
        "final_gold_count": len(in_final_ids),
        "lost_after_candidate": len(lost_papers_ids),
        "lost_papers": lost_papers
    }


def classify_drop_reason(
    gold_item: dict[str, Any] | str,
    candidate_pool: list[Any],
    selections: list[Any],
    final_papers: list[Any],
    max_candidates_for_rerank: int = 300,
    output_k: int = 3
) -> str:
    """
    分类诊断一篇 gold 论文被丢弃的原因。
    返回分类包括:
    - 'source_missing': 未被任何检索源召回 (不在候选池中)
    - 'candidate_rank_too_low': 被粗排淘汰 (排名 > max_candidates_for_rerank，未进入 Selection 阶段)
    - 'llm_selector_misjudge': LLM 相关度判定为 low/irrelevant 导致被弃
    - 'evidence_validator_downgrade': 因 evidence_validator 校验失败降级 (如无 evidence 或 must 约束缺失)
    - 'dynamic_k_cutoff': 在最终精排中分值足够但超出 Dynamic-K / TopK 截断线
    - 'unknown': 其他位置原因
    """
    # 1. 判断是否在候选池中
    found_paper_in_cand = None
    cand_rank = 9999
    
    for idx, p in enumerate(candidate_pool):
        if is_paper_match_gold(p, gold_item, strict=False):
            found_paper_in_cand = p
            cand_rank = idx + 1
            break
            
    if not found_paper_in_cand:
        return "source_missing"
        
    # 2. 判断是否在粗排过滤外
    if cand_rank > max_candidates_for_rerank:
        return "candidate_rank_too_low"
        
    # 3. 检查 selection 结果
    selection_map = {sel.paper_id: sel for sel in selections}
    sel = selection_map.get(found_paper_in_cand.paper_id)
    
    if not sel:
        # 虽在 top300 内，但没有被送入 selector (如 pipeline 里面限制只送前 max_selection 篇论文进行 selector，导致被截断)
        return "candidate_rank_too_low"
        
    # 4. 判断是不是 LLM 相关性直接判错
    if sel.relevance_level in ("low", "irrelevant"):
        # 检查是否因为校验失败而被 validator 降级
        # 如果 validator 改判过，那么 sel.validation_notes 中会含有 "downgrade" 或者 "violated" 等关键词
        validator_downgrade = False
        notes = getattr(sel, "validation_notes", [])
        for note in notes:
            if any(kw in note.lower() for kw in ("downgrade", "violation", "violated", "missing", "hallucinated")):
                validator_downgrade = True
                break
        if validator_downgrade:
            return "evidence_validator_downgrade"
        else:
            return "llm_selector_misjudge"
            
    # 5. 判断是不是在最终截断中被淘汰
    found_in_final = False
    final_rank = 9999
    for idx, p in enumerate(final_papers):
        if is_paper_match_gold(p, gold_item, strict=False):
            found_in_final = True
            final_rank = idx + 1
            break
            
    if not found_in_final:
        # 已经通过了 selector 校验且为 high/medium 却没有在 final_papers 列表中，证明是排序截断导致
        return "dynamic_k_cutoff"
        
    return "unknown"
