from __future__ import annotations

import re
from typing import Any

def _normalize_identifier(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:", "", text)
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text.strip() or None


def _normalize_title(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", text) or None


def _normalize_corpus_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^corpusid:", "", text, flags=re.IGNORECASE)
    return f"corpus:{text.lower()}"


def paper_match_keys(paper: Any, strict: bool = False) -> set[str]:
    keys = {
        _normalize_identifier(paper.paper_id),
        _normalize_identifier(paper.doi),
        _normalize_identifier(paper.arxiv_id),
        _normalize_corpus_id(paper.metadata.get("corpus_id")),
        _normalize_corpus_id(paper.metadata.get("corpusId")),
    }
    if not strict:
        keys.add(_normalize_title(paper.title))
    return {key for key in keys if key}


def gold_match_keys(item: dict[str, Any] | str, strict: bool = False) -> set[str]:
    if isinstance(item, str):
        keys = {_normalize_identifier(item)}
        if not strict:
            keys.add(_normalize_title(item))
    else:
        keys = {
            _normalize_identifier(item.get("paper_id")),
            _normalize_identifier(item.get("doi")),
            _normalize_identifier(item.get("arxiv_id")),
            _normalize_corpus_id(item.get("corpus_id")),
            _normalize_corpus_id(item.get("corpusid")),
            _normalize_corpus_id(item.get("semantic_scholar_corpus_id")),
        }
        if not strict:
            keys.add(_normalize_title(item.get("title")))
    return {key for key in keys if key}


def _merge_gold_key_sets(gold_key_sets: list[set[str]], paper_key_sets: list[set[str]]) -> list[set[str]]:
    parents = list(range(len(gold_key_sets)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_keys in enumerate(gold_key_sets):
        for right_index in range(left_index + 1, len(gold_key_sets)):
            if left_keys & gold_key_sets[right_index]:
                union(left_index, right_index)

    for paper_keys in paper_key_sets:
        matched_indexes = [index for index, gold_keys in enumerate(gold_key_sets) if paper_keys & gold_keys]
        for index in matched_indexes[1:]:
            union(matched_indexes[0], index)

    merged: dict[int, set[str]] = {}
    for index, gold_keys in enumerate(gold_key_sets):
        root = find(index)
        merged.setdefault(root, set()).update(gold_keys)
    return list(merged.values())


def is_paper_match_gold(paper: Any, gold_item: dict[str, Any] | str, strict: bool = False) -> bool:
    """判断单篇论文是否匹配某个金标项"""
    p_keys = paper_match_keys(paper, strict=strict)
    g_keys = gold_match_keys(gold_item, strict=strict)
    
    if p_keys & g_keys:
        return True
        
    # 对于 strict 模式下的标题精确相等校验（防 ID 缺失而误判）
    if strict:
        p_title = _normalize_title(paper.title)
        if isinstance(gold_item, str):
            g_title = _normalize_title(gold_item)
        else:
            g_title = _normalize_title(gold_item.get("title"))
        if p_title and g_title and p_title == g_title:
            return True
            
    # 对于 relaxed 模式，增加标题相似度兜底（支持编辑距离或包含关系）
    if not strict:
        p_title = _normalize_title(paper.title)
        if isinstance(gold_item, str):
            g_title = _normalize_title(gold_item)
        else:
            g_title = _normalize_title(gold_item.get("title"))
            
        if p_title and g_title:
            # 包含匹配或者相似度匹配
            if p_title in g_title or g_title in p_title:
                return True
                
    return False


def score_papers_against_gold(
    papers: list[Any],
    gold_items: list[dict[str, Any] | str],
    strict: bool = False
) -> dict[str, float | int]:
    """计算推荐论文与金标的评估指标"""
    gold_key_sets = [keys for item in gold_items if (keys := gold_match_keys(item, strict=strict))]
    paper_key_sets = [paper_match_keys(paper, strict=strict) for paper in papers]
    
    # 增加 strict 模式下的 Title 完全对齐校准
    if strict:
        # 补充严格的标题匹配 key
        for idx, paper in enumerate(papers):
            p_title = _normalize_title(paper.title)
            if p_title:
                paper_key_sets[idx].add(f"title_strict:{p_title}")
                
        for idx, item in enumerate(gold_items):
            if isinstance(item, str):
                g_title = _normalize_title(item)
            else:
                g_title = _normalize_title(item.get("title"))
            if g_title:
                gold_key_sets[idx].add(f"title_strict:{g_title}")

    gold_key_sets = _merge_gold_key_sets(gold_key_sets, paper_key_sets)
    matched_gold_indexes: set[int] = set()

    for paper_keys in paper_key_sets:
        for index, gold_keys in enumerate(gold_key_sets):
            if index in matched_gold_indexes:
                continue
            if paper_keys & gold_keys:
                matched_gold_indexes.add(index)
                break

    tp = len(matched_gold_indexes)
    predicted_count = len(papers)
    gold_count = len(gold_key_sets)

    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "true_positive": tp,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def calculate_hit_at_k(papers: list[Any], gold_items: list[dict[str, Any] | str], k: int) -> int:
    """计算 Hit@K (前 K 个推荐结果里有至少一个命中金标返回 1，否则返回 0)"""
    subset = papers[:k]
    for p in subset:
        for g in gold_items:
            if is_paper_match_gold(p, g, strict=False):
                return 1
    return 0


def calculate_precision_recall_f1_at_k(
    papers: list[Any],
    gold_items: list[dict[str, Any] | str],
    k: int,
    strict: bool = False
) -> dict[str, float]:
    """截取 papers[:k] 并计算 Precision, Recall, F1"""
    scores = score_papers_against_gold(papers[:k], gold_items, strict=strict)
    return {
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"]
    }


def compute_selector_stats(
    candidate_pool: list[Any],
    selections: list[Any],
    gold_items: list[dict[str, Any] | str]
) -> dict[str, int]:
    """
    计算 Selector 假阳性与假阴性统计。
    - False Positive: 被选择器判定为 high/medium，但不在金标中的推荐篇数。
    - False Negative: 金标论文已进入候选池，但被选择器判定为 low/irrelevant 导致丢弃的篇数。
    """
    # 建立 paper_id 到 selection 映射
    selection_map = {sel.paper_id: sel for sel in selections}
    
    # 查找候选池中属于金标的论文
    golds_in_candidates: list[Any] = []
    for p in candidate_pool:
        is_gold = False
        for g in gold_items:
            if is_paper_match_gold(p, g, strict=False):
                is_gold = True
                break
        if is_gold:
            golds_in_candidates.append(p)
            
    # False Negative: 金标在候选池里，但其大模型判定为 low / irrelevant
    fn_count = 0
    for gp in golds_in_candidates:
        sel = selection_map.get(gp.paper_id)
        if sel and sel.relevance_level in ("low", "irrelevant"):
            fn_count += 1
        elif not sel:
            # 候选池里有，但因为限制 max_papers_for_selector 根本没送去 selector 的也算 FN
            fn_count += 1
            
    # False Positive: 判定为 high / medium 却不属于金标的论文数量
    fp_count = 0
    for sel in selections:
        if sel.relevance_level in ("high", "medium"):
            # 查找这篇 selection.paper_id 对应的 paper 对象是否属于金标
            # 为方便，我们在 candidate_pool 中查找
            is_gold = False
            # 找到对应的 paper 对象
            p_obj = next((p for p in candidate_pool if p.paper_id == sel.paper_id), None)
            if p_obj:
                for g in gold_items:
                    if is_paper_match_gold(p_obj, g, strict=False):
                        is_gold = True
                        break
            if not is_gold:
                fp_count += 1

    return {
        "selector_false_positive": fp_count,
        "selector_false_negative": fn_count
    }
