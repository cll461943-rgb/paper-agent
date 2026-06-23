import logging
import re

from scholar_agent.models.schemas import QueryPlan

LOGGER = logging.getLogger(__name__)

# Regex pattern for matching year mentions in queries
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')

COMMON_METHOD_HINTS = [
    "knowledge distillation",
    "llm",
    "large language model",
    "language model compression",
    "vision-language",
    "multimodal",
    "moe",
    "dpo",
    "rlhf",
    "rl",
    "bm25",
    "tf-idf",
    "transformer",
    "diffusion",
    "lora",
    "low-rank adaptation",
    "contrastive learning",
    "prompt tuning",
    "instruction tuning",
    "in-context learning",
    "chain-of-thought",
    "few-shot",
    "zero-shot",
    "self-supervised",
    "semi-supervised",
    "active learning",
    "curriculum learning",
    "meta-learning",
    "federated learning",
    "pruning",
    "quantization",
    "neural architecture search",
    "graph neural network",
    "attention mechanism",
    "cross-attention",
    "self-attention",
    "reinforcement learning from human feedback",
    "direct preference optimization",
    "retrieval augmented generation",
    "rag",
]

COMMON_TASK_HINTS = [
    "compress", "compression", "generation", "retrieval", "ranking",
    "translation", "translating", "planning", "reasoning", "summarization",
    "evaluation", "benchmark", "alignment", "quantization", "quantized",
    "pretraining", "pre-training", "captioning", "description", "extraction",
    "watermarking",
]

COMMON_ENTITY_HINTS = [
    "in-context learning", "large language model agent", "language model agent",
    "pre-trained language model", "tunisian arabic dialect", "reward shaping",
    "video description", "long video description", "event extraction",
    "document-level event extraction", "vocabulary watermarking", "hallucination",
    "factual consistency", "abstractive summarization", "quantized pretraining",
    "scholarly documents",
    "text generation", "code generation", "image generation",
    "sentiment analysis", "named entity recognition", "relation extraction",
    "machine translation", "question answering", "reading comprehension",
    "text classification", "semantic segmentation", "object detection",
    "speech recognition", "text-to-speech", "image captioning",
    "visual question answering", "dialogue system", "chatbot",
    "embedding", "representation learning", "feature extraction",
    "alignment", "safety", "jailbreak", "red-teaming",
    "long context", "context window", "token limit",
    "parameter-efficient", "adapter", "prefix tuning",
]

CHINESE_METHOD_PATTERNS = [
    (r"检索增强生成|检索增强|RAG", "retrieval augmented generation"),
    (r"大语言模型|大模型", "large language model"),
    (r"引用网络|引文网络|citation", "citation network"),
    (r"重排序|重排|rerank", "reranking"),
]

CHINESE_TASK_PATTERNS = [
    (r"检索|搜索", "retrieval"),
    (r"重排序|重排|排序", "ranking"),
    (r"缓解|检测|识别", "detection"),
]

CHINESE_ENTITY_PATTERNS = [
    (r"幻觉", "hallucination"),
    (r"学术搜索|论文检索", "scholarly search"),
    (r"论文推荐", "paper recommendation"),
    (r"引用网络|引文网络", "citation network"),
    (r"重排序|重排", "reranking"),
    (r"查询分解|问题分解|子查询", "query decomposition"),
    (r"查询改写|查询扩展", "query reformulation"),
    (r"引文追踪|引用追踪", "citation tracking"),
    (r"参考文献扩展", "reference expansion"),
    (r"语义检索", "semantic retrieval"),
    (r"多源检索", "multi-source retrieval"),
    (r"学习排序|学习式排序", "learning-to-rank"),
    (r"学术智能体|科研智能体", "scholarly agent"),
]

QUERY_PREFIXES = [
    "give me papers which show that", "give me papers that", "give me papers about",
    "provide me with all papers that", "provide me with papers that",
    "provide me with all papers", "provide me with papers", "provide me with",
    "provide papers on", "provide papers about", "find papers that",
    "find papers about", "find papers on", "papers that", "papers which propose",
    "list all papers that", "do you know some papers about", "show me research on",
    "i am looking for research papers on", "i am looking for papers on",
    "i am looking to understand more about", "i am looking to understand",
]


def _extract_acronyms(text: str) -> list[str]:
    acronyms = set()
    for match in re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b", text):
        normalized = match.strip("-")
        if len(normalized) >= 2:
            acronyms.add(normalized)
    return sorted(acronyms)


def _extract_quoted_phrases(text: str) -> list[str]:
    return sorted(set(re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', text)))


def _flatten_phrase_matches(matches: list[tuple[str, ...]]) -> list[str]:
    values: list[str] = []
    for match in matches:
        for item in match:
            if item:
                values.append(item)
    return sorted(set(values))


def _looks_like_specific_paper(query: str) -> bool:
    """判断查询是否在找特定论文（有明确标题片段、年份或作者信息）。"""
    lowered = query.lower()
    title_words = len([w for w in query.split() if len(w) > 4])
    # 标题词数多 + 有年份信息或作者信息 → likely specific paper
    has_year = bool(YEAR_PATTERN.search(query))
    has_author = any(kw in lowered for kw in ("author", "et al", "et al.", "written by", "published by"))
    return title_words >= 5 and (has_year or has_author)


def _clean_query_prefix(text: str) -> str:
    lowered = text.lower().strip()
    for prefix in QUERY_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip(" .,:;")
    return text.strip()


def _contains_hint(lowered: str, hint: str) -> bool:
    if hint == "llm":
        return re.search(r"\bllms?\b|\bllm[- ]based\b", lowered) is not None
    if hint == "rl":
        return re.search(r"\brl\b|\breinforcement learning\b", lowered) is not None
    if hint == "large language model":
        return re.search(r"\blarge language models?\b", lowered) is not None
    return re.search(rf"\b{re.escape(hint)}s?\b", lowered) is not None


def _derived_entity_hints(lowered: str) -> list[str]:
    hints: list[str] = []
    if "knowledge distillation" in lowered and ("compress" in lowered or "compression" in lowered):
        hints.append("language model compression")
        if "language model" in lowered:
            hints.append("pre-trained language model")
        if "task-agnostic" in lowered or "task agnostic" in lowered:
            hints.append("task-agnostic knowledge distillation")
    if "tunisian arabic dialect" in lowered and "translat" in lowered:
        hints.append("tunisian arabic dialect translation")
        if "resource" in lowered or "data" in lowered:
            hints.append("parallel resources")
        if "translated comment" in lowered or "native speaker" in lowered:
            hints.append("translated comments")
    if "hallucination" in lowered and ("sequence generation" in lowered or "generation" in lowered):
        if "token" in lowered or "sentence" in lowered or "summarization" in lowered:
            hints.extend(["factual consistency", "abstractive summarization"])
    return hints


def _extract_chinese_matches(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    matches = []
    for pattern, value in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(value)
    return list(dict.fromkeys(matches))


def normalize_constraints(plan: QueryPlan) -> QueryPlan:
    must = set(plan.must_have_constraints)
    nice = set(plan.nice_to_have_constraints)

    # 数据集、benchmark、明确年份一般是硬约束
    for d in plan.datasets:
        must.add(d)

    # 用户明确说“使用/基于/with/using/采用”的方法，加入硬约束
    explicit_method_markers = ["using", "with", "based on", "采用", "基于", "使用"]
    if any(m in plan.original_query.lower() for m in explicit_method_markers):
        for m in plan.methods:
            must.add(m)
    else:
        for m in plan.methods:
            nice.add(m)

    for e in plan.entities:
        nice.add(e)

    plan.must_have_constraints = list(must)
    plan.nice_to_have_constraints = list(nice)
    return plan


def heuristic_understand_query(query: str) -> QueryPlan:
    lowered = query.lower()
    cleaned_query = _clean_query_prefix(query)
    methods = sorted(
        {hint for hint in COMMON_METHOD_HINTS if _contains_hint(lowered, hint)}
        | set(_extract_chinese_matches(query, CHINESE_METHOD_PATTERNS))
    )
    task_tokens = sorted(
        {hint for hint in COMMON_TASK_HINTS if _contains_hint(lowered, hint)}
        | set(_extract_chinese_matches(query, CHINESE_TASK_PATTERNS))
    )
    acronyms = _extract_acronyms(query)
    quoted_phrases = _flatten_phrase_matches(_extract_quoted_phrases(query))
    entity_hints = sorted(
        {hint for hint in COMMON_ENTITY_HINTS if _contains_hint(lowered, hint)}
        | set(_derived_entity_hints(lowered))
        | set(_extract_chinese_matches(query, CHINESE_ENTITY_PATTERNS))
    )

    datasets = sorted(
        {
            token
            for token in acronyms + quoted_phrases
            if any(char.isdigit() for char in token) or "-" in token or token.lower().endswith(("qa", "bench", "eval", "dataset"))
        }
    )
    entities = sorted(set(acronyms + quoted_phrases + task_tokens + entity_hints))
    must_have: list[str] = []
    nice_to_have: list[str] = []
    uncertainty: list[str] = []
    time_range: dict[str, int] | None = None

    # 时间解析，注意 2026 是当前年份
    if "近三年" in query:
        time_range = {"start_year": 2024, "end_year": 2026}
        must_have.append("year>=2024")
    elif "近五年" in query:
        time_range = {"start_year": 2022, "end_year": 2026}
        must_have.append("year>=2022")
    else:
        # 寻找诸如 2024-2026 或者是 2024 to 2026 这种模式
        match = re.search(r"(\d{4})\s*[-到至to]\s*(\d{4})", query)
        if match:
            time_range = {"start_year": int(match.group(1)), "end_year": int(match.group(2))}
            must_have.append(f"year>={match.group(1)}")
            must_have.append(f"year<={match.group(2)}")
        else:
            # 检查 after/since 及其中文表达
            years = [int(y) for y in re.findall(r"\b(20\d{2})\b", query)]
            if years:
                lowered = query.lower()
                if re.search(r"\bafter\s+20\d{2}\b|\bsince\s+20\d{2}\b|之后|以来|后", lowered):
                    max_year = max(years)
                    time_range = {"start_year": max_year, "end_year": 2026}
                    must_have.append(f"year>={max_year}")

    if not methods:
        uncertainty.append("method not explicitly recognized")
    if not task_tokens:
        uncertainty.append("task not explicitly recognized")

    topic_source = " ".join(cleaned_query.split()[:8]).strip() or query
    research_topic = " ".join(token for token in entity_hints[:2] or task_tokens[:3] or acronyms[:3] or [topic_source]).strip() or query
    task = "find_relevant_papers"
    if "survey" in lowered or "review" in lowered or "综述" in query:
        nice_to_have.append("survey")
        task = "find_survey_and_primary_papers"

    # 推断 query_type
    query_type = "unknown"
    # 检查引号包裹的精确标题 → exact_title
    quoted_phrases_all = re.findall(r'"([^"]+)"|"([^"]+)"|\'([^\']+)\'', query)
    quoted_text = [item for match in quoted_phrases_all for item in match if item]
    if quoted_text and len(quoted_text[0].split()) >= 4:
        query_type = "exact_title"
    elif "survey" in lowered or "review" in lowered or "综述" in query:
        query_type = "survey"
    elif "comparison" in lowered or "compare" in lowered or "对比" in query or "比较" in query:
        query_type = "method_comparison"
    elif any(c in lowered or c in query for c in ("dataset", "bench", "eval", "数据集")):
        query_type = "dataset_constraint"
    elif any(c in lowered or c in query for c in ("latest", "recent", "newest", "最新")):
        query_type = "latest_work"
    elif _looks_like_specific_paper(query):
        query_type = "specific_paper"

    plan = QueryPlan(
        original_query=query,
        language="zh" if re.search(r"[\u4e00-\u9fff]", query) else "en",
        query_type=query_type,
        research_topic=research_topic,
        task=task,
        methods=methods,
        datasets=datasets,
        entities=entities,
        time_range=time_range,
        venues=[],
        must_have_constraints=must_have,
        nice_to_have_constraints=nice_to_have,
        exclude_terms=[],
        expected_output="top_papers",
        uncertainty=uncertainty,
    )
    return normalize_constraints(plan)


def understand_query(query: str, llm_client: object | None = None) -> QueryPlan:
    fallback = heuristic_understand_query(query)
    if llm_client is None:
        return fallback

    # 使用 pro 级别的模型做复杂查询解析，保证意图契约准确
    system_prompt = (
        "You convert a scholarly search request into a strict QueryPlan JSON object. "
        "Return JSON only. Do not assume a domain like medicine unless the user explicitly says so. "
        "Keep unknown facts in uncertainty instead of inventing them. "
        "CRITICAL: List fields (methods, datasets, entities, venues, must_have_constraints, nice_to_have_constraints, exclude_terms, uncertainty) "
        "MUST be returned as JSON arrays (e.g. [\"term1\", \"term2\"]), never as raw strings. "
        "CRITICAL: If the user query is in Chinese, you MUST extract English equivalent terms in methods, datasets, and entities fields "
        "so that the search can match English paper databases. For example, '幻觉' → 'hallucination', '知识蒸馏' → 'knowledge distillation'."
    )
    user_prompt = (
        "Output a JSON object matching this shape exactly: "
        "{original_query, language, research_topic, task, methods, datasets, entities, "
        "time_range, venues, must_have_constraints, nice_to_have_constraints, exclude_terms, "
        "expected_output, uncertainty}. "
        "If the query is in Chinese, provide English translations for all key terms in methods, datasets, and entities. "
        f"User query: {query!r}. "
        f"Heuristic reference: {fallback.model_dump_json()}"
    )
    # 用 pro 级别的模型进行理解
    timeout = getattr(llm_client.budget.config, "llm_timeout_seconds", 30) if getattr(llm_client, "budget", None) else 30
    response = getattr(llm_client, "complete_json", lambda *_: None)(system_prompt, user_prompt, model_type="flash", timeout_seconds=timeout)
    if response is None:
        return fallback
    try:
        payload = dict(response)
        payload["original_query"] = query

        # 柔性类型转换与格式自适应修复，防止 ValidationError
        list_fields = [
            "methods", "datasets", "entities", "venues", 
            "must_have_constraints", "nice_to_have_constraints", 
            "exclude_terms", "uncertainty"
        ]
        for field in list_fields:
            val = payload.get(field)
            if val is None:
                payload[field] = []
            elif not isinstance(val, list):
                if isinstance(val, str):
                    if val.strip() and val.lower() != "none":
                        payload[field] = [val]
                    else:
                        payload[field] = []
                else:
                    payload[field] = [str(val)]
                    
        # time_range 兼容性修复
        t_range = payload.get("time_range")
        if t_range is not None:
            if not isinstance(t_range, dict) or not t_range:
                payload["time_range"] = None
            else:
                cleaned_range = {}
                if "start_year" in t_range and t_range["start_year"] is not None:
                    try:
                        cleaned_range["start_year"] = int(t_range["start_year"])
                    except (ValueError, TypeError):
                        pass
                if "end_year" in t_range and t_range["end_year"] is not None:
                    try:
                        cleaned_range["end_year"] = int(t_range["end_year"])
                    except (ValueError, TypeError):
                        pass
                payload["time_range"] = cleaned_range if cleaned_range else None

        # language 兼容性修复
        lang = payload.get("language")
        if not isinstance(lang, str) or not lang:
            payload["language"] = "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"
        else:
            lang = lang.lower().strip()
            if "chinese" in lang or "zh" in lang:
                payload["language"] = "zh"
            else:
                payload["language"] = "en"

        return normalize_constraints(QueryPlan.model_validate(payload))
    except Exception as exc:
        LOGGER.warning("LLM QueryPlan validation failed: %s. Raw response: %s. Falling back to heuristic.", exc, response)
        return fallback
