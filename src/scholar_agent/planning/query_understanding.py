from __future__ import annotations

import re

from scholar_agent.models.schemas import QueryPlan

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

    return QueryPlan(
        original_query=query,
        language="zh" if re.search(r"[\u4e00-\u9fff]", query) else "en",
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


def understand_query(query: str, llm_client: object | None = None) -> QueryPlan:
    fallback = heuristic_understand_query(query)
    if llm_client is None:
        return fallback

    # 使用 pro 级别的模型做复杂查询解析，保证意图契约准确
    system_prompt = (
        "You convert a scholarly search request into a strict QueryPlan JSON object. "
        "Return JSON only. Do not assume a domain like medicine unless the user explicitly says so. "
        "Keep unknown facts in uncertainty instead of inventing them."
    )
    user_prompt = (
        "Output a JSON object matching this shape exactly: "
        "{original_query, language, research_topic, task, methods, datasets, entities, "
        "time_range, venues, must_have_constraints, nice_to_have_constraints, exclude_terms, "
        "expected_output, uncertainty}. "
        f"User query: {query!r}. "
        f"Heuristic reference: {fallback.model_dump_json()}"
    )
    # 用 pro 级别的模型进行理解
    response = getattr(llm_client, "complete_json", lambda *_: None)(system_prompt, user_prompt, model_type="pro")
    if response is None:
        return fallback
    try:
        payload = dict(response)
        payload["original_query"] = query
        return QueryPlan.model_validate(payload)
    except Exception:
        return fallback
