import logging
import re

from scholar_agent.models.schemas import QueryPlan

LOGGER = logging.getLogger(__name__)

# Regex pattern for matching year mentions in queries
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')

COMMON_METHOD_HINTS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "deep neural network",
    "neural network",
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
    "generative adversarial network",
    "convolutional neural network",
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
    "watermarking", "detection", "classification", "recognition", "diagnosis",
    "diagnostics", "treatment", "therapy", "screening", "prediction",
    "training efficiency",
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
    "legal text analysis", "legal natural language processing", "legal nlp",
    "machine translation", "question answering", "reading comprehension",
    "text classification", "semantic segmentation", "object detection",
    "face recognition", "facial recognition", "masked face recognition",
    "occluded face recognition", "partial face recognition",
    "face recognition under occlusion", "real-time face recognition",
    "efficient face recognition", "lightweight face recognition",
    "medical imaging", "medical imaging diagnostics", "tumor detection",
    "cancer imaging", "lung cancer", "non-small cell lung cancer",
    "computer vision", "efficient inference", "model acceleration",
    "autonomous driving", "autonomous vehicle", "autonomous vehicles",
    "autonomous vehicle decision making", "robotic decision making",
    "driving decision making",
    "spam detection",
    "speech recognition", "speech data augmentation", "text-to-speech", "image captioning",
    "visual question answering", "dialogue system", "chatbot",
    "few-shot learning", "data-efficient natural language processing",
    "semi-supervised natural language processing", "self-supervised natural language processing",
    "network traffic analysis", "traffic classification", "intrusion detection",
    "real-time traffic classification", "time series classification",
    "multilingual language models", "cross-lingual transfer", "language barrier",
    "machine translation", "reinforcement learning", "long-term reward optimization",
    "sequential decision making",
    "cerebrospinal fluid", "cerebrospinal fluid biomarkers",
    "amyloid beta", "beta-amyloid", "tau protein", "Alzheimer's disease",
    "vaccine development", "mRNA vaccines", "emerging infectious diseases",
    "personalized immunotherapy", "cancer immunotherapy", "precision oncology",
    "precision medicine", "immune checkpoint blockade",
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
        return re.search(r"\blarge(?:[-\s]scale)?\s+language models?\b", lowered) is not None
    return re.search(rf"\b{re.escape(hint)}s?\b", lowered) is not None


def _derived_entity_hints(lowered: str) -> list[str]:
    hints: list[str] = []
    if "sentiment analysis" in lowered and ("context" in lowered or "accuracy" in lowered):
        hints.extend(["context-aware sentiment analysis", "aspect-level sentiment classification"])
    if "spam detection" in lowered:
        hints.append("spam detection")
    if "knowledge distillation" in lowered and ("compress" in lowered or "compression" in lowered):
        hints.append("language model compression")
        if "language model" in lowered:
            hints.append("pre-trained language model")
        if "task-agnostic" in lowered or "task agnostic" in lowered:
            hints.append("task-agnostic knowledge distillation")
    if "legal" in lowered and (
        "language model" in lowered or "large-scale language" in lowered or "llm" in lowered
    ):
        hints.extend(["legal large language models", "legal natural language processing", "automated legal text analysis"])
    if ("face recognition" in lowered or "facial recognition" in lowered) and (
        "mask" in lowered or "masked" in lowered or "occlusion" in lowered or "occluded" in lowered
    ):
        hints.extend([
            "masked face recognition",
            "occluded face recognition",
            "partial face recognition",
            "face recognition under occlusion",
        ])
        if "real-time" in lowered or "real time" in lowered or "processing time" in lowered or "speed" in lowered:
            hints.extend([
                "real-time face recognition",
                "efficient face recognition",
                "lightweight face recognition",
            ])
    if "lung cancer" in lowered:
        hints.append("non-small cell lung cancer")
        if "treatment" in lowered or "therapy" in lowered:
            hints.extend(["targeted lung cancer therapy", "immune checkpoint inhibitors"])
    if "medical imaging" in lowered and ("tumor" in lowered or "cancer" in lowered or "diagnos" in lowered):
        hints.extend(["medical imaging diagnostics", "tumor detection", "cancer imaging"])
    if ("autonomous" in lowered or "driving" in lowered) and (
        "decision" in lowered or "vehicle" in lowered or "robot" in lowered
    ):
        hints.extend(["autonomous driving", "autonomous vehicle decision making", "driving decision making"])
    if "speech recognition" in lowered and ("gan" in lowered or "generative adversarial" in lowered):
        hints.extend(["speech recognition", "speech data augmentation"])
    if ("few-shot" in lowered or "few shot" in lowered or "unlabeled" in lowered) and (
        "nlp" in lowered or "natural language" in lowered
    ):
        hints.extend(["few-shot learning", "data-efficient natural language processing", "semi-supervised natural language processing"])
    if "network traffic" in lowered or ("real-time" in lowered and "traffic" in lowered):
        hints.extend(["network traffic analysis", "traffic classification", "intrusion detection"])
    if "multilingual" in lowered or "language barrier" in lowered:
        hints.extend(["multilingual language models", "cross-lingual transfer", "machine translation"])
    if "reinforcement learning" in lowered and ("long-term" in lowered or "long term" in lowered or "reward" in lowered):
        hints.extend(["long-term reward optimization", "sequential decision making"])
    if ("inference" in lowered or "processing time" in lowered or "speed" in lowered) and (
        "computer vision" in lowered or "deep learning" in lowered or "deep neural network" in lowered
    ):
        hints.extend(["efficient inference", "model acceleration", "computer vision"])
    if (
        "cerebrospinal fluid" in lowered
        or "amyloid" in lowered
        or "β-amyloid" in lowered
        or "aβ" in lowered
    ) and ("alzheimer" in lowered or "tau" in lowered):
        hints.extend(["cerebrospinal fluid biomarkers", "amyloid beta", "tau protein", "Alzheimer's disease"])
    if "vaccine" in lowered and ("infectious" in lowered or "development" in lowered):
        hints.extend(["vaccine development", "mRNA vaccines", "emerging infectious diseases"])
    if "immunotherapy" in lowered and "cancer" in lowered:
        hints.extend(["personalized immunotherapy", "cancer immunotherapy", "precision oncology", "immune checkpoint blockade"])
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
    elif any(p in lowered for p in ("all papers", "papers that discuss", "papers about",
                                     "papers explaining", "papers that show",
                                     "list research", "show me research",
                                     "papers that share", "provide me with all",
                                     "could you list research", "papers on")) and not re.search(r'\([A-Z]{2,8}\)', query):
        # broad_topic only when no specific technique abbreviation (e.g. "(QAT)")
        # is present — those indicate focused method queries, not broad surveys
        query_type = "broad_topic"

    return QueryPlan(
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


def understand_query(
    query: str,
    llm_client: object | None = None,
    feedback: str | None = None,
    prev_plan: QueryPlan | None = None,
) -> QueryPlan:
    """LLM-first 查询理解。启发式仅作 fallback，不注入 LLM prompt 避免锚定。

    Args:
        query: 原始用户查询
        llm_client: LLM 客户端，None 时直接返回启发式结果
        feedback: 上一轮 LLM pool review 的修正建议（闭环迭代时传入）
        prev_plan: 上一轮的 QueryPlan（闭环迭代时传入，用于在追加语义上扩展而非重置）
    """
    fallback = heuristic_understand_query(query)
    if llm_client is None:
        return fallback

    # LLM-first：system prompt 引导独立分析，不提供启发式参考
    system_prompt = (
        "You are an expert scholarly search intent analyst. Your job is to INDEPENDENTLY analyze "
        "the user's research query and produce a strict QueryPlan JSON object. "
        "Do not guess or invent facts — keep uncertain aspects in the 'uncertainty' field. "
        "Think step by step: first identify the core research topic, then extract methods/datasets/entities, "
        "then determine the query_type, then list hard constraints. "
        "Return JSON only, no explanation text.\n"
        "CRITICAL: List fields (methods, datasets, entities, venues, must_have_constraints, "
        "nice_to_have_constraints, exclude_terms, uncertainty) MUST be JSON arrays, never raw strings. "
        "CRITICAL: If the user query is in Chinese, you MUST extract English equivalent terms in "
        "methods, datasets, and entities fields so the search can match English paper databases. "
        "For example, '幻觉' → 'hallucination', '知识蒸馏' → 'knowledge distillation'."
    )

    feedback_clause = ""
    if feedback:
        feedback_clause = (
            f"\n\nPREVIOUS ROUND FEEDBACK (from LLM pool review of candidate papers):\n{feedback}\n"
            "Based on this feedback, REFINE your understanding: add any missing methods/datasets/entities "
            "that the candidate pool failed to cover. Do NOT remove previously identified terms — only ADD. "
            "If the feedback suggests the query_type was wrong, correct it."
        )

    prev_plan_clause = ""
    if prev_plan is not None:
        prev_plan_clause = (
            f"\n\nPREVIOUS QueryPlan (refine on this basis, do not narrow):\n"
            f"{prev_plan.model_dump_json()}"
        )

    user_prompt = (
        "Analyze the following scholarly search query INDEPENDENTLY and output a JSON object matching "
        "this shape exactly: "
        "{original_query, language, research_topic, task, query_type, methods, datasets, entities, "
        "time_range, venues, must_have_constraints, nice_to_have_constraints, exclude_terms, "
        "expected_output, uncertainty}. "
        "query_type must be one of: 'exact_title', 'single_gold', 'specific_paper', "
        "'dataset_constraint', 'method_comparison', 'latest_work', 'broad_topic', 'survey', 'unknown'. "
        "Use 'survey' for comprehensive reviews/overviews, 'broad_topic' for wide research areas, "
        "'method_comparison' for comparing approaches, 'specific_paper' for finding particular papers, "
        "'exact_title'/'single_gold' when a specific title is mentioned. "
        "IMPORTANT: Do NOT use 'broad_topic' if the query mentions a specific technique, method, or acronym "
        "in parentheses (e.g. '(QAT)', '(BERT)') — use 'specific_paper' instead. "
        "If the query is in Chinese, provide English translations for all key terms in methods, datasets, "
        "and entities. "
        f"User query: {query!r}.{feedback_clause}{prev_plan_clause}"
    )

    response = getattr(llm_client, "complete_json", lambda *_: None)(system_prompt, user_prompt, model_type="flash")
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

        # 闭环迭代：如果传入了 prev_plan，合并语义字段（只追加不删除）
        if prev_plan is not None:
            for field in list_fields:
                prev_vals = set(getattr(prev_plan, field, []) or [])
                curr_vals = payload.get(field) or []
                # 追加 prev 中有但 curr 没有的，保留 curr 新增的
                merged = list(dict.fromkeys(curr_vals + [v for v in prev_vals if v not in set(curr_vals)]))
                payload[field] = merged
            # query_type：如果 LLM 改了就用 LLM 的，否则保留 prev
            if not payload.get("query_type") or payload.get("query_type") == "unknown":
                payload["query_type"] = prev_plan.query_type
        else:
            # 首轮：LLM 缺 query_type 时从 fallback 补
            if not payload.get("query_type") or payload.get("query_type") == "unknown":
                payload["query_type"] = fallback.query_type

        return QueryPlan.model_validate(payload)
    except Exception as exc:
        LOGGER.warning("LLM QueryPlan validation failed: %s. Raw response: %s. Falling back to heuristic.", exc, response)
        return fallback
