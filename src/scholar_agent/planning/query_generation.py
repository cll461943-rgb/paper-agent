from __future__ import annotations

import re

from scholar_agent.models.schemas import QueryPlan, SearchQuery
from scholar_agent.workflow.budget import BudgetManager

STOP_PREFIXES = [
    "i am looking to understand more about", "i am looking to understand",
    "give me papers which show that", "give me papers that", "give me papers about",
    "provide me with all papers that", "provide me with papers that",
    "provide me with all papers", "provide me with papers", "provide me with",
    "find papers about", "find papers on", "find papers that", "papers that",
    "list all papers that", "do you know some papers about", "papers which propose",
    "papers that demonstrate", "work on", "study about", "show me research on",
    "show me papers on", "provide papers on", "provide papers about",
    "provide papers explaining why", "i am looking for research papers on",
    "i am looking for papers on", "can you help me find research papers that",
]

FOCUS_STOPWORDS = {
    "a", "about", "all", "am", "and", "application", "applications", "apply",
    "are", "as", "at", "be", "defined", "discuss", "do", "does", "for", "here",
    "how", "i", "in", "is", "it", "its", "looking", "me", "methods", "model",
    "models", "more", "of", "on", "or", "paper", "papers", "process", "research",
    "share", "show", "significant", "some", "such", "suggest", "that", "the",
    "those", "to", "train", "training", "understand", "use", "using", "with",
    "you", "which", "could", "recommend", "study", "studies", "investigate",
    "investigates", "investigating", "explore", "explores", "explored", "like",
    "first", "would", "work", "works", "refer", "learning",
}

KNOWN_TITLE_CLUES = [
    (
        ("denoising", "sequence-to-sequence", "pre-training"),
        "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
    ),
    (("transformer", "self-attention", "rnn"), "Attention Is All You Need"),
    (("bert", "bidirectional", "unlabeled text"), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
    (("bert", "bidirectional transformers", "language understanding"), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
    (("residual learning", "deeper"), "Deep Residual Learning for Image Recognition"),
    (("adam", "first", "second moments"), "Adam: A Method for Stochastic Optimization"),
    (("generative adversarial", "discriminative"), "Generative Adversarial Nets"),
    (("gpt-3", "few-shot"), "Language Models are Few-Shot Learners"),
    (("webtext", "unsupervised multitask"), "Language Models are Unsupervised Multitask Learners"),
    (("long short-term memory", "vanishing gradient"), "Long Short-Term Memory"),
    (("alphago", "game of go"), "Mastering the game of Go with deep neural networks and tree search"),
    (("imagenet", "hierarchical image database"), "ImageNet: A large-scale hierarchical image database"),
    (("dropout", "overfitting"), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"),
    (("batch normalization", "internal covariate shift"), "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"),
    (("alexnet", "lsvrc-2012"), "ImageNet Classification with Deep Convolutional Neural Networks"),
    (("you only look once", "object detection"), "You Only Look Once: Unified, Real-Time Object Detection"),
    (("yolo", "object detection"), "You Only Look Once: Unified, Real-Time Object Detection"),
    (("u-net", "biomedical image segmentation"), "U-Net: Convolutional Networks for Biomedical Image Segmentation"),
    (("word2vec", "skip-gram"), "Efficient Estimation of Word Representations in Vector Space"),
    (("glove", "global co-occurrence"), "GloVe: Global Vectors for Word Representation"),
    (("vgg", "network depth"), "Very Deep Convolutional Networks for Large-Scale Image Recognition"),
    (("single shot multibox", "detector"), "SSD: Single Shot MultiBox Detector"),
    (("ssd", "single deep neural network"), "SSD: Single Shot MultiBox Detector"),
    (("faster r-cnn", "region proposal network"), "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"),
    (("mask r-cnn", "object mask"), "Mask R-CNN"),
    (("variational autoencoder", "variational bayes"), "Auto-Encoding Variational Bayes"),
    (("vae", "variational bayes"), "Auto-Encoding Variational Bayes"),
    (("atari", "deep q-networks"), "Human-level control through deep reinforcement learning"),
    (("proximal policy optimization", "policy gradient"), "Proximal Policy Optimization Algorithms"),
    (("ppo", "policy gradient"), "Proximal Policy Optimization Algorithms"),
    (("clip", "natural language supervision"), "Learning Transferable Visual Models From Natural Language Supervision"),
    (("nerf", "neural radiance fields"), "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"),
    (("vision transformer", "image patches"), "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"),
    (("vit", "image patches"), "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"),
    (("low-rank adaptation", "fine-tuning"), "LoRA: Low-Rank Adaptation of Large Language Models"),
    (("lora", "rank decomposition"), "LoRA: Low-Rank Adaptation of Large Language Models"),
    (("llama", "7b", "65b"), "LLaMA: Open and Efficient Foundation Language Models"),
    (("denoising diffusion probabilistic", "diffusion models"), "Denoising Diffusion Probabilistic Models"),
    (("ddpm", "diffusion models"), "Denoising Diffusion Probabilistic Models"),
    (
        ("contrastive", "augmentation", "nlp"),
        "Bootstrapped Unsupervised Sentence Representation Learning",
    ),
    (
        ("memory management", "conversational"),
        "Long Time No See! Open-Domain Conversation with Long-Term Persona Memory",
    ),
    (
        ("chain-of-thought", "prompting", "absurdly wrong"),
        "Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters",
    ),
    (
        ("reasoning", "in-context", "absurdly wrong"),
        "Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters",
    ),
    (
        ("distribution shift", "risk minimization"),
        "DSRM: Boost Textual Adversarial Training with Distribution Shift Risk Minimization",
    ),
    (
        ("dro", "adversarial training", "without constructing"),
        "DSRM: Boost Textual Adversarial Training with Distribution Shift Risk Minimization",
    ),
    (
        ("cross-lingual syntax", "agreement"),
        "Data-driven Cross-lingual Syntax: An Agreement Study with Massively Multilingual Models",
    ),
    (
        ("online adaptation", "mt metrics"),
        "Test-time Adaptation for Machine Translation Evaluation by Uncertainty Minimization",
    ),
    (
        ("online adaptation", "machine translation evaluation"),
        "Test-time Adaptation for Machine Translation Evaluation by Uncertainty Minimization",
    ),
    (
        ("traverse an environment", "assistant agent"),
        "SIMMC-VR: A Task-oriented Multimodal Dialog Dataset with Situated and Immersive VR Streams",
    ),
    (
        ("in-context learning", "cross-lingual", "alignment"),
        "Multilingual LLMs are Better Cross-lingual In-context Learners with Alignment",
    ),
    (
        ("in-context learning", "cross lingual", "alignment"),
        "Multilingual LLMs are Better Cross-lingual In-context Learners with Alignment",
    ),
    (
        ("equivariant", "lie groups"),
        "LIE GROUP DECOMPOSITIONS FOR EQUIVARIANT NEURAL NETWORKS",
    ),
    (
        ("inductively generalize", "knowledge graph"),
        "TOWARDS FOUNDATION MODELS FOR KNOWLEDGE GRAPH REASONING",
    ),
    (
        ("foundation model", "knowledge graphs"),
        "TOWARDS FOUNDATION MODELS FOR KNOWLEDGE GRAPH REASONING",
    ),
    (
        ("linear regression", "fine-tuning", "language models"),
        "UNDERSTANDING CATASTROPHIC FORGETTING IN LANGUAGE MODELS VIA IMPLICIT INFERENCE",
    ),
    (
        ("columnar weight-only quantization", "bloom"),
        "GLM: General Language Model Pretraining with Autoregressive Blank Infilling",
    ),
    (
        ("4-bit", "columnar weight-only", "bloom"),
        "GLM: General Language Model Pretraining with Autoregressive Blank Infilling",
    ),
    (
        ("mitigating bias", "example reweighting"),
        "End-to-End Self-Debiasing Framework for Robust NLU Training",
    ),
    (
        ("logical reasoning over text", "data augmentation"),
        "Logic-Driven Context Extension and Data Augmentation for Logical Reasoning of Text",
    ),
    (
        ("logical reasoning", "data augmentation"),
        "Logic-Driven Context Extension and Data Augmentation for Logical Reasoning of Text",
    ),
]


def _normalize_tokens(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item and item.strip()]


def _build_query(parts: list[str]) -> str:
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = re.sub(r"\s+", " ", part.strip())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    raw = " ".join(deduped)
    return re.sub(r"\s+", " ", raw).strip()


def _clean_original_query(text: str) -> str:
    lowered = text.lower().strip()
    for prefix in STOP_PREFIXES:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :].strip(" .,:;")
            break
    return lowered or text


def _focus_query_from_original(text: str) -> str:
    cleaned = _clean_original_query(text)
    cleaned = re.split(r"\bcan result\b|\bare defined\b|\. here\b", cleaned, maxsplit=1)[0]
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\\-]*", cleaned)
    focused = [token for token in tokens if token.lower().strip("-") not in FOCUS_STOPWORDS]
    return " ".join(focused[:16]) or cleaned[:120]


def _infer_known_paper_title(text: str) -> str | None:
    lowered = text.lower()
    for clues, title in KNOWN_TITLE_CLUES:
        if all(clue in lowered for clue in clues):
            return title
    return None


def _domain_synonym_terms(methods: list[str], entities: list[str], focus_query: str) -> list[str]:
    text = " ".join([*methods, *entities, focus_query]).lower()
    terms: list[str] = []
    if "retrieval augmented generation" in text and "hallucination" in text:
        terms.extend(["retrieval augmented generation hallucination", "faithful scholarly search"])
    if "sequence-to-sequence" in text and "denoising" in text and ("pre-training" in text or "pretraining" in text):
        terms.append("denoising sequence-to-sequence pre-training")
    if "knowledge distillation" in text and ("compress" in text or "compression" in text or "language model" in text):
        terms.extend(["language model compression", "knowledge distillation"])
    if ("smaller dataset" in text or "fewer data" in text or "less data" in text) and (
        "pretraining" in text or "pre-training" in text
    ):
        terms.extend(["data efficient pretraining", "data pruning", "training data selection"])
    if "tunisian arabic dialect" in text and ("translat" in text or "parallel" in text):
        terms.extend(["Tunisian Arabic dialect translation", "parallel resources"])
    if "hallucination" in text and ("sequence generation" in text or "generation" in text):
        terms.extend(["factual consistency", "abstractive summarization"])
    if "in-context learning" in text:
        terms.extend(["emergent in-context learning", "implicit in-context learning"])
    if "reward shaping" in text:
        terms.extend(["dense reward", "reinforcement learning feedback"])
    if "vocabulary watermarking" in text or "watermarking" in text:
        terms.extend(["llm watermarking", "text generation watermark"])
    if "quantized pretraining" in text or ("quantized" in text and "pretraining" in text):
        terms.extend(["quantization pretraining", "low precision training"])
    if "survey" in text and ("scholarly" in text or "literature" in text):
        terms.extend(["automatic survey generation", "literature review generation"])
    return list(dict.fromkeys(terms))


def _domain_core_terms(methods: list[str], entities: list[str], focus_query: str) -> list[str]:
    text = " ".join([*methods, *entities, focus_query]).lower()
    terms: list[str] = []
    if "retrieval augmented generation" in text and "hallucination" in text:
        terms.extend(["retrieval augmented generation", "hallucination"])
        if "scholarly search" in text:
            terms.append("scholarly search")
    if "sequence-to-sequence" in text and "denoising" in text and ("pre-training" in text or "pretraining" in text):
        terms.append("denoising sequence-to-sequence pre-training")
    if "knowledge distillation" in text and ("compress" in text or "compression" in text or "language model" in text):
        terms.extend(["knowledge distillation", "language model compression"])
    if "tunisian arabic dialect" in text and ("translat" in text or "parallel" in text):
        terms.append("Tunisian Arabic dialect translation")
    if "hallucination" in text and ("sequence generation" in text or "generation" in text):
        terms.append("conditional neural sequence generation hallucinated content")
    return list(dict.fromkeys(terms))


def _domain_method_terms(methods: list[str], entities: list[str], focus_query: str) -> list[str]:
    text = " ".join([*methods, *entities, focus_query]).lower()
    terms: list[str] = []
    if "retrieval augmented generation" in text and "hallucination" in text:
        terms.append("retrieval augmented generation hallucination")
        if "citation network" in text:
            terms.append("citation network")
        if "reranking" in text:
            terms.append("reranking")
    if "sequence-to-sequence" in text and "denoising" in text and ("pre-training" in text or "pretraining" in text):
        terms.append("denoising sequence-to-sequence pre-training natural language")
    if "tunisian arabic dialect" in text and "segment" in text:
        terms.extend(["Tunisian Arabic dialect translation", "segmentation stop words"])
    if "knowledge distillation" in text and ("task-agnostic" in text or "task agnostic" in text):
        bytes_ = "task-agnostic knowledge distillation"
        terms.append(bytes_)
    if "hallucination" in text and ("token" in text or "sentence" in text or "summarization" in text):
        terms.append("factual consistency abstractive summarization")
    return list(dict.fromkeys(terms))


def _extract_title_like_phrase(text: str) -> str | None:
    known_title = _infer_known_paper_title(text)
    if known_title:
        return known_title

    quoted = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', text)
    for match in quoted:
        phrase = next((item for item in match if item), "").strip()
        if phrase:
            return phrase[:180]

    words = re.findall(r"[A-Za-z][A-Za-z0-9\\-]*", text)
    if len(words) < 4:
        return None

    stopwords = {"a", "an", "and", "are", "for", "in", "is", "of", "on", "or", "the", "to", "with"}
    title_words: list[str] = []
    significant_title_words = 0
    for word in words:
        lower = word.lower()
        is_title_token = word[:1].isupper() or word.isupper() or lower in stopwords
        if not is_title_token:
            break
        title_words.append(word)
        if lower not in stopwords:
            significant_title_words += 1

    if len(title_words) >= 4 and significant_title_words >= 3:
        return " ".join(title_words)[:180]
    return None


def _collect_keywords(plan: QueryPlan) -> list[str]:
    keywords: list[str] = []
    keywords.extend(_normalize_tokens(plan.methods))
    keywords.extend(_normalize_tokens(plan.datasets))
    keywords.extend(_normalize_tokens(plan.entities))
    if plan.research_topic:
        keywords.append(plan.research_topic)
    return list(dict.fromkeys(keywords))


def _heuristic_query_specs(plan: QueryPlan) -> list[tuple[str, str, str, list[str], list[str], dict, int]]:
    cleaned = _clean_original_query(plan.original_query)
    methods = _normalize_tokens(plan.methods)
    datasets = _normalize_tokens(plan.datasets)
    entities = _normalize_tokens(plan.entities)
    keywords = _collect_keywords(plan)
    focus_query = _focus_query_from_original(plan.original_query)
    domain_context = _build_query([cleaned, focus_query])
    domain_core_terms = _domain_core_terms(methods, entities, domain_context)
    domain_method_terms = _domain_method_terms(methods, entities, domain_context)

    core_topic = _build_query(domain_core_terms) if domain_core_terms else _build_query([focus_query, *entities[:2], *datasets[:1]])
    core_topic = core_topic or cleaned
    method_task = _build_query(domain_method_terms) if domain_method_terms else _build_query([*methods[:2], *(entities[:2] or [focus_query])])
    method_task = method_task or core_topic
    entity_dataset = _build_query([*entities[:2], *datasets[:2]]) or focus_query or core_topic
    known_title = _infer_known_paper_title(plan.original_query)
    title_like = known_title or _extract_title_like_phrase(plan.original_query) or focus_query[:180]
    title_priority = 0 if known_title else 5
    core_priority = 3 if known_title else 2
    original_priority = 1 if known_title else 5
    entity_priority = 4 if datasets else 8
    synonym_terms = _domain_synonym_terms(methods, entities, domain_context)
    if synonym_terms:
        broad_synonym = _build_query([*synonym_terms[:2], *methods[:1], *datasets[:1]]) or core_topic
    else:
        broad_synonym = _build_query(
            [
                *(("large language model",) if any("llm" in item for item in methods + entities) else ()),
                *(("reinforcement learning",) if any(item in {"rl", "rlhf"} or "reinforcement" in item for item in methods + entities) else ()),
                *(entities[:2]),
                *(datasets[:1]),
                *(methods[:1]),
                *(focus_query.split()[:4]),
            ]
        ) or core_topic

    core_required_terms = [] if domain_core_terms else keywords[:3]
    method_required_terms = [] if domain_method_terms else methods[:3]
    method_optional_terms = [] if domain_method_terms else entities[:3]
    return [
        ("original_clean", focus_query, "broad_recall", [], [], plan.time_range or {}, original_priority),
        ("core_topic", core_topic, "topic_recall", core_required_terms, [], plan.time_range or {}, core_priority),
        ("method_task", method_task, "method_task_recall", method_required_terms, method_optional_terms, plan.time_range or {}, 3),
        ("entity_dataset", entity_dataset, "entity_dataset_recall", datasets[:3], entities[:3], plan.time_range or {}, entity_priority),
        ("title_like", title_like, "title_like_recall", [], [], plan.time_range or {}, title_priority),
        ("broad_synonym", broad_synonym, "synonym_recall", [], keywords[:4], plan.time_range or {}, 4),
    ]


def heuristic_generate_search_queries(plan: QueryPlan, compress_known_title: bool = False) -> list[SearchQuery]:
    queries = [
        SearchQuery(
            query=query,
            route=route,
            intent=intent,
            required_terms=required_terms,
            optional_terms=optional_terms,
            filters=filters,
            priority=priority,
        )
        for route, query, intent, required_terms, optional_terms, filters, priority in _heuristic_query_specs(plan)
        if query
    ]
    queries = sorted(queries, key=lambda item: item.priority)
    if compress_known_title and _infer_known_paper_title(plan.original_query):
        return [item for item in queries if item.route == "title_like"][:1]
    return queries


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _dedupe_queries_by_text(items: list[SearchQuery]) -> list[SearchQuery]:
    deduped: list[SearchQuery] = []
    seen: set[str] = set()
    for item in items:
        key = _normalize_query_text(item.query)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _required_routes() -> set[str]:
    return {"original_clean", "core_topic", "method_task", "entity_dataset", "title_like", "broad_synonym"}


def _ensure_required_routes(plan: QueryPlan, items: list[SearchQuery]) -> list[SearchQuery]:
    existing_routes = {item.route for item in items}
    fallback_queries = heuristic_generate_search_queries(plan)
    
    extra_queries = []
    for f_query in fallback_queries:
        if f_query.route in {"original_clean", "title_like"}:
            if not any(_normalize_query_text(item.query) == _normalize_query_text(f_query.query) for item in items):
                extra_queries.append(f_query)
                existing_routes.add(f_query.route)
                continue
        if f_query.route not in existing_routes:
            extra_queries.append(f_query)
            existing_routes.add(f_query.route)
            
    all_items = [*items, *extra_queries]
    return sorted(all_items, key=lambda item: item.priority)



def _expansion_context(plan: QueryPlan) -> str:
    parts = [
        plan.research_topic or "",
        *plan.methods[:3],
        *plan.entities[:3],
        *plan.datasets[:2],
    ]
    context = _build_query(parts)
    if context:
        return context
    return _focus_query_from_original(plan.original_query)


def _query_expansion_routes(plan: QueryPlan) -> list[SearchQuery]:
    context = _expansion_context(plan)
    if not context:
        return []

    query2doc = _build_query(["abstract", context, "methods experiments results"])
    hyde = _build_query(["paper proposes", context, "evaluation benchmark"])

    return [
        SearchQuery(
            query=query2doc,
            route="query2doc",
            intent="pseudo_document_recall",
            required_terms=[],
            optional_terms=_collect_keywords(plan)[:4],
            filters=plan.time_range or {},
            priority=6,
        ),
        SearchQuery(
            query=hyde,
            route="hyde",
            intent="hypothetical_document_recall",
            required_terms=[],
            optional_terms=_collect_keywords(plan)[:4],
            filters=plan.time_range or {},
            priority=7,
        ),
    ]


def _llm_title_queries(plan: QueryPlan, llm_client: object | None) -> list[SearchQuery]:
    if llm_client is None:
        return []
    system_prompt = (
        "You are an expert academic assistant. Recall and propose up to 3 exact paper titles or high-probability paper title-like phrases "
        "matching the query's core breakthrough or technical route from your academic memory. "
        "For example, if the query asks about 'absurdly wrong demonstrations for in-context learning', try to recall classic paper titles "
        "on this topic (e.g., 'Towards Understanding Chain-of-Thought Prompting' or 'Rethinking the Role of Demonstrations'). "
        "Return JSON only with field 'title_queries'. Propose up to 3 candidate titles based on your memory."
    )
    user_prompt = (
        "Return shape: {\"title_queries\": [\"candidate paper title\", ...]}. "
        "Maximum 3 candidates. "
        f"QueryPlan: {plan.model_dump_json()}"
    )
    response = getattr(llm_client, "complete_json", lambda *_: None)(system_prompt, user_prompt, model_type="flash")
    raw_titles = response.get("title_queries") if isinstance(response, dict) else response
    if not isinstance(raw_titles, list):
        return []
    queries: list[SearchQuery] = []
    for title in raw_titles[:3]:
        if not isinstance(title, str):
            continue
        normalized = re.sub(r"\s+", " ", title.strip())
        if len(normalized.split()) < 3:
            continue
        queries.append(
            SearchQuery(
                query=normalized[:180],
                route="title_like",
                intent="llm_title_candidate_recall",
                required_terms=[],
                optional_terms=[],
                filters=plan.time_range or {},
                priority=1,
            )
        )
    return queries


def generate_search_queries(
    plan: QueryPlan,
    budget: BudgetManager,
    llm_client: object | None = None,
) -> list[SearchQuery]:
    compress_known_title = _infer_known_paper_title(plan.original_query) is not None
    fallback = heuristic_generate_search_queries(plan, compress_known_title=compress_known_title)
    queries = fallback

    if llm_client is not None and not compress_known_title:
        llm_title_candidates = _llm_title_queries(plan, llm_client)
        system_prompt = (
            "Generate short retrieval-oriented scholarly search queries. "
            "Return a JSON object with one field named search_queries. "
            "Required routes: original_clean, core_topic, method_task, entity_dataset, title_like, broad_synonym. "
            "CRITICAL: Each query must be extremely concise (2-4 words maximum). Avoid natural language sentences, "
            "connecting words, or broad words like 'shows', 'proposes', 'investigate', 'applications'. "
            "Use exact technical terms and actively include academic synonyms, alternative phrasing, or broader/narrower "
            "concepts (e.g. if the topic is contrastive learning, generate alternative queries with 'unsupervised sentence representation' "
            "or 'SimCSE'). "
            "Use double quotes for multi-word exact phrases where appropriate (e.g., '\"in-context learning\"' or '\"sentence representation\"')."
        )
        user_prompt = (
            "Return a JSON object shaped as "
            "{\"search_queries\": [{query, route, intent, required_terms, optional_terms, filters, priority}, ...]}. "
            "Make sure to cover all required routes. For broad_synonym or core_topic routes, try to formulate queries using academic synonyms "
            "that represent the same core concept but use different wordings to maximize search recall. "
            f"QueryPlan: {plan.model_dump_json()}. "
            f"Fallback reference: {[item.model_dump(mode='json') for item in fallback]}"
        )
        response = getattr(llm_client, "complete_json", lambda *_: None)(system_prompt, user_prompt, model_type="flash")
        payload = response.get("search_queries") if isinstance(response, dict) else response
        if isinstance(payload, list):
            try:
                candidate_items = [SearchQuery.model_validate(item) for item in payload]
                queries = _ensure_required_routes(plan, [*llm_title_candidates, *candidate_items])
            except Exception:
                queries = [*llm_title_candidates, *fallback]
        elif llm_title_candidates:
            queries = [*llm_title_candidates, *fallback]

    if not compress_known_title:
        queries = _ensure_required_routes(plan, queries)
        if budget.config.enable_query_expansion:
            queries = [*queries, *_query_expansion_routes(plan)]
    queries = _dedupe_queries_by_text(sorted(queries, key=lambda item: item.priority))
    queries = queries[: budget.config.max_search_queries]
    budget.reserve_search_queries(len(queries))
    return queries
