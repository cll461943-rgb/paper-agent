# Implementation Plan: Enhancing Scholar Agent Generalization Performance (Recall & F1) and Latency Control

We will implement a comprehensive set of improvements to fix the critical generalization and retrieval bugs in the Scholar Agent. The goal is to maximize the recall and F1 metrics on the full evaluation dataset while strictly ensuring that the query wall time stays under 60 seconds.

## User Review Required

> [!IMPORTANT]
> 1. We will lift the strict truncation constraint (which currently limits OpenAlex and ArXiv queries to only the first 5 words). Instead, we will prioritize key terms (methods and entities) and allow up to 10 terms for title-like or specialized semantic queries.
> 2. We will increase the number of parallel queries processed by online retrievers from 4 to 6 in Round 1 to ensure that broader synonym expansion paths are queried.
> 3. We will change the evidence selection `batch_size` from 5 to 10. This halves the concurrent DeepSeek connections (from 4 down to 2), improving rate-limit resilience and network stability.

## Open Questions

None at the moment. We have diagnosed the root causes of the zero-recall issue on non-Case-1 queries.

---

## Proposed Changes

### Component: Query Understanding & Validation
#### [MODIFY] [query_understanding.py](file:///c:/Users/33316/Desktop/claude-code-src-main/paper-agent/src/scholar_agent/planning/query_understanding.py)
- Prevent the silent discarding of `QueryPlan` when Pydantic throws a `ValidationError`.
- Log the validation exception details for debugging.
- Add robust automatic type coercion for fields (e.g., if `uncertainty` is returned as a raw string `"low"` instead of a list, wrap it in a list `["low"]` before validation).
- Refine LLM system instructions to enforce strict JSON array structures for list fields.

---

### Component: Query Generation & Synonym Expansion
#### [MODIFY] [query_generation.py](file:///c:/Users/33316/Desktop/claude-code-src-main/paper-agent/src/scholar_agent/planning/query_generation.py)
- Refactor the query generation LLM prompt to actively encourage conceptual synonym expansion (e.g., expanding `"contrastive learning in NLP"` to `"sentence representation learning"`, `"SimCSE"`, `"sentence embeddings"`).
- Remove static case-specific heuristics and make fallback entity extraction more robust.

---

### Component: Retrieval Sanitization & AND Constraints
#### [MODIFY] [openalex.py](file:///c:/Users/33316/Desktop/claude-code-src-main/paper-agent/src/scholar_agent/retrieval/openalex.py)
- Fix the bug where queries longer than 5 words are blindly truncated, dropping crucial search terms.
- Skip raw concatenation of `required_terms` and `optional_terms` if they are already embedded in the query or if the query is an LLM-synthesized retrieval query.
- Prioritize core concepts (methods, entities) when sanitizing queries.

#### [MODIFY] [arxiv.py](file:///c:/Users/33316/Desktop/claude-code-src-main/paper-agent/src/scholar_agent/retrieval/arxiv.py)
- Align sanitization and length limits with OpenAlex, allowing longer queries when title-like or phrase-based.
- Preserve quoted phrases to execute exact phrase matching instead of breaking them into separate words.

---

### Component: Retrieval Dispatching
#### [MODIFY] [multi_route.py](file:///c:/Users/33316/Desktop/claude-code-src-main/paper-agent/src/scholar_agent/retrieval/multi_route.py)
- Increase online provider query limits from `queries[:4]` to `queries[:6]` in Round 1 to search broader semantic routes concurrently.

---

### Component: Concurrency & Reliability
#### [MODIFY] [evidence_selector.py](file:///c:/Users/33316/Desktop/claude-code-src-main/paper-agent/src/scholar_agent/selection/evidence_selector.py)
- Increase `batch_size` from 5 to 10 in `select_and_extract_evidence` to reduce concurrent HTTP threads and avoid hitting DeepSeek rate limits.
- Add an explicit timeout parameter to the LLM client request during selection (e.g. 20s) and immediately fallback to heuristic selections upon timeout.

---

## Verification Plan

### Automated Baseline and Post-Optimization Test
We run the 5-case evaluation benchmark:
```bash
C:\Users\33316\AppData\Local\conda\conda\envs\paper-agent\python.exe evaluate.py --cases 1,7,12,15,16
```
We will verify that:
1. **F1 & Recall**: F1 on Case 7, 12, 15, 16 is significantly boosted above 0.0.
2. **Latency**: Average query duration (`avg_wall_time`) is controlled within 60 seconds.
3. **Pydantic Validation**: Zero silent validation fallbacks occur.
