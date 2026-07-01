"""Query Expansion Engine — acronym expansion, synonym lookup, candidate feedback.

Core principle: expand queries using academic domain knowledge, not LLM.
After the first retrieval round, extract terms from top candidates to
generate feedback-based expansion queries for subsequent rounds.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any
from pathlib import Path

import yaml

from scholar_agent.models.schemas import Paper, QueryPlan, SearchQuery

LOGGER = logging.getLogger(__name__)

# Default path to the taxonomy file
DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "academic_taxonomy.yaml"

# Stopwords for term extraction
_STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "to", "for", "with", "on", "at",
    "by", "from", "that", "this", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must",
    "can", "shall", "or", "but", "not", "no", "nor", "so", "yet", "if",
    "then", "than", "too", "very", "just", "also", "only", "up", "out",
    "about", "into", "over", "after", "before", "between", "under",
    "through", "during", "above", "below", "off", "down", "more", "most",
    "some", "such", "each", "other", "its", "our", "your", "their",
    "we", "they", "it", "he", "she", "his", "her", "as", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "all",
    "both", "few", "many", "any", "each", "paper", "study", "based",
    "using", "via", "method", "approach", "model", "result", "show",
    "propose", "present", "introduce", "demonstrate", "achieve",
    "et", "al", "fig", "table", "section", "eq", "ref",
}


class QueryExpander:
    """Expand queries using academic taxonomy and candidate feedback.

    Usage:
        expander = QueryExpander()
        # Expand acronyms in a query
        expanded = expander.expand_query(query)
        # Generate feedback queries from candidates
        new_queries = expander.feedback_expand(original_queries, candidates, query_plan)
    """

    def __init__(self, taxonomy_path: str | Path | None = None) -> None:
        self._taxonomy: dict[str, Any] = {}
        self._acronym_map: dict[str, list[str]] = {}
        self._task_synonyms: dict[str, list[str]] = {}
        self._method_synonyms: dict[str, list[str]] = {}
        self._domain_keywords: dict[str, list[str]] = {}
        self._load(taxonomy_path or DEFAULT_TAXONOMY_PATH)

    def _load(self, path: Path) -> None:
        """Load taxonomy from YAML file."""
        try:
            if not path.exists():
                # Try relative to CWD
                alt = Path.cwd() / "configs" / "academic_taxonomy.yaml"
                if alt.exists():
                    path = alt
                else:
                    LOGGER.warning("Taxonomy file not found at %s, query expansion disabled", path)
                    return
            with open(path, "r", encoding="utf-8") as f:
                self._taxonomy = yaml.safe_load(f) or {}
            self._acronym_map = self._taxonomy.get("acronyms", {})
            self._task_synonyms = self._taxonomy.get("task_synonyms", {})
            self._method_synonyms = self._taxonomy.get("method_synonyms", {})
            self._domain_keywords = self._taxonomy.get("domain_keywords", {})
            LOGGER.info(
                "Loaded academic taxonomy: %d acronyms, %d task synonyms, %d method synonyms",
                len(self._acronym_map), len(self._task_synonyms), len(self._method_synonyms),
            )
        except Exception as exc:
            LOGGER.warning("Failed to load taxonomy from %s: %s. Query expansion disabled.", path, exc)

    def expand_acronyms(self, text: str) -> list[str]:
        """Find acronyms in text and return their full forms.

        Returns a list of expanded forms for acronyms found in the text.
        """
        if not text or not self._acronym_map:
            return []

        expansions: list[str] = []
        # Match uppercase acronyms (2-6 chars, may have digits)
        acronym_pattern = re.compile(r"\b([A-Z][A-Z0-9]{1,5}s?)\b")
        for match in acronym_pattern.finditer(text):
            acronym = match.group(1)
            # Remove trailing 's' for lookup
            base = acronym.rstrip("s")
            plural = acronym != base

            full_forms = self._acronym_map.get(acronym) or self._acronym_map.get(base)
            if full_forms:
                for form in full_forms:
                    # If original was plural, pluralize the expansion
                    if plural and not form.endswith("s"):
                        form = form + "s"
                    if form.lower() not in [e.lower() for e in expansions]:
                        expansions.append(form)

        return expansions

    def get_synonyms(self, term: str) -> list[str]:
        """Get synonyms for a task or method term."""
        if not term:
            return []

        lowered = term.lower().strip()
        # Check task synonyms
        if lowered in self._task_synonyms:
            return self._task_synonyms[lowered]
        # Check method synonyms
        if lowered in self._method_synonyms:
            return self._method_synonyms[lowered]

        # Also check if any key contains the term
        for syn_map in (self._task_synonyms, self._method_synonyms):
            for key, syns in syn_map.items():
                if lowered in key or key in lowered:
                    return syns

        return []

    def expand_query(self, query: SearchQuery) -> SearchQuery:
        """Expand a SearchQuery by adding acronym expansions to optional_terms.

        Returns a new SearchQuery with expanded optional_terms.
        Does not modify the original query.
        """
        # Gather text from query
        text_parts = [query.query] + list(query.required_terms or [])
        full_text = " ".join(text_parts)

        # Expand acronyms
        expansions = self.expand_acronyms(full_text)

        # Get synonyms for required terms
        for term in (query.required_terms or []):
            syns = self.get_synonyms(term)
            expansions.extend(syns)

        # Deduplicate and filter out terms already in the query
        existing_lower = {t.lower() for t in (query.required_terms or []) + (query.optional_terms or [])}
        new_terms = [e for e in expansions if e.lower() not in existing_lower]

        if not new_terms:
            return query

        # Add to optional_terms
        new_optional = list(query.optional_terms or []) + new_terms[:5]  # Cap at 5 new terms

        return SearchQuery(
            query=query.query,
            route=query.route,
            intent=query.intent,
            priority=query.priority,
            required_terms=query.required_terms,
            optional_terms=new_optional,
            filters=query.filters,
            sources=query.sources,
        )

    def extract_terms_from_candidates(
        self,
        candidates: list[Paper],
        top_k: int = 50,
        max_terms: int = 15,
    ) -> list[str]:
        """Extract key terms from the top-K candidates' titles and abstracts.

        Uses simple TF ranking with stopword removal.
        Returns a list of the most frequent meaningful terms.
        """
        if not candidates:
            return []

        # Take top-K candidates (assumed pre-sorted by rough score)
        top_papers = candidates[:top_k]

        # Extract 1-3 gram terms from titles and abstracts
        term_counter: Counter[str] = Counter()

        for paper in top_papers:
            text = f"{paper.title or ''} {paper.abstract or ''}"
            if not text.strip():
                continue

            # Extract n-grams (1-3 words)
            words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
            # Filter stopwords
            words = [w for w in words if w not in _STOPWORDS and len(w) >= 3]

            # 1-grams
            for w in words:
                term_counter[w] += 1

            # 2-grams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                term_counter[bigram] += 1

            # 3-grams (only for high-value phrases)
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                term_counter[trigram] += 1

        # Filter out very short or generic terms
        filtered = Counter()
        for term, count in term_counter.items():
            # Only keep terms that appear in at least 2 papers
            if count >= 2:
                # Boost multi-word terms (more specific)
                word_count = len(term.split())
                if word_count >= 2:
                    filtered[term] = count * 2  # Boost factor for multi-word
                else:
                    filtered[term] = count

        # Return top terms
        return [term for term, _ in filtered.most_common(max_terms)]

    def feedback_expand(
        self,
        original_queries: list[SearchQuery],
        candidate_pool: list[Paper],
        query_plan: QueryPlan | None = None,
        max_new_queries: int = 3,
    ) -> list[SearchQuery]:
        """Generate new queries based on candidate feedback.

        Extracts terms from the top candidates that are NOT already in
        the original queries, and creates new search queries using those
        terms combined with the core research topic.
        """
        if not candidate_pool or not original_queries:
            return []

        # Collect existing query terms to avoid duplication
        existing_terms: set[str] = set()
        for q in original_queries:
            existing_terms.update(t.lower() for t in (q.required_terms or []))
            existing_terms.update(t.lower() for t in (q.optional_terms or []))
            # Also add the query text itself
            for w in re.findall(r"[a-zA-Z]{3,}", q.query.lower()):
                existing_terms.add(w)

        # Extract terms from candidates
        candidate_terms = self.extract_terms_from_candidates(candidate_pool, top_k=50, max_terms=30)

        # Filter out terms already in the original queries
        new_terms = []
        for term in candidate_terms:
            # Check if this term (or its words) are already in existing terms
            words = term.lower().split()
            if all(w in existing_terms for w in words):
                continue
            new_terms.append(term)

        if not new_terms:
            return []

        # Get core topic from query_plan
        core_topic = ""
        if query_plan is not None:
            core_topic = getattr(query_plan, "research_topic", "") or ""

        # Generate new queries: combine core topic with new terms
        new_queries: list[SearchQuery] = []
        for term in new_terms[:max_new_queries]:
            if core_topic:
                query_text = f"{core_topic} {term}"
            else:
                query_text = term

            new_queries.append(SearchQuery(
                query=query_text,
                route="feedback_expansion",
                intent=f"feedback-based expansion: {term}",
                priority=2,
                required_terms=[],
                optional_terms=[term],
                filters={},
                sources=[],
            ))

        return new_queries

    def is_available(self) -> bool:
        """Check if the taxonomy was loaded successfully."""
        return bool(self._acronym_map or self._task_synonyms)
