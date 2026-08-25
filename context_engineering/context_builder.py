from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


# TOKEN ESTIMATION
class TokenCounter:
    """
    Default approximate token counter.
    chars / 4 is an estimate. Pass a custom tokenizer callable
    to ContextBuilder for exact model accounting.
    """

    @staticmethod
    def count(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


# STOP WORDS & NORMALIZATION
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "it", "its", "this", "that",
    "with", "be", "by", "from", "as", "into",
    "have", "has", "had", "not", "what", "how",
    "when", "where", "who", "which", "will",
    "would", "could", "should", "do", "does",
    "did", "i", "we", "you", "he", "she",
    "they", "their", "our", "your", "about",
    "up", "out", "if",
}


def _normalize_terms(text: str) -> set[str]:
    words = re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
        text.lower(),
    )

    return {word for word in words if word not in _STOP_WORDS and len(word) > 1}


# AUDIT RECORD
@dataclass
class ContextAudit:
    timestamp: str
    query: str

    chunks_in: int
    chunks_used: int
    chunks_discarded: int

    raw_tokens: int
    final_tokens: int
    token_budget: int

    compressed_chunks: int

    average_relevance: float
    max_relevance: float

    query_term_coverage: float
    length_score: float
    quality_score: float

    quality_threshold: float
    passed_gate: bool


# INTERNAL CHUNK REPRESENTATION
@dataclass
class RankedChunk:
    index: int
    text: str
    relevance: float
    original_tokens: int


# CONTEXT BUILDER
class ContextBuilder:
    """
    Optimized Retrieval Context Engineering Pipeline.
    """

    def __init__(
        self,
        *,
        token_budget: int = 1200,
        quality_threshold: float = 0.60,
        compression_llm: Any = None,
        token_counter: Callable[[str], int] | None = None,
        separator: str = "\n\n---\n\n",
        verbose: bool = True,
        min_quality_context_tokens: int = 80,
        minimum_remaining_tokens_for_compression: int = 40,
    ) -> None:

        if token_budget <= 0:
            raise ValueError("token_budget must be greater than zero.")

        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0 and 1.")

        self.token_budget = token_budget
        self.quality_threshold = quality_threshold
        self.compression_llm = compression_llm
        self.token_counter = token_counter or TokenCounter.count
        self.separator = separator
        self.verbose = verbose

        self.min_quality_context_tokens = min_quality_context_tokens
        self.minimum_remaining_tokens_for_compression = (
            minimum_remaining_tokens_for_compression
        )

        self._audit_logs: list[dict[str, Any]] = []

    # RELEVANCE & TOKEN HELPERS
    def _count_tokens(self, text: str) -> int:
        return self.token_counter(text)

    @staticmethod
    def _relevance_score(chunk: str, query: str) -> float:
        query_terms = _normalize_terms(query)

        if not query_terms:
            return 0.5

        chunk_terms = _normalize_terms(chunk)
        overlap = query_terms.intersection(chunk_terms)

        return round(len(overlap) / len(query_terms), 4)

    # LOST IN THE MIDDLE
    @staticmethod
    def _reorder_lost_in_the_middle(chunks: list[RankedChunk]) -> list[RankedChunk]:
        if len(chunks) <= 2:
            return chunks

        best = chunks[0]
        second_best = chunks[1]
        middle = chunks[2:]

        return [best, *middle, second_best]

    # COMPRESSION
    def _compress_chunk(self, chunk: str, query: str) -> str:
        if self.compression_llm is None:
            return ""

        prompt = f"""
                Compress the following retrieved research chunk for the query below.

                QUERY:
                {query}

                Rules:
                - Keep only information directly relevant to the query.
                - Preserve names, dates, and numbers exactly.
                - Preserve important qualifiers.
                - Do not invent information.
                - Remove repetition and irrelevant detail.
                - Return only the compressed content.

                CHUNK:
                {chunk}
                """

        try:
            response = self.compression_llm.invoke(prompt)
            content = getattr(response, "content", response)
            return str(content).strip() if content else ""
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"[ContextBuilder] Compression failed: {exc}")
            return ""

    # QUALITY SCORE
    def _quality_score(self, context: str, query: str) -> tuple[float, float, float]:
        query_terms = _normalize_terms(query)

        if not query_terms:
            query_coverage = 1.0
        else:
            context_terms = _normalize_terms(context)
            overlap = query_terms.intersection(context_terms)
            query_coverage = len(overlap) / len(query_terms)

        final_tokens = self._count_tokens(context)
        length_score = min(1.0, final_tokens / self.min_quality_context_tokens)

        quality = 0.70 * query_coverage + 0.30 * length_score

        return (
            round(query_coverage, 3),
            round(length_score, 3),
            round(min(1.0, quality), 3),
        )

    # BUILD PIPELINE
    def build(self, *, chunks: list[str], query: str) -> str:
        # 0. Clean input
        cleaned_chunks = [
            chunk.strip()
            for chunk in chunks
            if isinstance(chunk, str) and chunk.strip()
        ]

        if not cleaned_chunks:
            return ""

        # Precompute separator token cost
        separator_tokens = self._count_tokens(self.separator)

        # 1. Calculate raw token count (without separators)
        raw_tokens = sum(self._count_tokens(chunk) for chunk in cleaned_chunks)

        # 2. Relevance Ranking
        ranked_chunks: list[RankedChunk] = []

        for index, chunk in enumerate(cleaned_chunks):
            ranked_chunks.append(
                RankedChunk(
                    index=index,
                    text=chunk,
                    relevance=self._relevance_score(chunk, query),
                    original_tokens=self._count_tokens(chunk),
                )
            )

        ranked_chunks.sort(key=lambda x: x.relevance, reverse=True)

        # 3. Token-Budget Selection (Accounts for Separator Overhead)
        selected: list[RankedChunk] = []
        tokens_used = 0
        compressed_count = 0
        discarded_count = 0

        for chunk in ranked_chunks:
            # First chunk incurs no separator overhead; subsequent chunks do.
            added_sep_tokens = separator_tokens if selected else 0
            needed_tokens = chunk.original_tokens + added_sep_tokens

            # Chunk fits as-is
            if tokens_used + needed_tokens <= self.token_budget:
                selected.append(chunk)
                tokens_used += needed_tokens
                continue

            # Check if compression is viable
            remaining_budget = self.token_budget - tokens_used - added_sep_tokens
            if (
                self.compression_llm is None
                or remaining_budget < self.minimum_remaining_tokens_for_compression
            ):
                discarded_count += 1
                continue

            compressed_text = self._compress_chunk(chunk.text, query)
            if not compressed_text:
                discarded_count += 1
                continue

            compressed_tokens = self._count_tokens(compressed_text)
            needed_compressed_tokens = compressed_tokens + added_sep_tokens

            if tokens_used + needed_compressed_tokens <= self.token_budget:
                selected.append(
                    RankedChunk(
                        index=chunk.index,
                        text=compressed_text,
                        relevance=chunk.relevance,
                        original_tokens=compressed_tokens,
                    )
                )
                tokens_used += needed_compressed_tokens
                compressed_count += 1
            else:
                discarded_count += 1

        # 4. Lost-in-the-Middle Reordering
        selected.sort(key=lambda x: x.relevance, reverse=True)
        reordered = self._reorder_lost_in_the_middle(selected)

        # 5. Assemble Final Context
        final_context = self.separator.join(chunk.text for chunk in reordered)

        # 6. Quality Gate
        (
            query_term_coverage,
            length_score,
            quality_score,
        ) = self._quality_score(final_context, query)

        passed_gate = quality_score >= self.quality_threshold

        # 7. Relevance Statistics
        selected_scores = [chunk.relevance for chunk in selected]
        average_relevance = (
            sum(selected_scores) / len(selected_scores) if selected_scores else 0.0
        )
        max_relevance = max(selected_scores) if selected_scores else 0.0

        # 8. Audit Logging
        final_token_count = self._count_tokens(final_context)
        audit = ContextAudit(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            chunks_in=len(cleaned_chunks),
            chunks_used=len(selected),
            chunks_discarded=discarded_count,
            raw_tokens=raw_tokens,
            final_tokens=final_token_count,
            token_budget=self.token_budget,
            compressed_chunks=compressed_count,
            average_relevance=round(average_relevance, 3),
            max_relevance=round(max_relevance, 3),
            query_term_coverage=query_term_coverage,
            length_score=length_score,
            quality_score=quality_score,
            quality_threshold=self.quality_threshold,
            passed_gate=passed_gate,
        )

        self._audit_logs.append(asdict(audit))

        if self.verbose:
            print(
                f"\n[ContextBuilder]\n"
                f"  chunks: {audit.chunks_in} → {audit.chunks_used}\n"
                f"  raw tokens: {audit.raw_tokens}\n"
                f"  final tokens: {audit.final_tokens}/{audit.token_budget}\n"
                f"  compressed: {audit.compressed_chunks}\n"
                f"  discarded: {audit.chunks_discarded}\n"
                f"  avg relevance: {audit.average_relevance:.2f}\n"
                f"  max relevance: {audit.max_relevance:.2f}\n"
                f"  query coverage: {audit.query_term_coverage:.2f}\n"
                f"  length score: {audit.length_score:.2f}\n"
                f"  quality: {audit.quality_score:.2f}\n"
                f"  gate: {'PASS' if audit.passed_gate else 'FAIL'}"
            )

        if not passed_gate:
            return ""

        return final_context

    # AUDIT API
    @property
    def audit_logs(self) -> list[dict[str, Any]]:
        return list(self._audit_logs)

    def audit_json(self) -> str:
        return json.dumps(self._audit_logs, indent=2)

    def clear_audit(self) -> None:
        self._audit_logs.clear()
