"""Tokenizer-aware context budgeting utilities."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import tiktoken

from config import settings


Segment = Tuple[str, str]


class TokenBudget:
    """Count, truncate, select, and batch text using the configured model tokenizer."""

    def __init__(self, model: str | None = None):
        selected_model = model or settings.LLM_MODEL
        try:
            self.encoding = tiktoken.encoding_for_model(selected_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

    def select(self, segments: Sequence[Segment], max_tokens: int) -> List[Segment]:
        """Take ranked segments until the budget is exhausted."""
        selected: List[Segment] = []
        remaining = max_tokens
        for segment_id, text in segments:
            if remaining <= 0:
                break
            token_count = self.count(text)
            if token_count <= remaining:
                selected.append((segment_id, text))
                remaining -= token_count
                continue
            clipped = self.truncate(text, remaining)
            if clipped:
                selected.append((segment_id, clipped))
            break
        return selected

    def coverage_select(
        self,
        segments: Sequence[Segment],
        max_tokens: int,
        min_tokens_per_segment: int = 16,
    ) -> List[Segment]:
        """Allocate the total budget fairly without dropping a community.

        Fails explicitly when even the minimum evidence per report cannot fit.
        The allocation is deterministic and uses the actual tokenizer.
        """
        if max_tokens <= 0 or min_tokens_per_segment <= 0:
            raise ValueError("Token budgets must be positive")
        if not segments:
            return []
        lengths = [self.count(content) for _, content in segments]
        assigned = [min(length, min_tokens_per_segment) for length in lengths]
        remaining = max_tokens - sum(assigned)
        if remaining < 0:
            raise ValueError(
                "Global coverage budget cannot retain all community reports; "
                "increase GLOBAL_MAX_DATA_TOKENS or use relevance selection"
            )
        active = {index for index, length in enumerate(lengths) if length > assigned[index]}
        while remaining > 0 and active:
            grant = max(1, remaining // len(active))
            progressed = 0
            for index in sorted(active):
                additional = min(grant, lengths[index] - assigned[index], remaining)
                assigned[index] += additional
                remaining -= additional
                progressed += additional
                if remaining == 0:
                    break
            if not progressed:
                break
            active = {index for index in active if lengths[index] > assigned[index]}
        return [
            (segment_id, self.truncate(content, assigned[index]))
            for index, (segment_id, content) in enumerate(segments)
        ]

    def batches(self, segments: Sequence[Segment], max_tokens: int) -> List[List[Segment]]:
        """Create ordered batches that each stay within the token budget."""
        batches: List[List[Segment]] = []
        current: List[Segment] = []
        used = 0

        for segment_id, text in segments:
            clipped = self.truncate(text, max_tokens)
            cost = self.count(clipped)
            if current and used + cost > max_tokens:
                batches.append(current)
                current = []
                used = 0
            if clipped:
                current.append((segment_id, clipped))
                used += cost

        if current:
            batches.append(current)
        return batches


token_budget = TokenBudget()
