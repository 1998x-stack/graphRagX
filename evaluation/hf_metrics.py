"""Transparent, deterministic, reference-aware metrics.

Missing gold annotations are represented as null rather than spuriously
scoring perfect evidence recall. Global community exposure is not a citation
faithfulness claim.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from statistics import mean
from typing import Any, Iterable


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = "".join(character for character in text
                   if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(answer: str, references: list[str]) -> float | None:
    if not references:
        return None
    normalized = normalize_answer(answer)
    return float(any(normalized == normalize_answer(ref) for ref in references))


def token_f1(answer: str, references: list[str]) -> float | None:
    if not references:
        return None
    prediction = normalize_answer(answer).split()
    def f1(ref: str) -> float:
        gold = normalize_answer(ref).split()
        if not prediction or not gold:
            return float(prediction == gold)
        overlap = sum((Counter(prediction) & Counter(gold)).values())
        if not overlap:
            return 0.0
        precision = overlap / len(prediction)
        recall = overlap / len(gold)
        return 2 * precision * recall / (precision + recall)
    return max(f1(ref) for ref in references)


def rouge_l_f1(answer: str, references: list[str], max_words: int = 2048) -> float | None:
    if not references:
        return None
    predicted = normalize_answer(answer).split()[:max_words]
    def score(reference: str) -> float:
        gold = normalize_answer(reference).split()[:max_words]
        if not predicted or not gold:
            return float(predicted == gold)
        # O(m*n) time, O(min(m,n)) memory.
        short, long = (predicted, gold) if len(predicted) <= len(gold) else (gold, predicted)
        previous = [0] * (len(short) + 1)
        for word in long:
            current = [0]
            for i, token in enumerate(short, start=1):
                current.append(previous[i - 1] + 1 if token == word
                               else max(previous[i], current[i - 1]))
            previous = current
        lcs = previous[-1]
        return 2 * lcs / (len(predicted) + len(gold))
    return max(score(ref) for ref in references)


def evidence_metrics(
    gold_ids: Iterable[str],
    ranked_ids: Iterable[str],
    annotated: bool,
    k: int = 5,
) -> dict[str, float | None]:
    if not annotated:
        return {
            "evidence_recall": None, "evidence_precision": None,
            "evidence_mrr": None, "evidence_ndcg_at_k": None,
        }
    gold = set(gold_ids)
    if not gold:
        return {
            "evidence_recall": None, "evidence_precision": None,
            "evidence_mrr": None, "evidence_ndcg_at_k": None,
        }
    ranked = list(dict.fromkeys(ranked_ids))
    selected = set(ranked)
    hits = len(selected & gold)
    mrr = next((1.0 / (i + 1) for i, item in enumerate(ranked)
                if item in gold), 0.0)
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, item in enumerate(ranked[:k]) if item in gold
    )
    ideal = sum(1.0 / math.log2(position + 2)
                for position in range(min(len(gold), k)))
    return {
        "evidence_recall": hits / len(gold),
        "evidence_precision": hits / len(selected) if selected else 0.0,
        "evidence_mrr": mrr,
        "evidence_ndcg_at_k": dcg / ideal if ideal else None,
    }


def answer_metrics(
    answer: str, reference_answers: list[str], task: str,
) -> dict[str, float | None]:
    if task == "multi_document_summary":
        return {
            "exact_match": None, "answer_token_f1": None,
            "rouge_l_f1": rouge_l_f1(answer, reference_answers),
        }
    return {
        "exact_match": exact_match(answer, reference_answers),
        "answer_token_f1": token_f1(answer, reference_answers),
        "rouge_l_f1": rouge_l_f1(answer, reference_answers),
    }


def _mean_valid(rows: list[dict], key: str) -> float | None:
    values = [row.get(key) for row in rows
              if isinstance(row.get(key), (int, float))
              and not isinstance(row.get(key), bool)
              and math.isfinite(row[key])]
    return mean(values) if values else None


def aggregate_slices(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Build every metric by mode, dataset, difficulty and hop count.

    Macro-average by question, without treating missing gold labels as 1.
    """
    groups: dict[str, dict[str, list[dict]]] = {}
    metrics = (
        "exact_match", "answer_token_f1", "rouge_l_f1",
        "evidence_recall", "evidence_precision", "evidence_mrr",
        "evidence_ndcg_at_k", "latency_ms", "community_report_coverage",
    )
    for row in records:
        tags = {
            "condition": str(row.get("condition", "")),
            "dataset": str(row.get("dataset", "")),
            "task_type": str(row.get("task_type", "")),
            "difficulty": str(row.get("difficulty") or "unknown"),
            "hop_count": str(row.get("hop_count")
                             if row.get("hop_count") is not None else "unknown"),
        }
        for group_name, label in tags.items():
            groups.setdefault(group_name, {}).setdefault(label, []).append(row)
    result: dict[str, dict[str, dict]] = {}
    for group_name, buckets in groups.items():
        result[group_name] = {}
        for label, rows in sorted(buckets.items()):
            result[group_name][label] = {
                "count": len(rows),
                "success_rate": mean(float(bool(row.get("success"))) for row in rows),
                **{key: _mean_valid(rows, key) for key in metrics},
            }
    return result
