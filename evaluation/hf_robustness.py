"""Deterministic perturbations for retrieval robustness experiments.

Perturbations must not change the gold answer or silently convert a supported
question into an answerable/unanswerable label.
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterable

from evaluation.hf_datasets import BenchmarkCase, SourceDocument


def make_variant(
    case: BenchmarkCase,
    variant: str,
    *,
    seed: int = 42,
    noise_documents: int = 2,
) -> BenchmarkCase:
    """Produce source-identity-preserving graph evaluation perturbations."""
    if variant not in {"original", "shuffle", "distractor", "drop_support"}:
        raise ValueError(f"Unknown robustness variant: {variant}")
    if noise_documents < 1:
        raise ValueError("noise_documents must be positive")
    updated = case.model_copy(deep=True)
    updated.id = case.id + "__" + variant + "_" + str(seed)
    updated.metadata.update({
        "base_case_id": case.metadata.get("base_case_id", case.id),
        "robustness_variant": variant,
        "robustness_seed": seed,
    })
    rng_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{case.id}:{variant}".encode()).digest()[:8],
        "big",
    )
    rng = random.Random(rng_seed)
    if variant == "shuffle":
        rng.shuffle(updated.documents)
    elif variant == "distractor":
        used = {doc.id for doc in updated.documents}
        for index in range(noise_documents):
            doc_id = f"noise_{index}"
            if doc_id in used:
                raise ValueError("Synthetic distractor ID collides with original")
            updated.documents.append(SourceDocument(
                id=doc_id,
                title=f"Synthetic unrelated context {index}",
                text=(
                    f"This is synthetic distractor {index}. "
                    "It discusses stationery inventory and desk organization, "
                    "not the facts in the evaluation question."
                ),
            ))
    elif variant == "drop_support":
        if not updated.gold_document_ids:
            raise ValueError(
                "drop_support requires positive annotated gold source documents"
            )
        removed = set(updated.gold_document_ids)
        updated.documents = [
            doc for doc in updated.documents if doc.id not in removed
        ]
        if not updated.documents:
            raise ValueError("Cannot remove all documents from a benchmark case")
        updated.gold_document_ids = []
        updated.gold_sentence_refs = []
        updated.reference_answers = []
        updated.metadata["support_removed"] = True
        updated.metadata["answerability_after_removal"] = "unknown"
        # Remaining context may still entail the answer; this is not a
        # factuality/abstention ground truth label.
    return updated


def expand_variants(
    cases: Iterable[BenchmarkCase],
    variants: list[str],
    *,
    seed: int = 42,
    noise_documents: int = 2,
) -> list[BenchmarkCase]:
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate robustness variants")
    output = []
    for case in cases:
        for variant in variants:
            if variant == "drop_support" and not case.gold_document_ids:
                continue
            output.append(make_variant(
                case, variant, seed=seed,
                noise_documents=noise_documents,
            ))
    return output
