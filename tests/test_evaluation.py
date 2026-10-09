from evaluation.metrics import (
    aggregate_results,
    answer_term_coverage,
    source_recall,
)
from evaluation.models import EvaluationResult
from evaluation.runner import extract_source_ids


def test_deterministic_quality_metrics():
    assert source_recall(["a", "b"], ["b", "c"]) == 0.5
    assert answer_term_coverage(
        ["graph", "retrieval"],
        "Graph systems improve retrieval.",
    ) == 1.0


def test_extract_source_ids_covers_all_query_modes():
    local = extract_source_ids(
        "local",
        {
            "text_units": [{"id": "c1"}],
            "seed_entities": [{"name": "Acme"}],
            "claims": [{"id": "claim_1"}],
        },
    )
    assert local == ["c1", "entity:Acme", "claim_1"]

    global_ids = extract_source_ids(
        "global",
        {"community_summaries": [{"id": "L0_C0"}]},
    )
    assert global_ids == ["L0_C0"]

    drift = extract_source_ids(
        "drift",
        {
            "community_summaries": [{"id": "L0_C0"}],
            "drift": {
                "evidence": [
                    {"source_ids": ["c1", "entity:Acme", "claim_1"]}
                ]
            },
        },
    )
    assert drift == ["L0_C0", "c1", "entity:Acme", "claim_1"]


def test_aggregate_results_is_mode_scoped():
    results = [
        EvaluationResult(
            case_id="q1",
            mode="basic",
            success=True,
            source_recall=1.0,
            answer_term_coverage=0.5,
            latency_ms=10,
        ),
        EvaluationResult(
            case_id="q2",
            mode="basic",
            success=False,
            source_recall=0.0,
            answer_term_coverage=0.0,
            latency_ms=20,
        ),
    ]
    summary = aggregate_results(results)
    assert summary["basic"]["success_rate"] == 0.5
    assert summary["basic"]["avg_source_recall"] == 0.5
    assert summary["basic"]["avg_latency_ms"] == 15
