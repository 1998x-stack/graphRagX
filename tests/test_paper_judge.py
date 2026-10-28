"""Methodology and data-integrity tests for the optional paper judge."""
import pytest

from evaluation.paper_judge import CRITERIA, evaluate_pairs


@pytest.mark.asyncio
async def test_counterbalanced_pairwise_scores_and_ties():
    report = {
        "questions": {"q1": "What themes recur?"},
        "results": [
            {"case_id": "q1", "mode": "global", "success": True, "answer": "Long graph summary"},
            {"case_id": "q1", "mode": "basic", "success": True, "answer": "Short retrieval answer"},
        ],
    }
    seen = []

    async def judge(prompt):
        seen.append(prompt)
        return '{"winner":"TIE","reason":"equal coverage"}'

    result = await evaluate_pairs(report, repeats=2, judge=judge)
    assert result["num_cases"] == 1
    assert len(seen) == 2 * len(CRITERIA)
    for summary in result["summary"].values():
        assert summary["comparisons"] == 2
        assert summary["ties"] == 2
        assert summary["preference_rate_with_half_ties"] == 0.5


@pytest.mark.asyncio
async def test_paper_judge_rejects_malformed_model_output():
    report = {
        "questions": {"q": "Why?"},
        "results": [
            {"case_id": "q", "mode": "global", "success": True, "answer": "A"},
            {"case_id": "q", "mode": "basic", "success": True, "answer": "B"},
        ],
    }

    async def invalid(prompt):
        return "Candidate A is clearly better."

    with pytest.raises(ValueError, match="JSON object"):
        await evaluate_pairs(report, judge=invalid, repeats=1)


@pytest.mark.asyncio
async def test_pairwise_requires_question_provenance():
    report = {
        "results": [
            {"case_id": "q", "mode": "global", "success": True, "answer": "A"},
            {"case_id": "q", "mode": "basic", "success": True, "answer": "B"},
        ]
    }

    async def unused(prompt):
        return '{"winner":"TIE"}'

    with pytest.raises(ValueError, match="Question text missing"):
        await evaluate_pairs(report, judge=unused)
