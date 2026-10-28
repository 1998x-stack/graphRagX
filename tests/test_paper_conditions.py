"""C0-C3 / TS / SS orchestration and paper-label pairing tests."""
import pytest

from evaluation.models import EvaluationCase, EvaluationResult
from evaluation.paper_conditions import CONDITIONS, run_paper_conditions
from evaluation.paper_judge import evaluate_pairs


@pytest.mark.asyncio
async def test_all_six_conditions_use_one_frozen_case(monkeypatch):
    observed = []

    async def fake_run_case(case, mode):
        observed.append((case.index_id, case.query, mode, case.community_level))
        return EvaluationResult(
            case_id=case.id, mode=mode, success=True,
            answer=f"{mode} at level {case.community_level}",
        )

    monkeypatch.setattr("evaluation.paper_conditions.run_case", fake_run_case)
    case = EvaluationCase(
        id="paper_q1", index_id="frozen_001", query="Which themes recur?"
    )
    report = await run_paper_conditions([case])
    assert len(report["results"]) == 6
    assert list(report["summary"]) == list(CONDITIONS)
    assert [row["condition"] for row in report["results"]] == list(CONDITIONS)
    assert observed == [
        ("frozen_001", "Which themes recur?", "global", 0),
        ("frozen_001", "Which themes recur?", "global", 1),
        ("frozen_001", "Which themes recur?", "global", 2),
        ("frozen_001", "Which themes recur?", "global", 3),
        ("frozen_001", "Which themes recur?", "text_summary", None),
        ("frozen_001", "Which themes recur?", "basic", None),
    ]


@pytest.mark.asyncio
async def test_judge_can_compare_paper_condition_labels():
    results = [
        {"case_id": "q", "condition": label, "mode": mode,
         "success": True, "answer": label + " answer"}
        for label, mode in [("C2", "global"), ("SS", "basic")]
    ]

    async def tie(prompt):
        return '{"winner":"TIE"}'

    report = await evaluate_pairs(
        {"questions": {"q": "What are main trends?"}, "results": results},
        mode_a="C2", mode_b="SS", repeats=1, judge=tie,
    )
    assert report["num_cases"] == 1
    assert report["summary"]["comprehensiveness"]["ties"] == 1


@pytest.mark.asyncio
async def test_unknown_condition_is_rejected_before_model_calls():
    with pytest.raises(ValueError, match="Unknown paper evaluation conditions"):
        await run_paper_conditions([], conditions=["C0", "INVALID"])
