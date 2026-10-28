"""Offline adapter/metric tests; no Hugging Face download or API token needed."""
from pathlib import Path

import pytest

from evaluation.hf_datasets import (
    adapt_row, deterministic_sample, load_fixture, write_fixture,
)
from evaluation.hf_metrics import (
    aggregate_slices, answer_metrics, evidence_metrics, exact_match,
    rouge_l_f1, token_f1,
)
from evaluation.hf_statistics import paired_bootstrap


REV = "abcdef1234567890"


def test_hotpotqa_gold_titles_and_sentence_refs():
    row = {
        "id": "hotpot-1", "question": "How is A related to C?",
        "answer": "Through B", "type": "bridge", "level": "hard",
        "context": {
            "title": ["A page", "C page", "Noise"],
            "sentences": [
                ["A links to B.", "More A."],
                ["C is linked to B."],
                ["Random unrelated passage."],
            ],
        },
        "supporting_facts": {
            "title": ["A page", "C page"],
            "sent_id": [0, 0],
        },
    }
    case = adapt_row("hotpotqa", row, REV, "validation")[0]
    assert case.gold_document_ids == ["d0", "d1"]
    assert case.gold_sentence_refs == ["d0:s0", "d1:s0"]
    assert case.metadata["difficulty"] == "hard"
    assert case.metadata["oracle_context"] is True
    assert len(case.documents) == 3


def test_hotpotqa_bad_gold_not_silently_dropped():
    row = {
        "id": "broken", "question": "What?", "answer": "A",
        "context": {"title": ["Existing"], "sentences": [["x"]]},
        "supporting_facts": {"title": ["Absent"], "sent_id": [0]},
    }
    with pytest.raises(ValueError, match="supporting title"):
        adapt_row("hotpotqa", row, REV, "validation")


def test_musique_support_and_hops():
    row = {
        "id": "2hop_case", "question": "Who built the school?",
        "answer": "Engineer Z",
        "paragraphs": [
            {"idx": 0, "is_supporting": True, "title": "School", "paragraph_text": "School X was built by Engineer Z."},
            {"idx": 2, "is_supporting": False, "title": "Other", "paragraph_text": "Other source."},
        ],
        "question_decomposition": {
            "id": [1, 2], "question": ["What school?", "Who built it?"],
            "answer": ["School X", "Engineer Z"],
            "paragraph_support_idx": [0, 0],
        },
        "answerable": True,
    }
    case = adapt_row("musique", row, REV, "validation")[0]
    assert case.gold_document_ids == ["d0"]
    assert case.metadata["hop_count"] == 2
    assert case.documents[1].id == "d2"


def test_multi_news_is_summarization_not_gold_retrieval():
    row = {
        "document": "First news story. ||||| Second news story.",
        "summary": "Two stories cover similar events.",
    }
    case = adapt_row("multi_news", row, REV, "validation")[0]
    assert len(case.documents) == 2
    assert case.gold_document_ids == []
    assert case.metadata["gold_evidence_unlabeled"] is True
    assert case.task_type == "multi_document_summary"


def test_qasper_multiple_questions_and_answers():
    row = {
        "id": "paper001", "title": "Research Paper", "abstract": "A framework",
        "full_text": {
            "section_name": ["Intro"],
            "paragraphs": [["Key evidence here.", "Further work."]],
        },
        "qas": {
            "question": ["What is proposed?", "Is this novel?"],
            "question_id": ["q1", "q2"],
            "answers": [
                {"annotation_id": ["a1"], "answer": [{
                    "unanswerable": False, "extractive_spans": [],
                    "free_form_answer": "A framework", "yes_no": False,
                }], "worker_id": ["w1"]},
                {"annotation_id": ["a2"], "answer": [{
                    "unanswerable": False, "extractive_spans": [],
                    "free_form_answer": "", "yes_no": True,
                }], "worker_id": ["w2"]},
            ],
        },
    }
    cases = adapt_row("qasper", row, REV, "validation")
    assert len(cases) == 2
    assert cases[0].reference_answers == ["A framework"]
    assert cases[1].reference_answers == ["Yes"]
    assert all(case.gold_document_ids == [] for case in cases)


def test_sample_is_order_independent_and_fixture_tamper_rejected(tmp_path: Path):
    rows = [
        adapt_row("multi_news",
                  {"document": f"A{i}.|||||B{i}.", "summary": f"Summary {i}"},
                  REV, "validation")[0]
        for i in range(15)
    ]
    first = deterministic_sample(rows, count=5, seed=13)
    second = deterministic_sample(list(reversed(rows)), count=5, seed=13)
    assert [item.id for item in first] == [item.id for item in second]
    output = tmp_path / "dataset.jsonl"
    write_fixture(first, {"dataset": "multi_news", "revision": REV}, output)
    assert load_fixture(output) == first
    output.write_text(output.read_text() + "bad", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_fixture(output)


def test_metrics_distinguish_unlabeled_evidence_and_answer_gold():
    assert exact_match("The Answer", ["answer"]) == 1.0
    assert token_f1("alpha beta", ["alpha beta gamma"]) == pytest.approx(0.8)
    assert rouge_l_f1("alpha beta", ["alpha x beta"]) == pytest.approx(0.8)
    assert evidence_metrics([], ["d0"], annotated=False)["evidence_recall"] is None
    measured = evidence_metrics(["d0", "d1"], ["d1", "d2"], annotated=True)
    assert measured["evidence_recall"] == 0.5
    assert measured["evidence_precision"] == 0.5
    assert measured["evidence_mrr"] == 1.0
    assert answer_metrics("summary", ["summary"], "multi_document_summary")["exact_match"] is None


def test_slices_exclude_missing_ground_truth_from_averages():
    rows = [
        {"condition": "global", "dataset": "multi_news", "task_type": "multi_document_summary",
         "success": True, "evidence_recall": None, "rouge_l_f1": 0.7},
        {"condition": "global", "dataset": "hotpotqa", "task_type": "multihop_qa",
         "success": True, "evidence_recall": 0.5, "rouge_l_f1": 0.2},
    ]
    summary = aggregate_slices(rows)
    assert summary["condition"]["global"]["evidence_recall"] == 0.5
    assert summary["dataset"]["multi_news"]["evidence_recall"] is None


def test_paired_bootstrap_is_reproducible_and_rejects_unpaired():
    records = []
    for i in range(10):
        records.extend([
            {"case_id": f"q{i}", "dataset": "hotpotqa", "condition": "global",
             "success": True, "answer_token_f1": 0.9},
            {"case_id": f"q{i}", "dataset": "hotpotqa", "condition": "basic",
             "success": True, "answer_token_f1": 0.5},
        ])
    report = {"results": records}
    first = paired_bootstrap(
        report, candidate="global", baseline="basic",
        metric="answer_token_f1", dataset="hotpotqa", iterations=200
    )
    second = paired_bootstrap(
        report, candidate="global", baseline="basic",
        metric="answer_token_f1", dataset="hotpotqa", iterations=200
    )
    assert first == second
    assert first["candidate_minus_baseline"] == pytest.approx(0.4)
    assert first["ci_95_percentile"][0] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="at least two paired"):
        paired_bootstrap(
            {"results": records[:2]}, candidate="global", baseline="basic",
            metric="answer_token_f1", iterations=200
        )

def test_multi_news_rawfiles_restore_newlines_and_reject_truncation(tmp_path: Path):
    from evaluation.hf_datasets import multi_news_rows

    src = tmp_path / "val.src.cleaned"
    tgt = tmp_path / "val.tgt"
    src.write_text("FirstNEWLINE_CHARline.|||||Second article.\\n", encoding="utf-8")
    tgt.write_text("Summary one.\\n", encoding="utf-8")
    parsed = list(multi_news_rows(src, tgt))
    assert len(parsed) == 1
    assert parsed[0]["document"] == "First\\nline.|||||Second article."
    tgt.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="line-count mismatch"):
        list(multi_news_rows(src, tgt))


def test_multi_news_download_requires_license_acknowledgement(monkeypatch):
    import sys
    from types import SimpleNamespace
    from evaluation.hf_datasets import download_cases

    class FakeApi:
        def dataset_info(self, **kwargs):
            return SimpleNamespace(sha=REV)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        SimpleNamespace(
            HfApi=FakeApi,
            hf_hub_download=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("should not download without consent")
            ),
        ),
    )
    with pytest.raises(ValueError, match="non-commercial research"):
        download_cases("multi_news", count=1)
