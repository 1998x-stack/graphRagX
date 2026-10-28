"""Offline report generation is deterministic and guards unknown gold labels."""
import csv
from pathlib import Path

from scripts.hf_report import build_markdown_report, export_csv


def test_markdown_report_stratifies_tasks_and_disclaims_mock():
    report = {
        "mode": "offline_smoke",
        "settings": {
            "llm_provider": "mock", "llm_model": "mock",
            "embedding_provider": "hash", "embedding_model": "hash",
        },
        "results": [
            {"case_id": "a", "dataset": "hotpotqa", "condition": "basic",
             "success": True, "exact_match": 1.0, "answer_token_f1": 1.0,
             "rouge_l_f1": 1.0, "evidence_recall": 0.5, "latency_ms": 12},
            {"case_id": "b", "dataset": "multi_news", "condition": "global",
             "success": True, "exact_match": None, "answer_token_f1": None,
             "rouge_l_f1": 0.6, "evidence_recall": None, "latency_ms": 44},
        ],
    }
    output = build_markdown_report(report)
    assert "OFFLINE SMOKE" in output
    assert "## hotpotqa" in output
    assert "## multi_news" in output
    assert "N/A" in output
    assert "P95 ms" in output


def test_csv_exports_lists_as_json(tmp_path: Path):
    rows = [{"case_id": "q1", "sources": ["d0", "d1"], "latency_ms": 11.4}]
    file_path = tmp_path / "cases.csv"
    export_csv(rows, file_path)
    with file_path.open(encoding="utf-8", newline="") as handle:
        data = list(csv.DictReader(handle))
    assert data[0]["sources"] == '["d0", "d1"]'
    assert data[0]["case_id"] == "q1"
