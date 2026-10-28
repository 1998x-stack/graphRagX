"""Publish a model-aware, dataset-stratified Markdown and CSV benchmark report.

Plots are optional and only render if matplotlib is installed. The report
never promotes offline smoke outputs to measured model quality claims.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
    "success", "exact_match", "answer_token_f1", "rouge_l_f1",
    "evidence_recall", "evidence_precision", "evidence_mrr",
    "evidence_ndcg_at_k", "latency_ms", "community_report_coverage",
)


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_markdown_report(report: dict) -> str:
    settings = report.get("settings", {})
    smoke = report.get("mode") == "offline_smoke"
    rows = report.get("results", [])
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in rows:
        groups[str(item.get("dataset"))][str(item.get("condition"))].append(item)
    lines = [
        "# graphRagX Hugging Face Evaluation",
        "",
        f"**Status:** {'OFFLINE SMOKE — NOT SCIENTIFIC PERFORMANCE DATA' if smoke else 'REAL MODEL RUN — SEE LIMITATIONS'}",
        "",
        f"LLM: \`{settings.get('llm_provider', 'unknown')} / {settings.get('llm_model', 'unknown')}\`  ",
        f"Embeddings: \`{settings.get('embedding_provider', 'unknown')} / {settings.get('embedding_model', 'unknown')}\`  ",
        f"Chunking: \`{settings.get('chunk_unit', 'unknown')} / {settings.get('chunk_size', '?')} / {settings.get('chunk_overlap', '?')}\`",
        "",
    ]
    for dataset, modes in sorted(groups.items()):
        lines.extend([
            f"## {dataset}",
            "",
            "| Condition | N | Success | EM | Token F1 | ROUGE-L | Evidence recall | P95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        from evaluation.hf_metrics import _percentile_95
        for condition, items in sorted(modes.items()):
            def average(key):
                vals = [item[key] for item in items if isinstance(item.get(key), (int, float))]
                return mean(vals) if vals else None
            lines.append(
                "| " + " | ".join([
                    condition, str(len(items)),
                    _format(average("success")),
                    _format(average("exact_match")),
                    _format(average("answer_token_f1")),
                    _format(average("rouge_l_f1")),
                    _format(average("evidence_recall")),
                    _format(_percentile_95([item.get("latency_ms") for item in items])),
                ]) + " |"
            )
        lines.extend(["", "Missing labels are shown as N/A, never substituted with 1.0.", ""])
    lines.extend([
        "## Interpretation limits",
        "",
        "* The HotpotQA/MuSiQue/QASPER corpus is the question's oracle candidate context, not open-corpus search.",
        "* Global community-to-document mapping is candidate exposure, not claim-level citation faithfulness.",
        "* ROUGE-L is a lexical summary metric, not a semantic factuality judge.",
        "* No token-cost or dollar-cost numbers are reported without measured usage.",
        "* Compare the same question IDs and frozen LLM/embedding settings before statistical inference.",
        "",
    ])
    return "\n".join(lines)


def export_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [
        {
            key: json.dumps(row.get(key), ensure_ascii=False)
            if isinstance(row.get(key), (list, dict)) else row.get(key)
            for key in sorted({field for item in rows for field in item})
        }
        for row in rows
    ]
    if not flat_rows:
        raise ValueError("No rows available")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def plot_per_dataset(report: dict, output_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Install matplotlib to render figures: pip install matplotlib"
        ) from exc
    grouped = defaultdict(lambda: defaultdict(list))
    for row in report.get("results", []):
        dataset, condition = row.get("dataset"), row.get("condition")
        if dataset and condition:
            grouped[dataset][condition].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset, modes in sorted(grouped.items()):
        metric = "rouge_l_f1" if dataset == "multi_news" else "answer_token_f1"
        names, values = [], []
        for condition, rows in sorted(modes.items()):
            observed = [
                row.get(metric) for row in rows
                if isinstance(row.get(metric), (int, float))
            ]
            if observed:
                names.append(condition)
                values.append(mean(observed))
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(max(7, len(names)), 4))
        ax.bar(names, values)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.set_title(f"{dataset} — {metric} (per-question mean)")
        fig.tight_layout()
        path = output_dir / f"{dataset}_{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = args.output_dir / "summary.md"
    markdown.write_text(build_markdown_report(report), encoding="utf-8")
    export_csv(report.get("results", []), args.output_dir / "per_question.csv")
    print(f"Report: {markdown}")
    if args.plots:
        for path in plot_per_dataset(report, args.output_dir / "plots"):
            print(f"Figure: {path}")


if __name__ == "__main__":
    main()
