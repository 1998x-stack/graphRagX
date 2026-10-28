"""Download pinned HF fixtures, run a paired benchmark, or compare model modes.

Example:
  python -m scripts.hf_eval download --dataset hotpotqa --count 10 --output data/hf/hotpot.jsonl
  python -m scripts.hf_eval run --input data/hf/hotpot.jsonl --conditions basic local global \
    --output outputs/evaluation/hotpot.json
  python -m scripts.hf_eval compare --input outputs/evaluation/hotpot.json \
    --candidate global --baseline basic --dataset hotpotqa --output outputs/evaluation/diff.json

The default provider is mock/hash. Running an *actual* benchmark requires
real configured providers; --allow-mock-smoke is an explicit smoke-test opt-in.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evaluation.hf_benchmark import run_benchmark, write_report
from evaluation.hf_datasets import (
    DATASETS, download_cases, load_fixture, write_fixture,
)
from evaluation.hf_statistics import benchmark_comparison_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pinned Hugging Face multi-axis GraphRAG evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Fetch public HF corpus + labels")
    dl.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    dl.add_argument("--count", type=int, default=20)
    dl.add_argument("--seed", type=int, default=42)
    dl.add_argument("--revision", type=str, default=None)
    dl.add_argument("--split", type=str, default=None)
    dl.add_argument("--max-source-documents", type=int, default=50)
    dl.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="Index once and run paired query modes")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--conditions", nargs="+", default=[
        "basic", "local", "global", "drift", "text_summary"
    ])
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--allow-mock-smoke", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--output", type=Path, required=True)

    cmp = sub.add_parser("compare", help="Paired bootstrap and slice comparison")
    cmp.add_argument("--input", type=Path, required=True)
    cmp.add_argument("--candidate", type=str, default="global")
    cmp.add_argument("--baseline", type=str, default="basic")
    cmp.add_argument("--dataset", type=str, default=None)
    cmp.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    if args.command == "download":
        cases, metadata = download_cases(
            dataset=args.dataset, count=args.count,
            seed=args.seed, revision=args.revision,
            split=args.split, max_source_documents=args.max_source_documents,
        )
        write_fixture(cases, metadata, args.output)
        print(f"Saved {len(cases)} pinned benchmark cases to {args.output}")
        return
    if args.command == "run":
        cases = load_fixture(args.input)
        report = await run_benchmark(
            cases, args.conditions,
            allow_mock_smoke=args.allow_mock_smoke,
            overwrite=args.overwrite, k=args.top_k,
        )
        write_report(report, args.output)
        print(f"Mode: {report['mode']} | cases={len(cases)} | "
              f"rows={len(report['results'])} | output={args.output}")
        for condition, metrics in report["summary"]["condition"].items():
            print(
                f"{condition}: n={metrics['count']} success={metrics['success_rate']:.3f} "
                f"token_f1={metrics['answer_token_f1']} "
                f"evidence_recall={metrics['evidence_recall']} "
                f"latency_ms={metrics['latency_ms']}"
            )
        return
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report.get("mode") == "offline_smoke":
        raise RuntimeError(
            "Do not report benchmark quality comparisons from mock/hash smoke tests"
        )
    comparison = benchmark_comparison_matrix(
        report, candidate=args.candidate,
        baseline=args.baseline, dataset=args.dataset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote paired bootstrap comparison: {args.output}")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
