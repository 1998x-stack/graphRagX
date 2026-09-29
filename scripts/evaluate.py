"""CLI for deterministic multi-mode evaluation."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evaluation.runner import load_cases, run_dataset, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate graphRagX Basic/Local/Global/DRIFT modes."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/report.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset)
    report = await run_dataset(cases)
    write_report(report, args.output)
    print(f"Wrote evaluation report: {args.output}")
    for mode, metrics in report["summary"].items():
        print(
            f"{mode}: success={metrics['success_rate']:.3f} "
            f"source_recall={metrics['avg_source_recall']:.3f} "
            f"term_coverage={metrics['avg_answer_term_coverage']:.3f} "
            f"latency_ms={metrics['avg_latency_ms']:.1f}"
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
