"""Run the paper's six retrieval/summary conditions on one frozen index."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evaluation.paper_conditions import CONDITIONS, run_paper_conditions
from evaluation.runner import load_cases, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Edge et al. C0/C1/C2/C3/TS/SS comparison arms."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/evaluation/paper_conditions.json")
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=list(CONDITIONS),
        default=list(CONDITIONS)
    )
    parser.add_argument("--max-cases", type=int, default=125)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset)
    report = await run_paper_conditions(
        cases,
        conditions=args.conditions,
        max_cases=args.max_cases,
    )
    write_report(report, args.output)
    print(f"Wrote six-condition evaluation: {args.output}")
    for condition, metrics in report["summary"].items():
        print(
            f"{condition}: success={metrics['success_rate']:.3f}, "
            f"latency_ms={metrics['avg_latency_ms']:.1f}"
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
