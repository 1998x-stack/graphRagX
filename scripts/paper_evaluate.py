"""Run the paper's four criteria as blinded, repeated pairwise comparisons.

First create a cross-mode report with scripts.evaluate. This script requires an
actual evaluator model and will not silently grade answers using the mock.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evaluation.paper_judge import evaluate_pairs
from evaluation.runner import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blind pairwise Global vs Basic evaluation for arXiv:2404.16130"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/pairwise.json"))
    parser.add_argument("--mode-a", default="global")
    parser.add_argument("--mode-b", default="basic")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=125)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    paired = await evaluate_pairs(
        report,
        mode_a=args.mode_a,
        mode_b=args.mode_b,
        repeats=args.repeats,
        max_cases=args.max_cases,
    )
    write_report(paired, args.output)
    print(f"Wrote pairwise paper evaluation: {args.output}")
    for criterion, row in paired["summary"].items():
        print(f"{criterion}: {row['preference_rate_with_half_ties']:.3f} "
              f"({row['wins']}W/{row['losses']}L/{row['ties']}T)")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
