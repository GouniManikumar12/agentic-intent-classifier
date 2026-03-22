from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import EVALUATION_ARTIFACTS_DIR, KNOWN_FAILURE_CASES_PATH
from evaluation.regression_suite import evaluate_known_failure_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured known-failure regression checks.")
    parser.add_argument(
        "--cases-path",
        default=str(KNOWN_FAILURE_CASES_PATH),
        help="Structured known-failure case file to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_ARTIFACTS_DIR / "latest"),
        help="Directory to write regression artifacts into.",
    )
    args = parser.parse_args()

    summary = evaluate_known_failure_cases(Path(args.cases_path), Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
