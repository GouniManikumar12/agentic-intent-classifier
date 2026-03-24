from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    EVALUATION_ARTIFACTS_DIR,
    IAB_CROSS_VERTICAL_QUALITY_TARGET_CASES_PATH,
    IAB_QUALITY_TARGET_CASES_PATH,
)
from evaluation.regression_suite import (
    evaluate_iab_cross_vertical_quality_target_cases,
    evaluate_iab_quality_target_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curated IAB quality-target evaluation cases.")
    parser.add_argument(
        "--cases-path",
        default=str(IAB_QUALITY_TARGET_CASES_PATH),
        help="Curated IAB quality-target case file to execute.",
    )
    parser.add_argument(
        "--cross-vertical-cases-path",
        default=str(IAB_CROSS_VERTICAL_QUALITY_TARGET_CASES_PATH),
        help="Cross-vertical IAB quality-target case file to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_ARTIFACTS_DIR / "latest"),
        help="Directory to write evaluation artifacts into.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = {
        "curated_quality_targets": evaluate_iab_quality_target_cases(Path(args.cases_path), output_dir),
        "cross_vertical_quality_targets": evaluate_iab_cross_vertical_quality_target_cases(
            Path(args.cross_vertical_cases_path),
            output_dir,
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
