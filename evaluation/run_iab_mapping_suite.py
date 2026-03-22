from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import EVALUATION_ARTIFACTS_DIR, IAB_MAPPING_CASES_PATH
from evaluation.regression_suite import evaluate_iab_mapping_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curated IAB content mapping regression checks.")
    parser.add_argument(
        "--cases-path",
        default=str(IAB_MAPPING_CASES_PATH),
        help="Structured IAB mapping case file to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_ARTIFACTS_DIR / "latest"),
        help="Directory to write regression artifacts into.",
    )
    args = parser.parse_args()

    summary = evaluate_iab_mapping_cases(Path(args.cases_path), Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
