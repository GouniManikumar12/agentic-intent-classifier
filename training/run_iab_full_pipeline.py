from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def run_step(args: list[str]) -> None:
    print(f"\n==> Running: {' '.join(args)}")
    subprocess.run(args, cwd=BASE_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full IAB training/calibration/evaluation pipeline.")
    parser.add_argument("--epochs", type=float, default=2.0, help="Training epochs for the IAB head.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Per-device eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Training learning rate.")
    parser.add_argument(
        "--skip-full-eval",
        action="store_true",
        help="Skip the full evaluation suite and only run IAB-specific regression.",
    )
    args = parser.parse_args()

    python = sys.executable
    run_step([python, "training/build_iab_dataset.py"])
    run_step(
        [
            python,
            "training/train_iab_content.py",
            "--epochs",
            str(args.epochs),
            "--train-batch-size",
            str(args.train_batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--learning-rate",
            str(args.learning_rate),
        ]
    )
    run_step([python, "training/calibrate_confidence.py", "--head", "iab_content"])
    run_step([python, "evaluation/run_iab_mapping_suite.py"])
    if not args.skip_full_eval:
        run_step([python, "evaluation/run_regression_suite.py"])
        run_step([python, "evaluation/run_evaluation.py"])


if __name__ == "__main__":
    main()
