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
    parser = argparse.ArgumentParser(description="Run the full multi-head training pipeline.")
    parser.add_argument("--iab-epochs", type=float, default=2.0, help="Training epochs for the IAB head.")
    parser.add_argument("--iab-train-batch-size", type=int, default=16, help="Per-device training batch size for IAB.")
    parser.add_argument("--iab-eval-batch-size", type=int, default=16, help="Per-device eval batch size for IAB.")
    parser.add_argument("--iab-learning-rate", type=float, default=2e-5, help="Learning rate for IAB training.")
    parser.add_argument(
        "--iab-target-rows-per-label",
        type=int,
        default=0,
        help="Optional cap per IAB label. Use 0 for the uncapped full dataset.",
    )
    parser.add_argument(
        "--skip-full-eval",
        action="store_true",
        help="Skip the final full evaluation pass and only run calibration.",
    )
    args = parser.parse_args()

    python = sys.executable
    run_step([python, "training/build_full_intent_taxonomy_dataset.py"])
    run_step([python, "training/train.py"])
    run_step([python, "training/build_subtype_dataset.py"])
    run_step([python, "training/train_subtype.py"])
    run_step([python, "training/train_decision_phase.py"])
    run_step(
        [
            python,
            "training/build_iab_dataset.py",
            "--target-rows-per-label",
            str(args.iab_target_rows_per_label),
        ]
    )
    run_step(
        [
            python,
            "training/train_iab_content.py",
            "--epochs",
            str(args.iab_epochs),
            "--train-batch-size",
            str(args.iab_train_batch_size),
            "--eval-batch-size",
            str(args.iab_eval_batch_size),
            "--learning-rate",
            str(args.iab_learning_rate),
        ]
    )
    run_step([python, "training/calibrate_confidence.py", "--head", "all"])
    if not args.skip_full_eval:
        run_step([python, "evaluation/run_regression_suite.py"])
        run_step([python, "evaluation/run_iab_mapping_suite.py"])
        run_step([python, "evaluation/run_evaluation.py"])


if __name__ == "__main__":
    main()
