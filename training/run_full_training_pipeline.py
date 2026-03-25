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
    parser.add_argument(
        "--iab-embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for building the optional IAB shadow retrieval index.",
    )
    parser.add_argument(
        "--build-iab-shadow-index",
        action="store_true",
        help="Also rebuild the optional IAB retrieval shadow index.",
    )
    parser.add_argument(
        "--skip-full-eval",
        action="store_true",
        help="Skip the final full evaluation pass and only run calibration.",
    )
    args = parser.parse_args()

    python = sys.executable
    run_step([python, "training/build_full_intent_taxonomy_dataset.py"])
    run_step([python, "training/build_intent_type_difficulty_dataset.py"])
    run_step([python, "training/train_multitask_intent.py"])
    run_step([python, "training/build_subtype_dataset.py"])
    run_step([python, "training/build_subtype_difficulty_dataset.py"])
    # Subtype labels are trained as part of multitask intent training.
    run_step([python, "training/build_decision_phase_difficulty_dataset.py"])
    # Decision phase labels are trained as part of multitask intent training.
    run_step([python, "training/build_iab_difficulty_dataset.py"])
    run_step([python, "training/build_iab_cross_vertical_benchmark.py"])
    run_step([python, "training/train_iab.py"])
    run_step([python, "training/calibrate_confidence.py", "--head", "intent_type"])
    run_step([python, "training/calibrate_confidence.py", "--head", "intent_subtype"])
    run_step([python, "training/calibrate_confidence.py", "--head", "decision_phase"])
    run_step([python, "training/calibrate_confidence.py", "--head", "iab_content"])
    if args.build_iab_shadow_index:
        run_step(
            [
                python,
                "training/build_iab_taxonomy_embeddings.py",
                "--batch-size",
                str(args.iab_embedding_batch_size),
            ]
        )
    if not args.skip_full_eval:
        run_step([python, "evaluation/run_regression_suite.py"])
        run_step([python, "evaluation/run_iab_mapping_suite.py"])
        run_step([python, "evaluation/run_iab_quality_suite.py"])
        run_step([python, "evaluation/run_evaluation.py"])


if __name__ == "__main__":
    main()
