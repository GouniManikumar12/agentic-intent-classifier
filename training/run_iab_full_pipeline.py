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
    parser = argparse.ArgumentParser(description="Run the embedding-based IAB build/evaluation pipeline.")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for building local IAB taxonomy embeddings.",
    )
    parser.add_argument(
        "--skip-full-eval",
        action="store_true",
        help="Skip the full evaluation suite and only run IAB-specific regression.",
    )
    args = parser.parse_args()

    python = sys.executable
    run_step([python, "training/build_iab_difficulty_dataset.py"])
    run_step([python, "training/build_iab_cross_vertical_benchmark.py"])
    run_step(
        [
            python,
            "training/build_iab_taxonomy_embeddings.py",
            "--batch-size",
            str(args.embedding_batch_size),
        ]
    )
    run_step([python, "evaluation/run_iab_mapping_suite.py"])
    if not args.skip_full_eval:
        run_step([python, "evaluation/run_regression_suite.py"])
        run_step([python, "evaluation/run_evaluation.py"])


if __name__ == "__main__":
    main()
