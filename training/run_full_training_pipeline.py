from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def run_step(args: list[str]) -> None:
    cmd = " ".join(args)
    started_at = time.perf_counter()
    started_wall = datetime.now(timezone.utc).isoformat()
    print(f"\n==> Running: {cmd}\n    start: {started_wall}")
    subprocess.run(args, cwd=BASE_DIR, check=True)
    elapsed_s = time.perf_counter() - started_at
    ended_wall = datetime.now(timezone.utc).isoformat()
    print(f"    end:   {ended_wall}\n    took:  {elapsed_s:.2f}s")


def main() -> None:
    pipeline_start = time.perf_counter()
    pipeline_start_wall = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(
        description=(
            "Run the full multi-head training pipeline: multitask intent, IAB classifier, calibration for all heads, "
            "then eval suites. IAB train+calibrate are part of the default sequence; only the taxonomy shadow index is an extra step."
        )
    )
    parser.add_argument(
        "--iab-embedding-batch-size",
        type=int,
        default=32,
        help="Batch size when building the IAB taxonomy embedding shadow index (extra; for retrieval/eval tooling).",
    )
    parser.add_argument(
        "--build-iab-shadow-index",
        action="store_true",
        help="Also run build_iab_taxonomy_embeddings.py (shadow index for retrieval metrics; IAB classifier already trained above).",
    )
    parser.add_argument(
        "--skip-full-eval",
        action="store_true",
        help="Skip the final full evaluation pass and only run calibration.",
    )
    parser.add_argument(
        "--export-multitask-onnx",
        action="store_true",
        help="After training, export training/export_multitask_onnx.py to multitask_intent_model_output/.",
    )
    parser.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="Run training/pipeline_verify.py after the pipeline (checks weights + calibration files).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run combined_inference.py on a sample query after the pipeline.",
    )
    parser.add_argument(
        "--smoke-test-query",
        default="Which laptop should I buy for college?",
        help="Query string for --smoke-test.",
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Shorthand for --export-multitask-onnx --verify-artifacts --smoke-test.",
    )
    args = parser.parse_args()

    if args.complete:
        args.export_multitask_onnx = True
        args.verify_artifacts = True
        args.smoke_test = True

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

    if args.export_multitask_onnx:
        run_step([python, "training/export_multitask_onnx.py"])
    if args.verify_artifacts:
        run_step([python, "training/pipeline_verify.py"])
    if args.smoke_test:
        run_step([python, "combined_inference.py", args.smoke_test_query])

    pipeline_elapsed_s = time.perf_counter() - pipeline_start
    pipeline_end_wall = datetime.now(timezone.utc).isoformat()
    print(
        f"\n==> Pipeline complete\n"
        f"    start: {pipeline_start_wall}\n"
        f"    end:   {pipeline_end_wall}\n"
        f"    total: {pipeline_elapsed_s:.2f}s"
    )


if __name__ == "__main__":
    main()
