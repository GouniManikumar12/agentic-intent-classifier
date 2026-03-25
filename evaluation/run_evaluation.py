from __future__ import annotations

import os

# Quieter logs when TensorFlow/XLA are pulled in indirectly (common on Colab).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from combined_inference import classify_query
from config import (
    DEFAULT_BENCHMARK_PATH,
    EVALUATION_ARTIFACTS_DIR,
    HEAD_CONFIGS,
    IAB_HEAD_CONFIG,
    IAB_BEHAVIOR_LOCK_CASES_PATH,
    IAB_CROSS_VERTICAL_BEHAVIOR_LOCK_CASES_PATH,
    IAB_CROSS_VERTICAL_QUALITY_TARGET_CASES_PATH,
    IAB_QUALITY_TARGET_CASES_PATH,
    KNOWN_FAILURE_CASES_PATH,
    ensure_artifact_dirs,
)
from evaluation.regression_suite import (
    evaluate_iab_behavior_lock_cases,
    evaluate_iab_cross_vertical_behavior_lock_cases,
    evaluate_iab_cross_vertical_quality_target_cases,
    evaluate_iab_quality_target_cases,
    evaluate_known_failure_cases,
)
from evaluation.iab_quality import compute_path_metrics, evaluate_iab_views, path_from_label
from iab_classifier import predict_iab_content_classifier_batch
from model_runtime import get_head
from schemas import validate_classify_response


def _maybe_free_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_head_dataset(head_name: str, dataset_path: Path, suite_name: str, output_dir: Path) -> dict:
    head = get_head(head_name)
    config = head.config
    rows = load_jsonl(dataset_path)
    predictions = head.predict_batch([row["text"] for row in rows])

    y_true = [row[config.label_field] for row in rows]
    y_pred = [prediction["label"] for prediction in predictions]
    accepted = [prediction["meets_confidence_threshold"] for prediction in predictions]

    confusion = confusion_matrix(y_true, y_pred, labels=list(config.labels))
    confusion_df = pd.DataFrame(confusion, index=config.labels, columns=config.labels)
    confusion_path = output_dir / f"{head_name}_{suite_name}_confusion_matrix.csv"
    confusion_df.to_csv(confusion_path)

    accepted_total_count = sum(accepted)
    accepted_accuracy = (
        accuracy_score(
            [truth for truth, keep in zip(y_true, accepted) if keep],
            [pred for pred, keep in zip(y_pred, accepted) if keep],
        )
        if accepted_total_count
        else 0.0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(config.labels),
        output_dict=True,
        zero_division=0,
    )
    difficulty_breakdown = None
    if rows and all("difficulty" in row for row in rows):
        difficulty_breakdown = {}
        for difficulty in sorted({row["difficulty"] for row in rows}):
            indices = [idx for idx, row in enumerate(rows) if row["difficulty"] == difficulty]
            difficulty_true = [y_true[idx] for idx in indices]
            difficulty_pred = [y_pred[idx] for idx in indices]
            difficulty_accepted = [accepted[idx] for idx in indices]
            difficulty_accepted_count = sum(difficulty_accepted)
            difficulty_accepted_accuracy = (
                accuracy_score(
                    [truth for truth, keep in zip(difficulty_true, difficulty_accepted) if keep],
                    [pred for pred, keep in zip(difficulty_pred, difficulty_accepted) if keep],
                )
                if difficulty_accepted_count
                else 0.0
            )
            difficulty_breakdown[difficulty] = {
                "count": len(indices),
                "accuracy": round(float(accuracy_score(difficulty_true, difficulty_pred)), 4),
                "macro_f1": round(float(f1_score(difficulty_true, difficulty_pred, average="macro")), 4),
                "accepted_coverage": round(float(difficulty_accepted_count / len(indices)), 4),
                "accepted_accuracy": round(float(difficulty_accepted_accuracy), 4),
                "fallback_rate": round(float(1 - (difficulty_accepted_count / len(indices))), 4),
            }
    summary = {
        "head": head_name,
        "suite": suite_name,
        "dataset_path": str(dataset_path),
        "count": len(rows),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "accepted_coverage": round(float(accepted_total_count / len(rows)), 4),
        "accepted_accuracy": round(float(accepted_accuracy), 4),
        "fallback_rate": round(float(1 - (accepted_total_count / len(rows))), 4),
        "per_class_metrics": report,
        "confusion_matrix_path": str(confusion_path),
    }
    if difficulty_breakdown is not None:
        summary["difficulty_breakdown"] = difficulty_breakdown
    write_json(output_dir / f"{head_name}_{suite_name}_report.json", summary)
    return summary


def evaluate_iab_dataset(dataset_path: Path, suite_name: str, output_dir: Path) -> dict:
    rows = load_jsonl(dataset_path)
    true_paths = [path_from_label(row["iab_path"]) for row in rows]
    true_labels = [row["iab_path"] for row in rows]
    predictions = predict_iab_content_classifier_batch([row["text"] for row in rows])
    if not any(output is not None for output in predictions):
        raise RuntimeError(
            "IAB classifier artifacts are unavailable. Run `python3 training/train_iab.py` "
            "and `python3 training/calibrate_confidence.py --head iab_content` "
            "from the `agentic-intent-classifier` directory first."
        )

    pred_paths = [
        tuple(output["path"]) if output is not None else tuple()
        for output in predictions
    ]
    accepted = [bool(output and output["meets_confidence_threshold"]) for output in predictions]
    source = next((output["source"] for output in predictions if output is not None), "supervised_classifier")
    pred_labels = [" > ".join(path) if path else "__no_prediction__" for path in pred_paths]

    accepted_total_count = sum(accepted)
    accepted_accuracy = (
        sum(1 for truth, pred, keep in zip(true_paths, pred_paths, accepted) if keep and truth == pred) / accepted_total_count
        if accepted_total_count
        else 0.0
    )
    difficulty_breakdown = None
    if rows and all("difficulty" in row for row in rows):
        difficulty_breakdown = {}
        for difficulty in sorted({row["difficulty"] for row in rows}):
            indices = [idx for idx, row in enumerate(rows) if row["difficulty"] == difficulty]
            difficulty_true_paths = [true_paths[idx] for idx in indices]
            difficulty_pred_paths = [pred_paths[idx] for idx in indices]
            difficulty_true_labels = [true_labels[idx] for idx in indices]
            difficulty_pred_labels = [pred_labels[idx] for idx in indices]
            difficulty_accepted = [accepted[idx] for idx in indices]
            difficulty_accepted_count = sum(difficulty_accepted)
            difficulty_accepted_accuracy = (
                sum(
                    1
                    for truth, pred, keep in zip(difficulty_true_paths, difficulty_pred_paths, difficulty_accepted)
                    if keep and truth == pred
                )
                / difficulty_accepted_count
                if difficulty_accepted_count
                else 0.0
            )
            difficulty_breakdown[difficulty] = {
                "count": len(indices),
                "accuracy": round(
                    float(sum(1 for truth, pred in zip(difficulty_true_paths, difficulty_pred_paths) if truth == pred) / max(len(indices), 1)),
                    4,
                ),
                "macro_f1": round(float(f1_score(difficulty_true_labels, difficulty_pred_labels, average="macro")), 4),
                "accepted_coverage": round(float(difficulty_accepted_count / max(len(indices), 1)), 4),
                "accepted_accuracy": round(float(difficulty_accepted_accuracy), 4),
                "fallback_rate": round(float(1 - (difficulty_accepted_count / max(len(indices), 1))), 4),
            }
    summary = {
        "head": "iab_content",
        "suite": suite_name,
        "dataset_path": str(dataset_path),
        "count": len(rows),
        "accuracy": round(float(sum(1 for truth, pred in zip(true_paths, pred_paths) if truth == pred) / max(len(rows), 1)), 4),
        "macro_f1": round(float(f1_score(true_labels, pred_labels, average="macro")), 4),
        "accepted_coverage": round(float(accepted_total_count / max(len(rows), 1)), 4),
        "accepted_accuracy": round(float(accepted_accuracy), 4),
        "fallback_rate": round(float(1 - (accepted_total_count / max(len(rows), 1))), 4),
        "primary_source": source,
        "tier_metrics": compute_path_metrics(true_paths, pred_paths),
        "view_metrics": evaluate_iab_views(rows),
    }
    if difficulty_breakdown is not None:
        summary["difficulty_breakdown"] = difficulty_breakdown
    write_json(output_dir / f"iab_content_{suite_name}_report.json", summary)
    return summary


def evaluate_combined_benchmark(path: Path, output_dir: Path) -> dict:
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    outputs = []
    fallback_applied = 0
    for item in benchmark:
        payload = validate_classify_response(classify_query(item["input"]))
        if payload["model_output"].get("fallback"):
            fallback_applied += 1
        outputs.append(
            {
                "input": item["input"],
                "expected_behavior": item["expected_behavior"],
                "response": payload,
            }
        )
    write_json(output_dir / "combined_demo_benchmark.json", outputs)
    return {
        "benchmark_path": str(path),
        "count": len(outputs),
        "fallback_rate": round(fallback_applied / len(outputs), 4) if outputs else 0.0,
        "output_path": str(output_dir / "combined_demo_benchmark.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeatable evaluation for classifier heads and combined output.")
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_ARTIFACTS_DIR / "latest"),
        help="Directory to write evaluation artifacts into.",
    )
    parser.add_argument(
        "--skip-iab-train-eval",
        action="store_true",
        help="Skip the IAB train split (largest JSONL). Use on low-RAM hosts (e.g. Colab free tier).",
    )
    args = parser.parse_args()

    ensure_artifact_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"heads": {}, "combined": {}}
    for head_name, config in HEAD_CONFIGS.items():
        if head_name == "iab_content":
            continue
        head_summary = {}
        for split_name, split_path in config.split_paths.items():
            head_summary[split_name] = evaluate_head_dataset(head_name, split_path, split_name, output_dir)
        for suite_name, suite_path in config.stress_suite_paths.items():
            head_summary[suite_name] = evaluate_head_dataset(head_name, suite_path, suite_name, output_dir)
        summary["heads"][head_name] = head_summary
        gc.collect()
        _maybe_free_cuda_memory()

    iab_summary = {}
    for split_name, split_path in IAB_HEAD_CONFIG.split_paths.items():
        if args.skip_iab_train_eval and split_name == "train":
            continue
        iab_summary[split_name] = evaluate_iab_dataset(split_path, split_name, output_dir)
        gc.collect()
        _maybe_free_cuda_memory()
    for suite_name, suite_path in IAB_HEAD_CONFIG.stress_suite_paths.items():
        iab_summary[suite_name] = evaluate_iab_dataset(suite_path, suite_name, output_dir)
        gc.collect()
        _maybe_free_cuda_memory()
    summary["heads"]["iab_content"] = iab_summary

    summary["combined"]["demo_benchmark"] = evaluate_combined_benchmark(DEFAULT_BENCHMARK_PATH, output_dir)
    summary["combined"]["known_failure_regression"] = evaluate_known_failure_cases(KNOWN_FAILURE_CASES_PATH, output_dir)
    summary["combined"]["iab_behavior_lock_regression"] = evaluate_iab_behavior_lock_cases(
        IAB_BEHAVIOR_LOCK_CASES_PATH,
        output_dir,
    )
    summary["combined"]["iab_cross_vertical_behavior_lock_regression"] = evaluate_iab_cross_vertical_behavior_lock_cases(
        IAB_CROSS_VERTICAL_BEHAVIOR_LOCK_CASES_PATH,
        output_dir,
    )
    summary["combined"]["iab_quality_target_eval"] = evaluate_iab_quality_target_cases(
        IAB_QUALITY_TARGET_CASES_PATH,
        output_dir,
    )
    summary["combined"]["iab_cross_vertical_quality_target_eval"] = evaluate_iab_cross_vertical_quality_target_cases(
        IAB_CROSS_VERTICAL_QUALITY_TARGET_CASES_PATH,
        output_dir,
    )
    write_json(output_dir / "summary.json", summary)
    compact_summary = {
        "heads": {
            head_name: {
                "test": {
                    key: head_summary["test"][key]
                    for key in (
                        "count",
                        "accuracy",
                        "macro_f1",
                        "accepted_accuracy",
                        "accepted_coverage",
                        "fallback_rate",
                    )
                }
                | (
                    {"tier_metrics": head_summary["test"]["tier_metrics"]}
                    if "tier_metrics" in head_summary["test"]
                    else {}
                )
            }
            for head_name, head_summary in summary["heads"].items()
        },
        "combined": {
            "demo_benchmark": summary["combined"]["demo_benchmark"],
            "known_failure_regression": {
                "count": summary["combined"]["known_failure_regression"]["count"],
                "passed": summary["combined"]["known_failure_regression"]["passed"],
                "failed": summary["combined"]["known_failure_regression"]["failed"],
                "by_status": summary["combined"]["known_failure_regression"]["by_status"],
            },
            "iab_behavior_lock_regression": {
                "count": summary["combined"]["iab_behavior_lock_regression"]["count"],
                "passed": summary["combined"]["iab_behavior_lock_regression"]["passed"],
                "failed": summary["combined"]["iab_behavior_lock_regression"]["failed"],
                "by_status": summary["combined"]["iab_behavior_lock_regression"]["by_status"],
            },
            "iab_cross_vertical_behavior_lock_regression": {
                "count": summary["combined"]["iab_cross_vertical_behavior_lock_regression"]["count"],
                "passed": summary["combined"]["iab_cross_vertical_behavior_lock_regression"]["passed"],
                "failed": summary["combined"]["iab_cross_vertical_behavior_lock_regression"]["failed"],
                "by_status": summary["combined"]["iab_cross_vertical_behavior_lock_regression"]["by_status"],
            },
            "iab_quality_target_eval": {
                "count": summary["combined"]["iab_quality_target_eval"]["count"],
                "passed": summary["combined"]["iab_quality_target_eval"]["passed"],
                "failed": summary["combined"]["iab_quality_target_eval"]["failed"],
                "by_status": summary["combined"]["iab_quality_target_eval"]["by_status"],
            },
            "iab_cross_vertical_quality_target_eval": {
                "count": summary["combined"]["iab_cross_vertical_quality_target_eval"]["count"],
                "passed": summary["combined"]["iab_cross_vertical_quality_target_eval"]["passed"],
                "failed": summary["combined"]["iab_cross_vertical_quality_target_eval"]["failed"],
                "by_status": summary["combined"]["iab_cross_vertical_quality_target_eval"]["by_status"],
            },
        },
        "summary_path": str(output_dir / "summary.json"),
    }
    print(json.dumps(compact_summary, indent=2))


if __name__ == "__main__":
    main()
