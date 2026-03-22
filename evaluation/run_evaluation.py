from __future__ import annotations

import argparse
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
    IAB_MAPPING_CASES_PATH,
    KNOWN_FAILURE_CASES_PATH,
    ensure_artifact_dirs,
)
from evaluation.regression_suite import evaluate_iab_mapping_cases, evaluate_known_failure_cases
from model_runtime import get_head
from schemas import validate_classify_response


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

    accepted_count = sum(accepted)
    accepted_accuracy = (
        accuracy_score(
            [truth for truth, keep in zip(y_true, accepted) if keep],
            [pred for pred, keep in zip(y_pred, accepted) if keep],
        )
        if accepted_count
        else 0.0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(config.labels),
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "head": head_name,
        "suite": suite_name,
        "dataset_path": str(dataset_path),
        "count": len(rows),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "accepted_coverage": round(float(accepted_count / len(rows)), 4),
        "accepted_accuracy": round(float(accepted_accuracy), 4),
        "fallback_rate": round(float(1 - (accepted_count / len(rows))), 4),
        "per_class_metrics": report,
        "confusion_matrix_path": str(confusion_path),
    }
    write_json(output_dir / f"{head_name}_{suite_name}_report.json", summary)
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
    args = parser.parse_args()

    ensure_artifact_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"heads": {}, "combined": {}}
    for head_name, config in HEAD_CONFIGS.items():
        head_summary = {}
        for split_name, split_path in config.split_paths.items():
            head_summary[split_name] = evaluate_head_dataset(head_name, split_path, split_name, output_dir)
        for suite_name, suite_path in config.stress_suite_paths.items():
            head_summary[suite_name] = evaluate_head_dataset(head_name, suite_path, suite_name, output_dir)
        summary["heads"][head_name] = head_summary

    summary["combined"]["demo_benchmark"] = evaluate_combined_benchmark(DEFAULT_BENCHMARK_PATH, output_dir)
    summary["combined"]["known_failure_regression"] = evaluate_known_failure_cases(KNOWN_FAILURE_CASES_PATH, output_dir)
    summary["combined"]["iab_mapping_regression"] = evaluate_iab_mapping_cases(IAB_MAPPING_CASES_PATH, output_dir)
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
            "iab_mapping_regression": {
                "count": summary["combined"]["iab_mapping_regression"]["count"],
                "passed": summary["combined"]["iab_mapping_regression"]["passed"],
                "failed": summary["combined"]["iab_mapping_regression"]["failed"],
                "by_status": summary["combined"]["iab_mapping_regression"]["by_status"],
            },
        },
        "summary_path": str(output_dir / "summary.json"),
    }
    print(json.dumps(compact_summary, indent=2))


if __name__ == "__main__":
    main()
