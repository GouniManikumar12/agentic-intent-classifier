from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from combined_inference import classify_query
from config import EVALUATION_ARTIFACTS_DIR, IAB_MAPPING_CASES_PATH, IAB_HEAD_CONFIG, ensure_artifact_dirs
from evaluation.regression_suite import load_cases, resolve_path, write_json
from model_runtime import get_head
from training.common import load_labeled_rows


def evaluate_curated_cases(threshold: float, cases: list[dict]) -> dict:
    passed = 0
    classifier_taken = 0
    case_results = []
    for case in cases:
        payload = classify_query(case["text"], threshold_overrides={"iab_content": threshold})
        expected = case.get("expected", {})
        mismatches = []
        for dotted_path, expected_value in expected.items():
            actual_value = resolve_path(payload, dotted_path)
            if actual_value != expected_value:
                mismatches.append({"path": dotted_path, "expected": expected_value, "actual": actual_value})

        head_pred = get_head("iab_content").predict(case["text"], confidence_threshold=threshold)
        if head_pred["meets_confidence_threshold"]:
            classifier_taken += 1
        if not mismatches:
            passed += 1
        case_results.append(
            {
                "id": case["id"],
                "pass": not mismatches,
                "classifier_taken": head_pred["meets_confidence_threshold"],
                "classifier_label": head_pred["label"],
                "classifier_confidence": head_pred["confidence"],
                "mismatches": mismatches,
            }
        )
    return {
        "threshold": threshold,
        "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
        "passed": passed,
        "total": len(cases),
        "classifier_take_rate": round(classifier_taken / len(cases), 4) if cases else 0.0,
        "results": case_results,
    }


def evaluate_validation_set(threshold: float) -> dict:
    head = get_head("iab_content")
    rows = load_labeled_rows(IAB_HEAD_CONFIG.split_paths["val"], IAB_HEAD_CONFIG.label_field, IAB_HEAD_CONFIG.label2id)
    predictions = head.predict_batch([row["text"] for row in rows], confidence_threshold=threshold)
    accepted = [pred["meets_confidence_threshold"] for pred in predictions]
    accepted_count = sum(accepted)
    correct = [
        pred["label"] == head.config.id2label[row["label"]]
        for row, pred in zip(rows, predictions)
    ]
    accepted_accuracy = (
        sum(is_correct for is_correct, keep in zip(correct, accepted) if keep) / accepted_count
        if accepted_count
        else 0.0
    )
    return {
        "threshold": threshold,
        "accepted_coverage": round(accepted_count / len(rows), 4) if rows else 0.0,
        "accepted_accuracy": round(accepted_accuracy, 4),
        "fallback_rate": round(1 - (accepted_count / len(rows)), 4) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep IAB classifier confidence thresholds.")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7],
        help="Thresholds to evaluate.",
    )
    parser.add_argument(
        "--output-path",
        default=str(EVALUATION_ARTIFACTS_DIR / "latest" / "iab_threshold_sweep.json"),
        help="Artifact path for sweep output.",
    )
    args = parser.parse_args()

    ensure_artifact_dirs()
    cases = load_cases(IAB_MAPPING_CASES_PATH)
    threshold_values = [min(max(float(value), 0.0), 1.0) for value in args.thresholds]
    results = []
    for threshold in threshold_values:
        results.append(
            {
                "curated_cases": evaluate_curated_cases(threshold, cases),
                "validation": evaluate_validation_set(threshold),
            }
        )

    payload = {
        "head": "iab_content",
        "results": results,
    }
    output_path = Path(args.output_path)
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
