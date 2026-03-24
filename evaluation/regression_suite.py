from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from combined_inference import classify_query
from schemas import validate_classify_response


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(payload: dict, dotted_path: str):
    value = payload
    for part in dotted_path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def evaluate_case_file(cases_path: Path, output_dir: Path, artifact_name: str) -> dict:
    cases = load_cases(cases_path)
    results = []
    counts_by_status: dict[str, dict[str, int]] = {}

    for case in cases:
        payload = validate_classify_response(classify_query(case["text"]))
        mismatches = []
        expected = case.get("expected", {})
        actual_snapshot = {}
        for dotted_path, expected_value in expected.items():
            actual_value = resolve_path(payload, dotted_path)
            actual_snapshot[dotted_path] = actual_value
            if actual_value != expected_value:
                mismatches.append(
                    {
                        "path": dotted_path,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )

        status = case["status"]
        bucket = counts_by_status.setdefault(status, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if mismatches:
            bucket["failed"] += 1
        else:
            bucket["passed"] += 1

        results.append(
            {
                "id": case["id"],
                "status": status,
                "text": case["text"],
                "notes": case.get("notes", ""),
                "pass": not mismatches,
                "mismatches": mismatches,
                "expected": expected,
                "actual": actual_snapshot,
            }
        )

    summary = {
        "cases_path": str(cases_path),
        "count": len(results),
        "passed": sum(1 for item in results if item["pass"]),
        "failed": sum(1 for item in results if not item["pass"]),
        "by_status": counts_by_status,
        "results": results,
    }
    write_json(output_dir / artifact_name, summary)
    return summary


def evaluate_known_failure_cases(cases_path: Path, output_dir: Path) -> dict:
    return evaluate_case_file(cases_path, output_dir, "known_failure_regression.json")


def evaluate_iab_behavior_lock_cases(cases_path: Path, output_dir: Path) -> dict:
    return evaluate_case_file(cases_path, output_dir, "iab_behavior_lock_regression.json")


def evaluate_iab_cross_vertical_behavior_lock_cases(cases_path: Path, output_dir: Path) -> dict:
    return evaluate_case_file(cases_path, output_dir, "iab_cross_vertical_behavior_lock_regression.json")


def evaluate_iab_quality_target_cases(cases_path: Path, output_dir: Path) -> dict:
    return evaluate_case_file(cases_path, output_dir, "iab_quality_target_eval.json")


def evaluate_iab_cross_vertical_quality_target_cases(cases_path: Path, output_dir: Path) -> dict:
    return evaluate_case_file(cases_path, output_dir, "iab_cross_vertical_quality_target_eval.json")
