from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from combined_inference import classify_query
from config import BASE_DIR, INTENT_HEAD_CONFIG, ensure_artifact_dirs
from model_runtime import get_head
from schemas import validate_classify_response

DEFAULT_THRESHOLDS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
SAFE_INTENT_TYPES = {"ambiguous", "personal_reflection", "support"}
OBVIOUS_INTENT_TYPES = {"informational", "commercial", "transactional"}
SWEEP_SUITE_PATH = BASE_DIR / "examples" / "intent_threshold_sweep_suite.json"
OUTPUT_PATH = BASE_DIR / "artifacts" / "evaluation" / "intent_threshold_sweep.json"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def round_score(value: float) -> float:
    return round(float(value), 4)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_intent_head_threshold(threshold: float) -> dict:
    head = get_head("intent_type")
    dataset_specs = [
        ("val", INTENT_HEAD_CONFIG.split_paths["val"]),
        ("test", INTENT_HEAD_CONFIG.split_paths["test"]),
        ("hard_cases", BASE_DIR / "data" / "hard_cases.jsonl"),
        ("third_wave_cases", BASE_DIR / "data" / "third_wave_cases.jsonl"),
    ]
    rows = []
    for suite_name, path in dataset_specs:
        for item in load_jsonl(path):
            rows.append({"suite": suite_name, **item})

    predictions = head.predict_batch([row["text"] for row in rows], confidence_threshold=threshold)
    obvious_total = 0
    obvious_fallback = 0
    ambiguous_total = 0
    ambiguous_bad_allow = 0
    intent_only_safe_pred = 0

    for row, prediction in zip(rows, predictions):
        predicted_label = prediction["label"]
        head_would_fallback = (not prediction["meets_confidence_threshold"]) or (predicted_label in SAFE_INTENT_TYPES)

        if row[INTENT_HEAD_CONFIG.label_field] in OBVIOUS_INTENT_TYPES:
            obvious_total += 1
            if head_would_fallback:
                obvious_fallback += 1

        if row[INTENT_HEAD_CONFIG.label_field] == "ambiguous":
            ambiguous_total += 1
            if not head_would_fallback:
                ambiguous_bad_allow += 1

        if head_would_fallback and predicted_label in SAFE_INTENT_TYPES:
            intent_only_safe_pred += 1

    return {
        "obvious_prompt_count": obvious_total,
        "obvious_false_fallback_rate": round_score(obvious_fallback / obvious_total) if obvious_total else 0.0,
        "ambiguous_prompt_count": ambiguous_total,
        "ambiguous_bad_allow_rate": round_score(ambiguous_bad_allow / ambiguous_total) if ambiguous_total else 0.0,
        "safe_predicate_rate": round_score(intent_only_safe_pred / len(rows)) if rows else 0.0,
    }


def evaluate_combined_threshold(threshold: float) -> dict:
    suite = load_json(SWEEP_SUITE_PATH)
    benchmark = load_json(BASE_DIR / "examples" / "demo_prompt_suite.json")

    obvious_total = 0
    obvious_false_fallback = 0
    safe_total = 0
    safe_bad_allow = 0
    intent_only = 0
    phase_only = 0
    both = 0
    policy_safe = 0

    suite_outputs = []
    for item in suite:
        payload = validate_classify_response(classify_query(item["input"], threshold_overrides={"intent_type": threshold}))
        fallback = payload["model_output"].get("fallback")
        fallback_applied = fallback is not None
        failed_components = set((fallback or {}).get("failed_components", []))

        if item["expected_outcome"] == "pass":
            obvious_total += 1
            if fallback_applied:
                obvious_false_fallback += 1
        else:
            safe_total += 1
            if not fallback_applied:
                safe_bad_allow += 1

        if fallback_applied:
            if failed_components == {"intent_type"}:
                intent_only += 1
            elif failed_components == {"decision_phase"}:
                phase_only += 1
            elif failed_components == {"intent_type", "decision_phase"}:
                both += 1
            else:
                policy_safe += 1

        suite_outputs.append(
            {
                "input": item["input"],
                "expected_outcome": item["expected_outcome"],
                "fallback_applied": fallback_applied,
                "failed_components": sorted(failed_components),
                "intent_type": payload["model_output"]["classification"]["intent"]["type"],
                "decision_phase": payload["model_output"]["classification"]["intent"]["decision_phase"],
                "intent_confidence": payload["model_output"]["classification"]["intent"]["component_confidence"]["intent_type"]["confidence"],
                "phase_confidence": payload["model_output"]["classification"]["intent"]["component_confidence"]["decision_phase"]["confidence"],
            }
        )

    benchmark_fallbacks = []
    for item in benchmark:
        payload = validate_classify_response(classify_query(item["input"], threshold_overrides={"intent_type": threshold}))
        fallback = payload["model_output"].get("fallback")
        fallback_applied = fallback is not None
        failed_components = set((fallback or {}).get("failed_components", []))
        benchmark_fallbacks.append(
            {
                "fallback_applied": fallback_applied,
                "intent_only": failed_components == {"intent_type"},
                "phase_only": failed_components == {"decision_phase"},
                "both": failed_components == {"intent_type", "decision_phase"},
            }
        )

    total_suite_fallbacks = intent_only + phase_only + both + policy_safe
    benchmark_total_fallbacks = sum(1 for item in benchmark_fallbacks if item["fallback_applied"])
    return {
        "suite_path": str(SWEEP_SUITE_PATH),
        "obvious_prompt_count": obvious_total,
        "false_fallback_rate_on_obvious_prompts": round_score(obvious_false_fallback / obvious_total) if obvious_total else 0.0,
        "safe_prompt_count": safe_total,
        "bad_allow_rate_on_safe_prompts": round_score(safe_bad_allow / safe_total) if safe_total else 0.0,
        "fallback_responsibility": {
            "intent_only": intent_only,
            "phase_only": phase_only,
            "both": both,
            "policy_safe": policy_safe,
            "intent_share_of_threshold_fallbacks": round_score(
                (intent_only + both) / (intent_only + phase_only + both)
            )
            if (intent_only + phase_only + both)
            else 0.0,
            "phase_share_of_threshold_fallbacks": round_score(
                (phase_only + both) / (intent_only + phase_only + both)
            )
            if (intent_only + phase_only + both)
            else 0.0,
            "fallback_rate": round_score(total_suite_fallbacks / len(suite)) if suite else 0.0,
        },
        "benchmark_fallback_rate": round_score(benchmark_total_fallbacks / len(benchmark)) if benchmark else 0.0,
        "benchmark_intent_only_fallback_rate": round_score(
            sum(1 for item in benchmark_fallbacks if item["intent_only"]) / len(benchmark)
        )
        if benchmark
        else 0.0,
        "benchmark_phase_only_fallback_rate": round_score(
            sum(1 for item in benchmark_fallbacks if item["phase_only"]) / len(benchmark)
        )
        if benchmark
        else 0.0,
        "suite_outputs": suite_outputs,
    }


def pick_recommended_threshold(results: list[dict]) -> dict:
    return min(
        results,
        key=lambda item: (
            item["combined"]["bad_allow_rate_on_safe_prompts"],
            item["head"]["ambiguous_bad_allow_rate"],
            item["combined"]["false_fallback_rate_on_obvious_prompts"],
            abs(item["combined"]["fallback_responsibility"]["intent_share_of_threshold_fallbacks"] - 0.5),
            item["combined"]["benchmark_fallback_rate"],
            item["threshold"],
        ),
    )


def apply_threshold(threshold: float) -> None:
    calibration_path = INTENT_HEAD_CONFIG.calibration_path
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    payload["confidence_threshold"] = round_score(threshold)
    payload["threshold_selection_mode"] = "manual_sweep"
    write_json(calibration_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep candidate intent_type thresholds and compare end-to-end behavior.")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Candidate thresholds to evaluate.",
    )
    parser.add_argument(
        "--apply-threshold",
        type=float,
        default=None,
        help="Optional threshold to write into the intent_type calibration artifact after evaluation.",
    )
    args = parser.parse_args()

    ensure_artifact_dirs()
    thresholds = [round_score(value) for value in args.thresholds]
    results = []
    for threshold in thresholds:
        results.append(
            {
                "threshold": threshold,
                "head": evaluate_intent_head_threshold(threshold),
                "combined": evaluate_combined_threshold(threshold),
            }
        )

    recommended = pick_recommended_threshold(results)
    output = {
        "thresholds": thresholds,
        "results": results,
        "recommended_threshold": recommended["threshold"],
    }
    write_json(OUTPUT_PATH, output)

    if args.apply_threshold is not None:
        apply_threshold(args.apply_threshold)
        output["applied_threshold"] = round_score(args.apply_threshold)
        write_json(OUTPUT_PATH, output)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
