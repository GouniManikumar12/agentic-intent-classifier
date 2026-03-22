from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import HEAD_CONFIGS, ensure_artifact_dirs
from model_runtime import get_head
from training.common import load_labeled_rows, write_json


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = predictions == labels
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(accuracies[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += abs(bin_accuracy - bin_confidence) * float(np.mean(mask))
    return ece


def optimize_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_tensor / temperature.clamp(min=1e-3), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return max(float(temperature.detach().item()), 1e-3)


def select_threshold(confidences: np.ndarray, correct: np.ndarray, target_precision: float, step: float) -> dict:
    candidates = []
    threshold = 0.0
    while threshold <= 1.000001:
        accepted = confidences >= threshold
        coverage = float(np.mean(accepted))
        if coverage > 0:
            accepted_accuracy = float(np.mean(correct[accepted]))
            candidates.append(
                {
                    "threshold": round(float(threshold), 4),
                    "coverage": round(coverage, 4),
                    "accepted_accuracy": round(accepted_accuracy, 4),
                }
            )
        threshold += step

    eligible = [candidate for candidate in candidates if candidate["accepted_accuracy"] >= target_precision]
    if eligible:
        return max(eligible, key=lambda candidate: (candidate["coverage"], -candidate["threshold"]))
    return max(
        candidates,
        key=lambda candidate: (
            candidate["accepted_accuracy"] * candidate["coverage"],
            candidate["accepted_accuracy"],
            -candidate["threshold"],
        ),
    )


def summarize_threshold(confidences: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    accepted = confidences >= threshold
    coverage = float(np.mean(accepted))
    accepted_accuracy = float(np.mean(correct[accepted])) if coverage > 0 else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "coverage": round(coverage, 4),
        "accepted_accuracy": round(accepted_accuracy, 4),
    }


def collect_logits(head_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    head = get_head(head_name)
    config = head.config
    rows = load_labeled_rows(config.split_paths[split], config.label_field, config.label2id)
    texts = [row["text"] for row in rows]
    labels = np.array([row["label"] for row in rows], dtype=np.int64)

    inputs = head.tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config.max_length,
    )
    with torch.no_grad():
        logits = head.model(**inputs).logits.detach().cpu().numpy()
    return logits, labels


def calibrate_head(head_name: str, split: str, step: float) -> dict:
    head = get_head(head_name)
    logits, labels = collect_logits(head_name, split)
    raw_probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    raw_confidences = raw_probs.max(axis=1)
    raw_preds = raw_probs.argmax(axis=1)
    raw_correct = raw_preds == labels
    raw_nll = float(log_loss(labels, raw_probs, labels=list(range(len(head.config.labels)))))

    optimized_temperature = optimize_temperature(logits, labels)
    calibrated_probs = torch.softmax(torch.tensor(logits / optimized_temperature, dtype=torch.float32), dim=-1).numpy()
    calibrated_confidences = calibrated_probs.max(axis=1)
    calibrated_preds = calibrated_probs.argmax(axis=1)
    calibrated_correct = calibrated_preds == labels
    calibrated_nll = float(log_loss(labels, calibrated_probs, labels=list(range(len(head.config.labels)))))

    temperature = optimized_temperature
    used_temperature_scaling = calibrated_nll <= raw_nll
    if not used_temperature_scaling:
        temperature = 1.0
        calibrated_probs = raw_probs
        calibrated_confidences = raw_confidences
        calibrated_preds = raw_preds
        calibrated_correct = raw_correct
        calibrated_nll = raw_nll

    selected_threshold_summary = select_threshold(
        calibrated_confidences,
        calibrated_correct,
        target_precision=head.config.target_accept_precision,
        step=step,
    )
    applied_threshold = max(
        float(selected_threshold_summary["threshold"]),
        float(head.config.min_calibrated_confidence_threshold),
    )
    threshold_summary = summarize_threshold(calibrated_confidences, calibrated_correct, applied_threshold)

    payload = {
        "head": head_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calibrated": True,
        "temperature": round(float(temperature), 6),
        "temperature_scaling_applied": used_temperature_scaling,
        "optimized_temperature_candidate": round(float(optimized_temperature), 6),
        "confidence_threshold": threshold_summary["threshold"],
        "selection_target_precision": head.config.target_accept_precision,
        "selection_split": split,
        "minimum_threshold_floor": round(float(head.config.min_calibrated_confidence_threshold), 4),
        "metrics": {
            "raw_accuracy": round(float(accuracy_score(labels, raw_preds)), 4),
            "calibrated_accuracy": round(float(accuracy_score(labels, calibrated_preds)), 4),
            "raw_negative_log_likelihood": round(raw_nll, 4),
            "calibrated_negative_log_likelihood": round(calibrated_nll, 4),
            "raw_expected_calibration_error": round(float(expected_calibration_error(raw_probs, labels)), 4),
            "calibrated_expected_calibration_error": round(
                float(expected_calibration_error(calibrated_probs, labels)),
                4,
            ),
            "mean_raw_confidence": round(float(np.mean(raw_confidences)), 4),
            "mean_calibrated_confidence": round(float(np.mean(calibrated_confidences)), 4),
        },
        "selected_threshold_before_floor": selected_threshold_summary,
        "threshold_summary": threshold_summary,
    }
    write_json(head.config.calibration_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate confidence scores for classifier heads.")
    parser.add_argument(
        "--head",
        choices=["all", *HEAD_CONFIGS.keys()],
        default="all",
        help="Which head to calibrate.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which labeled split to use for calibration fitting.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Grid step for confidence threshold selection.",
    )
    args = parser.parse_args()

    ensure_artifact_dirs()
    head_names = list(HEAD_CONFIGS.keys()) if args.head == "all" else [args.head]
    summary = {
        head_name: calibrate_head(head_name, split=args.split, step=max(args.threshold_step, 0.001))
        for head_name in head_names
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
