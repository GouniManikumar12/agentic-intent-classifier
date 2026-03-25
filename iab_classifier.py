from __future__ import annotations

from functools import lru_cache

import torch

from config import IAB_PARENT_FALLBACK_CONFIDENCE_FLOOR
from iab_taxonomy import get_iab_taxonomy, parse_path_label, path_to_label
from model_runtime import get_head


def round_score(value: float) -> float:
    return round(float(value), 4)


@lru_cache(maxsize=1)
def _prefix_label_ids() -> dict[tuple[str, ...], list[int]]:
    head = get_head("iab_content")
    prefix_map: dict[tuple[str, ...], list[int]] = {}
    for label, label_id in head.config.label2id.items():
        path = parse_path_label(label)
        for depth in range(1, len(path) + 1):
            prefix_map.setdefault(path[:depth], []).append(label_id)
    return prefix_map


def _effective_exact_threshold(confidence_threshold: float | None) -> float:
    head = get_head("iab_content")
    if confidence_threshold is None:
        return float(head.calibration.confidence_threshold)
    return min(max(float(confidence_threshold), 0.0), 1.0)


def _effective_parent_threshold(exact_threshold: float) -> float:
    return min(max(IAB_PARENT_FALLBACK_CONFIDENCE_FLOOR, exact_threshold), 1.0)


def _build_prediction(
    accepted_path: tuple[str, ...],
    *,
    exact_label: str,
    confidence: float,
    raw_confidence: float,
    exact_threshold: float,
    calibrated: bool,
    meets_confidence_threshold: bool,
    mapping_mode: str,
    stopped_reason: str,
) -> dict:
    taxonomy = get_iab_taxonomy()
    return {
        "label": path_to_label(accepted_path),
        "exact_label": exact_label,
        "path": list(accepted_path),
        "confidence": round_score(confidence),
        "raw_confidence": round_score(raw_confidence),
        "confidence_threshold": round_score(exact_threshold),
        "calibrated": calibrated,
        "meets_confidence_threshold": meets_confidence_threshold,
        "content": taxonomy.build_content_object(
            accepted_path,
            mapping_mode=mapping_mode,
            mapping_confidence=confidence,
        ),
        "mapping_mode": mapping_mode,
        "mapping_confidence": round_score(confidence),
        "source": "supervised_classifier",
        "stopped_reason": stopped_reason,
    }


def predict_iab_content_classifier_batch(
    texts: list[str],
    confidence_threshold: float | None = None,
) -> list[dict | None]:
    if not texts:
        return []

    head = get_head("iab_content")
    if not head.config.model_dir.exists():
        return [None for _ in texts]

    raw_probs, calibrated_probs = head.predict_probs_batch(texts)
    prefix_map = _prefix_label_ids()
    exact_threshold = _effective_exact_threshold(confidence_threshold)
    parent_threshold = _effective_parent_threshold(exact_threshold)
    predictions: list[dict | None] = []

    for raw_row, calibrated_row in zip(raw_probs, calibrated_probs):
        pred_id = int(torch.argmax(calibrated_row).item())
        exact_label = head.model.config.id2label[pred_id]
        exact_path = parse_path_label(exact_label)
        exact_confidence = float(calibrated_row[pred_id].item())
        exact_raw_confidence = float(raw_row[pred_id].item())

        if exact_confidence >= exact_threshold:
            predictions.append(
                _build_prediction(
                    exact_path,
                    exact_label=exact_label,
                    confidence=exact_confidence,
                    raw_confidence=exact_raw_confidence,
                    exact_threshold=exact_threshold,
                    calibrated=head.calibration.calibrated,
                    meets_confidence_threshold=True,
                    mapping_mode="exact",
                    stopped_reason="exact_threshold_met",
                )
            )
            continue

        accepted_path = exact_path[:1]
        accepted_confidence = float(calibrated_row[prefix_map[accepted_path]].sum().item())
        accepted_raw_confidence = float(raw_row[prefix_map[accepted_path]].sum().item())
        meets_confidence_threshold = False
        stopped_reason = "top_level_safe_fallback"

        for depth in range(len(exact_path) - 1, 0, -1):
            prefix = exact_path[:depth]
            prefix_ids = prefix_map[prefix]
            prefix_confidence = float(calibrated_row[prefix_ids].sum().item())
            prefix_raw_confidence = float(raw_row[prefix_ids].sum().item())
            if prefix_confidence >= parent_threshold:
                accepted_path = prefix
                accepted_confidence = prefix_confidence
                accepted_raw_confidence = prefix_raw_confidence
                meets_confidence_threshold = True
                stopped_reason = "parent_fallback_threshold_met"
                break

        predictions.append(
            _build_prediction(
                accepted_path,
                exact_label=exact_label,
                confidence=accepted_confidence,
                raw_confidence=accepted_raw_confidence,
                exact_threshold=exact_threshold,
                calibrated=head.calibration.calibrated,
                meets_confidence_threshold=meets_confidence_threshold,
                mapping_mode="nearest_equivalent",
                stopped_reason=stopped_reason,
            )
        )

    return predictions


def predict_iab_content_classifier(text: str, confidence_threshold: float | None = None) -> dict | None:
    return predict_iab_content_classifier_batch([text], confidence_threshold=confidence_threshold)[0]
