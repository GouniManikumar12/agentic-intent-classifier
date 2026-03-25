from __future__ import annotations

import os
from collections import Counter

from combined_inference import classify_query
from iab_classifier import predict_iab_content_classifier_batch
from iab_retrieval import predict_iab_content_retrieval_batch
from iab_taxonomy import parse_path_label


def _include_shadow_retrieval_in_iab_views() -> bool:
    """Shadow retrieval loads Alibaba-NLP/gte-Qwen2-1.5B (~7GB) when the taxonomy index exists."""
    value = os.environ.get("IAB_EVAL_INCLUDE_SHADOW_RETRIEVAL", "0").strip().lower()
    return value in ("1", "true", "yes")


def path_from_content(content: dict) -> tuple[str, ...]:
    path = []
    for tier in ("tier1", "tier2", "tier3", "tier4"):
        if tier in content:
            path.append(content[tier]["label"])
    return tuple(path)


def path_from_label(label: str) -> tuple[str, ...]:
    return parse_path_label(label)


def is_parent_safe(true_path: tuple[str, ...], pred_path: tuple[str, ...]) -> bool:
    if not pred_path:
        return False
    if len(pred_path) > len(true_path):
        return False
    return true_path[: len(pred_path)] == pred_path


def error_bucket(true_path: tuple[str, ...], pred_path: tuple[str, ...]) -> str:
    if pred_path == true_path:
        return "exact_match"
    if not pred_path:
        return "no_prediction"
    if true_path[:1] != pred_path[:1]:
        return "wrong_tier1"
    if len(true_path) >= 2 and (len(pred_path) < 2 or true_path[:2] != pred_path[:2]):
        return "right_tier1_wrong_tier2"
    if is_parent_safe(true_path, pred_path):
        return "parent_safe_stop"
    return "wrong_deep_leaf"


def compute_path_metrics(true_paths: list[tuple[str, ...]], pred_paths: list[tuple[str, ...]]) -> dict:
    total = len(true_paths)
    if total == 0:
        return {
            "tier1_accuracy": 0.0,
            "tier2_accuracy": 0.0,
            "tier3_accuracy": 0.0,
            "tier4_accuracy": 0.0,
            "exact_path_accuracy": 0.0,
            "parent_safe_accuracy": 0.0,
            "average_prediction_depth": 0.0,
            "error_buckets": {},
        }

    tier_hits = {1: 0, 2: 0, 3: 0, 4: 0}
    tier_totals = {1: 0, 2: 0, 3: 0, 4: 0}
    exact_hits = 0
    parent_safe_hits = 0
    buckets = Counter()
    for true_path, pred_path in zip(true_paths, pred_paths):
        if pred_path == true_path:
            exact_hits += 1
        if is_parent_safe(true_path, pred_path):
            parent_safe_hits += 1
        buckets[error_bucket(true_path, pred_path)] += 1
        for level in range(1, 5):
            if len(true_path) < level:
                continue
            tier_totals[level] += 1
            if len(pred_path) >= level and true_path[:level] == pred_path[:level]:
                tier_hits[level] += 1

    return {
        "tier1_accuracy": round(tier_hits[1] / max(tier_totals[1], 1), 4),
        "tier2_accuracy": round(tier_hits[2] / max(tier_totals[2], 1), 4),
        "tier3_accuracy": round(tier_hits[3] / max(tier_totals[3], 1), 4),
        "tier4_accuracy": round(tier_hits[4] / max(tier_totals[4], 1), 4),
        "exact_path_accuracy": round(exact_hits / total, 4),
        "parent_safe_accuracy": round(parent_safe_hits / total, 4),
        "average_prediction_depth": round(sum(len(path) for path in pred_paths) / total, 4),
        "error_buckets": dict(sorted(buckets.items())),
    }


def evaluate_iab_views(rows: list[dict], max_combined_rows: int = 500) -> dict:
    texts = [row["text"] for row in rows]
    true_paths = [path_from_label(row["iab_path"]) for row in rows]
    classifier_outputs = predict_iab_content_classifier_batch(texts)
    if not any(output is not None for output in classifier_outputs):
        raise RuntimeError(
            "IAB classifier artifacts are unavailable. Run `python3 training/train_iab.py` "
            "and `python3 training/calibrate_confidence.py --head iab_content` "
            "from the `agentic-intent-classifier` directory first."
        )

    classifier_paths = [path_from_content(output["content"]) if output is not None else tuple() for output in classifier_outputs]
    views = {"classifier": compute_path_metrics(true_paths, classifier_paths)}

    if _include_shadow_retrieval_in_iab_views():
        retrieval_outputs = predict_iab_content_retrieval_batch(texts)
    else:
        retrieval_outputs = [None for _ in texts]
        views["shadow_embedding_retrieval"] = {
            "skipped": True,
            "reason": "disabled_by_default",
            "hint": "Set IAB_EVAL_INCLUDE_SHADOW_RETRIEVAL=1 to run shadow embedding retrieval (downloads/loads gte-Qwen2 when index is present).",
        }

    if any(output is not None for output in retrieval_outputs):
        retrieval_paths = [path_from_content(output["content"]) if output is not None else tuple() for output in retrieval_outputs]
        views["shadow_embedding_retrieval"] = compute_path_metrics(true_paths, retrieval_paths)

    if len(rows) > max_combined_rows:
        views["combined_path"] = {
            "skipped": True,
            "reason": "dataset_too_large_for_combined_view",
            "count": len(rows),
            "max_combined_rows": max_combined_rows,
        }
        views["disagreements"] = {
            "skipped": True,
            "reason": "dataset_too_large_for_combined_view",
            "count": len(rows),
            "max_combined_rows": max_combined_rows,
        }
        return views

    combined_payloads = [classify_query(text) for text in texts]
    combined_contents = [payload["model_output"]["classification"]["iab_content"] for payload in combined_payloads]
    combined_fallbacks = [bool(payload["model_output"].get("fallback")) for payload in combined_payloads]
    combined_paths = [path_from_content(content) for content in combined_contents]
    views["combined_path"] = {
        **compute_path_metrics(true_paths, combined_paths),
        "fallback_rate": round(sum(combined_fallbacks) / max(len(combined_fallbacks), 1), 4),
        "fallback_overuse_count": sum(combined_fallbacks),
    }
    disagreements = {
        "classifier_vs_combined": sum(1 for left, right in zip(classifier_paths, combined_paths) if left != right),
    }
    if any(output is not None for output in retrieval_outputs):
        disagreements["retrieval_vs_classifier"] = sum(
            1 for left, right in zip(retrieval_paths, classifier_paths) if left != right
        )
        disagreements["retrieval_vs_combined"] = sum(
            1 for left, right in zip(retrieval_paths, combined_paths) if left != right
        )
    views["disagreements"] = disagreements
    return views
