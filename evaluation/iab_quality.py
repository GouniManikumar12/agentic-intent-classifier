from __future__ import annotations

from collections import Counter

from combined_inference import classify_query
from iab_hierarchy import predict_iab_content_hierarchical
from iab_mapping import map_iab_content
from iab_taxonomy import parse_path_label
from model_runtime import get_head


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


def evaluate_iab_views(rows: list[dict]) -> dict:
    texts = [row["text"] for row in rows]
    true_paths = [path_from_label(row["iab_path"]) for row in rows]
    taxonomy = get_head("iab_content").config

    flat_head = get_head("iab_content")
    flat_predictions = flat_head.predict_batch(texts)
    flat_contents = [
        {
            "tier1": {"label": path_from_label(pred["label"])[0], "id": ""},
        }
        for pred in flat_predictions
    ]
    from iab_taxonomy import get_iab_taxonomy

    taxonomy_obj = get_iab_taxonomy()
    flat_contents = [
        taxonomy_obj.build_content_object_from_label(
            pred["label"],
            mapping_mode="exact",
            mapping_confidence=pred["confidence"],
        )
        for pred in flat_predictions
    ]

    intent_preds = get_head("intent_type").predict_batch(texts)
    subtype_preds = get_head("intent_subtype").predict_batch(texts)
    phase_preds = get_head("decision_phase").predict_batch(texts)
    mapper_contents = [
        map_iab_content(text, intent["label"], subtype["label"], phase["label"])
        for text, intent, subtype, phase in zip(texts, intent_preds, subtype_preds, phase_preds)
    ]
    combined_payloads = [classify_query(text) for text in texts]
    combined_contents = [payload["model_output"]["classification"]["iab_content"] for payload in combined_payloads]
    combined_fallbacks = [bool(payload["model_output"].get("fallback")) for payload in combined_payloads]

    hierarchical_outputs = [predict_iab_content_hierarchical(text) for text in texts]
    hierarchy_available = any(output is not None for output in hierarchical_outputs)
    hierarchical_contents = [output["content"] if output is not None else None for output in hierarchical_outputs]

    flat_paths = [path_from_content(content) for content in flat_contents]
    mapper_paths = [path_from_content(content) for content in mapper_contents]
    combined_paths = [path_from_content(content) for content in combined_contents]

    views = {
        "flat_head": {
            **compute_path_metrics(true_paths, flat_paths),
            "accepted_coverage": round(
                sum(pred["meets_confidence_threshold"] for pred in flat_predictions) / max(len(flat_predictions), 1),
                4,
            ),
        },
        "mapper_only": compute_path_metrics(true_paths, mapper_paths),
        "combined_path": {
            **compute_path_metrics(true_paths, combined_paths),
            "fallback_rate": round(sum(combined_fallbacks) / max(len(combined_fallbacks), 1), 4),
            "fallback_overuse_count": sum(combined_fallbacks),
        },
        "disagreements": {
            "flat_vs_mapper": sum(1 for left, right in zip(flat_paths, mapper_paths) if left != right),
            "flat_vs_combined": sum(1 for left, right in zip(flat_paths, combined_paths) if left != right),
            "mapper_vs_combined": sum(1 for left, right in zip(mapper_paths, combined_paths) if left != right),
        },
    }
    if hierarchy_available:
        hierarchy_paths = [path_from_content(content) if content is not None else tuple() for content in hierarchical_contents]
        views["hierarchical_model"] = compute_path_metrics(true_paths, hierarchy_paths)
        views["disagreements"]["hierarchical_vs_combined"] = sum(
            1 for left, right in zip(hierarchy_paths, combined_paths) if left != right
        )
    return views
