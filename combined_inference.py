import argparse
import json
import os

from config import (
    CAUTIONARY_SUBTYPES,
    COMMERCIAL_SCORE_MIN,
    HIGH_INTENT_SUBTYPES,
    INTENT_SCORE_WEIGHTS,
    LOW_SIGNAL_SUBTYPES,
    PHASE_SCORE_WEIGHTS,
    PROJECT_VERSION,
    SAFE_FALLBACK_INTENTS,
    SAFE_FALLBACK_SUBTYPE_FAMILIES,
    SUBTYPE_FAMILY_MAP,
    SUBTYPE_SCORE_WEIGHTS,
)
from inference_intent_type import predict as predict_intent_type
from inference_decision_phase import predict as predict_decision_phase
from inference_iab_classifier import predict as predict_iab_content_classifier
from inference_subtype import predict as predict_intent_subtype
from schemas import validate_classify_response

# Degraded fallback only: production requires `training/train_iab.py` and
# `calibrate_confidence.py --head iab_content`. Used when weights are missing or forced via --skip-iab.
_SKIPPED_IAB_CONTENT: dict = {
    "taxonomy": "IAB Content Taxonomy",
    "taxonomy_version": "3.0",
    "tier1": {"id": "skip_placeholder", "label": "Technology & computing"},
    "mapping_mode": "internal_extension",
    "mapping_confidence": 0.0,
}
_SKIPPED_IAB_PRED: dict = {"calibrated": False, "placeholder": True}


def _force_iab_placeholder(explicit: bool) -> bool:
    """Force placeholder IAB even when a trained classifier exists (tests / debugging)."""
    if explicit:
        return True
    return os.environ.get("SKIP_IAB_CLASSIFIER", "").strip().lower() in ("1", "true", "yes")


def round_score(value: float) -> float:
    return round(float(value), 4)


def iab_content_path(content: dict) -> tuple[str, ...]:
    path = []
    for tier in ("tier1", "tier2", "tier3", "tier4"):
        if tier in content:
            path.append(content[tier]["label"])
    return tuple(path)


def subtype_family(subtype: str) -> str:
    return SUBTYPE_FAMILY_MAP.get(subtype, "unknown")


def requires_subtype_threshold(subtype: str) -> bool:
    return subtype not in LOW_SIGNAL_SUBTYPES


def compute_commercial_score(intent_type: str, decision_phase: str, subtype: str) -> float:
    intent_weight = INTENT_SCORE_WEIGHTS.get(intent_type, 0.2)
    phase_weight = PHASE_SCORE_WEIGHTS.get(decision_phase, 0.2)
    subtype_weight = SUBTYPE_SCORE_WEIGHTS.get(subtype, 0.2)
    return round_score((intent_weight * 0.2) + (phase_weight * 0.35) + (subtype_weight * 0.45))


def build_summary(intent_type: str, decision_phase: str, subtype: str) -> str:
    return f"Classified as {intent_type} intent with subtype {subtype} in the {decision_phase} phase."


def build_overall_confidence(intent_pred: dict, subtype_pred: dict, phase_pred: dict) -> float:
    confidences = [intent_pred["confidence"], phase_pred["confidence"]]
    if requires_subtype_threshold(subtype_pred["label"]):
        confidences.append(subtype_pred["confidence"])
    return round_score(min(confidences))


def build_fallback(intent_pred: dict, subtype_pred: dict, phase_pred: dict) -> dict | None:
    intent_type = intent_pred["label"]
    subtype = subtype_pred["label"]
    subtype_group = subtype_family(subtype)
    failed_components = []
    if not intent_pred["meets_confidence_threshold"]:
        failed_components.append("intent_type")
    if requires_subtype_threshold(subtype) and not subtype_pred["meets_confidence_threshold"]:
        failed_components.append("intent_subtype")
    if not phase_pred["meets_confidence_threshold"]:
        failed_components.append("decision_phase")

    if intent_type == "ambiguous" or subtype == "follow_up":
        reason = "ambiguous_query"
        fallback_intent_type = "ambiguous"
        eligibility = "not_allowed"
    elif intent_type == "prohibited":
        reason = "policy_default"
        fallback_intent_type = "prohibited"
        eligibility = "not_allowed"
    elif intent_type in {"support", "chit_chat"}:
        reason = "policy_default"
        fallback_intent_type = intent_type
        eligibility = "not_allowed"
    elif intent_type == "personal_reflection" or subtype_group in SAFE_FALLBACK_SUBTYPE_FAMILIES:
        reason = "policy_default"
        fallback_intent_type = "personal_reflection" if subtype_group == "reflection" else "ambiguous"
        eligibility = "not_allowed"
    elif failed_components or intent_type in SAFE_FALLBACK_INTENTS:
        reason = "confidence_below_threshold"
        fallback_intent_type = "ambiguous"
        eligibility = "not_allowed"
    else:
        return None

    return {
        "applied": True,
        "fallback_intent_type": fallback_intent_type,
        "fallback_monetization_eligibility": eligibility,
        "reason": reason,
        "failed_components": failed_components,
    }


def build_policy(
    intent_type: str,
    decision_phase: str,
    subtype: str,
    commercial_score: float,
    iab_content: dict,
    fallback: dict | None,
    intent_pred: dict,
    subtype_pred: dict,
    phase_pred: dict,
) -> dict:
    subtype_group = subtype_family(subtype)
    applied_thresholds = {
        "commercial_score_min": COMMERCIAL_SCORE_MIN,
        "intent_type_confidence_min": intent_pred["confidence_threshold"],
        "intent_subtype_confidence_min": subtype_pred["confidence_threshold"],
        "decision_phase_confidence_min": phase_pred["confidence_threshold"],
    }
    if fallback is not None:
        if fallback["reason"] == "ambiguous_query":
            decision_basis = "fallback_ambiguous_intent"
        elif fallback["reason"] == "policy_default":
            decision_basis = "fallback_policy_default"
        else:
            decision_basis = "fallback_low_confidence"
        return {
            "monetization_eligibility": fallback["fallback_monetization_eligibility"],
            "eligibility_reason": fallback["reason"],
            "decision_basis": decision_basis,
            "applied_thresholds": applied_thresholds,
            "sensitivity": "high" if subtype_group in {"reflection", "support"} else "medium",
            "regulated_vertical": False,
        }

    if subtype in HIGH_INTENT_SUBTYPES and commercial_score >= 0.72:
        return {
            "monetization_eligibility": "allowed",
            "eligibility_reason": "high_intent_subtype_signal",
            "decision_basis": "score_threshold",
            "applied_thresholds": applied_thresholds,
            "sensitivity": "low",
            "regulated_vertical": False,
        }

    if intent_type == "commercial" and commercial_score >= COMMERCIAL_SCORE_MIN:
        reason = "commercial_decision_signal_present"
        if subtype == "product_discovery":
            reason = "commercial_discovery_signal_present"
        elif subtype in CAUTIONARY_SUBTYPES:
            reason = "commercial_comparison_signal_present"
        return {
            "monetization_eligibility": "allowed_with_caution",
            "eligibility_reason": reason,
            "decision_basis": "score_threshold",
            "applied_thresholds": applied_thresholds,
            "sensitivity": "medium" if subtype == "deal_seeking" else "low",
            "regulated_vertical": False,
        }

    if subtype in {"download"} and commercial_score >= 0.42:
        return {
            "monetization_eligibility": "allowed_with_caution",
            "eligibility_reason": "download_signal_present",
            "decision_basis": "score_threshold",
            "applied_thresholds": applied_thresholds,
            "sensitivity": "low",
            "regulated_vertical": False,
        }

    if subtype_group == "post_purchase":
        return {
            "monetization_eligibility": "restricted",
            "eligibility_reason": "post_purchase_setup_query",
            "decision_basis": "score_threshold",
            "applied_thresholds": applied_thresholds,
            "sensitivity": "low",
            "regulated_vertical": False,
        }

    if subtype == "task_execution":
        return {
            "monetization_eligibility": "restricted",
            "eligibility_reason": "operational_task_query",
            "decision_basis": "score_threshold",
            "applied_thresholds": applied_thresholds,
            "sensitivity": "low",
            "regulated_vertical": False,
        }

    return {
        "monetization_eligibility": "restricted",
        "eligibility_reason": "commercial_signal_below_threshold",
        "decision_basis": "score_threshold",
        "applied_thresholds": applied_thresholds,
        "sensitivity": "low",
        "regulated_vertical": False,
    }


def build_opportunity(subtype: str, fallback: dict | None) -> dict:
    if fallback is not None or subtype_family(subtype) in SAFE_FALLBACK_SUBTYPE_FAMILIES:
        return {"type": "none", "strength": "low"}

    if subtype in {"signup", "purchase", "booking", "contact_sales"}:
        return {"type": "transaction_trigger", "strength": "high"}
    if subtype == "provider_selection":
        return {"type": "decision_moment", "strength": "high"}
    if subtype in {"comparison", "evaluation"}:
        return {"type": "comparison_slot", "strength": "high" if subtype == "comparison" else "medium"}
    if subtype in {"product_discovery", "deal_seeking", "download", "onboarding_setup"}:
        return {"type": "soft_recommendation", "strength": "medium" if subtype != "onboarding_setup" else "low"}
    return {"type": "none", "strength": "low"}


def iab_path_labels(iab_content: dict) -> tuple[str | None, str | None, str | None, str | None]:
    return (
        iab_content.get("tier1", {}).get("label"),
        iab_content.get("tier2", {}).get("label"),
        iab_content.get("tier3", {}).get("label"),
        iab_content.get("tier4", {}).get("label"),
    )


def normalize_iab_label(label: str | None) -> str:
    return (label or "").strip().lower()


def is_buyable_iab_path(iab_content: dict) -> bool:
    tier1, tier2, tier3, tier4 = iab_path_labels(iab_content)
    labels = [normalize_iab_label(label) for label in (tier1, tier2, tier3, tier4) if label]
    if not labels:
        return False

    joined = " > ".join(labels)
    if any(
        term in joined
        for term in {
            "buying and selling",
            "shopping",
            "sales and promotions",
            "coupons and discounts",
            "laptops",
            "desktops",
            "smartphones",
            "tablets and e-readers",
            "cameras and camcorders",
            "wearable technology",
            "computer software and applications",
            "software and applications",
            "web hosting",
            "real estate renting and leasing",
            "hotels and motels",
            "air travel",
        }
    ):
        return True

    tier1_label = labels[0]
    tier2_label = labels[1] if len(labels) > 1 else ""
    return (
        tier1_label in {"automotive", "shopping", "real estate", "travel"}
        or (tier1_label == "technology & computing" and tier2_label in {"computing", "consumer electronics"})
    )


def should_override_low_confidence_fallback(
    fallback: dict | None,
    intent_pred: dict,
    subtype_pred: dict,
    phase_pred: dict,
    commercial_score: float,
    iab_content: dict,
) -> bool:
    if fallback is None or fallback.get("reason") != "confidence_below_threshold":
        return False
    failed_components = set(fallback.get("failed_components", []))
    if not failed_components or len(failed_components) > 2:
        return False
    if len(failed_components) == 2 and failed_components != {"intent_type", "decision_phase"}:
        return False
    if intent_pred["label"] != "commercial":
        return False
    if phase_pred["label"] not in {"consideration", "decision", "action"}:
        return False
    if subtype_family(subtype_pred["label"]) in SAFE_FALLBACK_SUBTYPE_FAMILIES:
        return False
    if subtype_pred["label"] not in {
        "product_discovery",
        "comparison",
        "evaluation",
        "deal_seeking",
        "provider_selection",
        "purchase",
        "booking",
        "contact_sales",
    }:
        return False
    if not is_buyable_iab_path(iab_content):
        return False
    mapping_confidence = iab_content.get("mapping_confidence", 0.0)
    subtype_threshold = subtype_pred["confidence_threshold"]
    subtype_confidence = subtype_pred["confidence"]

    if failed_components == {"intent_subtype"}:
        return (
            intent_pred["meets_confidence_threshold"]
            and phase_pred["meets_confidence_threshold"]
            and subtype_confidence >= max(0.2, subtype_threshold - 0.03)
            and commercial_score >= 0.78
            and mapping_confidence >= 0.8
        )

    if failed_components == {"intent_type", "decision_phase"}:
        return (
            subtype_pred["meets_confidence_threshold"]
            and commercial_score >= 0.72
            and mapping_confidence >= 0.72
        )

    return False


def build_iab_content(
    text: str,
    intent_type: str,
    subtype: str,
    decision_phase: str,
    confidence_threshold: float | None = None,
    *,
    force_placeholder: bool = False,
) -> tuple[dict, dict]:
    if force_placeholder:
        return _SKIPPED_IAB_CONTENT, _SKIPPED_IAB_PRED
    classifier_pred = predict_iab_content_classifier(text, confidence_threshold=confidence_threshold)
    if classifier_pred is None:
        # Missing IAB artifacts: valid JSON only; check meta.iab_mapping_is_placeholder. Train + calibrate IAB for production.
        return _SKIPPED_IAB_CONTENT, _SKIPPED_IAB_PRED
    return classifier_pred["content"], classifier_pred


def classify_query(
    text: str,
    threshold_overrides: dict[str, float] | None = None,
    *,
    force_iab_placeholder: bool = False,
) -> dict:
    threshold_overrides = threshold_overrides or {}
    force_iab_placeholder = _force_iab_placeholder(force_iab_placeholder)
    intent_pred = predict_intent_type(text, confidence_threshold=threshold_overrides.get("intent_type"))
    subtype_pred = predict_intent_subtype(text, confidence_threshold=threshold_overrides.get("intent_subtype"))
    phase_pred = predict_decision_phase(text, confidence_threshold=threshold_overrides.get("decision_phase"))

    intent_type = intent_pred["label"]
    subtype = subtype_pred["label"]
    decision_phase = phase_pred["label"]
    confidence = build_overall_confidence(intent_pred, subtype_pred, phase_pred)
    commercial_score = compute_commercial_score(intent_type, decision_phase, subtype)
    iab_content, iab_pred = build_iab_content(
        text,
        intent_type,
        subtype,
        decision_phase,
        confidence_threshold=threshold_overrides.get("iab_content"),
        force_placeholder=force_iab_placeholder,
    )
    fallback = build_fallback(intent_pred, subtype_pred, phase_pred)
    if should_override_low_confidence_fallback(
        fallback,
        intent_pred,
        subtype_pred,
        phase_pred,
        commercial_score,
        iab_content,
    ):
        fallback = None

    payload = {
        "model_output": {
            "classification": {
                "iab_content": iab_content,
                "intent": {
                    "type": intent_type,
                    "subtype": subtype,
                    "decision_phase": decision_phase,
                    "confidence": confidence,
                    "commercial_score": commercial_score,
                    "summary": build_summary(intent_type, decision_phase, subtype),
                    "component_confidence": {
                        "intent_type": {
                            "label": intent_pred["label"],
                            "confidence": intent_pred["confidence"],
                            "raw_confidence": intent_pred["raw_confidence"],
                            "confidence_threshold": intent_pred["confidence_threshold"],
                            "calibrated": intent_pred["calibrated"],
                            "meets_threshold": intent_pred["meets_confidence_threshold"],
                        },
                        "intent_subtype": {
                            "label": subtype_pred["label"],
                            "confidence": subtype_pred["confidence"],
                            "raw_confidence": subtype_pred["raw_confidence"],
                            "confidence_threshold": subtype_pred["confidence_threshold"],
                            "calibrated": subtype_pred["calibrated"],
                            "meets_threshold": subtype_pred["meets_confidence_threshold"],
                        },
                        "decision_phase": {
                            "label": phase_pred["label"],
                            "confidence": phase_pred["confidence"],
                            "raw_confidence": phase_pred["raw_confidence"],
                            "confidence_threshold": phase_pred["confidence_threshold"],
                            "calibrated": phase_pred["calibrated"],
                            "meets_threshold": phase_pred["meets_confidence_threshold"],
                        },
                        "overall_strategy": "min_required_component_confidence",
                    },
                }
            },
            "fallback": fallback,
        },
        "system_decision": {
            "policy": build_policy(
                intent_type,
                decision_phase,
                subtype,
                commercial_score,
                iab_content,
                fallback,
                intent_pred,
                subtype_pred,
                phase_pred,
            ),
            "opportunity": build_opportunity(subtype, fallback),
            "intent_trajectory": [decision_phase],
        },
        "meta": {
            "system_version": PROJECT_VERSION,
            "calibration_enabled": bool(
                intent_pred["calibrated"]
                or subtype_pred["calibrated"]
                or phase_pred["calibrated"]
                or (iab_pred is not None and iab_pred["calibrated"])
            ),
            "iab_mapping_is_placeholder": bool(iab_pred is not None and iab_pred.get("placeholder")),
        },
   }
    return validate_classify_response(payload)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run combined IAB + intent classification. Production requires trained+calibrated IAB "
            "under iab_classifier_model_output/; use meta.iab_mapping_is_placeholder to detect degraded mode."
        )
    )
    parser.add_argument("text", help="Raw query to classify")
    parser.add_argument(
        "--skip-iab",
        action="store_true",
        dest="force_iab_placeholder",
        help="Ignore the IAB classifier and return placeholder mapping (testing only).",
    )
    args = parser.parse_args()
    print(json.dumps(classify_query(args.text, force_iab_placeholder=args.force_iab_placeholder), indent=2))


if __name__ == "__main__":
    main()
