import argparse
import json

from config import (
    COMMERCIAL_SCORE_MIN,
    INTENT_SCORE_WEIGHTS,
    PHASE_SCORE_WEIGHTS,
    PROJECT_VERSION,
    SAFE_FALLBACK_INTENTS,
)
from inference import predict as predict_intent_type
from inference_decision_phase import predict as predict_decision_phase
from schemas import validate_classify_response


def round_score(value: float) -> float:
    return round(float(value), 4)


def compute_commercial_score(intent_type: str, decision_phase: str) -> float:
    intent_weight = INTENT_SCORE_WEIGHTS.get(intent_type, 0.2)
    phase_weight = PHASE_SCORE_WEIGHTS.get(decision_phase, 0.2)
    return round_score((intent_weight * 0.6) + (phase_weight * 0.4))


def build_summary(intent_type: str, decision_phase: str) -> str:
    return f"Classified as {intent_type} intent in the {decision_phase} phase."


def build_fallback(intent_pred: dict, phase_pred: dict) -> dict | None:
    intent_type = intent_pred["label"]
    failed_components = []
    if not intent_pred["meets_confidence_threshold"]:
        failed_components.append("intent_type")
    if not phase_pred["meets_confidence_threshold"]:
        failed_components.append("decision_phase")

    if not failed_components and intent_type not in SAFE_FALLBACK_INTENTS:
        return None

    if intent_type == "ambiguous":
        reason = "ambiguous_query"
        fallback_intent_type = "ambiguous"
        eligibility = "not_allowed"
    elif intent_type in {"support", "personal_reflection"}:
        reason = "policy_default"
        fallback_intent_type = intent_type
        eligibility = "not_allowed"
    else:
        reason = "confidence_below_threshold"
        fallback_intent_type = "ambiguous"
        eligibility = "not_allowed"

    return {
        "applied": True,
        "fallback_intent_type": fallback_intent_type,
        "fallback_monetization_eligibility": eligibility,
        "reason": reason,
        "failed_components": failed_components,
    }


def build_policy(intent_type: str, commercial_score: float, fallback: dict | None, intent_pred: dict, phase_pred: dict) -> dict:
    applied_thresholds = {
        "commercial_score_min": COMMERCIAL_SCORE_MIN,
        "intent_type_confidence_min": intent_pred["confidence_threshold"],
        "decision_phase_confidence_min": phase_pred["confidence_threshold"],
    }
    if fallback is not None:
        decision_basis = (
            "fallback_ambiguous_intent"
            if fallback["reason"] == "ambiguous_query"
            else "fallback_low_confidence"
        )
        return {
            "monetization_eligibility": fallback["fallback_monetization_eligibility"],
            "eligibility_reason": fallback["reason"],
            "decision_basis": decision_basis,
            "applied_thresholds": applied_thresholds,
            "sensitivity": "high" if intent_type == "personal_reflection" else "medium",
            "regulated_vertical": False,
        }

    if intent_type == "transactional" and commercial_score >= COMMERCIAL_SCORE_MIN:
        return {
            "monetization_eligibility": "allowed",
            "eligibility_reason": "high_intent_transactional_query",
            "decision_basis": "score_threshold",
            "applied_thresholds": applied_thresholds,
            "sensitivity": "low",
            "regulated_vertical": False,
        }

    if intent_type == "commercial" and commercial_score >= COMMERCIAL_SCORE_MIN:
        return {
            "monetization_eligibility": "allowed_with_caution",
            "eligibility_reason": "commercial_decision_signal_present",
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


def build_opportunity(intent_type: str, decision_phase: str, fallback: dict | None) -> dict:
    if fallback is not None:
        return {"type": "none", "strength": "low"}

    if intent_type == "transactional" and decision_phase == "action":
        return {"type": "transaction_trigger", "strength": "high"}
    if intent_type == "commercial" and decision_phase == "decision":
        return {"type": "decision_moment", "strength": "high"}
    if intent_type == "commercial" and decision_phase == "consideration":
        return {"type": "comparison_slot", "strength": "medium"}
    if decision_phase in {"research", "awareness"}:
        return {"type": "soft_recommendation", "strength": "low"}
    return {"type": "none", "strength": "low"}


def classify_query(text: str, threshold_overrides: dict[str, float] | None = None) -> dict:
    threshold_overrides = threshold_overrides or {}
    intent_pred = predict_intent_type(text, confidence_threshold=threshold_overrides.get("intent_type"))
    phase_pred = predict_decision_phase(text, confidence_threshold=threshold_overrides.get("decision_phase"))

    intent_type = intent_pred["label"]
    decision_phase = phase_pred["label"]
    confidence = round_score(min(intent_pred["confidence"], phase_pred["confidence"]))
    commercial_score = compute_commercial_score(intent_type, decision_phase)
    fallback = build_fallback(intent_pred, phase_pred)

    payload = {
        "model_output": {
            "classification": {
                "intent": {
                    "type": intent_type,
                    "decision_phase": decision_phase,
                    "confidence": confidence,
                    "commercial_score": commercial_score,
                    "summary": build_summary(intent_type, decision_phase),
                    "component_confidence": {
                        "intent_type": {
                            "label": intent_pred["label"],
                            "confidence": intent_pred["confidence"],
                            "raw_confidence": intent_pred["raw_confidence"],
                            "confidence_threshold": intent_pred["confidence_threshold"],
                            "calibrated": intent_pred["calibrated"],
                            "meets_threshold": intent_pred["meets_confidence_threshold"],
                        },
                        "decision_phase": {
                            "label": phase_pred["label"],
                            "confidence": phase_pred["confidence"],
                            "raw_confidence": phase_pred["raw_confidence"],
                            "confidence_threshold": phase_pred["confidence_threshold"],
                            "calibrated": phase_pred["calibrated"],
                            "meets_threshold": phase_pred["meets_confidence_threshold"],
                        },
                        "overall_strategy": "min_calibrated_component_confidence",
                    },
                }
            },
            "fallback": fallback,
        },
        "system_decision": {
            "policy": build_policy(intent_type, commercial_score, fallback, intent_pred, phase_pred),
            "opportunity": build_opportunity(intent_type, decision_phase, fallback),
            "intent_trajectory": [decision_phase],
        },
        "meta": {
            "system_version": PROJECT_VERSION,
            "calibration_enabled": bool(intent_pred["calibrated"] or phase_pred["calibrated"]),
        },
    }
    return validate_classify_response(payload)


def main():
    parser = argparse.ArgumentParser(description="Run combined intent + decision_phase classification.")
    parser.add_argument("text", help="Raw query to classify")
    args = parser.parse_args()
    print(json.dumps(classify_query(args.text), indent=2))


if __name__ == "__main__":
    main()
