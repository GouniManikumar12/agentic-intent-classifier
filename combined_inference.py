import argparse
import json

from inference import predict as predict_intent_type
from inference_decision_phase import predict as predict_decision_phase

CONFIDENCE_MIN = 0.45
COMMERCIAL_SCORE_MIN = 0.6

INTENT_SCORE_WEIGHTS = {
    "informational": 0.15,
    "commercial": 0.75,
    "transactional": 0.95,
    "personal_reflection": 0.0,
    "ambiguous": 0.1,
}

PHASE_SCORE_WEIGHTS = {
    "awareness": 0.1,
    "research": 0.35,
    "consideration": 0.7,
    "decision": 0.85,
    "action": 1.0,
    "post_purchase": 0.15,
    "support": 0.0,
}

SAFE_FALLBACK_INTENTS = {"ambiguous", "support", "personal_reflection"}


def round_score(value: float) -> float:
    return round(float(value), 4)


def compute_commercial_score(intent_type: str, decision_phase: str) -> float:
    intent_weight = INTENT_SCORE_WEIGHTS.get(intent_type, 0.2)
    phase_weight = PHASE_SCORE_WEIGHTS.get(decision_phase, 0.2)
    return round_score((intent_weight * 0.6) + (phase_weight * 0.4))


def build_summary(intent_type: str, decision_phase: str) -> str:
    return f"Classified as {intent_type} intent in the {decision_phase} phase."


def build_fallback(intent_type: str, confidence: float) -> dict | None:
    if confidence >= CONFIDENCE_MIN and intent_type not in SAFE_FALLBACK_INTENTS:
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
    }


def build_policy(intent_type: str, confidence: float, commercial_score: float, fallback: dict | None) -> dict:
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
            "applied_thresholds": {
                "commercial_score_min": COMMERCIAL_SCORE_MIN,
                "confidence_min": CONFIDENCE_MIN,
            },
            "sensitivity": "high" if intent_type == "personal_reflection" else "medium",
            "regulated_vertical": False,
        }

    if intent_type == "transactional" and commercial_score >= COMMERCIAL_SCORE_MIN:
        return {
            "monetization_eligibility": "allowed",
            "eligibility_reason": "high_intent_transactional_query",
            "decision_basis": "score_threshold",
            "applied_thresholds": {
                "commercial_score_min": COMMERCIAL_SCORE_MIN,
                "confidence_min": CONFIDENCE_MIN,
            },
            "sensitivity": "low",
            "regulated_vertical": False,
        }

    if intent_type == "commercial" and commercial_score >= COMMERCIAL_SCORE_MIN:
        return {
            "monetization_eligibility": "allowed_with_caution",
            "eligibility_reason": "commercial_decision_signal_present",
            "decision_basis": "score_threshold",
            "applied_thresholds": {
                "commercial_score_min": COMMERCIAL_SCORE_MIN,
                "confidence_min": CONFIDENCE_MIN,
            },
            "sensitivity": "low",
            "regulated_vertical": False,
        }

    return {
        "monetization_eligibility": "restricted",
        "eligibility_reason": "commercial_signal_below_threshold",
        "decision_basis": "score_threshold",
        "applied_thresholds": {
            "commercial_score_min": COMMERCIAL_SCORE_MIN,
            "confidence_min": CONFIDENCE_MIN,
        },
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


def classify_query(text: str) -> dict:
    intent_pred = predict_intent_type(text)
    phase_pred = predict_decision_phase(text)

    intent_type = intent_pred["label"]
    decision_phase = phase_pred["label"]
    confidence = round_score(min(intent_pred["confidence"], phase_pred["confidence"]))
    commercial_score = compute_commercial_score(intent_type, decision_phase)
    fallback = build_fallback(intent_type, confidence)

    model_output = {
        "classification": {
            "intent": {
                "type": intent_type,
                "decision_phase": decision_phase,
                "confidence": confidence,
                "commercial_score": commercial_score,
                "summary": build_summary(intent_type, decision_phase),
            }
        }
    }
    if fallback is not None:
        model_output["fallback"] = fallback

    return {
        "model_output": model_output,
        "system_decision": {
            "policy": build_policy(intent_type, confidence, commercial_score, fallback),
            "opportunity": build_opportunity(intent_type, decision_phase, fallback),
            "intent_trajectory": [decision_phase],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run combined intent + decision_phase classification.")
    parser.add_argument("text", help="Raw query to classify")
    args = parser.parse_args()
    print(json.dumps(classify_query(args.text), indent=2))


if __name__ == "__main__":
    main()
