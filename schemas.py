from __future__ import annotations

from copy import deepcopy

from config import DECISION_PHASE_LABELS, INTENT_TYPE_LABELS, PROJECT_VERSION, SUBTYPE_LABELS

API_SCHEMA_VERSION = "2026-03-22"
ALLOWED_MONETIZATION_ELIGIBILITY = {
    "allowed",
    "allowed_with_caution",
    "restricted",
    "not_allowed",
}
ALLOWED_DECISION_BASIS = {
    "score_threshold",
    "fallback_low_confidence",
    "fallback_ambiguous_intent",
    "fallback_policy_default",
}
ALLOWED_SENSITIVITY = {"low", "medium", "high"}
ALLOWED_OPPORTUNITY_TYPES = {
    "none",
    "transaction_trigger",
    "decision_moment",
    "comparison_slot",
    "soft_recommendation",
}
ALLOWED_OPPORTUNITY_STRENGTHS = {"low", "medium", "high"}
ALLOWED_FALLBACK_REASONS = {"ambiguous_query", "policy_default", "confidence_below_threshold"}
ALLOWED_IAB_MAPPING_MODES = {"exact", "nearest_equivalent", "internal_extension"}


class SchemaValidationError(Exception):
    def __init__(self, code: str, details: list[dict]):
        super().__init__(code)
        self.code = code
        self.details = details


def _detail(field: str, message: str, error_type: str = "validation_error") -> dict:
    return {"field": field, "message": message, "type": error_type}


def _expect_dict(value, field: str, errors: list[dict]) -> dict | None:
    if not isinstance(value, dict):
        errors.append(_detail(field, "must be an object", "type_error"))
        return None
    return value


def _expect_list(value, field: str, errors: list[dict]) -> list | None:
    if not isinstance(value, list):
        errors.append(_detail(field, "must be an array", "type_error"))
        return None
    return value


def _expect_bool(value, field: str, errors: list[dict]) -> bool | None:
    if not isinstance(value, bool):
        errors.append(_detail(field, "must be a boolean", "type_error"))
        return None
    return value


def _expect_str(value, field: str, errors: list[dict], *, min_length: int = 0, max_length: int | None = None) -> str | None:
    if not isinstance(value, str):
        errors.append(_detail(field, "must be a string", "type_error"))
        return None
    cleaned = value.strip()
    if len(cleaned) < min_length:
        errors.append(_detail(field, f"must be at least {min_length} characters", "value_error"))
    if max_length is not None and len(cleaned) > max_length:
        errors.append(_detail(field, f"must be at most {max_length} characters", "value_error"))
    return cleaned


def _expect_float(value, field: str, errors: list[dict], *, minimum: float = 0.0, maximum: float = 1.0) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(_detail(field, "must be a number", "type_error"))
        return None
    coerced = float(value)
    if coerced < minimum or coerced > maximum:
        errors.append(_detail(field, f"must be between {minimum} and {maximum}", "value_error"))
    return coerced


def _expect_member(value, field: str, allowed: set[str] | tuple[str, ...], errors: list[dict]) -> str | None:
    member = _expect_str(value, field, errors, min_length=1)
    if member is not None and member not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        errors.append(_detail(field, f"must be one of: {allowed_values}", "value_error"))
    return member


def validate_classify_request(payload) -> dict:
    errors: list[dict] = []
    payload_dict = _expect_dict(payload, "body", errors)
    if payload_dict is None:
        raise SchemaValidationError("request_validation_failed", errors)

    extra_keys = sorted(set(payload_dict) - {"text"})
    if extra_keys:
        errors.append(_detail("body", f"unexpected fields: {', '.join(extra_keys)}", "value_error"))

    text = _expect_str(payload_dict.get("text"), "text", errors, min_length=1, max_length=5000)
    if errors:
        raise SchemaValidationError("request_validation_failed", errors)
    return {"text": text}


def _validate_head_confidence(payload, field: str, labels: tuple[str, ...], errors: list[dict]) -> None:
    data = _expect_dict(payload, field, errors)
    if data is None:
        return
    _expect_member(data.get("label"), f"{field}.label", labels, errors)
    _expect_float(data.get("confidence"), f"{field}.confidence", errors)
    _expect_float(data.get("raw_confidence"), f"{field}.raw_confidence", errors)
    _expect_float(data.get("confidence_threshold"), f"{field}.confidence_threshold", errors)
    _expect_bool(data.get("calibrated"), f"{field}.calibrated", errors)
    _expect_bool(data.get("meets_threshold"), f"{field}.meets_threshold", errors)


def _validate_iab_level(payload, field: str, errors: list[dict]) -> None:
    data = _expect_dict(payload, field, errors)
    if data is None:
        return
    _expect_str(data.get("id"), f"{field}.id", errors, min_length=1)
    _expect_str(data.get("label"), f"{field}.label", errors, min_length=1)


def _validate_iab_content(payload, field: str, errors: list[dict]) -> None:
    data = _expect_dict(payload, field, errors)
    if data is None:
        return
    taxonomy = _expect_str(data.get("taxonomy"), f"{field}.taxonomy", errors, min_length=1)
    if taxonomy is not None and taxonomy != "IAB Content Taxonomy":
        errors.append(_detail(f"{field}.taxonomy", "must equal 'IAB Content Taxonomy'", "value_error"))
    _expect_str(data.get("taxonomy_version"), f"{field}.taxonomy_version", errors, min_length=1)
    _validate_iab_level(data.get("tier1"), f"{field}.tier1", errors)
    tier2 = data.get("tier2")
    if tier2 is not None:
        _validate_iab_level(tier2, f"{field}.tier2", errors)
    tier3 = data.get("tier3")
    if tier3 is not None:
        _validate_iab_level(tier3, f"{field}.tier3", errors)
    tier4 = data.get("tier4")
    if tier4 is not None:
        _validate_iab_level(tier4, f"{field}.tier4", errors)
    _expect_member(data.get("mapping_mode"), f"{field}.mapping_mode", ALLOWED_IAB_MAPPING_MODES, errors)
    _expect_float(data.get("mapping_confidence"), f"{field}.mapping_confidence", errors)


def _validate_fallback(payload, field: str, errors: list[dict]) -> None:
    if payload is None:
        return
    data = _expect_dict(payload, field, errors)
    if data is None:
        return
    _expect_bool(data.get("applied"), f"{field}.applied", errors)
    _expect_member(data.get("fallback_intent_type"), f"{field}.fallback_intent_type", INTENT_TYPE_LABELS, errors)
    _expect_member(
        data.get("fallback_monetization_eligibility"),
        f"{field}.fallback_monetization_eligibility",
        {"not_allowed"},
        errors,
    )
    _expect_member(data.get("reason"), f"{field}.reason", ALLOWED_FALLBACK_REASONS, errors)
    failed_components = _expect_list(data.get("failed_components"), f"{field}.failed_components", errors)
    if failed_components is not None:
        for index, item in enumerate(failed_components):
            _expect_member(
                item,
                f"{field}.failed_components[{index}]",
                {"intent_type", "intent_subtype", "decision_phase"},
                errors,
            )


def _validate_policy(payload, field: str, errors: list[dict]) -> None:
    data = _expect_dict(payload, field, errors)
    if data is None:
        return
    _expect_member(data.get("monetization_eligibility"), f"{field}.monetization_eligibility", ALLOWED_MONETIZATION_ELIGIBILITY, errors)
    _expect_str(data.get("eligibility_reason"), f"{field}.eligibility_reason", errors, min_length=1)
    _expect_member(data.get("decision_basis"), f"{field}.decision_basis", ALLOWED_DECISION_BASIS, errors)
    thresholds = _expect_dict(data.get("applied_thresholds"), f"{field}.applied_thresholds", errors)
    if thresholds is not None:
        _expect_float(thresholds.get("commercial_score_min"), f"{field}.applied_thresholds.commercial_score_min", errors)
        _expect_float(thresholds.get("intent_type_confidence_min"), f"{field}.applied_thresholds.intent_type_confidence_min", errors)
        _expect_float(
            thresholds.get("intent_subtype_confidence_min"),
            f"{field}.applied_thresholds.intent_subtype_confidence_min",
            errors,
        )
        _expect_float(thresholds.get("decision_phase_confidence_min"), f"{field}.applied_thresholds.decision_phase_confidence_min", errors)
    _expect_member(data.get("sensitivity"), f"{field}.sensitivity", ALLOWED_SENSITIVITY, errors)
    _expect_bool(data.get("regulated_vertical"), f"{field}.regulated_vertical", errors)


def _validate_opportunity(payload, field: str, errors: list[dict]) -> None:
    data = _expect_dict(payload, field, errors)
    if data is None:
        return
    _expect_member(data.get("type"), f"{field}.type", ALLOWED_OPPORTUNITY_TYPES, errors)
    _expect_member(data.get("strength"), f"{field}.strength", ALLOWED_OPPORTUNITY_STRENGTHS, errors)


def validate_classify_response(payload) -> dict:
    errors: list[dict] = []
    response = _expect_dict(payload, "response", errors)
    if response is None:
        raise SchemaValidationError("response_validation_failed", errors)

    model_output = _expect_dict(response.get("model_output"), "model_output", errors)
    if model_output is not None:
        classification = _expect_dict(model_output.get("classification"), "model_output.classification", errors)
        if classification is not None:
            _validate_iab_content(
                classification.get("iab_content"),
                "model_output.classification.iab_content",
                errors,
            )
            intent = _expect_dict(classification.get("intent"), "model_output.classification.intent", errors)
            if intent is not None:
                _expect_member(intent.get("type"), "model_output.classification.intent.type", INTENT_TYPE_LABELS, errors)
                _expect_member(intent.get("subtype"), "model_output.classification.intent.subtype", SUBTYPE_LABELS, errors)
                _expect_member(
                    intent.get("decision_phase"),
                    "model_output.classification.intent.decision_phase",
                    DECISION_PHASE_LABELS,
                    errors,
                )
                _expect_float(intent.get("confidence"), "model_output.classification.intent.confidence", errors)
                _expect_float(intent.get("commercial_score"), "model_output.classification.intent.commercial_score", errors)
                _expect_str(intent.get("summary"), "model_output.classification.intent.summary", errors, min_length=1)
                component_confidence = _expect_dict(
                    intent.get("component_confidence"),
                    "model_output.classification.intent.component_confidence",
                    errors,
                )
                if component_confidence is not None:
                    _validate_head_confidence(
                        component_confidence.get("intent_type"),
                        "model_output.classification.intent.component_confidence.intent_type",
                        INTENT_TYPE_LABELS,
                        errors,
                    )
                    _validate_head_confidence(
                        component_confidence.get("intent_subtype"),
                        "model_output.classification.intent.component_confidence.intent_subtype",
                        SUBTYPE_LABELS,
                        errors,
                    )
                    _validate_head_confidence(
                        component_confidence.get("decision_phase"),
                        "model_output.classification.intent.component_confidence.decision_phase",
                        DECISION_PHASE_LABELS,
                        errors,
                    )
                    _expect_member(
                        component_confidence.get("overall_strategy"),
                        "model_output.classification.intent.component_confidence.overall_strategy",
                        {"min_required_component_confidence"},
                        errors,
                    )
        _validate_fallback(model_output.get("fallback"), "model_output.fallback", errors)

    system_decision = _expect_dict(response.get("system_decision"), "system_decision", errors)
    if system_decision is not None:
        _validate_policy(system_decision.get("policy"), "system_decision.policy", errors)
        _validate_opportunity(system_decision.get("opportunity"), "system_decision.opportunity", errors)
        intent_trajectory = _expect_list(system_decision.get("intent_trajectory"), "system_decision.intent_trajectory", errors)
        if intent_trajectory is not None:
            for index, item in enumerate(intent_trajectory):
                _expect_member(item, f"system_decision.intent_trajectory[{index}]", DECISION_PHASE_LABELS, errors)

    meta = _expect_dict(response.get("meta"), "meta", errors)
    if meta is not None:
        _expect_str(meta.get("system_version"), "meta.system_version", errors, min_length=1)
        _expect_bool(meta.get("calibration_enabled"), "meta.calibration_enabled", errors)

    if errors:
        raise SchemaValidationError("response_validation_failed", errors)
    return deepcopy(response)


def validate_health_response(payload) -> dict:
    errors: list[dict] = []
    response = _expect_dict(payload, "response", errors)
    if response is None:
        raise SchemaValidationError("response_validation_failed", errors)

    _expect_member(response.get("status"), "status", {"ok"}, errors)
    _expect_str(response.get("system_version"), "system_version", errors, min_length=1)
    heads = _expect_list(response.get("heads"), "heads", errors)
    if heads is not None:
        for index, item in enumerate(heads):
            head = _expect_dict(item, f"heads[{index}]", errors)
            if head is None:
                continue
            _expect_str(head.get("head"), f"heads[{index}].head", errors, min_length=1)
            _expect_str(head.get("model_path"), f"heads[{index}].model_path", errors, min_length=1)
            _expect_str(head.get("calibration_path"), f"heads[{index}].calibration_path", errors, min_length=1)
            _expect_bool(head.get("ready"), f"heads[{index}].ready", errors)
            _expect_bool(head.get("calibrated"), f"heads[{index}].calibrated", errors)

    if errors:
        raise SchemaValidationError("response_validation_failed", errors)
    return deepcopy(response)


def validate_version_response(payload) -> dict:
    errors: list[dict] = []
    response = _expect_dict(payload, "response", errors)
    if response is None:
        raise SchemaValidationError("response_validation_failed", errors)

    _expect_str(response.get("system_version"), "system_version", errors, min_length=1)
    _expect_member(response.get("api_schema_version"), "api_schema_version", {API_SCHEMA_VERSION}, errors)
    if errors:
        raise SchemaValidationError("response_validation_failed", errors)
    return deepcopy(response)


def default_version_payload() -> dict:
    return {"system_version": PROJECT_VERSION, "api_schema_version": API_SCHEMA_VERSION}
