from __future__ import annotations

from functools import lru_cache

from config import IAB_HIERARCHY_HEAD_CONFIGS, HeadConfig
from iab_taxonomy import get_iab_taxonomy, parse_path_label
from model_runtime import SequenceClassifierHead

MIN_CANDIDATE_MASS_BY_LEVEL = {
    1: 0.2,
    2: 0.16,
    3: 0.12,
    4: 0.1,
}


def _has_model_artifacts(config: HeadConfig) -> bool:
    return config.model_dir.exists() and (
        (config.model_dir / "model.safetensors").exists()
        or (config.model_dir / "pytorch_model.bin").exists()
    )


def format_iab_hierarchy_text(text: str, parent_path: tuple[str, ...]) -> str:
    if not parent_path:
        return text.strip()
    return f"[IAB_PARENT] {' > '.join(parent_path)} [QUERY] {text.strip()}"


class HierarchicalIabPredictor:
    def __init__(self):
        self.taxonomy = get_iab_taxonomy()
        self._heads = {level: SequenceClassifierHead(config) for level, config in IAB_HIERARCHY_HEAD_CONFIGS.items()}

    def available_levels(self) -> list[int]:
        return [level for level, config in IAB_HIERARCHY_HEAD_CONFIGS.items() if _has_model_artifacts(config)]

    def ready(self) -> bool:
        available = set(self.available_levels())
        return {1, 2}.issubset(available)

    def _predict_level(self, level: int, text: str, parent_path: tuple[str, ...]) -> dict | None:
        children = self.taxonomy.immediate_children(parent_path)
        if not children:
            return None

        head = self._heads[level]
        candidate_labels = [node.path_label for node in children]
        prediction = head.predict_candidates(
            format_iab_hierarchy_text(text, parent_path),
            candidate_labels,
            confidence_threshold=head.config.default_confidence_threshold,
        )
        if prediction["label"] is None:
            return None

        candidate_mass_threshold = MIN_CANDIDATE_MASS_BY_LEVEL.get(level, 0.1)
        path = parse_path_label(prediction["label"])
        return {
            **prediction,
            "level": level,
            "path": path,
            "parent_path": parent_path,
            "candidate_count": len(candidate_labels),
            "candidate_mass_threshold": candidate_mass_threshold,
            "meets_candidate_mass_threshold": prediction["candidate_mass"] >= candidate_mass_threshold,
        }

    def predict(self, text: str) -> dict | None:
        available_levels = self.available_levels()
        if not available_levels:
            return None

        parent_path: tuple[str, ...] = ()
        selected_path: tuple[str, ...] | None = None
        tier_predictions: list[dict] = []
        stopped_reason = "no_prediction"

        for level in sorted(available_levels):
            if level != len(parent_path) + 1:
                break

            prediction = self._predict_level(level, text, parent_path)
            if prediction is None:
                stopped_reason = "no_children"
                break

            tier_predictions.append(
                {
                    "level": prediction["level"],
                    "label": prediction["label"],
                    "path": list(prediction["path"]),
                    "confidence": prediction["confidence"],
                    "raw_confidence": prediction["raw_confidence"],
                    "candidate_mass": prediction["candidate_mass"],
                    "candidate_count": prediction["candidate_count"],
                    "confidence_threshold": prediction["confidence_threshold"],
                    "candidate_mass_threshold": prediction["candidate_mass_threshold"],
                    "meets_confidence_threshold": prediction["meets_confidence_threshold"],
                    "meets_candidate_mass_threshold": prediction["meets_candidate_mass_threshold"],
                }
            )
            if not prediction["meets_confidence_threshold"]:
                stopped_reason = "confidence_below_threshold"
                break
            if not prediction["meets_candidate_mass_threshold"]:
                stopped_reason = "candidate_mass_below_threshold"
                break

            selected_path = prediction["path"]
            parent_path = selected_path
            stopped_reason = "accepted"

        if selected_path is None:
            return None

        node_children = self.taxonomy.immediate_children(selected_path)
        mapping_mode = "exact" if not node_children else "nearest_equivalent"

        mapping_confidence = min(item["confidence"] for item in tier_predictions if item["path"][: len(selected_path)] == list(selected_path))
        content = self.taxonomy.build_content_object(
            path=selected_path,
            mapping_mode=mapping_mode,
            mapping_confidence=mapping_confidence,
        )
        return {
            "content": content,
            "path": selected_path,
            "mapping_mode": mapping_mode,
            "mapping_confidence": mapping_confidence,
            "tier_predictions": tier_predictions,
            "stopped_reason": stopped_reason,
            "available_levels": available_levels,
        }


@lru_cache(maxsize=1)
def get_iab_hierarchy_predictor() -> HierarchicalIabPredictor:
    return HierarchicalIabPredictor()


def predict_iab_content_hierarchical(text: str) -> dict | None:
    predictor = get_iab_hierarchy_predictor()
    if not predictor.ready():
        return None
    return predictor.predict(text)
