from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PROJECT_VERSION = "0.6.0-phase4"

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CALIBRATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "calibration"
EVALUATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "evaluation"

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8008
DEFAULT_BENCHMARK_PATH = BASE_DIR / "examples" / "demo_prompt_suite.json"
KNOWN_FAILURE_CASES_PATH = BASE_DIR / "examples" / "known_failure_cases.json"
IAB_TAXONOMY_VERSION = "3.0"
IAB_TAXONOMY_PATH = BASE_DIR / "data" / "iab-content" / "Content Taxonomy 3.0.tsv"
IAB_MAPPING_CASES_PATH = BASE_DIR / "examples" / "iab_mapping_cases.json"

INTENT_TYPE_LABELS = (
    "informational",
    "exploratory",
    "commercial",
    "transactional",
    "support",
    "personal_reflection",
    "creative_generation",
    "chit_chat",
    "ambiguous",
    "prohibited",
)

DECISION_PHASE_LABELS = (
    "awareness",
    "research",
    "consideration",
    "decision",
    "action",
    "post_purchase",
    "support",
)

SUBTYPE_LABELS = (
    "education",
    "product_discovery",
    "comparison",
    "evaluation",
    "deal_seeking",
    "provider_selection",
    "signup",
    "purchase",
    "booking",
    "download",
    "contact_sales",
    "task_execution",
    "onboarding_setup",
    "troubleshooting",
    "account_help",
    "billing_help",
    "follow_up",
    "emotional_reflection",
)

def _load_iab_taxonomy_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))

    header = rows[1]
    data_rows = rows[2:]
    parsed_rows = []
    for row in data_rows:
        padded = row + [""] * (len(header) - len(row))
        parsed_rows.append(dict(zip(header, padded)))
    return parsed_rows


def _build_iab_labels(path: Path) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in _load_iab_taxonomy_rows(path):
        parts = [
            row.get(key, "").strip()
            for key in ("Tier 1", "Tier 2", "Tier 3", "Tier 4")
            if row.get(key, "").strip()
        ]
        if not parts:
            continue
        label = " > ".join(parts)
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return tuple(labels)


IAB_LABELS = _build_iab_labels(IAB_TAXONOMY_PATH)


def build_label_maps(labels: tuple[str, ...]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


@dataclass(frozen=True)
class HeadConfig:
    slug: str
    task_name: str
    model_name: str
    model_dir: Path
    data_dir: Path
    label_field: str
    labels: tuple[str, ...]
    max_length: int
    default_confidence_threshold: float
    target_accept_precision: float
    min_calibrated_confidence_threshold: float
    stress_suite_paths: dict[str, Path]

    @property
    def label2id(self) -> dict[str, int]:
        return build_label_maps(self.labels)[0]

    @property
    def id2label(self) -> dict[int, str]:
        return build_label_maps(self.labels)[1]

    @property
    def calibration_path(self) -> Path:
        return CALIBRATION_ARTIFACTS_DIR / f"{self.slug}.json"

    @property
    def split_paths(self) -> dict[str, Path]:
        return {
            "train": self.data_dir / "train.jsonl",
            "val": self.data_dir / "val.jsonl",
            "test": self.data_dir / "test.jsonl",
        }


INTENT_HEAD_CONFIG = HeadConfig(
    slug="intent_type",
    task_name="intent.type",
    model_name="distilbert-base-uncased",
    model_dir=BASE_DIR / "model_output",
    data_dir=BASE_DIR / "data",
    label_field="intent_type",
    labels=INTENT_TYPE_LABELS,
    max_length=64,
    default_confidence_threshold=0.7,
    target_accept_precision=0.8,
    min_calibrated_confidence_threshold=0.4,
    stress_suite_paths={
        "hard_cases": BASE_DIR / "data" / "hard_cases.jsonl",
        "third_wave_cases": BASE_DIR / "data" / "third_wave_cases.jsonl",
    },
)

DECISION_PHASE_HEAD_CONFIG = HeadConfig(
    slug="decision_phase",
    task_name="intent.decision_phase",
    model_name="distilbert-base-uncased",
    model_dir=BASE_DIR / "decision_phase_model_output",
    data_dir=BASE_DIR / "data" / "decision_phase",
    label_field="decision_phase",
    labels=DECISION_PHASE_LABELS,
    max_length=64,
    default_confidence_threshold=0.5,
    target_accept_precision=0.75,
    min_calibrated_confidence_threshold=0.22,
    stress_suite_paths={
        "hard_cases": BASE_DIR / "data" / "decision_phase" / "hard_cases.jsonl",
        "final_wave_cases": BASE_DIR / "data" / "decision_phase" / "final_wave_cases.jsonl",
    },
)

SUBTYPE_HEAD_CONFIG = HeadConfig(
    slug="intent_subtype",
    task_name="intent.subtype",
    model_name="distilbert-base-uncased",
    model_dir=BASE_DIR / "subtype_model_output",
    data_dir=BASE_DIR / "data" / "subtype",
    label_field="intent_subtype",
    labels=SUBTYPE_LABELS,
    max_length=72,
    default_confidence_threshold=0.45,
    target_accept_precision=0.75,
    min_calibrated_confidence_threshold=0.25,
    stress_suite_paths={
        "hard_cases": BASE_DIR / "data" / "subtype" / "hard_cases.jsonl",
        "extended_cases": BASE_DIR / "data" / "subtype" / "extended_cases.jsonl",
    },
)

IAB_HEAD_CONFIG = HeadConfig(
    slug="iab_content",
    task_name="content.iab",
    model_name="distilbert-base-uncased",
    model_dir=BASE_DIR / "iab_model_output",
    data_dir=BASE_DIR / "data" / "iab",
    label_field="iab_path",
    labels=IAB_LABELS,
    max_length=96,
    default_confidence_threshold=0.55,
    target_accept_precision=0.8,
    min_calibrated_confidence_threshold=0.7,
    stress_suite_paths={
        "hard_cases": BASE_DIR / "data" / "iab" / "hard_cases.jsonl",
        "extended_cases": BASE_DIR / "data" / "iab" / "extended_cases.jsonl",
    },
)

HEAD_CONFIGS = {
    INTENT_HEAD_CONFIG.slug: INTENT_HEAD_CONFIG,
    SUBTYPE_HEAD_CONFIG.slug: SUBTYPE_HEAD_CONFIG,
    DECISION_PHASE_HEAD_CONFIG.slug: DECISION_PHASE_HEAD_CONFIG,
    IAB_HEAD_CONFIG.slug: IAB_HEAD_CONFIG,
}

COMMERCIAL_SCORE_MIN = 0.6
SAFE_FALLBACK_INTENTS = {"ambiguous", "support", "personal_reflection", "chit_chat", "prohibited"}

INTENT_SCORE_WEIGHTS = {
    "informational": 0.15,
    "exploratory": 0.35,
    "commercial": 0.75,
    "transactional": 0.95,
    "support": 0.0,
    "personal_reflection": 0.0,
    "creative_generation": 0.0,
    "chit_chat": 0.0,
    "ambiguous": 0.1,
    "prohibited": 0.0,
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

SUBTYPE_SCORE_WEIGHTS = {
    "education": 0.05,
    "product_discovery": 0.58,
    "comparison": 0.74,
    "evaluation": 0.68,
    "deal_seeking": 0.71,
    "provider_selection": 0.9,
    "signup": 0.92,
    "purchase": 1.0,
    "booking": 0.94,
    "download": 0.46,
    "contact_sales": 0.95,
    "task_execution": 0.22,
    "onboarding_setup": 0.18,
    "troubleshooting": 0.0,
    "account_help": 0.0,
    "billing_help": 0.0,
    "follow_up": 0.05,
    "emotional_reflection": 0.0,
}

SUBTYPE_FAMILY_MAP = {
    "education": "informational",
    "product_discovery": "commercial",
    "comparison": "commercial",
    "evaluation": "commercial",
    "deal_seeking": "commercial",
    "provider_selection": "commercial",
    "signup": "transactional",
    "purchase": "transactional",
    "booking": "transactional",
    "download": "transactional",
    "contact_sales": "transactional",
    "task_execution": "transactional",
    "onboarding_setup": "post_purchase",
    "troubleshooting": "support",
    "account_help": "support",
    "billing_help": "support",
    "follow_up": "ambiguous",
    "emotional_reflection": "reflection",
}

SAFE_FALLBACK_SUBTYPE_FAMILIES = {"support", "ambiguous", "reflection"}
HIGH_INTENT_SUBTYPES = {"provider_selection", "signup", "purchase", "booking", "contact_sales"}
CAUTIONARY_SUBTYPES = {"comparison", "evaluation", "deal_seeking", "download"}
LOW_SIGNAL_SUBTYPES = {"education", "follow_up", "onboarding_setup", "task_execution"}


def ensure_artifact_dirs() -> None:
    CALIBRATION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
