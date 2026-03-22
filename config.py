from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_VERSION = "0.2.0-phase1"

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CALIBRATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "calibration"
EVALUATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "evaluation"

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8008
DEFAULT_BENCHMARK_PATH = BASE_DIR / "examples" / "demo_prompt_suite.json"

INTENT_TYPE_LABELS = (
    "informational",
    "commercial",
    "transactional",
    "personal_reflection",
    "ambiguous",
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
    default_confidence_threshold=0.55,
    target_accept_precision=0.8,
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
)

HEAD_CONFIGS = {
    INTENT_HEAD_CONFIG.slug: INTENT_HEAD_CONFIG,
    DECISION_PHASE_HEAD_CONFIG.slug: DECISION_PHASE_HEAD_CONFIG,
}

COMMERCIAL_SCORE_MIN = 0.6
SAFE_FALLBACK_INTENTS = {"ambiguous", "support", "personal_reflection"}

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

INTENT_STRESS_SUITE_PATHS = {
    "hard_cases": BASE_DIR / "data" / "hard_cases.jsonl",
    "third_wave_cases": BASE_DIR / "data" / "third_wave_cases.jsonl",
}

DECISION_PHASE_STRESS_SUITE_PATHS = {
    "hard_cases": BASE_DIR / "data" / "decision_phase" / "hard_cases.jsonl",
    "final_wave_cases": BASE_DIR / "data" / "decision_phase" / "final_wave_cases.jsonl",
}


def ensure_artifact_dirs() -> None:
    CALIBRATION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
