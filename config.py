from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_VERSION = "0.6.0-phase4"

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CALIBRATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "calibration"
EVALUATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "evaluation"
IAB_ARTIFACTS_DIR = ARTIFACTS_DIR / "iab"
FULL_INTENT_TAXONOMY_DATA_DIR = BASE_DIR / "data" / "full_intent_taxonomy"
INTENT_TYPE_DIFFICULTY_DATA_DIR = BASE_DIR / "data" / "intent_type_difficulty"
INTENT_TYPE_BENCHMARK_PATH = BASE_DIR / "data" / "intent_type_benchmark.jsonl"
DECISION_PHASE_DIFFICULTY_DATA_DIR = BASE_DIR / "data" / "decision_phase_difficulty"
DECISION_PHASE_BENCHMARK_PATH = BASE_DIR / "data" / "decision_phase_benchmark.jsonl"
SUBTYPE_DIFFICULTY_DATA_DIR = BASE_DIR / "data" / "subtype_difficulty"
SUBTYPE_BENCHMARK_PATH = BASE_DIR / "data" / "subtype_benchmark.jsonl"
IAB_DIFFICULTY_DATA_DIR = BASE_DIR / "data" / "iab_difficulty"
IAB_BENCHMARK_PATH = BASE_DIR / "data" / "iab_benchmark.jsonl"
IAB_CROSS_VERTICAL_BENCHMARK_PATH = BASE_DIR / "data" / "iab_cross_vertical_benchmark.jsonl"

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8008
DEFAULT_BENCHMARK_PATH = BASE_DIR / "examples" / "demo_prompt_suite.json"
KNOWN_FAILURE_CASES_PATH = BASE_DIR / "examples" / "known_failure_cases.json"
IAB_TAXONOMY_VERSION = os.environ.get("IAB_TAXONOMY_VERSION_OVERRIDE", "3.0")
_DEFAULT_IAB_TAXONOMY_PATH = Path(BASE_DIR / "data" / "iab-content" / f"Content Taxonomy {IAB_TAXONOMY_VERSION}.tsv")


def _resolve_iab_taxonomy_path() -> Path:
    # 1) Explicit override always wins.
    override = os.environ.get("IAB_TAXONOMY_PATH_OVERRIDE", "").strip()
    if override:
        return Path(override)

    # 2) Local repo file (normal local dev/training path).
    if _DEFAULT_IAB_TAXONOMY_PATH.exists():
        return _DEFAULT_IAB_TAXONOMY_PATH

    # 3) HF trust_remote_code fallback: dynamic module cache may not include data files.
    repo_id = os.environ.get("ADMESH_MODEL_REPO_ID", "admesh/agentic-intent-classifier").strip() or "admesh/agentic-intent-classifier"
    revision = os.environ.get("ADMESH_MODEL_REVISION", "").strip() or None
    filename = f"data/iab-content/Content Taxonomy {IAB_TAXONOMY_VERSION}.tsv"
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            revision=revision,
        )
        return Path(downloaded)
    except Exception:
        # Keep previous behavior: downstream code will raise clear file-not-found
        # if neither local nor hub fallback is available.
        return _DEFAULT_IAB_TAXONOMY_PATH


IAB_TAXONOMY_PATH = _resolve_iab_taxonomy_path()
IAB_TAXONOMY_GRAPH_PATH = IAB_ARTIFACTS_DIR / "taxonomy_graph.json"
IAB_TAXONOMY_NODES_PATH = IAB_ARTIFACTS_DIR / "taxonomy_nodes.json"
IAB_TAXONOMY_EMBEDDINGS_PATH = IAB_ARTIFACTS_DIR / "taxonomy_embeddings.pt"
IAB_DATASET_SUMMARY_PATH = IAB_ARTIFACTS_DIR / "dataset_summary.json"
MULTITASK_INTENT_MODEL_DIR = BASE_DIR / "multitask_intent_model_output"
IAB_CLASSIFIER_MODEL_DIR = BASE_DIR / "iab_classifier_model_output"
IAB_RETRIEVAL_LOCAL_MODEL_DIR = BASE_DIR / "iab_embedding_model_output"
IAB_QUALITY_TARGET_CASES_PATH = BASE_DIR / "examples" / "iab_mapping_cases.json"
IAB_CROSS_VERTICAL_QUALITY_TARGET_CASES_PATH = BASE_DIR / "examples" / "iab_cross_vertical_mapping_cases.json"
IAB_BEHAVIOR_LOCK_CASES_PATH = BASE_DIR / "examples" / "iab_behavior_lock_cases.json"
IAB_CROSS_VERTICAL_BEHAVIOR_LOCK_CASES_PATH = BASE_DIR / "examples" / "iab_cross_vertical_behavior_lock_cases.json"
IAB_RETRIEVAL_SPLIT_PATHS = {
    "train": BASE_DIR / "data" / "iab" / "train.jsonl",
    "val": BASE_DIR / "data" / "iab" / "val.jsonl",
    "test": BASE_DIR / "data" / "iab" / "test.jsonl",
}
IAB_RETRIEVAL_STRESS_SUITE_PATHS = {
    "hard_cases": BASE_DIR / "data" / "iab" / "hard_cases.jsonl",
    "extended_cases": BASE_DIR / "data" / "iab" / "extended_cases.jsonl",
    "difficulty_benchmark": IAB_BENCHMARK_PATH,
    "cross_vertical_benchmark": IAB_CROSS_VERTICAL_BENCHMARK_PATH,
}
IAB_RETRIEVAL_FALLBACK_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
IAB_RETRIEVAL_MODEL_MAX_LENGTH = 2048
IAB_RETRIEVAL_TOP_K = 16
IAB_RETRIEVAL_DEPTH_BONUS = 0.01
IAB_RETRIEVAL_PREFIX_CONFIDENCE_THRESHOLDS = {
    1: 0.5,
    2: 0.54,
    3: 0.58,
    4: 0.62,
}
IAB_PARENT_FALLBACK_CONFIDENCE_FLOOR = 0.3

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

def build_label_maps(labels: tuple[str, ...]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _looks_like_local_hf_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
    )


def _load_iab_path_labels(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)

    header = rows[1]
    labels: list[str] = []
    for row in rows[2:]:
        padded = row + [""] * (len(header) - len(row))
        item = dict(zip(header, padded))
        path_parts = [
            item.get(key, "").strip()
            for key in ("Tier 1", "Tier 2", "Tier 3", "Tier 4")
            if item.get(key, "").strip()
        ]
        if path_parts:
            labels.append(" > ".join(path_parts))
    return tuple(labels)


IAB_PATH_LABELS = _load_iab_path_labels(IAB_TAXONOMY_PATH)


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
        "difficulty_benchmark": INTENT_TYPE_BENCHMARK_PATH,
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
        "difficulty_benchmark": DECISION_PHASE_BENCHMARK_PATH,
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
        "difficulty_benchmark": SUBTYPE_BENCHMARK_PATH,
    },
)

IAB_HEAD_CONFIG = HeadConfig(
    slug="iab_content",
    task_name="iab.content",
    model_name="distilbert-base-uncased",
    model_dir=IAB_CLASSIFIER_MODEL_DIR,
    data_dir=BASE_DIR / "data" / "iab",
    label_field="iab_path",
    labels=IAB_PATH_LABELS,
    max_length=96,
    default_confidence_threshold=0.2,
    target_accept_precision=0.7,
    min_calibrated_confidence_threshold=0.12,
    stress_suite_paths=IAB_RETRIEVAL_STRESS_SUITE_PATHS,
)

IAB_RETRIEVAL_MODEL_NAME = os.environ.get(
    "IAB_RETRIEVAL_MODEL_NAME_OVERRIDE",
    str(IAB_RETRIEVAL_LOCAL_MODEL_DIR)
    if _looks_like_local_hf_model_dir(IAB_RETRIEVAL_LOCAL_MODEL_DIR)
    else IAB_RETRIEVAL_FALLBACK_MODEL_NAME,
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

INTENT_TYPE_TRAINING_WEIGHTS = {
    "informational": 1.0,
    "exploratory": 1.0,
    "commercial": 1.7,
    "transactional": 1.9,
    "support": 1.6,
    "personal_reflection": 0.85,
    "creative_generation": 0.75,
    "chit_chat": 0.7,
    "ambiguous": 1.1,
    "prohibited": 2.2,
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

DECISION_PHASE_TRAINING_WEIGHTS = {
    "awareness": 0.9,
    "research": 1.0,
    "consideration": 1.35,
    "decision": 1.8,
    "action": 1.55,
    "post_purchase": 1.15,
    "support": 1.5,
}

SUBTYPE_TRAINING_WEIGHTS = {
    "education": 0.95,
    "product_discovery": 1.55,
    "comparison": 1.65,
    "evaluation": 1.1,
    "deal_seeking": 1.7,
    "provider_selection": 1.75,
    "signup": 1.6,
    "purchase": 1.9,
    "booking": 1.45,
    "download": 1.1,
    "contact_sales": 1.55,
    "task_execution": 1.0,
    "onboarding_setup": 1.05,
    "troubleshooting": 1.4,
    "account_help": 1.55,
    "billing_help": 1.6,
    "follow_up": 0.9,
    "emotional_reflection": 0.85,
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
    IAB_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
