from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

try:
    from .config import (  # type: ignore
        CALIBRATION_ARTIFACTS_DIR,
        DECISION_PHASE_HEAD_CONFIG,
        INTENT_HEAD_CONFIG,
        MULTITASK_INTENT_MODEL_DIR,
        SUBTYPE_HEAD_CONFIG,
    )
    from .multitask_model import MultiTaskIntentModel, MultiTaskLabelSizes  # type: ignore
except ImportError:
    from config import (
        CALIBRATION_ARTIFACTS_DIR,
        DECISION_PHASE_HEAD_CONFIG,
        INTENT_HEAD_CONFIG,
        MULTITASK_INTENT_MODEL_DIR,
        SUBTYPE_HEAD_CONFIG,
    )
    from multitask_model import MultiTaskIntentModel, MultiTaskLabelSizes


def round_score(value: float) -> float:
    return round(float(value), 4)


TASK_TO_CONFIG = {
    "intent_type": INTENT_HEAD_CONFIG,
    "intent_subtype": SUBTYPE_HEAD_CONFIG,
    "decision_phase": DECISION_PHASE_HEAD_CONFIG,
}

TASK_TO_LOGIT_KEY = {
    "intent_type": "intent_type_logits",
    "intent_subtype": "intent_subtype_logits",
    "decision_phase": "decision_phase_logits",
}


@dataclass(frozen=True)
class CalibrationState:
    calibrated: bool
    temperature: float
    confidence_threshold: float


class MultiTaskRuntime:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._tokenizer = None
        self._model = None
        self._metadata = None
        self._predict_batch_size = 32

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            metadata_path = self.model_dir / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Missing multitask metadata at {metadata_path}. Run python3 training/train_multitask_intent.py first."
                )
            self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return self._metadata

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        return self._tokenizer

    @property
    def model(self) -> MultiTaskIntentModel:
        if self._model is None:
            weights_path = self.model_dir / "multitask_model.pt"
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Missing multitask weights at {weights_path}. Run python3 training/train_multitask_intent.py first."
                )
            payload = torch.load(weights_path, map_location="cpu")
            label_sizes = MultiTaskLabelSizes(
                intent_type=len(TASK_TO_CONFIG["intent_type"].labels),
                intent_subtype=len(TASK_TO_CONFIG["intent_subtype"].labels),
                decision_phase=len(TASK_TO_CONFIG["decision_phase"].labels),
            )
            model = MultiTaskIntentModel(self.metadata["base_model_name"], label_sizes)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            self._model = model
        return self._model

    def _encode(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def _predict_logits(self, task: str, texts: list[str]) -> torch.Tensor:
        config = TASK_TO_CONFIG[task]
        inputs = self._encode(texts, config.max_length)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs[TASK_TO_LOGIT_KEY[task]]

    def predict_all_heads_batch(
        self, texts: list[str]
    ) -> dict[str, torch.Tensor]:
        """Single encoder pass returning logits for all three heads at once.

        This is the hot-path entry point.  Compared with calling
        ``_predict_logits`` once per head it cuts the number of DistilBERT
        forward passes from 3 → 1, roughly halving CPU latency for a single
        query.

        Returns
        -------
        dict with keys ``intent_type_logits``, ``intent_subtype_logits``,
        ``decision_phase_logits`` — raw (pre-softmax) float tensors of shape
        ``(len(texts), n_classes_for_head)``.
        """
        # Use the maximum of the three head max_lengths so all heads see the
        # same truncation boundary.
        max_len = max(cfg.max_length for cfg in TASK_TO_CONFIG.values())
        inputs = self._encode(texts, max_len)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return {
            "intent_type_logits": outputs["intent_type_logits"],
            "intent_subtype_logits": outputs["intent_subtype_logits"],
            "decision_phase_logits": outputs["decision_phase_logits"],
        }


class MultiTaskHeadProxy:
    def __init__(self, task: str):
        if task not in TASK_TO_CONFIG:
            raise ValueError(f"Unsupported multitask head: {task}")
        self.task = task
        self.config = TASK_TO_CONFIG[task]
        self.runtime = get_multitask_runtime()
        self._calibration = None

    @property
    def tokenizer(self):
        return self.runtime.tokenizer

    @property
    def model(self):
        proxy = self

        class _TaskModelView:
            config = type("ConfigView", (), {"id2label": proxy.config.id2label})()

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                with torch.inference_mode():
                    outputs = proxy.runtime.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs[TASK_TO_LOGIT_KEY[proxy.task]]
                return type("OutputView", (), {"logits": logits})()

            __call__ = forward

        return _TaskModelView()

    @property
    def forward_arg_names(self) -> set[str]:
        return {"input_ids", "attention_mask"}

    @property
    def calibration(self) -> CalibrationState:
        if self._calibration is None:
            calibrated = False
            temperature = 1.0
            confidence_threshold = self.config.default_confidence_threshold
            calibration_path = CALIBRATION_ARTIFACTS_DIR / f"{self.task}.json"
            if calibration_path.exists():
                payload = json.loads(calibration_path.read_text(encoding="utf-8"))
                calibrated = bool(payload.get("calibrated", True))
                temperature = float(payload.get("temperature", 1.0))
                confidence_threshold = float(payload.get("confidence_threshold", confidence_threshold))
            self._calibration = CalibrationState(
                calibrated=calibrated,
                temperature=max(temperature, 1e-3),
                confidence_threshold=min(max(confidence_threshold, 0.0), 1.0),
            )
        return self._calibration

    def _predict_probs(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.runtime._predict_logits(self.task, texts)
        with torch.inference_mode():
            raw_probs = torch.softmax(logits, dim=-1)
            calibrated_probs = torch.softmax(logits / self.calibration.temperature, dim=-1)
        return raw_probs, calibrated_probs

    def predict_probs_from_logits(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute calibrated probs from pre-computed logits (hot-path helper).

        Called by ``classify_query_fused`` after a single shared encoder pass
        so that each ``MultiTaskHeadProxy`` does not re-run the encoder.
        """
        with torch.inference_mode():
            raw_probs = torch.softmax(logits, dim=-1)
            calibrated_probs = torch.softmax(logits / self.calibration.temperature, dim=-1)
        return raw_probs, calibrated_probs

    def predict_from_logits(
        self, logits: torch.Tensor, confidence_threshold: float | None = None
    ) -> dict:
        """Return a single prediction dict from pre-computed logits."""
        effective_threshold = (
            self.calibration.confidence_threshold
            if confidence_threshold is None
            else min(max(float(confidence_threshold), 0.0), 1.0)
        )
        raw_probs, calibrated_probs = self.predict_probs_from_logits(logits.unsqueeze(0))
        raw_row = raw_probs[0]
        calibrated_row = calibrated_probs[0]
        pred_id = int(torch.argmax(calibrated_row).item())
        confidence = float(calibrated_row[pred_id].item())
        raw_confidence = float(raw_row[pred_id].item())
        return {
            "label": self.config.id2label[pred_id],
            "confidence": round_score(confidence),
            "raw_confidence": round_score(raw_confidence),
            "confidence_threshold": round_score(effective_threshold),
            "calibrated": self.calibration.calibrated,
            "meets_confidence_threshold": confidence >= effective_threshold,
        }

    def predict_probs_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not texts:
            empty = torch.empty((0, len(self.config.labels)), dtype=torch.float32)
            return empty, empty
        raw_chunks: list[torch.Tensor] = []
        calibrated_chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), self.runtime._predict_batch_size):
            batch = texts[start : start + self.runtime._predict_batch_size]
            raw, calibrated = self._predict_probs(batch)
            raw_chunks.append(raw.detach().cpu())
            calibrated_chunks.append(calibrated.detach().cpu())
        return torch.cat(raw_chunks, dim=0), torch.cat(calibrated_chunks, dim=0)

    def predict_batch(self, texts: list[str], confidence_threshold: float | None = None) -> list[dict]:
        if not texts:
            return []
        effective_threshold = (
            self.calibration.confidence_threshold
            if confidence_threshold is None
            else min(max(float(confidence_threshold), 0.0), 1.0)
        )
        predictions: list[dict] = []
        for start in range(0, len(texts), self.runtime._predict_batch_size):
            batch = texts[start : start + self.runtime._predict_batch_size]
            raw_probs, calibrated_probs = self._predict_probs(batch)
            for raw_row, calibrated_row in zip(raw_probs, calibrated_probs):
                pred_id = int(torch.argmax(calibrated_row).item())
                confidence = float(calibrated_row[pred_id].item())
                raw_confidence = float(raw_row[pred_id].item())
                predictions.append(
                    {
                        "label": self.config.id2label[pred_id],
                        "confidence": round_score(confidence),
                        "raw_confidence": round_score(raw_confidence),
                        "confidence_threshold": round_score(effective_threshold),
                        "calibrated": self.calibration.calibrated,
                        "meets_confidence_threshold": confidence >= effective_threshold,
                    }
                )
        return predictions

    def predict(self, text: str, confidence_threshold: float | None = None) -> dict:
        return self.predict_batch([text], confidence_threshold=confidence_threshold)[0]

    def status(self) -> dict:
        return {
            "head": self.task,
            "model_path": str(self.runtime.model_dir),
            "calibration_path": str(CALIBRATION_ARTIFACTS_DIR / f"{self.task}.json"),
            "ready": (self.runtime.model_dir / "multitask_model.pt").exists(),
            "calibrated": self.calibration.calibrated,
        }


@lru_cache(maxsize=1)
def get_multitask_runtime() -> MultiTaskRuntime:
    return MultiTaskRuntime(MULTITASK_INTENT_MODEL_DIR)
