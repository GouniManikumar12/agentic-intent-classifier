from __future__ import annotations

import json
import os
import inspect
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import HEAD_CONFIGS, HeadConfig, _looks_like_local_hf_model_dir

_TRAIN_SCRIPT_HINTS: dict[str, str] = {
    "intent_type": "python3 training/train.py",
    "decision_phase": "python3 training/train_decision_phase.py",
    "intent_subtype": "python3 training/train_subtype.py",
    "iab_content": "python3 training/train_iab.py",
}


def _resolved_model_dir(config: HeadConfig) -> Path:
    return Path(config.model_dir).expanduser().resolve()


def _missing_head_weights_message(config: HeadConfig) -> str:
    path = _resolved_model_dir(config)
    train_hint = _TRAIN_SCRIPT_HINTS.get(
        config.slug,
        "See the `training/` directory for the matching `train_*.py` script.",
    )
    return (
        f"Classifier weights for head '{config.slug}' are missing or incomplete at {path}. "
        f"Expected a Hugging Face model directory with config.json and "
        f"model.safetensors (or pytorch_model.bin), plus tokenizer files. "
        f"From the `agentic-intent-classifier` directory, run: {train_hint}. "
        f"Note: training only `train_iab.py` does not populate `model_output`; "
        f"full `classify_query` / evaluation also needs the intent, subtype, and decision-phase heads."
    )


def round_score(value: float) -> float:
    return round(float(value), 4)


@dataclass(frozen=True)
class CalibrationState:
    calibrated: bool
    temperature: float
    confidence_threshold: float


class SequenceClassifierHead:
    def __init__(self, config: HeadConfig):
        self.config = config
        self._tokenizer = None
        self._model = None
        self._calibration = None
        self._predict_batch_size = 32
        self._forward_arg_names = None

    def _weights_dir(self) -> Path:
        return _resolved_model_dir(self.config)

    def _require_local_weights(self) -> Path:
        weights_dir = self._weights_dir()
        if not _looks_like_local_hf_model_dir(weights_dir):
            raise FileNotFoundError(_missing_head_weights_message(self.config))
        return weights_dir

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            weights_dir = self._require_local_weights()
            self._tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            weights_dir = self._require_local_weights()
            self._model = AutoModelForSequenceClassification.from_pretrained(str(weights_dir))
            self._model.eval()
        return self._model

    @property
    def forward_arg_names(self) -> set[str]:
        if self._forward_arg_names is None:
            self._forward_arg_names = set(inspect.signature(self.model.forward).parameters)
        return self._forward_arg_names

    @property
    def calibration(self) -> CalibrationState:
        if self._calibration is None:
            calibrated = False
            temperature = 1.0
            confidence_threshold = self.config.default_confidence_threshold
            if self.config.calibration_path.exists():
                payload = json.loads(self.config.calibration_path.read_text())
                calibrated = bool(payload.get("calibrated", True))
                temperature = float(payload.get("temperature", 1.0))
                confidence_threshold = float(
                    payload.get("confidence_threshold", self.config.default_confidence_threshold)
                )
            self._calibration = CalibrationState(
                calibrated=calibrated,
                temperature=max(temperature, 1e-3),
                confidence_threshold=min(max(confidence_threshold, 0.0), 1.0),
            )
        return self._calibration

    def status(self) -> dict:
        weights_dir = self._weights_dir()
        return {
            "head": self.config.slug,
            "model_path": str(weights_dir),
            "calibration_path": str(self.config.calibration_path),
            "ready": _looks_like_local_hf_model_dir(weights_dir),
            "calibrated": self.calibration.calibrated,
        }

    def _encode(self, texts: list[str]):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        )
        return {
            key: value
            for key, value in encoded.items()
            if key in self.forward_arg_names
        }

    def _predict_probs(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._encode(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_probs = torch.softmax(outputs.logits, dim=-1)
            calibrated_probs = torch.softmax(outputs.logits / self.calibration.temperature, dim=-1)
        return raw_probs, calibrated_probs

    def predict_probs_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not texts:
            empty = torch.empty((0, len(self.config.labels)), dtype=torch.float32)
            return empty, empty
        raw_chunks: list[torch.Tensor] = []
        calibrated_chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), self._predict_batch_size):
            batch_texts = texts[start : start + self._predict_batch_size]
            raw_probs, calibrated_probs = self._predict_probs(batch_texts)
            raw_chunks.append(raw_probs.detach().cpu())
            calibrated_chunks.append(calibrated_probs.detach().cpu())
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

        for start in range(0, len(texts), self._predict_batch_size):
            batch_texts = texts[start : start + self._predict_batch_size]
            raw_probs, calibrated_probs = self._predict_probs(batch_texts)
            for raw_row, calibrated_row in zip(raw_probs, calibrated_probs):
                pred_id = int(torch.argmax(calibrated_row).item())
                confidence = float(calibrated_row[pred_id].item())
                raw_confidence = float(raw_row[pred_id].item())
                predictions.append(
                    {
                        "label": self.model.config.id2label[pred_id],
                        "confidence": round_score(confidence),
                        "raw_confidence": round_score(raw_confidence),
                        "confidence_threshold": round_score(effective_threshold),
                        "calibrated": self.calibration.calibrated,
                        "meets_confidence_threshold": confidence >= effective_threshold,
                    }
                )
        return predictions

    def predict_candidate_batch(
        self,
        texts: list[str],
        candidate_labels: list[list[str]],
        confidence_threshold: float | None = None,
    ) -> list[dict]:
        if not texts:
            return []
        if len(texts) != len(candidate_labels):
            raise ValueError("texts and candidate_labels must have the same length")

        effective_threshold = (
            self.calibration.confidence_threshold
            if confidence_threshold is None
            else min(max(float(confidence_threshold), 0.0), 1.0)
        )
        predictions: list[dict] = []

        for start in range(0, len(texts), self._predict_batch_size):
            batch_texts = texts[start : start + self._predict_batch_size]
            batch_candidates = candidate_labels[start : start + self._predict_batch_size]
            raw_probs, calibrated_probs = self._predict_probs(batch_texts)
            for raw_row, calibrated_row, labels in zip(raw_probs, calibrated_probs, batch_candidates):
                label_ids = [self.config.label2id[label] for label in labels if label in self.config.label2id]
                if not label_ids:
                    predictions.append(
                        {
                            "label": None,
                            "confidence": 0.0,
                            "raw_confidence": 0.0,
                            "candidate_mass": 0.0,
                            "confidence_threshold": round_score(effective_threshold),
                            "calibrated": self.calibration.calibrated,
                            "meets_confidence_threshold": False,
                        }
                    )
                    continue

                calibrated_slice = calibrated_row[label_ids]
                raw_slice = raw_row[label_ids]
                calibrated_mass = float(calibrated_slice.sum().item())
                raw_mass = float(raw_slice.sum().item())
                if calibrated_mass <= 0:
                    predictions.append(
                        {
                            "label": labels[0],
                            "confidence": 0.0,
                            "raw_confidence": 0.0,
                            "candidate_mass": 0.0,
                            "confidence_threshold": round_score(effective_threshold),
                            "calibrated": self.calibration.calibrated,
                            "meets_confidence_threshold": False,
                        }
                    )
                    continue

                normalized_calibrated = calibrated_slice / calibrated_mass
                normalized_raw = raw_slice / max(raw_mass, 1e-9)
                pred_offset = int(torch.argmax(normalized_calibrated).item())
                pred_id = label_ids[pred_offset]
                confidence = float(normalized_calibrated[pred_offset].item())
                raw_confidence = float(normalized_raw[pred_offset].item())
                predictions.append(
                    {
                        "label": self.model.config.id2label[pred_id],
                        "confidence": round_score(confidence),
                        "raw_confidence": round_score(raw_confidence),
                        "candidate_mass": round_score(calibrated_mass),
                        "confidence_threshold": round_score(effective_threshold),
                        "calibrated": self.calibration.calibrated,
                        "meets_confidence_threshold": confidence >= effective_threshold,
                    }
                )
        return predictions

    def predict(self, text: str, confidence_threshold: float | None = None) -> dict:
        return self.predict_batch([text], confidence_threshold=confidence_threshold)[0]

    def predict_candidates(
        self,
        text: str,
        candidate_labels: list[str],
        confidence_threshold: float | None = None,
    ) -> dict:
        return self.predict_candidate_batch([text], [candidate_labels], confidence_threshold=confidence_threshold)[0]


@lru_cache(maxsize=None)
def get_head(head_name: str) -> SequenceClassifierHead:
    if head_name not in HEAD_CONFIGS:
        raise ValueError(f"Unknown head: {head_name}")
    return SequenceClassifierHead(HEAD_CONFIGS[head_name])
