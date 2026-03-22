from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import HEAD_CONFIGS, HeadConfig


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

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.config.model_dir)
            self._model.eval()
        return self._model

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
        return {
            "head": self.config.slug,
            "model_path": str(self.config.model_dir),
            "calibration_path": str(self.config.calibration_path),
            "ready": self.config.model_dir.exists(),
            "calibrated": self.calibration.calibrated,
        }

    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        )

    def _predict_probs(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._encode(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_probs = torch.softmax(outputs.logits, dim=-1)
            calibrated_probs = torch.softmax(outputs.logits / self.calibration.temperature, dim=-1)
        return raw_probs, calibrated_probs

    def predict_batch(self, texts: list[str], confidence_threshold: float | None = None) -> list[dict]:
        if not texts:
            return []

        raw_probs, calibrated_probs = self._predict_probs(texts)
        effective_threshold = (
            self.calibration.confidence_threshold
            if confidence_threshold is None
            else min(max(float(confidence_threshold), 0.0), 1.0)
        )
        predictions: list[dict] = []
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

    def predict(self, text: str, confidence_threshold: float | None = None) -> dict:
        return self.predict_batch([text], confidence_threshold=confidence_threshold)[0]


@lru_cache(maxsize=None)
def get_head(head_name: str) -> SequenceClassifierHead:
    if head_name not in HEAD_CONFIGS:
        raise ValueError(f"Unknown head: {head_name}")
    return SequenceClassifierHead(HEAD_CONFIGS[head_name])
