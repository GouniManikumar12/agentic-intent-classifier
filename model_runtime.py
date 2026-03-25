from __future__ import annotations

import json
import os
import inspect
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
        self._predict_batch_size = 32
        self._forward_arg_names = None

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
        return {
            "head": self.config.slug,
            "model_path": str(self.config.model_dir),
            "calibration_path": str(self.config.calibration_path),
            "ready": self.config.model_dir.exists(),
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
        raw_probs, calibrated_probs = self._predict_probs(texts)
        return raw_probs.detach().cpu(), calibrated_probs.detach().cpu()

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
