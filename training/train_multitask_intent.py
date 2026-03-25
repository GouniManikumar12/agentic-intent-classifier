from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    DECISION_PHASE_DIFFICULTY_DATA_DIR,
    DECISION_PHASE_HEAD_CONFIG,
    FULL_INTENT_TAXONOMY_DATA_DIR,
    INTENT_HEAD_CONFIG,
    INTENT_TYPE_DIFFICULTY_DATA_DIR,
    MULTITASK_INTENT_MODEL_DIR,
    SUBTYPE_DIFFICULTY_DATA_DIR,
    SUBTYPE_HEAD_CONFIG,
)
from multitask_model import MultiTaskIntentModel, MultiTaskLabelSizes  # noqa: E402
from training.common import write_json  # noqa: E402


IGNORE_INDEX = -100


@dataclass
class MultiTaskRow:
    text: str
    intent_type: int = IGNORE_INDEX
    intent_subtype: int = IGNORE_INDEX
    decision_phase: int = IGNORE_INDEX


def _load_task_rows(path: Path, label_field: str, label2id: dict[str, int]) -> list[tuple[str, int]]:
    if not path.exists():
        return []
    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append((item["text"], label2id[item[label_field]]))
    return rows


def _merge_rows(
    split: str,
    include_full_intent: bool = True,
    include_difficulty: bool = True,
) -> list[dict]:
    merged: dict[str, MultiTaskRow] = {}

    def upsert(task_key: str, text: str, label: int) -> None:
        row = merged.get(text)
        if row is None:
            row = MultiTaskRow(text=text)
            merged[text] = row
        setattr(row, task_key, int(label))

    # Base split rows
    for text, label in _load_task_rows(
        INTENT_HEAD_CONFIG.split_paths[split],
        INTENT_HEAD_CONFIG.label_field,
        INTENT_HEAD_CONFIG.label2id,
    ):
        upsert("intent_type", text, label)
    for text, label in _load_task_rows(
        SUBTYPE_HEAD_CONFIG.split_paths[split],
        SUBTYPE_HEAD_CONFIG.label_field,
        SUBTYPE_HEAD_CONFIG.label2id,
    ):
        upsert("intent_subtype", text, label)
    for text, label in _load_task_rows(
        DECISION_PHASE_HEAD_CONFIG.split_paths[split],
        DECISION_PHASE_HEAD_CONFIG.label_field,
        DECISION_PHASE_HEAD_CONFIG.label2id,
    ):
        upsert("decision_phase", text, label)

    if include_full_intent:
        full_path = FULL_INTENT_TAXONOMY_DATA_DIR / f"{split}.jsonl"
        for text, label in _load_task_rows(full_path, "intent_type", INTENT_HEAD_CONFIG.label2id):
            upsert("intent_type", text, label)
        for text, label in _load_task_rows(full_path, "intent_subtype", SUBTYPE_HEAD_CONFIG.label2id):
            upsert("intent_subtype", text, label)
        for text, label in _load_task_rows(full_path, "decision_phase", DECISION_PHASE_HEAD_CONFIG.label2id):
            upsert("decision_phase", text, label)

    if include_difficulty:
        for text, label in _load_task_rows(
            INTENT_TYPE_DIFFICULTY_DATA_DIR / f"{split}.jsonl",
            "intent_type",
            INTENT_HEAD_CONFIG.label2id,
        ):
            upsert("intent_type", text, label)
        for text, label in _load_task_rows(
            SUBTYPE_DIFFICULTY_DATA_DIR / f"{split}.jsonl",
            "intent_subtype",
            SUBTYPE_HEAD_CONFIG.label2id,
        ):
            upsert("intent_subtype", text, label)
        for text, label in _load_task_rows(
            DECISION_PHASE_DIFFICULTY_DATA_DIR / f"{split}.jsonl",
            "decision_phase",
            DECISION_PHASE_HEAD_CONFIG.label2id,
        ):
            upsert("decision_phase", text, label)

    return [
        {
            "text": row.text,
            "intent_type": row.intent_type,
            "intent_subtype": row.intent_subtype,
            "decision_phase": row.decision_phase,
        }
        for row in merged.values()
    ]


def _prepare_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_list(rows)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, loss_weights: dict[str, float], **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_type = inputs.pop("intent_type")
        labels_subtype = inputs.pop("intent_subtype")
        labels_phase = inputs.pop("decision_phase")
        outputs = model(**inputs)
        loss_type = self.ce(outputs["intent_type_logits"], labels_type)
        loss_subtype = self.ce(outputs["intent_subtype_logits"], labels_subtype)
        loss_phase = self.ce(outputs["decision_phase_logits"], labels_phase)
        loss = (
            (self.loss_weights["intent_type"] * loss_type)
            + (self.loss_weights["intent_subtype"] * loss_subtype)
            + (self.loss_weights["decision_phase"] * loss_phase)
        )
        return (loss, outputs) if return_outputs else loss


def _masked_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = labels != IGNORE_INDEX
    if not np.any(mask):
        return {"accuracy": 0.0, "macro_f1": 0.0, "count": 0}
    preds = np.argmax(logits[mask], axis=-1)
    true = labels[mask]
    return {
        "accuracy": float(accuracy_score(true, preds)),
        "macro_f1": float(f1_score(true, preds, average="macro")),
        "count": int(mask.sum()),
    }


def _compute_metrics(eval_pred):
    predictions, labels = eval_pred
    intent_logits, subtype_logits, phase_logits = predictions
    intent_labels, subtype_labels, phase_labels = labels
    intent_metrics = _masked_metrics(intent_logits, intent_labels)
    subtype_metrics = _masked_metrics(subtype_logits, subtype_labels)
    phase_metrics = _masked_metrics(phase_logits, phase_labels)
    return {
        "intent_type_accuracy": intent_metrics["accuracy"],
        "intent_type_macro_f1": intent_metrics["macro_f1"],
        "intent_subtype_accuracy": subtype_metrics["accuracy"],
        "intent_subtype_macro_f1": subtype_metrics["macro_f1"],
        "decision_phase_accuracy": phase_metrics["accuracy"],
        "decision_phase_macro_f1": phase_metrics["macro_f1"],
    }


def main() -> None:
    train_rows = _merge_rows("train", include_full_intent=True, include_difficulty=True)
    val_rows = _merge_rows("val", include_full_intent=True, include_difficulty=True)
    test_rows = _merge_rows("test", include_full_intent=False, include_difficulty=False)

    tokenizer = AutoTokenizer.from_pretrained(INTENT_HEAD_CONFIG.model_name)
    max_length = max(
        INTENT_HEAD_CONFIG.max_length,
        SUBTYPE_HEAD_CONFIG.max_length,
        DECISION_PHASE_HEAD_CONFIG.max_length,
    )
    train_dataset = _prepare_dataset(train_rows, tokenizer, max_length=max_length)
    val_dataset = _prepare_dataset(val_rows, tokenizer, max_length=max_length)
    test_dataset = _prepare_dataset(test_rows, tokenizer, max_length=max_length)

    model = MultiTaskIntentModel(
        INTENT_HEAD_CONFIG.model_name,
        MultiTaskLabelSizes(
            intent_type=len(INTENT_HEAD_CONFIG.labels),
            intent_subtype=len(SUBTYPE_HEAD_CONFIG.labels),
            decision_phase=len(DECISION_PHASE_HEAD_CONFIG.labels),
        ),
    )

    training_args = TrainingArguments(
        output_dir=str(MULTITASK_INTENT_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none",
        label_names=["intent_type", "intent_subtype", "decision_phase"],
    )
    loss_weights = {"intent_type": 1.0, "intent_subtype": 1.0, "decision_phase": 1.0}
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        loss_weights=loss_weights,
    )

    print(f"Loaded multitask splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(val_metrics)
    print(test_metrics)

    MULTITASK_INTENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(MULTITASK_INTENT_MODEL_DIR)
    torch.save({"state_dict": model.state_dict()}, MULTITASK_INTENT_MODEL_DIR / "multitask_model.pt")
    metadata = {
        "format": "admesh_multitask_intent_v1",
        "base_model_name": INTENT_HEAD_CONFIG.model_name,
        "max_length": max_length,
        "label_maps": {
            "intent_type": {"label2id": INTENT_HEAD_CONFIG.label2id, "id2label": INTENT_HEAD_CONFIG.id2label},
            "intent_subtype": {"label2id": SUBTYPE_HEAD_CONFIG.label2id, "id2label": SUBTYPE_HEAD_CONFIG.id2label},
            "decision_phase": {"label2id": DECISION_PHASE_HEAD_CONFIG.label2id, "id2label": DECISION_PHASE_HEAD_CONFIG.id2label},
        },
    }
    (MULTITASK_INTENT_MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(
        MULTITASK_INTENT_MODEL_DIR / "train_metrics.json",
        {
            "head": "multitask_intent",
            "loss_weights": loss_weights,
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "test_count": len(test_rows),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )


if __name__ == "__main__":
    main()
