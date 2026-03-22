from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def load_labeled_rows(path: Path, label_field: str, label2id: dict[str, int]) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append({"text": item["text"], "label": label2id[item[label_field]]})
    return rows


def load_labeled_rows_from_paths(paths: list[Path], label_field: str, label2id: dict[str, int]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(load_labeled_rows(path, label_field, label2id))
    return rows


def prepare_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_list(rows)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset


def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
