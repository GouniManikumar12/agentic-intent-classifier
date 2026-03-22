import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "decision_phase"
MODEL_OUTPUT_DIR = BASE_DIR / "decision_phase_model_output"

label2id = {
    "awareness": 0,
    "research": 1,
    "consideration": 2,
    "decision": 3,
    "action": 4,
    "post_purchase": 5,
    "support": 6,
}
id2label = {v: k for k, v in label2id.items()}


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append(
                {
                    "text": item["text"],
                    "label": label2id[item["decision_phase"]],
                }
            )
    return rows


train_rows = load_jsonl(DATA_DIR / "train.jsonl")
val_rows = load_jsonl(DATA_DIR / "val.jsonl")
test_rows = load_jsonl(DATA_DIR / "test.jsonl")

train_dataset = Dataset.from_list(train_rows)
val_dataset = Dataset.from_list(val_rows)
test_dataset = Dataset.from_list(test_rows)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


training_args = TrainingArguments(
    output_dir=str(MODEL_OUTPUT_DIR),
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print(
    f"Loaded decision_phase splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
)
trainer.train()
val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
print(val_metrics)
print(test_metrics)

model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
