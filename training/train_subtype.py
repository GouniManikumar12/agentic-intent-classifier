import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    FULL_INTENT_TAXONOMY_DATA_DIR,
    SUBTYPE_DIFFICULTY_DATA_DIR,
    SUBTYPE_HEAD_CONFIG,
    SUBTYPE_TRAINING_WEIGHTS,
)
from training.common import (
    build_label_weight_tensor,
    compute_classification_metrics,
    load_labeled_rows,
    load_labeled_rows_from_paths,
    prepare_dataset,
    write_json,
)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


train_rows = load_labeled_rows_from_paths(
    [
        SUBTYPE_HEAD_CONFIG.split_paths["train"],
        FULL_INTENT_TAXONOMY_DATA_DIR / "train.jsonl",
        SUBTYPE_DIFFICULTY_DATA_DIR / "train.jsonl",
    ],
    SUBTYPE_HEAD_CONFIG.label_field,
    SUBTYPE_HEAD_CONFIG.label2id,
)
val_rows = load_labeled_rows_from_paths(
    [
        SUBTYPE_HEAD_CONFIG.split_paths["val"],
        FULL_INTENT_TAXONOMY_DATA_DIR / "val.jsonl",
        SUBTYPE_DIFFICULTY_DATA_DIR / "val.jsonl",
    ],
    SUBTYPE_HEAD_CONFIG.label_field,
    SUBTYPE_HEAD_CONFIG.label2id,
)
test_rows = load_labeled_rows(
    SUBTYPE_HEAD_CONFIG.split_paths["test"],
    SUBTYPE_HEAD_CONFIG.label_field,
    SUBTYPE_HEAD_CONFIG.label2id,
)

tokenizer = AutoTokenizer.from_pretrained(SUBTYPE_HEAD_CONFIG.model_name)

train_dataset = prepare_dataset(train_rows, tokenizer, SUBTYPE_HEAD_CONFIG.max_length)
val_dataset = prepare_dataset(val_rows, tokenizer, SUBTYPE_HEAD_CONFIG.max_length)
test_dataset = prepare_dataset(test_rows, tokenizer, SUBTYPE_HEAD_CONFIG.max_length)
class_weights = build_label_weight_tensor(SUBTYPE_HEAD_CONFIG.labels, SUBTYPE_TRAINING_WEIGHTS)

model = AutoModelForSequenceClassification.from_pretrained(
    SUBTYPE_HEAD_CONFIG.model_name,
    num_labels=len(SUBTYPE_HEAD_CONFIG.labels),
    id2label=SUBTYPE_HEAD_CONFIG.id2label,
    label2id=SUBTYPE_HEAD_CONFIG.label2id,
)

training_args = TrainingArguments(
    output_dir=str(SUBTYPE_HEAD_CONFIG.model_dir),
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

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_classification_metrics,
    class_weights=class_weights,
)

print(
    f"Loaded subtype splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
)
print(f"Subtype weights: {[round(float(x), 3) for x in class_weights.tolist()]}")
trainer.train()
val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
print(val_metrics)
print(test_metrics)

SUBTYPE_HEAD_CONFIG.model_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SUBTYPE_HEAD_CONFIG.model_dir)
tokenizer.save_pretrained(SUBTYPE_HEAD_CONFIG.model_dir)
write_json(
    SUBTYPE_HEAD_CONFIG.model_dir / "train_metrics.json",
    {
        "head": SUBTYPE_HEAD_CONFIG.slug,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    },
)
