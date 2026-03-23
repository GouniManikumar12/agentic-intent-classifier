from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import IAB_HIERARCHY_HEAD_CONFIGS
from training.common import (
    build_balanced_class_weights,
    compute_classification_metrics,
    load_labeled_rows,
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


def train_level(level: int, epochs: float, train_batch_size: int, eval_batch_size: int, learning_rate: float) -> dict:
    config = IAB_HIERARCHY_HEAD_CONFIGS[level]
    train_rows = load_labeled_rows(config.split_paths["train"], config.label_field, config.label2id)
    val_rows = load_labeled_rows(config.split_paths["val"], config.label_field, config.label2id)
    test_rows = load_labeled_rows(config.split_paths["test"], config.label_field, config.label2id)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = prepare_dataset(train_rows, tokenizer, config.max_length)
    val_dataset = prepare_dataset(val_rows, tokenizer, config.max_length)
    test_dataset = prepare_dataset(test_rows, tokenizer, config.max_length)
    class_weights = build_balanced_class_weights(train_rows, len(config.labels))

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.labels),
        id2label=config.id2label,
        label2id=config.label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(config.model_dir),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
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
        f"Loaded hierarchical IAB level {level} splits: "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )
    print(
        f"Hierarchical IAB tier{level} weights: "
        f"min={round(float(class_weights.min().item()), 3)} "
        f"max={round(float(class_weights.max().item()), 3)}"
    )
    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(val_metrics)
    print(test_metrics)

    config.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.model_dir)
    tokenizer.save_pretrained(config.model_dir)
    payload = {
        "head": config.slug,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "training_args": {
            "epochs": epochs,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": learning_rate,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    write_json(config.model_dir / "train_metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hierarchical IAB content heads.")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Per-device eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=[1, 2, 3, 4],
        help="Hierarchy levels to train.",
    )
    args = parser.parse_args()

    results = {}
    for level in args.levels:
        if level not in IAB_HIERARCHY_HEAD_CONFIGS:
            raise ValueError(f"Unsupported IAB hierarchy level: {level}")
        results[f"tier{level}"] = train_level(
            level,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
        )

    summary_path = BASE_DIR / "iab_hierarchy_model_output" / "summary.json"
    write_json(summary_path, results)
    print(f"Wrote hierarchical IAB summary to {summary_path}")


if __name__ == "__main__":
    main()
