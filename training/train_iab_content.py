from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import IAB_HEAD_CONFIG
from training.common import (
    compute_classification_metrics,
    load_labeled_rows,
    prepare_dataset,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the full-taxonomy IAB content classifier.")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Per-device eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    args = parser.parse_args()

    train_rows = load_labeled_rows(
        IAB_HEAD_CONFIG.split_paths["train"],
        IAB_HEAD_CONFIG.label_field,
        IAB_HEAD_CONFIG.label2id,
    )
    val_rows = load_labeled_rows(
        IAB_HEAD_CONFIG.split_paths["val"],
        IAB_HEAD_CONFIG.label_field,
        IAB_HEAD_CONFIG.label2id,
    )
    test_rows = load_labeled_rows(
        IAB_HEAD_CONFIG.split_paths["test"],
        IAB_HEAD_CONFIG.label_field,
        IAB_HEAD_CONFIG.label2id,
    )

    tokenizer = AutoTokenizer.from_pretrained(IAB_HEAD_CONFIG.model_name)
    train_dataset = prepare_dataset(train_rows, tokenizer, IAB_HEAD_CONFIG.max_length)
    val_dataset = prepare_dataset(val_rows, tokenizer, IAB_HEAD_CONFIG.max_length)
    test_dataset = prepare_dataset(test_rows, tokenizer, IAB_HEAD_CONFIG.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        IAB_HEAD_CONFIG.model_name,
        num_labels=len(IAB_HEAD_CONFIG.labels),
        id2label=IAB_HEAD_CONFIG.id2label,
        label2id=IAB_HEAD_CONFIG.label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(IAB_HEAD_CONFIG.model_dir),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_classification_metrics,
    )

    print(f"Loaded IAB splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(val_metrics)
    print(test_metrics)

    IAB_HEAD_CONFIG.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(IAB_HEAD_CONFIG.model_dir)
    tokenizer.save_pretrained(IAB_HEAD_CONFIG.model_dir)
    write_json(
        IAB_HEAD_CONFIG.model_dir / "train_metrics.json",
        {
            "head": IAB_HEAD_CONFIG.slug,
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "test_count": len(test_rows),
            "training_args": {
                "epochs": args.epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
            },
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )


if __name__ == "__main__":
    main()
