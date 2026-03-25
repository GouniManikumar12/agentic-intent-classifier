from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import MULTITASK_INTENT_MODEL_DIR  # noqa: E402
from multitask_runtime import get_multitask_runtime  # noqa: E402


class _OnnxMultiTaskWrapper(torch.nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.model = runtime.model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return (
            outputs["intent_type_logits"],
            outputs["intent_subtype_logits"],
            outputs["decision_phase_logits"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multitask intent model to ONNX.")
    parser.add_argument(
        "--output-path",
        default=str(MULTITASK_INTENT_MODEL_DIR / "multitask_intent.onnx"),
        help="Output ONNX file path.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    args = parser.parse_args()

    runtime = get_multitask_runtime()
    tokenizer = runtime.tokenizer
    wrapper = _OnnxMultiTaskWrapper(runtime)
    wrapper.eval()

    sample = tokenizer(
        ["sample query for intent classification"],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=int(runtime.metadata.get("max_length", 96)),
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            (sample["input_ids"], sample["attention_mask"]),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=[
                "intent_type_logits",
                "intent_subtype_logits",
                "decision_phase_logits",
            ],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "intent_type_logits": {0: "batch_size"},
                "intent_subtype_logits": {0: "batch_size"},
                "decision_phase_logits": {0: "batch_size"},
            },
            opset_version=args.opset,
        )
    except ModuleNotFoundError as e:
        # Newer torch ONNX exporter requires `onnxscript` (and `onnx`).
        if e.name == "onnxscript" or "onnxscript" in str(e).lower():
            print(
                "Skipping ONNX export: missing dependency `onnxscript`.\n"
                "Install with: `pip install onnx onnxscript`",
                file=sys.stderr,
            )
            return
        raise
    print(f"Exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()
