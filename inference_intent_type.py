import argparse
import json

from model_runtime import get_head


def predict(text: str, confidence_threshold: float | None = None):
    return get_head("intent_type").predict(text, confidence_threshold=confidence_threshold)


examples = [
    "What is CRM?",
    "Best CRM for small teams",
    "HubSpot vs Zoho CRM",
    "Tell me more",
]


def main():
    parser = argparse.ArgumentParser(description="Run intent_type inference for one query or the built-in examples.")
    parser.add_argument("text", nargs="?", help="Optional query text to classify.")
    args = parser.parse_args()

    if args.text:
        print(json.dumps(predict(args.text), indent=2))
        return

    for text in examples:
        print(f"\nInput: {text}")
        print("Prediction:", json.dumps(predict(text), indent=2))


if __name__ == "__main__":
    main()
