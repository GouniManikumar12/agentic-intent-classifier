import json

try:
    from .model_runtime import get_head  # type: ignore
except ImportError:
    from model_runtime import get_head


def predict(text: str, confidence_threshold: float | None = None):
    return get_head("intent_subtype").predict(text, confidence_threshold=confidence_threshold)


examples = [
    "What is CRM software?",
    "HubSpot vs Zoho for a small team",
    "Which CRM should I buy for a 3-person startup?",
    "How do I reset my password?",
]

if __name__ == "__main__":
    for text in examples:
        print(f"\nInput: {text}")
        print("Prediction:", json.dumps(predict(text), indent=2))
