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

if __name__ == "__main__":
    for text in examples:
        print(f"\nInput: {text}")
        print("Prediction:", json.dumps(predict(text), indent=2))
