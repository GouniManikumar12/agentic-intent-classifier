import json

from model_runtime import get_head


def predict(text: str, confidence_threshold: float | None = None):
    return get_head("iab_content").predict(text, confidence_threshold=confidence_threshold)


examples = [
    "What is CRM software?",
    "How do I reset my password?",
    "Book a table for 2 tonight",
    "what is best vodka drink should i try",
]

if __name__ == "__main__":
    for text in examples:
        print(f"\nInput: {text}")
        print("Prediction:", json.dumps(predict(text), indent=2))
