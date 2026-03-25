import json

try:
    from .iab_classifier import predict_iab_content_classifier  # type: ignore
except ImportError:
    from iab_classifier import predict_iab_content_classifier


def predict(text: str, confidence_threshold: float | None = None):
    return predict_iab_content_classifier(text, confidence_threshold=confidence_threshold)


examples = [
    "Which laptop should I buy for college?",
    "What is CRM software?",
    "Book a table for two tonight",
]

if __name__ == "__main__":
    for text in examples:
        print(f"\nInput: {text}")
        print("Prediction:", json.dumps(predict(text), indent=2))
