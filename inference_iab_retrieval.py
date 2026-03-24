import json

from iab_retrieval import predict_iab_content_retrieval


def predict(text: str):
    return predict_iab_content_retrieval(text)


examples = [
    "What is CRM software?",
    "How do I reset my password?",
    "Best project management tool for a startup team",
    "best laptop for coding",
]

if __name__ == "__main__":
    for text in examples:
        print(f"\nInput: {text}")
        print("Prediction:", json.dumps(predict(text), indent=2))
