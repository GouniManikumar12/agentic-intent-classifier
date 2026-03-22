import json

from model_runtime import get_head


def predict(text: str, confidence_threshold: float | None = None):
    return get_head("decision_phase").predict(text, confidence_threshold=confidence_threshold)


examples = [
    "What is CRM software?",
    "What are some CRM options for startups?",
    "HubSpot vs Zoho for a small team",
    "Which CRM should I buy for a 3-person startup?",
    "Start my free trial",
    "How do I set up my new CRM?",
    "I cannot log into my account",
]

if __name__ == "__main__":
    for text in examples:
        print(f"\nInput: {text}")
        print("Prediction:", json.dumps(predict(text), indent=2))
