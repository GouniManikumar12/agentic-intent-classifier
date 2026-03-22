from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "decision_phase_model_output"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    return {
        "label": model.config.id2label[pred_id],
        "confidence": round(confidence, 4),
    }


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
        print("Prediction:", predict(text))
