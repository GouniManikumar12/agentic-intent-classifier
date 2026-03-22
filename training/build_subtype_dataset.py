from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import SUBTYPE_HEAD_CONFIG

OUTPUT_DIR = SUBTYPE_HEAD_CONFIG.data_dir

INTENT_SOURCE_PATHS = {
    "train": BASE_DIR / "data" / "train.jsonl",
    "val": BASE_DIR / "data" / "val.jsonl",
    "test": BASE_DIR / "data" / "test.jsonl",
    "hard_cases": BASE_DIR / "data" / "hard_cases.jsonl",
    "extended_cases": BASE_DIR / "data" / "third_wave_cases.jsonl",
}

PHASE_SOURCE_PATHS = {
    "train": BASE_DIR / "data" / "decision_phase" / "train.jsonl",
    "val": BASE_DIR / "data" / "decision_phase" / "val.jsonl",
    "test": BASE_DIR / "data" / "decision_phase" / "test.jsonl",
    "hard_cases": BASE_DIR / "data" / "decision_phase" / "hard_cases.jsonl",
    "extended_cases": BASE_DIR / "data" / "decision_phase" / "final_wave_cases.jsonl",
}

EXACT_SUBTYPE_OVERRIDES = {
    "Download the customer onboarding checklist": "download",
    "Intercom vs Zendesk for onboarding support": "comparison",
    "Why is customer retention important?": "education",
    "This is not working": "troubleshooting",
    "How should I fix it": "troubleshooting",
    "Show me options": "product_discovery",
    "What do you recommend": "provider_selection",
    "Give me the best one": "provider_selection",
    "Request pricing from the sales team": "contact_sales",
    "Sign me up for the demo": "contact_sales",
    "Book a demo for the analytics platform": "contact_sales",
    "Book a meeting with sales": "contact_sales",
    "Contact sales to schedule a product walkthrough": "contact_sales",
    "What costs less HubSpot or Zoho?": "deal_seeking",
    "Which car to buy in 2026": "provider_selection",
    "Which electric car should I buy this year?": "provider_selection",
    "Toyota Corolla vs Honda Civic for commuting": "comparison",
    "Best SUV to buy in 2026": "product_discovery",
    "Best hybrid SUV for a family": "product_discovery",
    "Book a test drive for the new SUV": "booking",
    "Schedule a test drive for this sedan": "booking",
    "Reserve a test drive for tomorrow": "booking",
}

MANUAL_TRAIN_ROWS = [
    {"text": "Talk to sales about enterprise pricing", "intent_subtype": "contact_sales"},
    {"text": "Put me in touch with an account executive", "intent_subtype": "contact_sales"},
    {"text": "Connect me with sales for a custom quote", "intent_subtype": "contact_sales"},
    {"text": "Download the buyer comparison guide", "intent_subtype": "download"},
    {"text": "Export the onboarding checklist as a PDF", "intent_subtype": "download"},
    {"text": "Download the setup worksheet", "intent_subtype": "download"},
    {"text": "Open a trial workspace for my team", "intent_subtype": "signup"},
    {"text": "Register a new account for our startup", "intent_subtype": "signup"},
    {"text": "Create a free workspace for evaluation", "intent_subtype": "signup"},
    {"text": "Why was my payment charged twice", "intent_subtype": "billing_help"},
    {"text": "My renewal invoice is incorrect", "intent_subtype": "billing_help"},
    {"text": "How do I fix a failed subscription payment", "intent_subtype": "billing_help"},
    {"text": "The reset password email never shows up", "intent_subtype": "account_help"},
    {"text": "I cannot unlock my account", "intent_subtype": "account_help"},
    {"text": "The login page keeps rejecting my sign in code", "intent_subtype": "account_help"},
    {"text": "The integration sync keeps crashing", "intent_subtype": "troubleshooting"},
    {"text": "Our export job fails every time", "intent_subtype": "troubleshooting"},
    {"text": "Why does the API connection keep erroring", "intent_subtype": "troubleshooting"},
    {"text": "Which car to buy in 2026", "intent_subtype": "provider_selection"},
    {"text": "Which electric car should I buy this year?", "intent_subtype": "provider_selection"},
    {"text": "Toyota Corolla vs Honda Civic for commuting", "intent_subtype": "comparison"},
    {"text": "Best SUV to buy in 2026", "intent_subtype": "product_discovery"},
    {"text": "Best hybrid SUV for a family", "intent_subtype": "product_discovery"},
    {"text": "Book a test drive for the new SUV", "intent_subtype": "booking"},
    {"text": "Schedule a test drive for this sedan", "intent_subtype": "booking"},
]

PHRASE_PATTERNS = {
    "billing_help": [
        "invoice",
        "billing",
        "credit card",
        "payment",
        "charged twice",
        "renewal",
        "declined",
    ],
    "account_help": [
        "log into",
        "login",
        "password",
        "credentials",
        "account",
        "workspace",
        "reset link",
        "reset email",
        "invites",
        "invite",
        "unlock",
        "sign in code",
    ],
    "troubleshooting": [
        "broken",
        "crashing",
        "failing",
        "failed",
        "error",
        "not loading",
        "not working",
        "stuck",
        "recover deleted",
    ],
    "onboarding_setup": [
        "set up",
        "configure",
        "connect",
        "import contacts",
        "onboard",
        "roll this tool out",
        "usage analytics",
        "invite teammates",
        "user permissions",
        "connect email",
        "features should we enable first",
        "what should we do first after",
        "what should i do first after",
    ],
    "download": ["download", "export"],
    "contact_sales": ["contact sales", "sales team", "quote", "product walkthrough", "demo", "account executive"],
    "booking": ["book", "schedule", "reserve", "appointment", "seat for", "table for", "meeting"],
    "purchase": ["buy", "checkout", "order", "annual plan", "monthly plan", "subscribe", "subscription"],
    "signup": ["free trial", "create an account", "create my account", "register", "waitlist", "sign up", "activate"],
    "task_execution": [
        "install",
        "upload",
        "run",
        "save",
        "deploy",
        "generate",
        "submit",
        "sync",
        "approve",
        "create a ",
        "annotate",
    ],
    "comparison": ["vs", "compare", "versus"],
    "deal_seeking": ["cheap", "cheaper", "affordable", "lower-cost", "costs less", "budget", "pricing", "under $"],
    "provider_selection": [
        "which should i buy",
        "should i choose",
        "which vendor",
        "best value",
        "go with",
        "commit to",
        "which one should",
        "should we pick",
        "right crm",
        "worth it",
    ],
    "product_discovery": [
        "best ",
        "top ",
        "alternatives",
        "options",
        "choices",
        "what tools",
        "what software",
        "what platforms",
        "worth considering",
        "what crm platforms",
        "what kinds of crm",
    ],
    "education": [
        "what is",
        "what does",
        "how does",
        "how do",
        "explain",
        "walk me through",
        "teach me",
        "show me how",
        "help me understand",
        "basics",
        "fundamentals",
        "difference between",
        "used for",
    ],
}

AMBIGUOUS_FOLLOW_UPS = {
    "tell me more",
    "what about this one",
    "can you help",
    "i need something better",
    "is this good",
    "can you explain",
    "what should i do next",
    "go on",
    "what do you mean",
    "show me more",
    "can you explain that",
    "which one",
    "why",
    "okay and then",
    "and then what",
    "tell me a bit more",
    "show me a little more",
    "which one do you mean",
    "why though",
    "how so exactly",
    "go deeper",
    "say more about that",
    "expand on that a little",
    "why do you say that",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def has_phrase(text: str, phrases: list[str]) -> bool:
    lowered = normalize(text)
    for phrase in phrases:
        token = phrase.lower()
        if token.startswith(" ") or token.endswith(" "):
            if token in lowered:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered):
            return True
        if token in lowered:
            return True
    return False


def infer_subtype(text: str, intent_type: str | None = None, decision_phase: str | None = None) -> str:
    if text in EXACT_SUBTYPE_OVERRIDES:
        return EXACT_SUBTYPE_OVERRIDES[text]

    lowered = normalize(text)
    if intent_type == "support":
        if has_phrase(text, PHRASE_PATTERNS["billing_help"]):
            return "billing_help"
        if has_phrase(text, PHRASE_PATTERNS["account_help"]):
            return "account_help"
        return "troubleshooting"
    if intent_type == "personal_reflection":
        return "emotional_reflection"
    if intent_type == "chit_chat":
        return "follow_up"
    if intent_type == "prohibited":
        return "follow_up"
    if intent_type == "creative_generation":
        return "task_execution"
    if lowered in AMBIGUOUS_FOLLOW_UPS or intent_type == "ambiguous":
        return "follow_up"

    if decision_phase == "support":
        if has_phrase(text, PHRASE_PATTERNS["billing_help"]):
            return "billing_help"
        if has_phrase(text, PHRASE_PATTERNS["account_help"]):
            return "account_help"
        return "troubleshooting"

    if decision_phase == "post_purchase" or has_phrase(text, PHRASE_PATTERNS["onboarding_setup"]):
        if has_phrase(text, PHRASE_PATTERNS["download"]):
            return "download"
        return "onboarding_setup"

    if has_phrase(text, PHRASE_PATTERNS["billing_help"]):
        return "billing_help"
    if has_phrase(text, PHRASE_PATTERNS["troubleshooting"]):
        return "troubleshooting"
    if has_phrase(text, PHRASE_PATTERNS["account_help"]):
        return "account_help"

    if intent_type == "transactional" or decision_phase == "action":
        if has_phrase(text, PHRASE_PATTERNS["download"]):
            return "download"
        if has_phrase(text, PHRASE_PATTERNS["contact_sales"]):
            return "contact_sales"
        if has_phrase(text, PHRASE_PATTERNS["booking"]):
            return "booking"
        if has_phrase(text, PHRASE_PATTERNS["purchase"]):
            return "purchase"
        if has_phrase(text, PHRASE_PATTERNS["signup"]):
            return "signup"
        return "task_execution"

    if intent_type == "commercial" or decision_phase in {"consideration", "decision"}:
        if has_phrase(text, PHRASE_PATTERNS["comparison"]):
            return "comparison"
        if has_phrase(text, PHRASE_PATTERNS["deal_seeking"]):
            return "deal_seeking"
        if decision_phase == "decision" or has_phrase(text, PHRASE_PATTERNS["provider_selection"]):
            return "provider_selection"
        if has_phrase(text, PHRASE_PATTERNS["product_discovery"]):
            return "product_discovery"
        return "evaluation"

    if intent_type == "exploratory" or decision_phase == "research":
        if has_phrase(text, PHRASE_PATTERNS["comparison"]):
            return "comparison"
        if has_phrase(text, PHRASE_PATTERNS["product_discovery"]):
            return "product_discovery"
        return "evaluation"

    return "education"


def merge_split(split_name: str) -> list[dict]:
    merged: dict[str, dict] = {}
    for source_name, source_paths in (
        ("intent_type", INTENT_SOURCE_PATHS),
        ("decision_phase", PHASE_SOURCE_PATHS),
    ):
        for row in load_jsonl(source_paths[split_name]):
            item = merged.setdefault(
                row["text"],
                {
                    "text": row["text"],
                    "intent_subtype": None,
                    "source_heads": [],
                    "source_splits": [],
                },
            )
            item["source_heads"].append(source_name)
            item["source_splits"].append(split_name)
            if "intent_type" in row:
                item["intent_type"] = row["intent_type"]
            if "decision_phase" in row:
                item["decision_phase"] = row["decision_phase"]

    rows: list[dict] = []
    for text in sorted(merged):
        row = merged[text]
        row["source_heads"] = sorted(set(row["source_heads"]))
        row["source_splits"] = sorted(set(row["source_splits"]))
        row["intent_subtype"] = infer_subtype(
            text=row["text"],
            intent_type=row.get("intent_type"),
            decision_phase=row.get("decision_phase"),
        )
        rows.append(row)

    if split_name == "train":
        for seed in MANUAL_TRAIN_ROWS:
            rows.append(
                {
                    "text": seed["text"],
                    "intent_subtype": seed["intent_subtype"],
                    "source_heads": ["manual_seed"],
                    "source_splits": ["train"],
                }
            )
        rows = sorted(rows, key=lambda item: item["text"].lower())
    return rows


def main() -> None:
    split_names = ("train", "val", "test", "hard_cases", "extended_cases")
    summary = {}
    for split_name in split_names:
        rows = merge_split(split_name)
        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        write_jsonl(output_path, rows)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["intent_subtype"]] = counts.get(row["intent_subtype"], 0) + 1
        summary[split_name] = {
            "count": len(rows),
            "output_path": str(output_path),
            "subtype_counts": dict(sorted(counts.items())),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
