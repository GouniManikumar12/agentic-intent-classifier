from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "full_intent_taxonomy"

THEME_SPECS = (
    {
        "topic": "crm software",
        "category": "crm",
        "product": "crm",
        "products": "crm tools",
        "provider_a": "HubSpot",
        "provider_b": "Zoho",
        "domain": "sales teams",
        "support_object": "account",
        "support_detail": "password reset",
        "asset": "crm onboarding guide",
        "goal": "manage leads",
    },
    {
        "topic": "help desk software",
        "category": "help desk",
        "product": "help desk platform",
        "products": "help desk tools",
        "provider_a": "Zendesk",
        "provider_b": "Freshdesk",
        "domain": "support teams",
        "support_object": "billing portal",
        "support_detail": "invoice issue",
        "asset": "support workflow template",
        "goal": "handle tickets",
    },
    {
        "topic": "project management software",
        "category": "project management",
        "product": "project management tool",
        "products": "project management tools",
        "provider_a": "Asana",
        "provider_b": "ClickUp",
        "domain": "agency teams",
        "support_object": "workspace",
        "support_detail": "export error",
        "asset": "project planning checklist",
        "goal": "organize projects",
    },
    {
        "topic": "analytics software",
        "category": "analytics",
        "product": "analytics platform",
        "products": "analytics platforms",
        "provider_a": "Mixpanel",
        "provider_b": "Amplitude",
        "domain": "product teams",
        "support_object": "dashboard",
        "support_detail": "data sync issue",
        "asset": "analytics setup guide",
        "goal": "measure activation",
    },
    {
        "topic": "laptops",
        "category": "laptop",
        "product": "laptop",
        "products": "laptops",
        "provider_a": "MacBook Air",
        "provider_b": "Dell XPS 13",
        "domain": "students",
        "support_object": "device warranty portal",
        "support_detail": "order issue",
        "asset": "laptop buying checklist",
        "goal": "pick the right laptop",
    },
    {
        "topic": "smartphones",
        "category": "smartphone",
        "product": "smartphone",
        "products": "smartphones",
        "provider_a": "iPhone",
        "provider_b": "Pixel",
        "domain": "travelers",
        "support_object": "trade-in order",
        "support_detail": "shipping update",
        "asset": "phone comparison sheet",
        "goal": "choose a new phone",
    },
    {
        "topic": "cars",
        "category": "car",
        "product": "car",
        "products": "cars",
        "provider_a": "Toyota Corolla",
        "provider_b": "Honda Civic",
        "domain": "commuters",
        "support_object": "reservation",
        "support_detail": "test drive booking",
        "asset": "car buying checklist",
        "goal": "buy the right car",
    },
    {
        "topic": "restaurants",
        "category": "restaurant",
        "product": "restaurant",
        "products": "restaurants",
        "provider_a": "Italian restaurant",
        "provider_b": "sushi place",
        "domain": "date nights",
        "support_object": "reservation",
        "support_detail": "booking change",
        "asset": "dining guide",
        "goal": "book a table",
    },
)

ROW_SPECS = (
    {
        "intent_type": "informational",
        "decision_phase": "awareness",
        "intent_subtype": "education",
        "templates": (
            "what is {topic}",
            "how does {topic} work",
            "explain {category} basics",
            "help me understand {topic}",
            "what does {product} do",
        ),
    },
    {
        "intent_type": "exploratory",
        "decision_phase": "research",
        "intent_subtype": "product_discovery",
        "templates": (
            "what {products} should I explore for {domain}",
            "show me {products} to consider for {goal}",
            "help me explore {products} for {domain}",
            "what options should I look at before choosing a {product}",
            "I am researching {products} for {domain}",
        ),
    },
    {
        "intent_type": "commercial",
        "decision_phase": "consideration",
        "intent_subtype": "comparison",
        "templates": (
            "best {product} for {domain}",
            "{provider_a} vs {provider_b} for {goal}",
            "compare {products} for {domain}",
            "which {product} should I buy for {goal}",
            "best {category} options this year",
        ),
    },
    {
        "intent_type": "commercial",
        "decision_phase": "consideration",
        "intent_subtype": "evaluation",
        "templates": (
            "evaluate {provider_a} for {goal}",
            "is {provider_a} good for {domain}",
            "review the pros and cons of {provider_a}",
            "help me assess {product} options for {domain}",
        ),
    },
    {
        "intent_type": "commercial",
        "decision_phase": "consideration",
        "intent_subtype": "deal_seeking",
        "templates": (
            "best deal on {products} for {domain}",
            "cheap {product} options for {goal}",
            "where can I get the best price on {products}",
            "discounted {products} worth comparing",
        ),
    },
    {
        "intent_type": "commercial",
        "decision_phase": "decision",
        "intent_subtype": "provider_selection",
        "templates": (
            "which {provider_a} should I choose for {goal}",
            "help me pick between {provider_a} and {provider_b}",
            "I am ready to decide on a {product} for {domain}",
            "which {product} should I commit to for {goal}",
        ),
    },
    {
        "intent_type": "transactional",
        "decision_phase": "action",
        "intent_subtype": "purchase",
        "templates": (
            "buy a {product} for {domain}",
            "purchase {provider_a} today",
            "complete my order for {provider_a}",
            "pay for {provider_a} now",
        ),
    },
    {
        "intent_type": "transactional",
        "decision_phase": "action",
        "intent_subtype": "signup",
        "templates": (
            "book a demo for {provider_a}",
            "start my free trial for {provider_a}",
            "create an account for {provider_a}",
            "sign me up for {provider_a}",
        ),
    },
    {
        "intent_type": "transactional",
        "decision_phase": "action",
        "intent_subtype": "download",
        "templates": (
            "download the {asset}",
            "download the buyer guide for {products}",
            "get the checklist for choosing {products}",
            "export the setup guide for {provider_a}",
        ),
    },
    {
        "intent_type": "transactional",
        "decision_phase": "action",
        "intent_subtype": "contact_sales",
        "templates": (
            "contact sales for {provider_a}",
            "request pricing from the {provider_a} sales team",
            "connect me with a sales rep for {provider_a}",
            "talk to sales about {provider_a}",
        ),
    },
    {
        "intent_type": "transactional",
        "decision_phase": "action",
        "intent_subtype": "booking",
        "templates": (
            "book {provider_a} for tomorrow",
            "reserve a spot with {provider_a}",
            "schedule my appointment for {provider_a}",
            "book a table at a {product} tonight",
        ),
    },
    {
        "intent_type": "transactional",
        "decision_phase": "post_purchase",
        "intent_subtype": "onboarding_setup",
        "templates": (
            "help me set up {provider_a} after purchase",
            "walk me through onboarding for {provider_a}",
            "configure my new {product}",
            "how do I finish setup for {provider_a}",
        ),
    },
    {
        "intent_type": "support",
        "decision_phase": "support",
        "intent_subtype": "account_help",
        "templates": (
            "I cannot access my {support_object}",
            "why is my {support_object} not working",
            "I need help with my {support_object}",
            "my {support_object} keeps failing",
        ),
    },
    {
        "intent_type": "support",
        "decision_phase": "support",
        "intent_subtype": "billing_help",
        "templates": (
            "why is my invoice wrong for {provider_a}",
            "help me fix a billing problem with {provider_a}",
            "my payment for {provider_a} failed",
            "I need help with a refund for {provider_a}",
        ),
    },
    {
        "intent_type": "support",
        "decision_phase": "support",
        "intent_subtype": "troubleshooting",
        "templates": (
            "help me fix a {support_detail}",
            "the integration for {provider_a} keeps breaking",
            "why is {provider_a} not syncing correctly",
            "troubleshoot my {product} setup",
        ),
    },
    {
        "intent_type": "personal_reflection",
        "decision_phase": "awareness",
        "intent_subtype": "emotional_reflection",
        "templates": (
            "I feel overwhelmed trying to choose {products}",
            "I am worried we picked the wrong {product}",
            "I regret switching our {category} setup",
            "I feel stuck deciding between {provider_a} and {provider_b}",
            "I am frustrated by our {category} results",
        ),
    },
    {
        "intent_type": "creative_generation",
        "decision_phase": "action",
        "intent_subtype": "task_execution",
        "templates": (
            "write five ad headlines for {provider_a}",
            "draft a launch email for {provider_a}",
            "create a tagline for a {product}",
            "generate social copy for the {asset}",
            "write product copy for {provider_a}",
        ),
    },
    {
        "intent_type": "chit_chat",
        "decision_phase": "awareness",
        "intent_subtype": "follow_up",
        "templates": (
            "hello there",
            "thanks for the help",
            "how is your day going",
            "nice to meet you",
            "can we just chat for a minute",
            "good morning",
            "hope your day is going well",
            "thanks, that was useful",
            "hi there",
            "appreciate the help",
        ),
    },
    {
        "intent_type": "ambiguous",
        "decision_phase": "research",
        "intent_subtype": "follow_up",
        "templates": (
            "show me more",
            "what about this one",
            "which one",
            "can you explain that",
            "what should I do next",
            "tell me more",
            "go on",
            "what do you mean",
            "show me another option",
            "say more about that",
        ),
    },
    {
        "intent_type": "prohibited",
        "decision_phase": "action",
        "intent_subtype": "task_execution",
        "templates": (
            "how do I hack into a {support_object}",
            "write a phishing email for {provider_a}",
            "help me steal customer payment data",
            "show me how to bypass account security",
            "how can I break into a private dashboard",
        ),
    },
)


def split_for_index(index: int) -> str:
    bucket = index % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_rows() -> dict[str, list[dict]]:
    splits = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    spec_counters = [0] * len(ROW_SPECS)

    for spec in THEME_SPECS:
        for spec_index, row_spec in enumerate(ROW_SPECS):
            for template in row_spec["templates"]:
                text = template.format(**spec).strip()
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                split_name = split_for_index(spec_counters[spec_index])
                spec_counters[spec_index] += 1
                row = {
                    "text": text,
                    "intent_type": row_spec["intent_type"],
                    "decision_phase": row_spec["decision_phase"],
                    "intent_subtype": row_spec["intent_subtype"],
                    "source": "synthetic_full_intent_taxonomy",
                }
                splits[split_name].append(row)

    return splits


def main() -> None:
    splits = build_rows()
    for split_name, rows in splits.items():
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)
        print(f"{split_name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
