from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "decision_phase_difficulty"
BENCHMARK_PATH = BASE_DIR / "data" / "decision_phase_benchmark.jsonl"

TRAIN_THEMES = (
    {
        "topic": "crm software",
        "product": "CRM",
        "products": "CRM tools",
        "provider_a": "HubSpot",
        "provider_b": "Zoho",
        "domain": "a small sales team",
        "goal": "manage leads better",
        "support_object": "account",
        "support_detail": "password reset",
        "asset": "CRM onboarding guide",
    },
    {
        "topic": "analytics software",
        "product": "analytics platform",
        "products": "analytics platforms",
        "provider_a": "Mixpanel",
        "provider_b": "Amplitude",
        "domain": "a product team",
        "goal": "measure activation",
        "support_object": "dashboard",
        "support_detail": "data sync issue",
        "asset": "analytics setup guide",
    },
    {
        "topic": "laptops",
        "product": "laptop",
        "products": "laptops",
        "provider_a": "MacBook Air",
        "provider_b": "Dell XPS 13",
        "domain": "college work",
        "goal": "choose the right laptop",
        "support_object": "order",
        "support_detail": "delivery delay",
        "asset": "laptop buying checklist",
    },
)

BENCHMARK_THEMES = {
    "easy": {
        "topic": "help desk software",
        "product": "help desk platform",
        "products": "help desk tools",
        "provider_a": "Zendesk",
        "provider_b": "Freshdesk",
        "domain": "a support team",
        "goal": "handle tickets faster",
        "support_object": "billing portal",
        "support_detail": "invoice issue",
        "asset": "help desk buyer guide",
    },
    "medium": {
        "topic": "cars",
        "product": "car",
        "products": "cars",
        "provider_a": "Toyota Corolla",
        "provider_b": "Honda Civic",
        "domain": "daily commuting",
        "goal": "choose the right car",
        "support_object": "reservation",
        "support_detail": "test drive booking",
        "asset": "car buying worksheet",
    },
    "hard": {
        "topic": "hosting platforms",
        "product": "hosting platform",
        "products": "hosting providers",
        "provider_a": "Vercel",
        "provider_b": "Netlify",
        "domain": "a startup launch",
        "goal": "ship a new website",
        "support_object": "deployment",
        "support_detail": "domain setup problem",
        "asset": "hosting migration guide",
    },
}

PHASE_TEMPLATES = {
    "awareness": {
        "easy": (
            "What is {topic}?",
            "Explain {topic}.",
            "How does {product} work?",
            "What does {provider_a} do?",
            "Give me the basics of {topic}.",
        ),
        "medium": (
            "Help me understand what problem {provider_a} solves.",
            "What should a beginner know about {products}?",
            "Before I look at options, what is a {product}?",
            "What is the purpose of {topic} in {domain}?",
            "What does a {product} actually help with?",
        ),
        "hard": (
            "I am not shopping yet, I just want to understand what {topic} is.",
            "Before I evaluate anything, what role does {topic} play in {domain}?",
            "I keep hearing about {provider_a}; what is it actually for?",
            "Can you clarify what people mean by {topic} in practice?",
            "I only need an overview of {topic} right now.",
        ),
    },
    "research": {
        "easy": (
            "What {products} should I explore for {domain}?",
            "Show me options to consider for {goal}.",
            "What tools should I look at for {domain}?",
            "Help me research {products}.",
            "Where should I start with {products}?",
        ),
        "medium": (
            "I am early in the process and want to explore {products}.",
            "Give me a shortlist of {products} worth researching.",
            "What directions should I investigate for {goal}?",
            "What categories should I look at before narrowing down?",
            "What are some promising {products} for {domain}?",
        ),
        "hard": (
            "I am not ready to compare vendors yet, just help me scope the market.",
            "What should I research first if I am only beginning to look at {products}?",
            "I need a landscape view before I make a shortlist.",
            "What are the main options in this space before I decide anything?",
            "Help me map the market for {products} without recommending one yet.",
        ),
    },
    "consideration": {
        "easy": (
            "Best {product} for {domain}.",
            "{provider_a} vs {provider_b}.",
            "Compare {products} for {goal}.",
            "Which {product} looks best for {domain}?",
            "What are some {products} worth considering?",
        ),
        "medium": (
            "Compare {provider_a} and {provider_b} for {goal}.",
            "What are the pros and cons of {provider_a}?",
            "Help me evaluate the best {products} for {domain}.",
            "Which {product} seems worth considering right now?",
            "I am comparing options for {goal}; what should be on the shortlist?",
        ),
        "hard": (
            "I am past basic research and now weighing tradeoffs between {provider_a} and {provider_b}.",
            "I want to compare serious options before committing to one.",
            "Help me think through the tradeoffs in the current shortlist.",
            "What looks strongest if I am narrowing down to a few options?",
            "I have done research, now help me compare the finalists.",
        ),
    },
    "decision": {
        "easy": (
            "Which {product} should I choose?",
            "Should I pick {provider_a} or {provider_b}?",
            "Which option should I commit to?",
            "What is the best fit for me right now?",
            "Which plan should I choose today?",
        ),
        "medium": (
            "I am ready to decide between {provider_a} and {provider_b}.",
            "Help me pick the final option for {goal}.",
            "Which {product} should I commit to this week?",
            "I need to make the call now; which option fits best?",
            "What should I choose if I need to decide today?",
        ),
        "hard": (
            "I have a shortlist and need to commit to one vendor now.",
            "I am at the point of commitment and need a final recommendation.",
            "Which option should we sign off on before next week?",
            "I have enough information; tell me which one to go with.",
            "I need the final pick, not another round of comparison.",
        ),
    },
    "action": {
        "easy": (
            "Start my free trial.",
            "Book a demo with {provider_a}.",
            "Create my account.",
            "Buy {provider_a} now.",
            "Download the {asset}.",
        ),
        "medium": (
            "Take me to checkout for {provider_a}.",
            "Get me signed up for {provider_a}.",
            "Reserve my spot with {provider_a}.",
            "I want to purchase {provider_a} today.",
            "Send me the download link for the {asset}.",
        ),
        "hard": (
            "I am ready to move forward now, where do I start the purchase?",
            "Help me complete the signup flow for {provider_a}.",
            "I want to act on this immediately and get access now.",
            "Can you help me finish the order for {provider_a}?",
            "I have decided, now let me complete the next step.",
        ),
    },
    "post_purchase": {
        "easy": (
            "How do I set up my new {product}?",
            "Show me how to import contacts into {provider_a}.",
            "How do I onboard my team after purchase?",
            "What should I enable first after signup?",
            "How do I configure my account now that I signed up?",
        ),
        "medium": (
            "We already subscribed; how do we get value quickly?",
            "What is the best way to roll this out after purchase?",
            "Help me configure {provider_a} now that we bought it.",
            "How do I invite teammates after signing up?",
            "What should I do first after we activate the plan?",
        ),
        "hard": (
            "We already made the purchase, now I need guidance on rollout and setup.",
            "This is not a buying decision anymore; I need post-purchase onboarding help.",
            "I need adoption guidance now that the contract is signed.",
            "What is the right onboarding sequence after we commit to {provider_a}?",
            "We are past checkout and need implementation help.",
        ),
    },
    "support": {
        "easy": (
            "I cannot log into my {support_object}.",
            "How do I reset my password?",
            "My invoice is wrong.",
            "The integration keeps failing.",
            "Our dashboard is not loading.",
        ),
        "medium": (
            "Can you help me fix a {support_detail}?",
            "I am stuck because my {support_object} keeps breaking.",
            "My password reset link is not working.",
            "I need support with my {support_object}.",
            "Why is {provider_a} not syncing correctly?",
        ),
        "hard": (
            "I am not evaluating anything, I just need this issue fixed.",
            "This is a live support problem, not a buying question.",
            "Please help me resolve a problem with my existing account.",
            "I cannot continue because something is broken in my setup.",
            "I need troubleshooting help, not recommendations.",
        ),
    },
}


def split_for_index(index: int) -> str:
    bucket = index % 5
    if bucket < 3:
        return "train"
    if bucket == 3:
        return "val"
    return "test"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def render_text(template: str, spec: dict[str, str]) -> str:
    return template.format(**spec).strip()


def build_training_splits() -> dict[str, list[dict]]:
    splits = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    counters = {(phase, difficulty): 0 for phase in PHASE_TEMPLATES for difficulty in PHASE_TEMPLATES[phase]}

    for phase, difficulty_map in PHASE_TEMPLATES.items():
        for difficulty, templates in difficulty_map.items():
            for template in templates:
                theme_specs = TRAIN_THEMES if "{" in template else ({},)
                for spec in theme_specs:
                    text = render_text(template, spec)
                    key = text.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    split_name = split_for_index(counters[(phase, difficulty)])
                    counters[(phase, difficulty)] += 1
                    splits[split_name].append(
                        {
                            "text": text,
                            "decision_phase": phase,
                            "difficulty": difficulty,
                            "source": "synthetic_decision_phase_difficulty",
                        }
                    )
    return splits


def build_benchmark_rows() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for phase, difficulty_map in PHASE_TEMPLATES.items():
        for difficulty, templates in difficulty_map.items():
            spec = BENCHMARK_THEMES.get(difficulty, {})
            for template in templates:
                text = render_text(template, spec)
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "text": text,
                        "decision_phase": phase,
                        "difficulty": difficulty,
                        "source": "decision_phase_benchmark",
                    }
                )
    return rows


def main() -> None:
    splits = build_training_splits()
    for split_name, rows in splits.items():
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)
        print(f"{split_name}: {len(rows)} rows")

    benchmark_rows = build_benchmark_rows()
    write_jsonl(BENCHMARK_PATH, benchmark_rows)
    print(f"benchmark: {len(benchmark_rows)} rows")


if __name__ == "__main__":
    main()
