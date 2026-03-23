from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import IAB_BENCHMARK_PATH, IAB_DIFFICULTY_DATA_DIR


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def shopping_prompts(fields: dict[str, str]) -> dict[str, tuple[str, ...]]:
    return {
        "easy": (
            f"best {fields['item_plural']}",
            f"which {fields['item']} should i buy in {fields['year']}",
            f"{fields['provider_a']} vs {fields['provider_b']}",
            f"{fields['item']} buying guide",
        ),
        "medium": (
            f"best {fields['item']} for {fields['audience']}",
            f"compare {fields['provider_a']} and {fields['provider_b']} before buying",
            f"affordable {fields['item_plural']} for {fields['audience']}",
            f"what {fields['item_plural']} are worth considering for {fields['audience']}",
        ),
        "hard": (
            f"i am replacing my current {fields['item']} and need the right option for {fields['audience']}",
            f"help me narrow down {fields['item_plural']} for {fields['audience']} without wasting money",
            f"which option makes more sense between {fields['provider_a']} and {fields['provider_b']} for {fields['audience']}",
            f"i need a shortlist of {fields['item_plural']} that fit {fields['constraint']}",
        ),
    }


def software_prompts(fields: dict[str, str]) -> dict[str, tuple[str, ...]]:
    return {
        "easy": (
            f"best {fields['item_plural']} for {fields['audience']}",
            f"what is {fields['item']}",
            f"{fields['provider_a']} vs {fields['provider_b']}",
            f"{fields['item']} for {fields['goal']}",
        ),
        "medium": (
            f"compare {fields['provider_a']} and {fields['provider_b']} for {fields['audience']}",
            f"best {fields['item_plural']} for {fields['goal']}",
            f"how does {fields['item']} work for {fields['audience']}",
            f"which {fields['item']} should a {fields['audience']} choose",
        ),
        "hard": (
            f"i am evaluating software for {fields['goal']} and need the right category",
            f"what tools should i shortlist before picking between {fields['provider_a']} and {fields['provider_b']}",
            f"we need a platform for {fields['goal']} and are not sure which branch this falls into",
            f"help me assess {fields['provider_a']} versus other options for {fields['audience']}",
        ),
    }


def business_it_prompts(fields: dict[str, str]) -> dict[str, tuple[str, ...]]:
    return {
        "easy": (
            "how do i reset my password",
            "business login security tools",
            "identity management software",
            f"{fields['provider_a']} vs {fields['provider_b']} for access management",
        ),
        "medium": (
            "best software for employee password resets",
            "how does single sign-on work for a small company",
            "access management platform for remote employees",
            f"compare {fields['provider_a']} and {fields['provider_b']} for workforce identity",
        ),
        "hard": (
            "our team keeps getting locked out and we need better identity software",
            "what category covers employee account security and access provisioning",
            "we need business software for login, permissions, and access control",
            "help me evaluate identity tooling for company account security",
        ),
    }


def dining_prompts(fields: dict[str, str]) -> dict[str, tuple[str, ...]]:
    return {
        "easy": (
            "book a table for dinner",
            "best restaurants for date night",
            "where should i eat tonight",
            "reserve a table for two",
        ),
        "medium": (
            f"{fields['area']} restaurant options for a birthday dinner",
            "family friendly restaurants near me",
            "compare brunch spots for a weekend meetup",
            "where can i book dinner for four tonight",
        ),
        "hard": (
            "i need a place to eat and want something i can reserve tonight",
            "what category covers restaurants and booking a table",
            "help me find a dinner spot for a client meeting",
            "i want dining options, not recipes",
        ),
    }


def beverage_prompts(fields: dict[str, str]) -> dict[str, tuple[str, ...]]:
    return {
        "easy": (
            "best vodka drink to try",
            "whiskey cocktail ideas",
            "what is a martini",
            "bourbon vs rye for beginners",
        ),
        "medium": (
            "best whiskey cocktail for a dinner party",
            "vodka drinks for beginners",
            "compare bourbon and scotch flavor profiles",
            "how does gin differ from vodka in cocktails",
        ),
        "hard": (
            "i want alcoholic drink recommendations, not restaurant suggestions",
            "help me understand beginner-friendly cocktails with bourbon",
            "what should i try if i want a spirit-forward drink",
            "compare vodka cocktails with tequila cocktails",
        ),
    }


def ai_prompts(fields: dict[str, str]) -> dict[str, tuple[str, ...]]:
    return {
        "easy": (
            "what is intent classification in nlp",
            "machine learning basics",
            "how does natural language processing work",
            "what are large language models",
        ),
        "medium": (
            "best ai methods for text classification",
            "nlp model comparison for intent detection",
            "how do llms handle classification tasks",
            "ai tools for labeling text data",
        ),
        "hard": (
            "i want the ai concept behind intent models, not software shopping",
            "help me understand the machine learning side of nlp classification",
            "compare transformer-based approaches for intent detection",
            "what branch covers language-model research topics",
        ),
    }


KIND_TO_BUILDER = {
    "shopping": shopping_prompts,
    "software": software_prompts,
    "business_it": business_it_prompts,
    "dining": dining_prompts,
    "beverage": beverage_prompts,
    "ai": ai_prompts,
}


AUGMENTATION_SCENARIOS = {
    "Automotive > Auto Buying and Selling": [
        {
            "kind": "shopping",
            "item": "car",
            "item_plural": "cars",
            "provider_a": "Toyota Corolla",
            "provider_b": "Honda Civic",
            "audience": "a commuter",
            "constraint": "a practical budget",
            "year": "2026",
        },
        {
            "kind": "shopping",
            "item": "suv",
            "item_plural": "suvs",
            "provider_a": "Toyota RAV4",
            "provider_b": "Honda CR-V",
            "audience": "a growing family",
            "constraint": "daily driving and storage needs",
            "year": "2026",
        },
        {
            "kind": "shopping",
            "item": "electric car",
            "item_plural": "electric cars",
            "provider_a": "Tesla Model 3",
            "provider_b": "Hyundai Ioniq 5",
            "audience": "a first-time ev buyer",
            "constraint": "reasonable range and price",
            "year": "2026",
        },
    ],
    "Business and Finance > Business > Sales": [
        {
            "kind": "software",
            "item": "crm software",
            "item_plural": "crm tools",
            "provider_a": "HubSpot",
            "provider_b": "Zoho",
            "audience": "small sales teams",
            "goal": "lead management",
        },
        {
            "kind": "software",
            "item": "sales engagement software",
            "item_plural": "sales platforms",
            "provider_a": "Apollo",
            "provider_b": "Outreach",
            "audience": "outbound teams",
            "goal": "pipeline generation",
        },
        {
            "kind": "software",
            "item": "customer relationship management software",
            "item_plural": "crm systems",
            "provider_a": "Pipedrive",
            "provider_b": "Freshsales",
            "audience": "growing startups",
            "goal": "deal tracking",
        },
    ],
    "Business and Finance > Business > Marketing and Advertising": [
        {
            "kind": "software",
            "item": "marketing software",
            "item_plural": "marketing tools",
            "provider_a": "Semrush",
            "provider_b": "Ahrefs",
            "audience": "content teams",
            "goal": "organic growth",
        },
        {
            "kind": "software",
            "item": "seo platform",
            "item_plural": "seo tools",
            "provider_a": "Surfer",
            "provider_b": "Clearscope",
            "audience": "editorial teams",
            "goal": "content optimization",
        },
        {
            "kind": "software",
            "item": "advertising analytics software",
            "item_plural": "marketing analytics tools",
            "provider_a": "Triple Whale",
            "provider_b": "Northbeam",
            "audience": "performance marketers",
            "goal": "campaign measurement",
        },
    ],
    "Business and Finance > Business > Business I.T.": [
        {"kind": "business_it", "provider_a": "Okta", "provider_b": "Microsoft Entra"},
        {"kind": "business_it", "provider_a": "JumpCloud", "provider_b": "Okta"},
        {"kind": "business_it", "provider_a": "Duo", "provider_b": "OneLogin"},
    ],
    "Food & Drink > Dining Out": [
        {"kind": "dining", "area": "downtown"},
        {"kind": "dining", "area": "midtown"},
        {"kind": "dining", "area": "the waterfront"},
    ],
    "Food & Drink > Alcoholic Beverages": [
        {"kind": "beverage"},
        {"kind": "beverage"},
        {"kind": "beverage"},
    ],
    "Technology & Computing > Artificial Intelligence": [
        {"kind": "ai"},
        {"kind": "ai"},
        {"kind": "ai"},
    ],
    "Technology & Computing > Computing > Computer Software and Applications": [
        {
            "kind": "software",
            "item": "software platform",
            "item_plural": "software applications",
            "provider_a": "Notion",
            "provider_b": "Airtable",
            "audience": "operations teams",
            "goal": "workflow management",
        },
        {
            "kind": "software",
            "item": "project management software",
            "item_plural": "software tools",
            "provider_a": "Asana",
            "provider_b": "ClickUp",
            "audience": "remote teams",
            "goal": "project planning",
        },
        {
            "kind": "software",
            "item": "business software",
            "item_plural": "software products",
            "provider_a": "Monday.com",
            "provider_b": "Notion",
            "audience": "startup operators",
            "goal": "team coordination",
        },
    ],
    "Technology & Computing > Computing > Computer Software and Applications > Communication": [
        {
            "kind": "software",
            "item": "communication software",
            "item_plural": "communication tools",
            "provider_a": "Slack",
            "provider_b": "Microsoft Teams",
            "audience": "remote teams",
            "goal": "team communication",
        },
        {
            "kind": "software",
            "item": "team chat software",
            "item_plural": "messaging platforms",
            "provider_a": "Slack",
            "provider_b": "Discord",
            "audience": "distributed startups",
            "goal": "internal collaboration",
        },
        {
            "kind": "software",
            "item": "workplace communication platform",
            "item_plural": "communication apps",
            "provider_a": "Google Chat",
            "provider_b": "Microsoft Teams",
            "audience": "cross-functional teams",
            "goal": "company messaging",
        },
    ],
    "Technology & Computing > Computing > Internet > Web Hosting": [
        {
            "kind": "software",
            "item": "web hosting",
            "item_plural": "hosting providers",
            "provider_a": "Vercel",
            "provider_b": "Netlify",
            "audience": "startup launch teams",
            "goal": "site hosting",
        },
        {
            "kind": "software",
            "item": "hosting platform",
            "item_plural": "hosting services",
            "provider_a": "Cloudflare Pages",
            "provider_b": "Render",
            "audience": "developers",
            "goal": "website deployment",
        },
        {
            "kind": "software",
            "item": "managed hosting",
            "item_plural": "hosting options",
            "provider_a": "WP Engine",
            "provider_b": "Kinsta",
            "audience": "content teams",
            "goal": "site performance",
        },
    ],
    "Technology & Computing > Computing > Laptops": [
        {
            "kind": "shopping",
            "item": "laptop",
            "item_plural": "laptops",
            "provider_a": "MacBook Air",
            "provider_b": "Dell XPS 13",
            "audience": "work and study",
            "constraint": "battery life and portability",
            "year": "2026",
        },
        {
            "kind": "shopping",
            "item": "gaming laptop",
            "item_plural": "gaming laptops",
            "provider_a": "Asus ROG Zephyrus",
            "provider_b": "Lenovo Legion Slim",
            "audience": "gamers",
            "constraint": "performance under a reasonable budget",
            "year": "2026",
        },
        {
            "kind": "shopping",
            "item": "student laptop",
            "item_plural": "student laptops",
            "provider_a": "Acer Swift Go",
            "provider_b": "HP Pavilion Aero",
            "audience": "college students",
            "constraint": "price and portability",
            "year": "2026",
        },
    ],
    "Technology & Computing > Computing > Desktops": [
        {
            "kind": "shopping",
            "item": "desktop",
            "item_plural": "desktops",
            "provider_a": "iMac",
            "provider_b": "Dell Inspiron Desktop",
            "audience": "home offices",
            "constraint": "everyday productivity",
            "year": "2026",
        },
        {
            "kind": "shopping",
            "item": "gaming desktop",
            "item_plural": "gaming desktops",
            "provider_a": "Alienware Aurora",
            "provider_b": "Lenovo Legion Tower",
            "audience": "pc gamers",
            "constraint": "strong graphics performance",
            "year": "2026",
        },
        {
            "kind": "shopping",
            "item": "desktop pc",
            "item_plural": "desktop computers",
            "provider_a": "HP Envy Desktop",
            "provider_b": "Acer Aspire TC",
            "audience": "families",
            "constraint": "value for money",
            "year": "2026",
        },
    ],
    "Technology & Computing > Consumer Electronics > Smartphones": [
        {
            "kind": "shopping",
            "item": "smartphone",
            "item_plural": "smartphones",
            "provider_a": "iPhone 17",
            "provider_b": "Samsung Galaxy S26",
            "audience": "everyday users",
            "constraint": "camera quality and battery life",
            "year": "2026",
        },
        {
            "kind": "shopping",
            "item": "budget phone",
            "item_plural": "budget smartphones",
            "provider_a": "Pixel 10a",
            "provider_b": "Galaxy A57",
            "audience": "budget-conscious buyers",
            "constraint": "under midrange pricing",
            "year": "2026",
        },
        {
            "kind": "shopping",
            "item": "android phone",
            "item_plural": "android phones",
            "provider_a": "OnePlus 15",
            "provider_b": "Pixel 10",
            "audience": "power users",
            "constraint": "performance and clean software",
            "year": "2026",
        },
    ],
}


BENCHMARK_SCENARIOS = {
    "Automotive > Auto Buying and Selling": {
        "kind": "shopping",
        "item": "car",
        "item_plural": "vehicles",
        "provider_a": "Mazda CX-5",
        "provider_b": "Subaru Forester",
        "audience": "a first-time buyer",
        "constraint": "safety and price",
        "year": "2026",
    },
    "Business and Finance > Business > Sales": {
        "kind": "software",
        "item": "crm platform",
        "item_plural": "sales tools",
        "provider_a": "Copper",
        "provider_b": "Salesforce Essentials",
        "audience": "small revenue teams",
        "goal": "managing leads",
    },
    "Business and Finance > Business > Marketing and Advertising": {
        "kind": "software",
        "item": "marketing platform",
        "item_plural": "marketing tools",
        "provider_a": "Moz",
        "provider_b": "SE Ranking",
        "audience": "brand teams",
        "goal": "search visibility",
    },
    "Business and Finance > Business > Business I.T.": {
        "kind": "business_it",
        "provider_a": "Rippling",
        "provider_b": "JumpCloud",
    },
    "Food & Drink > Dining Out": {"kind": "dining", "area": "uptown"},
    "Food & Drink > Alcoholic Beverages": {"kind": "beverage"},
    "Technology & Computing > Artificial Intelligence": {"kind": "ai"},
    "Technology & Computing > Computing > Computer Software and Applications": {
        "kind": "software",
        "item": "workflow software",
        "item_plural": "productivity apps",
        "provider_a": "Basecamp",
        "provider_b": "Asana",
        "audience": "small teams",
        "goal": "organizing work",
    },
    "Technology & Computing > Computing > Computer Software and Applications > Communication": {
        "kind": "software",
        "item": "communication platform",
        "item_plural": "team messaging tools",
        "provider_a": "Mattermost",
        "provider_b": "Slack",
        "audience": "engineering teams",
        "goal": "workplace communication",
    },
    "Technology & Computing > Computing > Internet > Web Hosting": {
        "kind": "software",
        "item": "web hosting service",
        "item_plural": "hosting platforms",
        "provider_a": "Fly.io",
        "provider_b": "Render",
        "audience": "product builders",
        "goal": "deploying websites",
    },
    "Technology & Computing > Computing > Laptops": {
        "kind": "shopping",
        "item": "laptop",
        "item_plural": "portable computers",
        "provider_a": "Surface Laptop",
        "provider_b": "Framework Laptop",
        "audience": "knowledge workers",
        "constraint": "portability and repairability",
        "year": "2026",
    },
    "Technology & Computing > Computing > Desktops": {
        "kind": "shopping",
        "item": "desktop computer",
        "item_plural": "desktop pcs",
        "provider_a": "Mac Studio",
        "provider_b": "HP Omen 45L",
        "audience": "creators",
        "constraint": "performance and reliability",
        "year": "2026",
    },
    "Technology & Computing > Consumer Electronics > Smartphones": {
        "kind": "shopping",
        "item": "smartphone",
        "item_plural": "mobile phones",
        "provider_a": "Nothing Phone 4",
        "provider_b": "Pixel 10 Pro",
        "audience": "everyday buyers",
        "constraint": "camera and battery performance",
        "year": "2026",
    },
}


def build_rows(label: str, scenarios: list[dict], include_difficulty: bool) -> list[dict]:
    rows = []
    seen = set()
    for scenario in scenarios:
        prompts_by_difficulty = KIND_TO_BUILDER[scenario["kind"]](scenario)
        for difficulty, prompts in prompts_by_difficulty.items():
            for text in prompts:
                normalized = " ".join(text.strip().lower().split())
                key = (label, normalized)
                if key in seen:
                    continue
                seen.add(key)
                row = {"text": normalized, "iab_path": label}
                if include_difficulty:
                    row["difficulty"] = difficulty
                rows.append(row)
    return rows


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    total = len(rows)
    val_count = max(1, total // 6)
    test_count = max(1, total // 6)
    test_rows = rows[:test_count]
    val_rows = rows[test_count : test_count + val_count]
    train_rows = rows[test_count + val_count :]
    return train_rows, val_rows, test_rows


def main() -> None:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    benchmark_rows: list[dict] = []

    for label, scenarios in AUGMENTATION_SCENARIOS.items():
        rows = build_rows(label, scenarios, include_difficulty=True)
        train_split, val_split, test_split = split_rows(rows)
        train_rows.extend(train_split)
        val_rows.extend(val_split)
        test_rows.extend(test_split)

    for label, scenario in BENCHMARK_SCENARIOS.items():
        benchmark_rows.extend(build_rows(label, [scenario], include_difficulty=True))

    write_jsonl(IAB_DIFFICULTY_DATA_DIR / "train.jsonl", train_rows)
    write_jsonl(IAB_DIFFICULTY_DATA_DIR / "val.jsonl", val_rows)
    write_jsonl(IAB_DIFFICULTY_DATA_DIR / "test.jsonl", test_rows)
    write_jsonl(IAB_BENCHMARK_PATH, benchmark_rows)

    print(f"train: {len(train_rows)} rows")
    print(f"val: {len(val_rows)} rows")
    print(f"test: {len(test_rows)} rows")
    print(f"benchmark: {len(benchmark_rows)} rows")


if __name__ == "__main__":
    main()
